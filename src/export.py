"""Export a trained checkpoint as a self-contained fp16 TorchScript module for run.py.

The traced module takes uint8 images (B, H, W, 3) and returns probabilities (B, 40) in
our ATTRIBUTES order, with normalisation, flip-TTA (logit average), calibrated offsets
(N4), sigmoid and structure decoding (N5) inside.

Usage: python src/export.py --run results/<run> [--ckpt best.pt] [--calib calib.json] [--no-flip]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from data import ATTRIBUTES, ROOT  # noqa: E402
from decode import EPS  # noqa: E402
from train import load_run_model  # noqa: E402


class InferenceModule(nn.Module):
    def __init__(self, net, offsets, groups, flip: bool):
        super().__init__()
        self.net = net
        self.register_buffer("offsets", torch.as_tensor(offsets, dtype=torch.float32))
        self.groups = [list(g) for g in groups]
        self.flip = flip

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).to(self.offsets.dtype)  # uint8 BHWC -> float BCHW in [0,255]
        z = self.net(x).float()
        if self.flip:
            z = (z + self.net(x.flip(3)).float()) / 2
        p = torch.sigmoid(z + self.offsets.float())
        for idx in self.groups:
            sub = p[:, idx]
            mx, arg = sub.max(1, keepdim=True)
            is_max = torch.zeros_like(sub, dtype=torch.bool).scatter_(1, arg, True)
            sub = torch.where((mx <= 0.5) & is_max, sub.clamp(min=0.5 + EPS), sub)
            p = torch.cat([p[:, :idx[0]], sub, p[:, idx[-1] + 1:]], 1)  # groups are contiguous
        return p


def export(run, ckpt="best.pt", calib_file="calib.json", flip=True, out_dir=None, half=True, use_calib=True):
    net, cfg = load_run_model(run, ckpt)
    offsets, groups = [0.0] * 40, []
    cpath = os.path.join(run, calib_file)
    if use_calib and os.path.exists(cpath):
        with open(cpath) as fh:
            c = json.load(fh)
        offsets, groups = c["offsets"], list(c["groups"].values())
        assert bool(c.get("flip", flip)) == flip, "calibration was fitted with a different flip-TTA setting"
    for g in groups:
        assert g == list(range(g[0], g[-1] + 1)), "decode groups must be contiguous"
    dev = torch.device("cuda")
    mod = InferenceModule(net, offsets, groups, flip).to(dev).eval()
    if half:
        mod.half()
    ex = torch.randint(0, 256, (4, cfg["h"], cfg["w"], 3), dtype=torch.uint8, device=dev)
    with torch.no_grad():
        traced = torch.jit.trace(mod, ex, check_trace=False)
        # batch-size independence check
        for b in (1, 7, 64):
            xb = torch.randint(0, 256, (b, cfg["h"], cfg["w"], 3), dtype=torch.uint8, device=dev)
            assert torch.allclose(traced(xb), mod(xb), atol=1e-3), "traced output mismatch"
    out_dir = out_dir or os.path.join(ROOT, "submission")
    os.makedirs(os.path.join(out_dir, "assets"), exist_ok=True)
    traced.save(os.path.join(out_dir, "assets", "model.pt"))
    meta = {"attribute_names": ATTRIBUTES, "h": cfg["h"], "w": cfg["w"], "run": os.path.basename(run.rstrip("/\\")),
            "flip": flip, "offsets": offsets, "groups": groups, "half": half}
    with open(os.path.join(out_dir, "assets", "config.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    shutil.copy(os.path.join(ROOT, "starter_kit", "examples", "task1", "sample_code_submission", "metadata.yaml"),
                os.path.join(out_dir, "metadata.yaml"))
    print("exported to", out_dir)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--calib", default="calib.json")
    ap.add_argument("--no-calib", action="store_true")
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    export(a.run, a.ckpt, a.calib, not a.no_flip, a.out, not a.fp32, not a.no_calib)
