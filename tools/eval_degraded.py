"""Robustness proxy for N3: evaluate runs on val images with a FIXED surveillance-style
degradation (down-scale to 40% + JPEG q=40), raw (offsets=0) and with each run's calibration.

Usage: python tools/eval_degraded.py --runs clip_b16_short clip_b16_short_nodegrade
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.io import decode_jpeg, encode_jpeg

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import CachedPAR  # noqa: E402
from decode import apply_decoding  # noqa: E402
from metrics import format_row, protocol_metrics  # noqa: E402
from train import load_run_model, predict_logits  # noqa: E402


class Degrade:
    def __init__(self, scale=0.4, quality=40):
        self.scale, self.quality = scale, quality

    def __call__(self, x):
        c, h, w = x.shape
        s = F.interpolate(x[None].float(), size=(int(h * self.scale), int(w * self.scale)), mode="bilinear",
                          antialias=True, align_corners=False)
        x = F.interpolate(s, size=(h, w), mode="bilinear", align_corners=False)[0].round().clamp(0, 255).to(torch.uint8)
        return decode_jpeg(encode_jpeg(x, quality=self.quality))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--scale", type=float, default=0.4)
    ap.add_argument("--quality", type=int, default=40)
    a = ap.parse_args()
    out = {}
    for r in a.runs:
        run = os.path.join(ROOT, "results", r)
        m, cfg = load_run_model(run)
        m = m.cuda()
        ds = CachedPAR(["val"], cfg["h"], cfg["w"], Degrade(a.scale, a.quality))
        z = predict_logits(m, DataLoader(ds, 256, num_workers=6, pin_memory=True))
        c = json.load(open(os.path.join(run, "calib.json")))
        raw = protocol_metrics(ds.labels, 1 / (1 + np.exp(-z)), ds.sources)
        cal = protocol_metrics(ds.labels, apply_decoding(1 / (1 + np.exp(-(z + np.array(c["offsets"])))), c["groups"]),
                               ds.sources)
        print(format_row(f"{r} degraded raw", raw))
        print(format_row(f"{r} degraded +N4N5", cal))
        out[r] = {"raw": raw["mean"]["Challenge_Avg"], "calibrated": cal["mean"]["Challenge_Avg"]}
        del m
        torch.cuda.empty_cache()
    with open(os.path.join(ROOT, "results", "degraded_eval.json"), "w") as fh:
        json.dump({"scale": a.scale, "quality": a.quality, "results": out}, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
