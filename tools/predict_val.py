"""Val logits (with and without flip-TTA) for a run's checkpoint, e.g. after stopping a run early.

Usage: python tools/predict_val.py --run results/<run> [--ckpt best.pt]
"""
import argparse
import os
import sys

import numpy as np
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import CachedPAR  # noqa: E402
from metrics import format_row, protocol_metrics  # noqa: E402
from train import load_run_model, predict_logits  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    a = ap.parse_args()
    m, cfg = load_run_model(a.run, a.ckpt)
    m = m.cuda()
    ds = CachedPAR(["val"], cfg["h"], cfg["w"])
    dl = DataLoader(ds, 256, shuffle=False, num_workers=6, pin_memory=True)
    for flip, fn in [(False, "val_logits.npz"), (True, "val_logits_flip.npz")]:
        z = predict_logits(m, dl, flip=flip)
        np.savez(os.path.join(a.run, fn), logits=z, labels=ds.labels, sources=ds.sources)
        print(format_row(f"{a.ckpt} flip={flip}", protocol_metrics(ds.labels, 1 / (1 + np.exp(-z)), ds.sources)))
