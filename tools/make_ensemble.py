"""Create an ensemble "run" folder: ensemble.json (members, checkpoints, weights) and the weighted
average of the members' val logits, so calibrate.py and export.py work on it like on a single run.

Usage: python tools/make_ensemble.py --name ens_b16_l14 --members clip_b16 clip_l14 --weights 0.4 0.6
       [--ckpts best.pt best.pt] [--logits val_logits.npz]

For a train+val ensemble (no val logits), pass --no-logits and export with the --calib of the
matching train-only ensemble.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import ROOT  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--ckpts", nargs="+", default=None)
    ap.add_argument("--logits", default="val_logits.npz")
    ap.add_argument("--no-logits", action="store_true")
    a = ap.parse_args()
    assert len(a.members) == len(a.weights), "one weight per member"
    ckpts = a.ckpts or ["best.pt"] * len(a.members)
    out = os.path.join(ROOT, "results", a.name)
    os.makedirs(out, exist_ok=True)
    members = [{"run": r, "ckpt": c, "weight": w} for r, c, w in zip(a.members, ckpts, a.weights)]
    with open(os.path.join(out, "ensemble.json"), "w") as fh:
        json.dump({"members": members}, fh, indent=1)
    if not a.no_logits:
        ds = [np.load(os.path.join(ROOT, "results", r, a.logits), allow_pickle=True) for r in a.members]
        for d in ds[1:]:
            assert (d["labels"] == ds[0]["labels"]).all(), "members were evaluated on different data"
        z = sum(w * d["logits"] for w, d in zip(a.weights, ds))
        np.savez(os.path.join(out, a.logits), logits=z, labels=ds[0]["labels"], sources=ds[0]["sources"])
    print("wrote", out)
