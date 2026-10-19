"""N4: protocol-matched calibration.

Learn one global per-attribute logit offset delta_k so that sigmoid(z_k + delta_k) > 0.5 is
the decision that maximises the evaluation protocol (Challenge_Avg per source dataset,
averaged over datasets). Coordinate ascent over a grid; then shrink delta <- lambda * delta
with lambda chosen on a held-out half of val. Offsets never depend on dataset identity
at test time: one vector for every image.

Also selects N5 decoding groups (decode.select_groups) on the calibrated probabilities.

Usage: python src/calibrate.py --run results/<run>   (reads val_logits.npz, writes calib.json)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from decode import GROUPS, apply_decoding, select_groups  # noqa: E402
from metrics import EPS, format_row, protocol_metrics  # noqa: E402


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class _Dom:
    """Per-dataset running state for fast single-column updates."""

    def __init__(self, z, y):
        self.z, self.y = z, y.astype(bool)
        self.n_gt = self.y.sum(1).astype(np.float64)
        self.gt_pos = self.y.sum(0).astype(np.float64)
        self.gt_neg = (~self.y).sum(0).astype(np.float64)

    def set(self, delta):
        self.pred = (self.z + delta) > 0
        self.inter = (self.pred & self.y).sum(1).astype(np.float64)
        self.n_pred = self.pred.sum(1).astype(np.float64)
        tp = (self.pred & self.y).sum(0)
        tn = (~self.pred & ~self.y).sum(0)
        self.ma_k = (tp / (self.gt_pos + EPS) + tn / (self.gt_neg + EPS)) / 2

    def scores_for_column(self, k, grid):
        """Challenge_Avg of this dataset for every candidate offset of column k (vectorised)."""
        newp = (self.z[:, k:k + 1] + grid[None, :]) > 0           # (N, G)
        yk = self.y[:, k:k + 1]
        oldp = self.pred[:, k:k + 1]
        inter = self.inter[:, None] - (oldp & yk) + (newp & yk)
        npred = self.n_pred[:, None] - oldp + newp
        prec = (inter / (npred + EPS)).mean(0)
        rec = (inter / (self.n_gt[:, None] + EPS)).mean(0)
        f1 = 2 * prec * rec / (prec + rec + EPS)
        tp = (newp & yk).sum(0)
        tn = (~newp & ~yk).sum(0)
        ma_new = (tp / (self.gt_pos[k] + EPS) + tn / (self.gt_neg[k] + EPS)) / 2
        ma = (self.ma_k.sum() - self.ma_k[k] + ma_new) / len(self.ma_k)
        return 2 * ma * f1 / (ma + f1 + EPS)


def fit_offsets(z, y, src, lo=-3.0, hi=3.0, step=0.05, passes=3, log=print):
    grid = np.round(np.arange(lo, hi + 1e-9, step), 4)
    doms = [_Dom(z[src == s], y[src == s]) for s in sorted(set(src.tolist()))]
    delta = np.zeros(z.shape[1])
    for d in doms:
        d.set(delta)
    for p in range(passes):
        changed = 0
        for k in range(z.shape[1]):
            score = np.mean([d.scores_for_column(k, grid) for d in doms], axis=0)
            # prefer the offset closest to 0 among ties (less overfitting)
            best = np.flatnonzero(score >= score.max() - 1e-12)
            b = grid[best[np.argmin(np.abs(grid[best]))]]
            if b != delta[k]:
                changed += 1
            delta[k] = b
            for d in doms:
                d.set(delta)
        cur = np.mean([2 * d.ma_k.mean() * _f1(d) / (d.ma_k.mean() + _f1(d) + EPS) for d in doms])
        log(f"  pass {p + 1}: {changed} offsets changed, fit Challenge_Avg {cur:.4f}")
        if changed == 0:
            break
    return delta


def _f1(d):
    p = (d.inter / (d.n_pred + EPS)).mean()
    r = (d.inter / (d.n_gt + EPS)).mean()
    return 2 * p * r / (p + r + EPS)


def split_halves(src, seed=0):
    """50/50 split of val per source dataset."""
    rng = np.random.default_rng(seed)
    a = np.zeros(len(src), dtype=bool)
    for s in set(src.tolist()):
        idx = np.flatnonzero(src == s)
        rng.shuffle(idx)
        a[idx[: len(idx) // 2]] = True
    return a, ~a


def calibrate(z, y, src, log=print, lambdas=(0.0, 0.25, 0.5, 0.75, 1.0), decode_candidates=None):
    z = z.astype(np.float64)
    A, B = split_halves(src)
    log("fit offsets on half A")
    dA = fit_offsets(z[A], y[A], src[A], log=log)
    rows = []
    for lam in lambdas:
        r = protocol_metrics(y[B], sigmoid(z[B] + lam * dA), src[B])["mean"]
        rows.append((lam, r["Challenge_Avg"]))
        log(f"  lambda {lam:.2f}: half-B Challenge_Avg {r['Challenge_Avg']:.4f} (mA {r['mA']:.4f}, F1 {r['Instance_F1']:.4f})")
    lam = max(rows, key=lambda t: (t[1], -abs(t[0] - 0.5)))[0]
    log(f"chosen lambda {lam}")
    # N5 group selection on calibrated probabilities (fit on A, confirm on B)
    pA, pB = sigmoid(z[A] + lam * dA), sigmoid(z[B] + lam * dA)
    groups = select_groups(pA, y[A], src[A], pB, y[B], src[B], decode_candidates, log=log)
    # final offsets: refit on all of val, shrink with the held-out lambda
    log("refit offsets on full val")
    dFull = fit_offsets(z, y, src, log=log)
    return {"offsets": (lam * dFull).tolist(), "offsets_unshrunk": dFull.tolist(), "lambda": lam,
            "offsets_halfA": dA.tolist(), "groups": groups, "lambda_scores": rows}


def report(z, y, src, calib, log=print):
    """Before/after rows (note: 'after' is in-sample on val except the half-B check above)."""
    base = protocol_metrics(y, sigmoid(z), src)
    cal = protocol_metrics(y, sigmoid(z + np.array(calib["offsets"])), src)
    dec = protocol_metrics(y, apply_decoding(sigmoid(z + np.array(calib["offsets"])), calib["groups"]), src)
    for n, r in [("raw (offsets=0)", base), ("+N4 calibration", cal), ("+N4+N5 decoding", dec)]:
        log(format_row(n, r))
    return {"raw": base, "calibrated": cal, "decoded": dec}


def honest_eval(z, y, src, log=print):
    """Out-of-sample estimate: fit everything on half A, evaluate on half B (and vice versa)."""
    out = []
    for fitm, evm in [split_halves(src), split_halves(src)[::-1]]:
        d = fit_offsets(z[fitm], y[fitm], src[fitm], log=lambda *_: None)
        p_raw = sigmoid(z[evm])
        p_cal = sigmoid(z[evm] + d)
        g = select_groups(sigmoid(z[fitm] + d), y[fitm], src[fitm], p_cal, y[evm], src[evm], log=lambda *_: None)
        out.append([protocol_metrics(y[evm], p, src[evm])["mean"]["Challenge_Avg"]
                    for p in (p_raw, p_cal, apply_decoding(p_cal, g))])
    m = np.mean(out, 0)
    log(f"held-out (2-fold) Challenge_Avg: raw {m[0]:.4f}  +N4 {m[1]:.4f}  +N4+N5 {m[2]:.4f}")
    return m.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--logits", default="val_logits.npz")
    ap.add_argument("--no-hair", action="store_true")
    a = ap.parse_args()
    d = np.load(os.path.join(a.run, a.logits), allow_pickle=True)
    z, y, src = d["logits"], d["labels"], d["sources"].astype(str)
    cands = {k: v for k, v in GROUPS.items() if not (a.no_hair and k == "hair")}
    calib = calibrate(z, y, src, decode_candidates=cands)
    calib["held_out"] = honest_eval(z, y, src)
    res = report(z, y, src, calib)
    calib["report"] = {k: {s: {m: v for m, v in r[s].items() if m != "mA_per_attr"} for s in r} for k, r in res.items()}
    with open(os.path.join(a.run, "calib.json"), "w") as fh:
        json.dump(calib, fh, indent=1)
    print("wrote", os.path.join(a.run, "calib.json"))


if __name__ == "__main__":
    main()
