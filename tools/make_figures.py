"""Figures + ablation table for the report from results/<run>/ folders.

Usage: python tools/make_figures.py --runs convnext_t clip_b16 [...]
Writes figures/*.png and results/ablations.csv.
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import ATTRIBUTES, SOURCES  # noqa: E402

FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")


def curves(runs):
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in runs:
        p = os.path.join(RES, r, "log.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        for tag, ls in [("ema", "-"), ("raw", "--")]:
            c = f"{tag}_mean_CA"
            if c in d:
                ax.plot(d["epoch"] + 1, d[c], ls, marker="o", ms=3, label=f"{r} ({tag})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val Challenge_Avg (protocol, offsets=0)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "val_curves.png"), dpi=150)


def per_attribute(run):
    p = os.path.join(RES, run, "per_attribute.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(d))
    for i, s in enumerate(SOURCES):
        ax.bar(x + (i - 1) * 0.27, d[f"mA_{s}"], 0.27, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(d["attribute"], rotation=90, fontsize=7)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("mA (val)")
    ax.set_title(f"Per-attribute mA by source dataset — {run}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"per_attribute_{run}.png"), dpi=150)


def calibration(run):
    p = os.path.join(RES, run, "calib.json")
    if not os.path.exists(p):
        return None
    c = json.load(open(p))
    rep = c["report"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 2.4]})
    stages = ["raw", "calibrated", "decoded"]
    for i, m in enumerate(["mA", "Instance_F1", "Challenge_Avg"]):
        axes[0].bar(np.arange(3) + (i - 1) * 0.27, [rep[s]["mean"][m] for s in stages], 0.27, label=m)
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(["offsets=0", "+N4", "+N4+N5"])
    axes[0].set_ylim(0.6, 0.9)
    axes[0].legend(fontsize=7)
    axes[0].set_title("val (in-sample offsets)")
    axes[1].bar(range(40), c["offsets"])
    axes[1].set_xticks(range(40))
    axes[1].set_xticklabels(ATTRIBUTES, rotation=90, fontsize=6)
    axes[1].set_ylabel("logit offset δ_k (after shrink)")
    axes[1].set_title(f"learned offsets, λ={c['lambda']}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"calibration_{run}.png"), dpi=150)
    return c


def table(runs):
    rows = []
    for r in runs:
        mp = os.path.join(RES, r, "val_metrics.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        row = {"run": r, "best_epoch": m["epoch"], "weights": m["tag"]}
        for s in SOURCES + ["mean"]:
            row[f"{s}_CA"] = round(m[s]["Challenge_Avg"], 4)
        row["mean_mA"] = round(m["mean"]["mA"], 4)
        row["mean_F1"] = round(m["mean"]["Instance_F1"], 4)
        cp = os.path.join(RES, r, "calib.json")
        if os.path.exists(cp):
            c = json.load(open(cp))
            ho = c.get("held_out", [None] * 3)
            row.update({"heldout_raw": round(ho[0], 4), "heldout_N4": round(ho[1], 4), "heldout_N4N5": round(ho[2], 4),
                        "tta_flip": c.get("flip"), "groups": "+".join(c["groups"])})
        cfgp = os.path.join(RES, r, "config.yaml")
        if os.path.exists(cfgp):
            import yaml
            cfg = yaml.safe_load(open(cfgp))
            row.update({k: cfg.get(k) for k in ["model", "epochs", "balanced", "degrade", "text_init", "epoch_fraction"]})
        rows.append(row)
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(RES, "ablations.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(pd.DataFrame(rows).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    curves(a.runs)
    for r in a.runs:
        per_attribute(r)
        calibration(r)
    table(a.runs)


if __name__ == "__main__":
    main()
