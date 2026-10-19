"""Challenge metrics for UPAR Track 1.

The official evaluation script is not public. This reproduces the formulas on
the Codabench Evaluation page using the standard PAR evaluation code
(Jia et al., "Rethinking of PAR", get_pedestrian_metrics), which the UPAR
organizers' earlier baselines use. Edge cases (empty predicted / true sets) are
handled by a small epsilon in every denominator, which is our best reproduction.

Instance_F1 is computed from the image-averaged precision and recall (as in the
reference code); `instance_f1_mode="mean"` gives the per-image F1 average instead.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-20
THRESHOLD = 0.5  # fixed by the evaluator: present iff prob > 0.5


def binarize(probs: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    return (np.asarray(probs) > threshold).astype(np.int64)


def pedestrian_metrics(gt: np.ndarray, pred: np.ndarray, instance_f1_mode: str = "pr") -> dict:
    """gt, pred: (N, K) binary arrays. Returns label-based and instance-based metrics."""
    gt = np.asarray(gt).astype(np.int64)
    pred = np.asarray(pred).astype(np.int64)
    assert gt.shape == pred.shape and gt.ndim == 2

    # label-based
    gt_pos = (gt == 1).sum(0)
    gt_neg = (gt == 0).sum(0)
    tp = ((gt == 1) & (pred == 1)).sum(0)
    tn = ((gt == 0) & (pred == 0)).sum(0)
    fp = ((gt == 0) & (pred == 1)).sum(0)
    pos_recall = tp / (gt_pos + EPS)
    neg_recall = tn / (gt_neg + EPS)
    label_ma_per_attr = (pos_recall + neg_recall) / 2
    label_prec = tp / (tp + fp + EPS)
    label_f1_per_attr = 2 * label_prec * pos_recall / (label_prec + pos_recall + EPS)

    # instance-based
    inter = ((gt == 1) & (pred == 1)).sum(1)
    union = ((gt == 1) | (pred == 1)).sum(1)
    n_gt = (gt == 1).sum(1)
    n_pred = (pred == 1).sum(1)
    inst_acc = inter / (union + EPS)
    inst_prec = inter / (n_pred + EPS)
    inst_rec = inter / (n_gt + EPS)
    if instance_f1_mode == "pr":
        p, r = inst_prec.mean(), inst_rec.mean()
        inst_f1 = 2 * p * r / (p + r + EPS)
    elif instance_f1_mode == "mean":
        inst_f1 = (2 * inst_prec * inst_rec / (inst_prec + inst_rec + EPS)).mean()
    else:
        raise ValueError(instance_f1_mode)

    ma = float(label_ma_per_attr.mean())
    f1 = float(inst_f1)
    return {
        "mA": ma,
        "Label_F1": float(label_f1_per_attr.mean()),
        "Instance_Acc": float(inst_acc.mean()),
        "Instance_Prec": float(inst_prec.mean()),
        "Instance_Rec": float(inst_rec.mean()),
        "Instance_F1": f1,
        "Challenge_Avg": 2 * ma * f1 / (ma + f1 + EPS),
        "mA_per_attr": label_ma_per_attr,
    }


SCALARS = ["mA", "Label_F1", "Instance_Acc", "Instance_Prec", "Instance_Rec", "Instance_F1", "Challenge_Avg"]


def protocol_metrics(gt: np.ndarray, probs: np.ndarray, sources: np.ndarray, threshold: float = THRESHOLD,
                     instance_f1_mode: str = "pr") -> dict:
    """Every metric per source dataset, then averaged over datasets (evaluation protocol).

    Returns {"mean": {...}, "<source>": {...}}. Challenge_Avg in "mean" is the dataset
    average of per-dataset Challenge_Avg (the page says every metric is averaged over
    datasets).
    """
    pred = binarize(probs, threshold)
    sources = np.asarray(sources)
    out = {}
    for s in sorted(set(sources.tolist())):
        m = sources == s
        out[s] = pedestrian_metrics(gt[m], pred[m], instance_f1_mode)
    keys = [k for k in out]
    out["mean"] = {k: float(np.mean([out[s][k] for s in keys])) for k in SCALARS}
    out["mean"]["mA_per_attr"] = np.mean([out[s]["mA_per_attr"] for s in keys], axis=0)
    return out


def challenge_avg(gt, probs, sources, threshold: float = THRESHOLD) -> float:
    return protocol_metrics(gt, probs, sources, threshold)["mean"]["Challenge_Avg"]


def format_row(name: str, res: dict) -> str:
    parts = [f"{name:<28}"]
    for s in [k for k in res if k != "mean"] + ["mean"]:
        r = res[s]
        parts.append(f"{s}: mA {r['mA']:.4f} F1 {r['Instance_F1']:.4f} CA {r['Challenge_Avg']:.4f}")
    return " | ".join(parts)
