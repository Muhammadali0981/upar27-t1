"""N5: structure-aware decoding for (near-)exclusive attribute groups.

Rule ("force"): if no member of a group exceeds 0.5, raise the arg-max member to 0.5 + eps.
Optional "cap": members other than the arg-max that are more than `margin` below the
arg-max are capped at 0.5 - eps (disabled unless margin is set).
Pure torch, traceable, applied to probabilities after calibration.
"""
from __future__ import annotations

import numpy as np
import torch

GROUPS = {
    "age": [0, 1, 2],
    "hair": [4, 5, 6],
    "ub_color": list(range(8, 20)),
    "lb_color": list(range(21, 33)),
    "lb_type": [33, 34],
}
EPS = 1e-3


def apply_group(p: torch.Tensor, idx: list[int], cap_margin: float | None = None) -> torch.Tensor:
    sub = p[:, idx]
    mx = sub.max(1, keepdim=True)[0]
    is_max = sub >= mx  # arg-max mask (exact float ties are negligible)
    need = (mx <= 0.5) & is_max
    sub = torch.where(need, sub.clamp(min=0.5 + EPS), sub)
    if cap_margin is not None:
        cap = (~is_max) & (sub < mx - cap_margin)
        sub = torch.where(cap, sub.clamp(max=0.5 - EPS), sub)
    p = p.clone()
    p[:, idx] = sub
    return p


def apply_decoding(p, groups: dict, cap_margins: dict | None = None):
    """p: (N,40) probabilities (torch or numpy). groups: {name: indices} to enforce."""
    is_np = isinstance(p, np.ndarray)
    t = torch.as_tensor(p, dtype=torch.float32)
    for name, idx in groups.items():
        t = apply_group(t, idx, (cap_margins or {}).get(name))
    return t.numpy() if is_np else t


def select_groups(probs_fit, gt_fit, src_fit, probs_chk, gt_chk, src_chk, candidates=None, log=print):
    """Greedy per-group selection: keep a group's rule only if it improves protocol
    Challenge_Avg on the fit half AND does not hurt it on the check half."""
    from metrics import challenge_avg
    candidates = candidates or GROUPS
    chosen = {}
    base_fit = challenge_avg(gt_fit, probs_fit, src_fit)
    base_chk = challenge_avg(gt_chk, probs_chk, src_chk)
    for name, idx in candidates.items():
        trial = {**chosen, name: idx}
        f = challenge_avg(gt_fit, apply_decoding(probs_fit, trial), src_fit)
        c = challenge_avg(gt_chk, apply_decoding(probs_chk, trial), src_chk)
        keep = f > base_fit and c >= base_chk
        log(f"  decode {name:<9} fit {base_fit:.4f}->{f:.4f}  check {base_chk:.4f}->{c:.4f}  {'KEEP' if keep else 'drop'}")
        if keep:
            chosen, base_fit, base_chk = trial, f, c
    return chosen
