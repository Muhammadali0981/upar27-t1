"""Weighted BCE (standard PAR choice): positives weighted exp(1 - r_k), negatives exp(r_k)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCE(nn.Module):
    def __init__(self, pos_rate, weighted=True, label_smoothing=0.0):
        super().__init__()
        r = torch.as_tensor(pos_rate, dtype=torch.float32)
        self.register_buffer("w_pos", torch.exp(1 - r) if weighted else torch.ones_like(r))
        self.register_buffer("w_neg", torch.exp(r) if weighted else torch.ones_like(r))
        self.ls = label_smoothing

    def forward(self, logits, target):
        w = torch.where(target > 0.5, self.w_pos, self.w_neg)
        if self.ls > 0:
            target = target * (1 - self.ls) + 0.5 * self.ls
        return (F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none") * w).sum(1).mean()
