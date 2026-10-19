"""Train-time augmentation on uint8 CHW tensors (runs in DataLoader workers).

Base: horizontal flip (no left/right attributes) + light pad-and-crop.
N3 surveillance degradation: down/up-scaling, JPEG, Gaussian/motion blur, mild
brightness/contrast, random erasing. No hue/saturation jitter (24/40 attributes are colours).
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torchvision.io import decode_jpeg, encode_jpeg


def _hflip(x):
    return x.flip(-1)


def _pad_crop(x, pad_h=12, pad_w=6):
    c, h, w = x.shape
    xp = F.pad(x.unsqueeze(0).float(), (pad_w, pad_w, pad_h, pad_h), mode="replicate")[0]
    i = random.randint(0, 2 * pad_h)
    j = random.randint(0, 2 * pad_w)
    return xp[:, i:i + h, j:j + w].round().clamp(0, 255).to(torch.uint8)


def _downup(x, lo=0.25, hi=0.75):
    c, h, w = x.shape
    f = random.uniform(lo, hi)
    small = F.interpolate(x.unsqueeze(0).float(), size=(max(8, int(h * f)), max(4, int(w * f))),
                          mode="bilinear", align_corners=False, antialias=True)
    mode = random.choice(["bilinear", "bicubic", "nearest"])
    kw = {} if mode == "nearest" else {"align_corners": False}
    big = F.interpolate(small, size=(h, w), mode=mode, **kw)[0]
    return big.round().clamp(0, 255).to(torch.uint8)


def _jpeg(x, lo=30, hi=90):
    return decode_jpeg(encode_jpeg(x.contiguous(), quality=random.randint(lo, hi)))


def _gauss_blur(x):
    sigma = random.uniform(0.4, 1.5)
    k = int(2 * math.ceil(2 * sigma) + 1)
    t = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    xf = x.unsqueeze(0).float()
    xf = F.conv2d(F.pad(xf, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    xf = F.conv2d(F.pad(xf, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)
    return xf[0].round().clamp(0, 255).to(torch.uint8)


def _motion_blur(x):
    k = random.choice([3, 5, 7])
    ker = torch.zeros(k, k)
    ang = random.uniform(0, math.pi)
    c = k // 2
    for t in torch.linspace(-c, c, 4 * k):
        i = int(round(c + float(t) * math.sin(ang)))
        j = int(round(c + float(t) * math.cos(ang)))
        ker[i, j] = 1
    ker = ker / ker.sum()
    xf = F.pad(x.unsqueeze(0).float(), (c, c, c, c), mode="replicate")
    xf = F.conv2d(xf, ker.view(1, 1, k, k).repeat(3, 1, 1, 1), groups=3)
    return xf[0].round().clamp(0, 255).to(torch.uint8)


def _bright_contrast(x, b=0.2, c=0.2):
    xf = x.float()
    mean = xf.mean()
    xf = (xf - mean) * random.uniform(1 - c, 1 + c) + mean
    xf = xf * random.uniform(1 - b, 1 + b)
    return xf.round().clamp(0, 255).to(torch.uint8)


def _erase(x, area=(0.02, 0.2), ratio=(0.3, 3.3)):
    c, h, w = x.shape
    for _ in range(10):
        a = random.uniform(*area) * h * w
        r = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
        eh, ew = int(round(math.sqrt(a * r))), int(round(math.sqrt(a / r)))
        if eh < h and ew < w:
            i, j = random.randint(0, h - eh), random.randint(0, w - ew)
            x = x.clone()
            x[:, i:i + eh, j:j + ew] = torch.randint(0, 256, (c, eh, ew), dtype=torch.uint8)
            return x
    return x


class TrainAugment:
    def __init__(self, degrade: bool = True, p: float = 0.3):
        self.degrade = degrade
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            x = _hflip(x)
        if random.random() < 0.5:
            x = _pad_crop(x)
        if self.degrade:
            p = self.p
            if random.random() < p:
                x = _downup(x)
            if random.random() < p:
                x = _gauss_blur(x) if random.random() < 0.5 else _motion_blur(x)
            if random.random() < p:
                x = _bright_contrast(x)
            if random.random() < p:
                x = _jpeg(x)
            if random.random() < p:
                x = _erase(x)
        return x
