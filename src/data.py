"""gt.csv parsing, cached uint8 memmap dataset, domain-balanced sampler (N2)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "starter_kit", "data")
CACHE_DIR = os.path.join(ROOT, "cache")
SOURCES = ["Market1501", "PA100k", "PETA"]

ATTRIBUTES = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female",
    "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald", "UpperBody-Length-Short",
    *[f"UpperBody-Color-{c}" for c in ["Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink",
                                         "Purple", "Red", "White", "Yellow", "Other"]],
    "LowerBody-Length-Short",
    *[f"LowerBody-Color-{c}" for c in ["Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink",
                                         "Purple", "Red", "White", "Yellow", "Other"]],
    "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack", "Accessory-Bag", "Accessory-Glasses-Normal", "Accessory-Glasses-Sun", "Accessory-Hat",
]
assert len(ATTRIBUTES) == 40


def read_gt(split: str):
    """Returns (paths, labels uint8 (N,40), sources, attribute_names) for 'train' or 'val'."""
    path = os.path.join(DATA_DIR, "annotations", "task1", split, "gt.csv")
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip()
    assert header.startswith("#"), header
    names = [h.strip() for h in header.lstrip("#").split(",")][1:]
    assert names == ATTRIBUTES, "attribute order in gt.csv differs from ATTRIBUTES"
    df = pd.read_csv(path, skiprows=1, header=None)
    paths = df.iloc[:, 0].astype(str).tolist()
    labels = df.iloc[:, 1:].to_numpy(dtype=np.uint8)
    assert labels.shape[1] == 40 and set(np.unique(labels)) <= {0, 1}
    sources = np.array([p.split("/")[0] for p in paths])
    assert set(sources) <= set(SOURCES), set(sources)
    return paths, labels, sources, names


def cache_path(split: str, h: int, w: int) -> str:
    return os.path.join(CACHE_DIR, f"{split}_{h}x{w}.u8")


def open_cache(split: str, n: int, h: int, w: int) -> np.memmap:
    return np.memmap(cache_path(split, h, w), dtype=np.uint8, mode="r", shape=(n, h, w, 3))


class CachedPAR(Dataset):
    """Returns (uint8 CHW tensor, float labels, source idx). Augmentation runs on uint8 tensors."""

    def __init__(self, splits, h=256, w=128, transform=None):
        self.items = []  # (split, local index)
        labels, sources, paths = [], [], []
        self.n = {}
        for split in splits:
            p, y, s, _ = read_gt(split)
            self.n[split] = len(p)
            self.items += [(split, i) for i in range(len(p))]
            labels.append(y)
            sources.append(s)
            paths += p
        self.labels = np.concatenate(labels)
        self.sources = np.concatenate(sources)
        self.source_idx = np.array([SOURCES.index(s) for s in self.sources])
        self.paths = paths
        self.h, self.w = h, w
        self.transform = transform
        self._mm = None

    def _maps(self):
        if self._mm is None:  # opened lazily per worker
            self._mm = {s: open_cache(s, n, self.h, self.w) for s, n in self.n.items()}
        return self._mm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        split, j = self.items[i]
        img = torch.from_numpy(np.ascontiguousarray(self._maps()[split][j])).permute(2, 0, 1)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.from_numpy(self.labels[i].astype(np.float32)), int(self.source_idx[i])


def balanced_sampler(source_idx: np.ndarray, num_samples: int | None = None, seed: int = 0):
    """Each source dataset gets an equal share of the samples per epoch (N2)."""
    counts = np.bincount(source_idx, minlength=len(SOURCES)).astype(np.float64)
    present = counts > 0
    w = np.zeros_like(counts)
    w[present] = 1.0 / (counts[present] * present.sum())
    weights = torch.as_tensor(w[source_idx], dtype=torch.double)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples or len(source_idx), replacement=True, generator=g)


def pos_rate(labels: np.ndarray) -> np.ndarray:
    return labels.mean(0).astype(np.float64)
