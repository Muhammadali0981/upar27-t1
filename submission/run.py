"""UPAR 2027 Track 1 submission: TorchScript PAR model (CLIP/ConvNeXt), batched inference.

assets/model.pt     traced module: uint8 (B,H,W,3) -> probabilities (B,40), with normalisation,
                    flip-TTA, calibrated offsets and structure decoding inside.
assets/config.json  attribute order of the model outputs and input size.
Nothing here depends on file paths or directory structure of the test data.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 256

_MODEL = None
_CFG: dict = {}
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_POOL = ThreadPoolExecutor(max_workers=16)


def load_model() -> None:
    global _MODEL, _CFG
    with open(os.path.join(HERE, "assets", "config.json"), "r", encoding="utf-8") as fh:
        _CFG = json.load(fh)
    torch.backends.cudnn.benchmark = True
    model = torch.jit.load(os.path.join(HERE, "assets", "model.pt"), map_location=_DEVICE)
    if _DEVICE.type == "cpu":
        model = model.float()  # fp16 weights are meant for the GPU
    _MODEL = model.eval()
    with torch.inference_mode():  # warm-up (cudnn autotune, lazy init)
        for b in (BATCH_SIZE, 1):
            _MODEL(torch.zeros((b, _CFG["h"], _CFG["w"], 3), dtype=torch.uint8, device=_DEVICE))
    if _DEVICE.type == "cuda":
        torch.cuda.synchronize()


def _load(path: str) -> np.ndarray:
    h, w = _CFG["h"], _CFG["w"]
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.uint8)
    except Exception:  # unreadable image: neutral grey instead of failing the whole run
        return np.full((h, w, 3), 128, dtype=np.uint8)


def _order(names: list) -> list:
    ours = _CFG["attribute_names"]
    assert sorted(names) == sorted(ours), "attribute names differ from the model's"
    return [ours.index(n) for n in names]


def predict_batch(samples: list) -> list:
    if _MODEL is None:
        load_model()
    imgs = list(_POOL.map(_load, [s["image_path"] for s in samples]))
    x = torch.from_numpy(np.stack(imgs)).to(_DEVICE, non_blocking=True)
    with torch.inference_mode():
        p = _MODEL(x).float().cpu().numpy()
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    out = []
    for s, row in zip(samples, p):
        idx = _order(list(s["attribute_names"]))
        out.append([float(row[i]) for i in idx])
    return out


def predict_image(sample: dict) -> list:
    return predict_batch([sample])[0]
