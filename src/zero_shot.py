"""CLIP zero-shot PAR on val (no PAR training) + text-anchored head init for N1.

For each attribute k: w_k = mean normalised text embedding of its positive prompts
(variant "pos") or normalised (pos - neg) (variant "diff"). Score cos(f, w_k).
Bias b_k is set from TRAIN data only: the threshold at which the fraction of predicted
positives on a train subset equals the (domain-balanced) train positive rate r_k.
The better variant (val Challenge_Avg) is saved as results/zeroshot/text_weights.npz.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(__file__))
from data import ATTRIBUTES, ROOT, CachedPAR  # noqa: E402
from metrics import format_row, protocol_metrics  # noqa: E402
from models import CLIP_MEAN, CLIP_STD, Normalize, resize_clip_pos_embed  # noqa: E402
from prompts import NEG, POS  # noqa: E402
from train import effective_pos_rate  # noqa: E402


@torch.no_grad()
def text_embeddings(model, tokenizer, table):
    out = []
    for name in ATTRIBUTES:
        t = model.encode_text(tokenizer(table[name]).cuda()).float()
        out.append(F.normalize(F.normalize(t, dim=-1).mean(0), dim=-1))
    return torch.stack(out)


@torch.no_grad()
def image_features(visual, norm, ds, bs=256):
    feats = []
    for x, _, _ in DataLoader(ds, bs, num_workers=6):
        with torch.autocast("cuda", dtype=torch.float16):
            feats.append(F.normalize(visual(norm(x.cuda().float())).float(), dim=-1).cpu())
    return torch.cat(feats)


def main(h=256, w=128, n_train=8000, scale=1 / 0.07):
    import open_clip
    out_dir = os.path.join(ROOT, "results", "zeroshot")
    os.makedirs(out_dir, exist_ok=True)
    model = open_clip.create_model("ViT-B-16", pretrained="openai", force_quick_gelu=True).cuda().eval()
    tok = open_clip.get_tokenizer("ViT-B-16")
    tpos, tneg = text_embeddings(model, tok, POS), text_embeddings(model, tok, NEG)
    resize_clip_pos_embed(model.visual, h, w)
    model.visual.cuda()
    norm = Normalize(CLIP_MEAN, CLIP_STD).cuda()

    tr = CachedPAR(["train"], h, w)
    r = effective_pos_rate(tr.labels, tr.source_idx, balanced=True)
    rng = np.random.default_rng(0)
    sub = np.concatenate([rng.choice(np.flatnonzero(tr.source_idx == s), n_train // 3, replace=False)
                          for s in np.unique(tr.source_idx)])
    f_tr = image_features(model.visual, norm, Subset(tr, sub))
    va = CachedPAR(["val"], h, w)
    f_va = image_features(model.visual, norm, va)

    results = {}
    best = None
    for variant, W in [("pos", tpos), ("diff", F.normalize(tpos - tneg, dim=-1))]:
        W = W.cpu()
        cos_tr = (f_tr @ W.t()).numpy()
        thr = np.array([np.quantile(cos_tr[:, k], 1 - r[k]) for k in range(40)])
        bias = -scale * thr
        z = scale * (f_va @ W.t()).numpy() + bias
        res = protocol_metrics(va.labels, 1 / (1 + np.exp(-z)), va.sources)
        print(format_row(f"zero-shot {variant}", res), flush=True)
        results[variant] = {s: {k: v for k, v in res[s].items() if k != "mA_per_attr"} for s in res}
        ca = res["mean"]["Challenge_Avg"]
        if best is None or ca > best[0]:
            best = (ca, variant, W.numpy(), bias)
    print(f"chosen text init: {best[1]} (val CA {best[0]:.4f})")
    np.savez(os.path.join(out_dir, "text_weights.npz"), weight=best[2], bias=best[3].astype(np.float32),
             scale=np.float32(scale), variant=best[1])
    with open(os.path.join(out_dir, "zeroshot.json"), "w") as fh:
        json.dump({"results": results, "chosen": best[1]}, fh, indent=1)


if __name__ == "__main__":
    main()
