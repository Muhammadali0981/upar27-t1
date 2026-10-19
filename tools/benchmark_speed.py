"""Split run.py time into image decoding (CPU threads) and model forward (GPU).

Usage: python tools/benchmark_speed.py --package submission [--n 3000] [--threads 16]
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import DATA_DIR, read_gt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=os.path.join(ROOT, "submission"))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--threads", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--batch", type=int, nargs="+", default=[128, 256])
    a = ap.parse_args()
    import json
    cfg = json.load(open(os.path.join(a.package, "assets", "config.json")))
    h, w = cfg["h"], cfg["w"]
    paths = [os.path.join(DATA_DIR, p) for p in read_gt("val")[0]]
    rng = np.random.default_rng(1)
    paths = [paths[i] for i in rng.choice(len(paths), a.n, replace=False)]

    def load(p):
        with Image.open(p) as im:
            return np.asarray(im.convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.uint8)

    for t in a.threads:
        with ThreadPoolExecutor(t) as ex:
            t0 = time.time()
            imgs = list(ex.map(load, paths))
            dt = time.time() - t0
        print(f"decode+resize, {t} threads: {a.n / dt:.0f} img/s")
    x_all = torch.from_numpy(np.stack(imgs))
    m = torch.jit.load(os.path.join(a.package, "assets", "model.pt"), map_location="cuda").eval()
    for b in a.batch:
        with torch.inference_mode():
            m(x_all[:b].cuda())
            torch.cuda.synchronize()
            t0 = time.time()
            for s in range(0, a.n, b):
                m(x_all[s:s + b].cuda(non_blocking=True)).float().cpu()
            torch.cuda.synchronize()
            dt = time.time() - t0
        print(f"model forward (incl. H2D/D2H), batch {b}: {a.n / dt:.0f} img/s")


if __name__ == "__main__":
    main()
