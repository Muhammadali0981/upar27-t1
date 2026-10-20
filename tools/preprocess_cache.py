"""Resize every train/val image once to HxW (PIL bilinear, same as run.py) into a uint8 memmap."""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import CACHE_DIR, DATA_DIR, cache_path, read_gt  # noqa: E402


def load(args):
    path, h, w = args
    with Image.open(os.path.join(DATA_DIR, path)) as im:
        return np.asarray(im.convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=128)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(CACHE_DIR, exist_ok=True)
    for split in a.splits:
        paths, _, _, _ = read_gt(split)
        out = cache_path(split, a.h, a.w)
        tmp = out + ".partial"
        mm = np.memmap(tmp, dtype=np.uint8, mode="w+", shape=(len(paths), a.h, a.w, 3))
        t0 = time.time()
        with ProcessPoolExecutor(a.workers) as ex:
            jobs = ex.map(load, [(p, a.h, a.w) for p in paths], chunksize=256)
            for i, arr in enumerate(tqdm(jobs, total=len(paths), desc=split, unit="img", dynamic_ncols=True)):
                mm[i] = arr
        mm.flush()
        del mm
        os.replace(tmp, out)
        print(f"{split}: wrote {out} ({len(paths)} images) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
