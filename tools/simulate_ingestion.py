"""Simulate the Codabench ingestion: fake input_data from N val images, import run.py from
the package folder, call load_model + predict_batch, validate outputs, score, time it.

Usage: python tools/simulate_ingestion.py --package submission [--n 2000] [--target 30000]
Runs with only torch/torchvision/numpy/Pillow needed by the package itself (metric code is ours).
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from metrics import format_row, protocol_metrics  # noqa: E402


def read_val():
    path = os.path.join(ROOT, "starter_kit", "data", "annotations", "task1", "val", "gt.csv")
    with open(path, encoding="utf-8") as fh:
        names = [h.strip() for h in fh.readline().lstrip("#").strip().split(",")][1:]
        rows = [r for r in csv.reader(fh)]
    return names, [r[0] for r in rows], np.array([[int(v) for v in r[1:]] for r in rows])


def build_input(n, out, seed=0):
    names, paths, labels = read_val()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(paths), size=min(n, len(paths)), replace=False))
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "images"))
    rows = []
    for k, i in enumerate(idx):
        # anonymised flat names: nothing about the source dataset leaks into the path
        dst = f"images/{k:06d}{os.path.splitext(paths[i])[1]}"
        shutil.copy(os.path.join(ROOT, "starter_kit", "data", paths[i]), os.path.join(out, dst))
        rows.append(dst)
    with open(os.path.join(out, "images.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image"])
        w.writerows([[r] for r in rows])
    with open(os.path.join(out, "task.json"), "w") as fh:
        json.dump({"attribute_names": names}, fh)
    src = np.array([paths[i].split("/")[0] for i in idx])
    return names, rows, labels[idx], src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=os.path.join(ROOT, "submission"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--target", type=int, default=30000)
    ap.add_argument("--input", default=os.path.join(ROOT, "cache", "sim_input"))
    ap.add_argument("--max-seconds", type=float, default=150)
    a = ap.parse_args()

    names, rows, labels, src = build_input(a.n, a.input)
    pkg = os.path.abspath(a.package)
    os.chdir(pkg)  # the platform's working dir is the submission folder
    sys.path.insert(0, pkg)
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("run", os.path.join(pkg, "run.py"))
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    if hasattr(run, "load_model"):
        run.load_model()
    t_load = time.time() - t0
    bs = getattr(run, "BATCH_SIZE", 32)
    samples = [{"index": i, "image": r, "image_path": os.path.join(a.input, r), "attribute_names": list(names)}
               for i, r in enumerate(rows)]
    preds = []
    t1 = time.time()
    if hasattr(run, "predict_batch"):
        for s in range(0, len(samples), bs):
            preds += run.predict_batch(samples[s:s + bs])
    else:
        preds = [run.predict_image(s) for s in samples]
    t_pred = time.time() - t1

    p = np.array(preds, dtype=np.float64)
    assert p.shape == (len(samples), 40), p.shape
    assert np.isfinite(p).all(), "NaN/inf in predictions"
    assert (p >= 0).all() and (p <= 1).all(), "predictions out of [0,1]"
    res = protocol_metrics(labels, p, src)
    print(format_row("simulated ingestion", res))
    ips = len(samples) / t_pred
    proj = t_load + a.target / ips
    print(f"load {t_load:.1f}s | {len(samples)} imgs in {t_pred:.1f}s = {ips:.0f} img/s (batch {bs}) | "
          f"projected {a.target} imgs: {proj:.0f}s on this machine")
    if proj > a.max_seconds:
        print(f"FAIL: projected time {proj:.0f}s > {a.max_seconds}s")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
