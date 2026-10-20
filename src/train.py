"""Training: AMP, EMA, layer-wise LR decay (CLIP), domain-balanced sampling, per-epoch
protocol-matched validation, checkpoints and logs in results/<name>/.

Usage: python src/train.py --config configs/convnext_t.yaml [--set key=value ...]
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from augment import TrainAugment  # noqa: E402
from data import ATTRIBUTES, ROOT, SOURCES, CachedPAR, balanced_sampler  # noqa: E402
from losses import WeightedBCE  # noqa: E402
from metrics import format_row, protocol_metrics  # noqa: E402
from models import build_model  # noqa: E402

DEFAULTS = dict(
    name="run", model="convnext_t", h=256, w=128, epochs=10, batch_size=64, accum=1,
    lr=2e-4, head_lr=2e-3, wd=0.05, layer_decay=0.7, warmup_epochs=1.0, ema_decay=0.999,
    balanced=True, degrade=True, degrade_p=0.3, weighted_loss=True, label_smoothing=0.0,
    train_splits=["train"], num_workers=8, seed=0, text_init=None, grad_ckpt=False,
    max_steps=0, eval_raw=True, dropout=0.2, init_scale=1 / 0.07, epoch_fraction=1.0, balance_alpha=0.0,
)
# progress bars refresh every 0.5 s in a terminal, every 30 s when output goes to a log file
BAR_INTERVAL = 0.5 if sys.stderr.isatty() else 30


class _Tee:
    def __init__(self, stream, path):
        self.stream, self.fh = stream, open(path, "a", encoding="utf-8")

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return self.stream.isatty()


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def effective_pos_rate(labels, source_idx, balanced, alpha=0.0):
    """Positive rate under the sampling distribution (source share proportional to n_s**alpha)."""
    if not balanced:
        return labels.mean(0).astype(np.float64)
    srcs = np.unique(source_idx)
    share = np.array([(source_idx == s).sum() ** alpha for s in srcs], dtype=np.float64)
    share /= share.sum()
    return np.sum([w * labels[source_idx == s].mean(0) for w, s in zip(share, srcs)], axis=0)


def make_model(cfg, pos_rate=None, load_pretrained=True):
    text_w = bias = None
    if cfg["model"] == "clip" and cfg.get("text_init") and load_pretrained:
        t = np.load(os.path.join(ROOT, cfg["text_init"]))
        text_w = t["weight"]
        bias = t["bias"] if "bias" in t else None
        cfg["init_scale"] = float(t["scale"]) if "scale" in t else cfg["init_scale"]
    if cfg["model"] == "clip" and bias is None and pos_rate is not None:
        bias = np.log(np.clip(pos_rate, 1e-4, 1 - 1e-4) / (1 - np.clip(pos_rate, 1e-4, 1 - 1e-4)))
    return build_model(cfg, pos_rate, text_w, bias, load_pretrained=load_pretrained)


def load_run_model(run, ckpt="best.pt"):
    with open(os.path.join(run, "config.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    model = make_model(cfg, load_pretrained=False)
    sd = torch.load(os.path.join(run, ckpt), map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    return model.eval(), cfg


class EMA:
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))
        ema_p = [p for p in self.module.state_dict().values() if p.dtype.is_floating_point]
        cur_p = [p for p in model.state_dict().values() if p.dtype.is_floating_point]
        torch._foreach_lerp_(ema_p, cur_p, 1 - d)


@torch.no_grad()
def predict_logits(model, loader, flip=False, channels_last=False, desc="predict"):
    model.eval()
    out = []
    for x, _, _ in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, mininterval=BAR_INTERVAL):
        x = x.cuda(non_blocking=True).float()
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            z = model(x).float()
            if flip:
                z = (z + model(x.flip(3)).float()) / 2
        out.append(z.cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override, e.g. epochs=3 balanced=false")
    a = ap.parse_args()
    def load_cfg(path):  # configs may inherit (recursively) from a base config in the same folder
        with open(path) as fh:
            c = yaml.safe_load(fh)
        if "base" in c:
            return {**load_cfg(os.path.join(os.path.dirname(path), c.pop("base"))), **c}
        return c

    cfg = {**DEFAULTS, **load_cfg(a.config)}
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    seed_all(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run = os.path.join(ROOT, "results", cfg["name"])
    os.makedirs(run, exist_ok=True)
    cfg["git"] = git_hash()
    # printed results also go to results/<name>/train.log (the progress bar is on stderr and stays in the terminal)
    sys.stdout = _Tee(sys.stdout, os.path.join(run, "train.log"))
    with open(os.path.join(run, "config.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh)

    train_ds = CachedPAR(cfg["train_splits"], cfg["h"], cfg["w"], TrainAugment(cfg["degrade"], cfg["degrade_p"]))
    has_val = "val" not in cfg["train_splits"]
    pos_rate = effective_pos_rate(train_ds.labels, train_ds.source_idx, cfg["balanced"], cfg["balance_alpha"])
    n_epoch = int(len(train_ds) * cfg["epoch_fraction"])
    sampler = balanced_sampler(train_ds.source_idx, n_epoch, cfg["seed"], cfg["balance_alpha"]) if cfg["balanced"] else \
        torch.utils.data.RandomSampler(train_ds, num_samples=n_epoch)
    nw = cfg["num_workers"]
    train_dl = DataLoader(train_ds, cfg["batch_size"], sampler=sampler, num_workers=nw, pin_memory=True,
                          drop_last=True, persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)
    if has_val:
        val_ds = CachedPAR(["val"], cfg["h"], cfg["w"])
        val_dl = DataLoader(val_ds, 256, shuffle=False, num_workers=min(nw, 6), pin_memory=True,
                            persistent_workers=nw > 0)
        val_src = val_ds.sources

    model = make_model(cfg, pos_rate).cuda()
    channels_last = cfg["model"] == "convnext_t"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if cfg["grad_ckpt"] and cfg["model"] == "clip":
        model.visual.set_grad_checkpointing(True)
    groups = model.param_groups(cfg["lr"], cfg["head_lr"], cfg["wd"], layer_decay=cfg["layer_decay"])
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda")
    crit = WeightedBCE(pos_rate, cfg["weighted_loss"], cfg["label_smoothing"]).cuda()
    steps_per_epoch = len(train_dl) // cfg["accum"]
    total = steps_per_epoch * cfg["epochs"]
    warm = int(steps_per_epoch * cfg["warmup_epochs"])
    ema_decay = cfg["ema_decay"]
    ema = EMA(model, ema_decay)
    print(f"[{cfg['name']}] train {len(train_ds)} imgs, {n_epoch}/epoch, {steps_per_epoch} steps/epoch, "
          f"{total} total, warmup {warm}, balanced={cfg['balanced']} degrade={cfg['degrade']}", flush=True)

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    log_path = os.path.join(run, "log.csv")
    best = -1.0
    step = 0
    t_start = time.time()
    # one bar over every training batch of the run: its ETA is the remaining training time (val passes excluded)
    bar = tqdm(total=len(train_dl) * cfg["epochs"], desc=cfg["name"], unit="it", dynamic_ncols=True,
               mininterval=BAR_INTERVAL, smoothing=0.05)
    log = bar.write
    for epoch in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        run_loss, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        bar.set_description(f"{cfg['name']} ep {epoch + 1}/{cfg['epochs']}")
        for i, (x, y, _) in enumerate(train_dl):
            x = x.cuda(non_blocking=True).float()
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            y = y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y) / cfg["accum"]
            scaler.scale(loss).backward()
            if (i + 1) % cfg["accum"] == 0:
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * lr_at(step)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update(model, step)
                step += 1
            run_loss += loss.item() * cfg["accum"]
            n += 1
            bar.update(1)
            if i % 20 == 0:
                el = time.time() - t0
                bar.set_postfix(loss=f"{run_loss / n:.4f}",
                                img_s=f"{(i + 1) * cfg['batch_size'] / max(el, 1e-6):.0f}", refresh=False)
            if cfg["max_steps"] and step >= cfg["max_steps"]:
                break
        ep_time = time.time() - t0
        row = {"epoch": epoch, "train_loss": run_loss / max(n, 1), "epoch_time_s": round(ep_time, 1)}
        cands = {"ema": ema.module}
        if cfg["eval_raw"]:
            cands["raw"] = model
        if has_val:
            for tag, m in cands.items():
                z = predict_logits(m, val_dl, channels_last=channels_last, desc=f"val {tag}")
                res = protocol_metrics(val_ds.labels, 1 / (1 + np.exp(-z)), val_src)
                log(format_row(f"ep{epoch} {tag}", res))
                for s in SOURCES + ["mean"]:
                    row[f"{tag}_{s}_CA"] = round(res[s]["Challenge_Avg"], 5)
                    row[f"{tag}_{s}_mA"] = round(res[s]["mA"], 5)
                    row[f"{tag}_{s}_F1"] = round(res[s]["Instance_F1"], 5)
                ca = res["mean"]["Challenge_Avg"]
                if ca > best:
                    best = ca
                    torch.save({"model": m.state_dict(), "epoch": epoch, "tag": tag, "val_CA": ca},
                               os.path.join(run, "best.pt"))
                    np.savez(os.path.join(run, "val_logits.npz"), logits=z, labels=val_ds.labels, sources=val_src)
                    with open(os.path.join(run, "val_metrics.json"), "w") as fh:
                        json.dump({"epoch": epoch, "tag": tag,
                                   **{s: {k: v for k, v in res[s].items() if k != "mA_per_attr"} for s in res}},
                                  fh, indent=1)
                    with open(os.path.join(run, "per_attribute.csv"), "w", newline="") as fh:
                        w = csv.writer(fh)
                        w.writerow(["attribute"] + [f"mA_{s}" for s in SOURCES] + ["mA_mean"])
                        for k, name in enumerate(ATTRIBUTES):
                            w.writerow([name] + [round(float(res[s]["mA_per_attr"][k]), 4) for s in SOURCES]
                                       + [round(float(res["mean"]["mA_per_attr"][k]), 4)])
        torch.save({"model": ema.module.state_dict(), "epoch": epoch}, os.path.join(run, "last_ema.pt"))
        torch.save({"model": model.state_dict(), "epoch": epoch}, os.path.join(run, "last_raw.pt"))
        row["elapsed_min"] = round((time.time() - t_start) / 60, 1)
        new = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        log(f"epoch {epoch} done in {ep_time / 60:.1f} min; best val CA {best:.4f}")
        if cfg["max_steps"] and step >= cfg["max_steps"]:
            break
    bar.close()

    if has_val and os.path.exists(os.path.join(run, "best.pt")):
        m, _ = load_run_model(run, "best.pt")
        m = m.cuda()
        if channels_last:
            m = m.to(memory_format=torch.channels_last)
        z = predict_logits(m, val_dl, flip=True, channels_last=channels_last, desc="val best +flip")
        np.savez(os.path.join(run, "val_logits_flip.npz"), logits=z, labels=val_ds.labels, sources=val_src)
        print(format_row("best +flipTTA", protocol_metrics(val_ds.labels, 1 / (1 + np.exp(-z)), val_src)), flush=True)
    print(f"done. best val CA (offsets=0, no TTA) {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
