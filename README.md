# UPAR 2027 Track 1 — Language-anchored, domain-balanced CLIP-PAR with protocol-matched calibration

Code for our entry to Track 1 (Pedestrian Attribute Recognition) of the UPAR Challenge
@ WACV 2027 RWS workshop (Codabench 18180). Trained only on the organizers' train (and val) data.

## Project layout

| Folder | What is in it | In git? |
|---|---|---|
| `src/` | training, models, data, calibration (N4), decoding (N5), export | yes |
| `configs/` | one YAML per training run (`base:` inherits from another config) | yes |
| `tools/` | packaging, ingestion simulator, speed benchmark, figures, cache builder | yes |
| `submission/` | the package that gets zipped: `run.py`, `metadata.yaml`, `assets/` (exported model) | code only |
| `results/<run>/` | one folder per training run: `train.log`, `log.csv`, metrics, `calib.json`, checkpoints | weights ignored |
| `upload/` | **the zips to upload to Codabench** | no |
| `figures/` | report figures | yes |
| `data/raw/` | Market1501, PA100k, PETA and annotations (moved out of the starter kit) | no |
| `data/cache/` | 256x128 uint8 memmaps used for training | no |
| `logs/` | full console output of each run (`<config>.log`), plus download/cache-build logs | no |
| `starter_kit/` | the organizers' repo (download script, sample submission) | no |

## Setup (Windows/Linux, Python 3.11, CUDA 12.1 build)

```bash
python -m venv .venv && . .venv/Scripts/activate        # Linux: . .venv/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
git clone https://github.com/speckean/UPAR-Challenge-2027.git starter_kit
cd starter_kit && python download_datasets.py && cd ..   # needs gdown==5.2.0 (fuzzy= arg)
mkdir data && mv starter_kit/data data/raw
python tools/preprocess_cache.py                         # 256x128 uint8 memmaps in data/cache/
pytest -q tests
```

## Pipeline

| Step | Command |
|---|---|
| Zero-shot CLIP + text-head init (N1) | `python src/zero_shot.py` |
| Train Model A (ConvNeXt-T) | `python src/train.py --config configs/convnext_t.yaml` |
| Train Model B (CLIP ViT-B/16) | `python src/train.py --config configs/clip_b16.yaml` |
| Calibrate offsets (N4) + select decoding groups (N5) | `python src/calibrate.py --run results/<run> --logits val_logits_flip.npz` |
| Export TorchScript fp16 package | `python src/export.py --run results/<run>` |
| Simulate Codabench ingestion (validity, score, speed) | `python tools/simulate_ingestion.py --package submission` |
| Zip | `python tools/package.py --src submission --out upload/<name>.zip` |

Training shows a progress bar over the whole run, with the remaining training time, loss and img/s. Validation
passes have their own short bar, and each epoch's results are printed and saved to `results/<run>/train.log`.

## Components

- **N1 Language-anchored head**: `logit_k = s·cos(f, w_k) + b_k`, `w_k` initialised from CLIP text embeddings of
  attribute prompts (`src/prompts.py`), `b_k` from train positive rates. Ablation: random `w_k`.
- **N2 Domain-balanced sampling**: each source dataset contributes 1/3 of every epoch (the metric averages per dataset).
- **N3 Surveillance degradation augmentation**: down/up-scaling, JPEG, blur, brightness/contrast, random erasing.
- **N4 Protocol-matched calibration**: one global per-attribute logit offset fitted by coordinate ascent on the exact
  protocol (Challenge_Avg per dataset, averaged), shrunk by λ chosen on a held-out half of val; baked into the model.
- **N5 Structure-aware decoding**: for (near-)exclusive groups, force the arg-max above 0.5 if nothing is; kept per group
  only if it improves held-out val.

## Results

All numbers are Challenge_Avg on the official val split, averaged over Market1501, PA100k and PETA (the test
protocol). "Held-out" means N4 offsets and N5 groups fitted on one half of val and scored on the other (2-fold mean).
Full table: `results/ablations.csv`; figures: `figures/`.

| Model | val, offsets=0 | held-out +N4+N5 |
|---|---|---|
| Prior-only baseline (organizers) | 0.530 | – |
| CLIP ViT-B/16 zero-shot (no training) | 0.638 | – |
| Model A: ConvNeXt-T, N2+N3, 10 ep (best ep 5), +flip-TTA | 0.8452 | 0.8569 |
| **Model B: CLIP ViT-B/16, N1+N2+N3, 6-ep schedule stopped after ep 3** | **0.8574** | **0.8742** |
| Model B recipe with a 3-epoch schedule | 0.8551 | 0.8710 |
| Ensemble 0.7·B + 0.3·A (logit average) | 0.8585 | 0.8765 |

Ablations (2-epoch CLIP ViT-B/16, full recipe = 0.8709 held-out):

| Component removed / changed | held-out +N4+N5 | Effect of the component |
|---|---|---|
| N1: random head instead of text-initialised | 0.8675 | +0.0034 |
| N2: natural sampling instead of equal per dataset | 0.8665 | +0.0044 |
| N2 variant: sampling ∝ sqrt(dataset size) | 0.8694 | −0.0015 vs equal |
| N3: no degradation, clean val | 0.8717 | −0.0008 (noise) |
| N3: no degradation, degraded val (0.4x scale, JPEG q40) | 0.8605 vs 0.8665 | +0.0060 |
| N4: no calibration (Model B) | 0.8575 | +0.0160 |
| N5: no structured decoding | – | +0.0001 to +0.0007 |

## Submissions (`upload/`)

| Zip | Contents | Val held-out | Test |
|---|---|---|---|
| `sub01_smoke_convnext.zip` | ConvNeXt after 10% of an epoch (pipeline test) | ~0.766 | not uploaded |
| `sub02_convnext_t_calib.zip` | Model A + N4 + N5 + flip-TTA | 0.8569 | not uploaded |
| `sub03_clip_b16_calib.zip` | Model B + N4 (λ=1) + N5 | 0.8742 | uploaded 2026-10-20, pending |
| `sub04_clip_b16_lam05.zip` | Model B + N4 shrunk to λ=0.5 + N5 | ~0.867 | not uploaded |
| `sub05_clip_b16_trainval.zip` | Model B recipe trained on train+val, sub03's offsets | n/a (val seen) | not uploaded |

Rebuild sub05: `python src/train.py --config configs/clip_b16_trainval.yaml`, then
`python src/export.py --run results/clip_b16_trainval --ckpt last_ema.pt --calib ../clip_b16/calib.json --no-flip`.
