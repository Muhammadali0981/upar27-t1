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
| `logs/` | logs from runs started before the per-run `train.log` existed | no |
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

(filled in as runs finish — see `results/`)
