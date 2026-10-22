# Deliverables: UPAR Challenge 2027, Track 1

**Group:** Muhammad Ali, Fasih Hasan Khan

| # | Deliverable | Where |
|---|---|---|
| 1 | Brief description of the task and approach | this file (below) |
| 2 | Working code and experimental results | repository root: `src/`, `configs/`, `tools/`, `results/` (metrics, `ablations.csv`), `figures/`, [`../README.md`](../README.md) |
| 3 | Report, 2-3 pages (LaTeX source and PDF) | [`report/report.tex`](report/report.tex), [`report/report.pdf`](report/report.pdf) |
| 4 | Proof of successful submission | [`proof_of_submission/`](proof_of_submission/): screenshot of sub03, Codabench record and run logs |
| 5 | Short demo presentation | [`presentation/UPAR_Track1_presentation.pptx`](presentation/UPAR_Track1_presentation.pptx) (speaker notes included) |

## Task

Track 1 of the UPAR Challenge (WACV 2027 Real-World Surveillance Workshop, Codabench 18180) asks for a model that
predicts 40 binary attributes of a pedestrian (age, gender, hair, clothing length, type and colour, accessories) from a
cropped surveillance image. The organizers provide training data from Market1501, PA100k and PETA. Submissions are
code, run on a hidden test set of 29,248 images on an A10 GPU within 5 minutes. The ranking metric is the harmonic mean
of label-based mA and instance F1, averaged over datasets, with a fixed 0.5 threshold.

## Approach

We fine-tune the CLIP ViT-B/16 image encoder (and, later, ViT-L/14) with five additions:

- **N1 Language-anchored head:** each attribute's classifier starts as CLIP's text embedding of attribute prompts.
- **N2 Domain-balanced sampling:** each dataset makes up a third of every epoch.
- **N3 Surveillance degradation augmentation:** low resolution, JPEG artefacts, blur and lighting changes.
- **N4 Protocol-matched calibration:** per-attribute logit offsets fitted on the exact challenge metric, which
  matters because the 0.5 threshold is fixed.
- **N5 Structure-aware decoding** for exclusive attribute groups.

Everything is exported as one fp16 TorchScript file.

## Results

| | Val (held-out, +N4+N5) | Hidden test |
|---|---|---|
| ConvNeXt-T baseline | 0.8569 | - |
| **CLIP ViT-B/16 (sub03, submitted)** | 0.8742 | **0.7699** |
| CLIP ViT-L/14 (sub06) | 0.8730 | not scored |
| ViT-B + ViT-L ensemble (sub07) | **0.8803** | not scored |

We built four more submissions after sub03: train+val retraining, ViT-L/14, and two ensembles. They could not be scored.
One upload's prediction step completed but was never scored, and from 21 Oct 13:45 UTC codabench.org stopped
responding, until after the phase closed (22 Oct 12:00 UTC). See the report, Section 4.

## Rebuilding the report and slides

```bash
cd deliverables/report && pdflatex report.tex && pdflatex report.tex
cd deliverables/presentation && npm install pptxgenjs@3.12.0 && node build_deck.js
```

The report and slides include `proof_of_submission/sub03_screenshot.png` automatically when the file is present.
