#!/bin/sh
# Short-schedule (2-epoch) CLIP ablations, run one after another; then N4/N5 calibration of each.
# Each run writes results/<name>/train.log; console output also goes to logs/<config>.log.
cd "$(dirname "$0")/.."
for c in clip_b16_short clip_b16_ablate_randhead clip_b16_ablate_natural clip_b16_ablate_nodegrade clip_b16_ablate_sqrt; do
  ./.venv/Scripts/python -u src/train.py --config configs/$c.yaml > logs/$c.log 2>&1
done
for r in clip_b16_short clip_b16_short_randhead clip_b16_short_natural clip_b16_short_nodegrade clip_b16_short_sqrt; do
  ./.venv/Scripts/python -u src/calibrate.py --run results/$r
done
