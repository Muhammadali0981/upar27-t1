#!/bin/sh
# Short-schedule (2-epoch) CLIP ablations, run sequentially after the main CLIP run.
cd "$(dirname "$0")/.."
until grep -qE "^done\.|Traceback" results_clip_b16.log; do sleep 30; done
for c in clip_b16_short clip_b16_ablate_randhead clip_b16_ablate_natural clip_b16_ablate_nodegrade; do
  ./.venv/Scripts/python -u src/train.py --config configs/$c.yaml > results_$c.log 2>&1
done
