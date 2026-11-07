#!/bin/bash
# Score every registered selector on the visible train slice.
#
# Same protocol, experiments, seeds and metrics as the held-out test slice,
# answered from the visible labels in data/benchmarks/*_train.json.
#
# Usage: bash scripts/evaluate_train.sh      (no arguments)
#   SELECTORS_DIR=<dir> overrides where selector manifests are read from.

set -euo pipefail

cd /home/user

python3 eval/train/evaluate.py \
  --selectors-dir "${SELECTORS_DIR:-outputs/selectors}" \
  --output "${TRAIN_SCORES_OUT:-scoring/scores_train.json}" \
  --episodes-output "${TRAIN_EPISODES_OUT:-logs/eval_train/episodes.json}" \
  --log-dir logs/eval_train
