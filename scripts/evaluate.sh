#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# Runs every selector registered in outputs/selectors/*.json (see
# scripts/register_selector.sh and data/selection_protocol.py) against the
# visible label oracle on data/benchmarks/*_train.json, for the six
# experiments of scoring/reference.json and seeds 0-4, and writes
# scoring/scores.json (the reproduction numbers, visible slice).
# The evaluator drives the label queries itself, records the selected model
# in every round and recomputes all metrics; it never imports method/.
#
# Usage: bash scripts/evaluate.sh            (no arguments)
#   SELECTORS_DIR=<dir> overrides where selector manifests are read from.

set -euo pipefail

cd /home/user

python3 eval/train/evaluate.py \
  --selectors-dir "${SELECTORS_DIR:-outputs/selectors}" \
  --output scoring/scores.json \
  --episodes-output logs/eval_main/episodes.json \
  --log-dir logs/eval_main
