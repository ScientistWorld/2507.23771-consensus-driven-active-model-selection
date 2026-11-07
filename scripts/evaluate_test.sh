#!/bin/bash
# Score every registered selector on the held-out test slice.
#
# The held-out labels live only in eval/test/. Each selector registered in
# outputs/selectors/*.json is run against the hidden label oracle; the
# evaluator records its selection every round, counts revealed labels and
# recomputes every metric. Protocol violations score as failed runs.
#
# Usage: bash scripts/evaluate_test.sh       (no arguments)
#   SELECTORS_DIR=<dir> overrides where selector manifests are read from.

set -euo pipefail

cd /home/user

python3 eval/test/evaluate.py \
  --selectors-dir "${SELECTORS_DIR:-outputs/selectors}" \
  --output "${TEST_SCORES_OUT:-scoring/scores_test.json}" \
  --episodes-output "${TEST_EPISODES_OUT:-logs/eval_test/episodes.json}" \
  --log-dir logs/eval_test
