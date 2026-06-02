#!/bin/bash
# Evaluate the visible train slice of method-produced active-selection traces.
#
# With no task/experiment arguments, evaluates every configured experiment.
#   ./scripts/evaluate_train.sh [active_results_dir] [task] [experiment]

set -e

cd /home/user

mkdir -p scoring

RESULTS_DIR="${1:-outputs/coda}"
TASK="${2:-}"
EXPERIMENT="${3:-}"
DATA_DIR="${DATA_DIR:-data/downloads/modelselector}"
UNSUPERVISED_RESULTS_DIR="${UNSUPERVISED_RESULTS_DIR:-outputs/coda_unsupervised}"

METHOD_ARGS=()
if [ -n "${METHODS:-}" ]; then
  METHOD_ARGS=(--methods $METHODS)
fi

MERGE_ARGS=()
if [ "${MERGE_EXISTING:-0}" = "1" ]; then
  MERGE_ARGS=(--merge-existing)
fi

PY_ARGS=(
  --results-dir "$RESULTS_DIR" \
  --unsupervised-results-dir "$UNSUPERVISED_RESULTS_DIR" \
  --data-dir "$DATA_DIR" \
  --output scoring/scores_train.json
)

if [ -n "$TASK" ]; then
  PY_ARGS+=(--task "$TASK")
fi
if [ -n "$EXPERIMENT" ]; then
  PY_ARGS+=(--experiment "$EXPERIMENT")
fi

python eval/train/evaluate.py \
  "${PY_ARGS[@]}" \
  "${METHOD_ARGS[@]}" \
  "${MERGE_ARGS[@]}"
