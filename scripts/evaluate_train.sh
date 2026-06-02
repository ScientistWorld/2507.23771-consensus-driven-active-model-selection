#!/bin/bash
# Evaluate the visible train slice of method-produced active-selection traces.
#
# Mirrors scripts/evaluate.sh:
#   ./scripts/evaluate_train.sh <results_dir> <task> <experiment>

set -e

cd /home/user

mkdir -p scoring

RESULTS_DIR="${1:-outputs/coda}"
TASK="${2:-glue_rte}"
EXPERIMENT="${3:-glue_rte_active_selection}"
DATA_DIR="${DATA_DIR:-data/downloads/modelselector}"

METHOD_ARGS=()
if [ -n "${METHODS:-}" ]; then
  METHOD_ARGS=(--methods $METHODS)
fi

MERGE_ARGS=()
if [ "${MERGE_EXISTING:-0}" = "1" ]; then
  MERGE_ARGS=(--merge-existing)
fi

python eval/train/evaluate.py \
  --results-dir "$RESULTS_DIR" \
  --data-dir "$DATA_DIR" \
  --task "$TASK" \
  --experiment "$EXPERIMENT" \
  "${METHOD_ARGS[@]}" \
  "${MERGE_ARGS[@]}" \
  --output scoring/scores_train.json
