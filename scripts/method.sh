#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

TASK="${1:-glue_rte}"
BUDGET="${2:-100}"
SEEDS="${3:-5}"
GRID="${4:-128}"
DATA_DIR="${DATA_DIR:-data/downloads/modelselector}"
START_SEED="${START_SEED:-0}"
SKIP_ARGS=()
if [ "${SKIP_EXISTING:-0}" = "1" ]; then
  SKIP_ARGS=(--skip-existing)
fi

python method/coda_reproduction.py \
  --data-dir "$DATA_DIR" \
  --task "$TASK" \
  --method coda \
  --budget "$BUDGET" \
  --seeds "$SEEDS" \
  --start-seed "$START_SEED" \
  "${SKIP_ARGS[@]}" \
  --coda-grid "$GRID" \
  --output-dir outputs/coda

bash scripts/evaluate.sh outputs/coda "$TASK" "${TASK}_active_selection"
