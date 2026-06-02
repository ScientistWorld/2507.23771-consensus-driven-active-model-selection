#!/bin/bash
# Run unsupervised model selection methods at step 0.
#
# Usage:
#   bash scripts/unsupervised.sh <task> <grid>

set -e

cd /home/user

TASK="${1:-glue_rte}"
GRID="${2:-128}"
DATA_DIR="${DATA_DIR:-data/downloads/modelselector}"

python method/coda_reproduction.py \
  --data-dir "$DATA_DIR" \
  --task "$TASK" \
  --method coda \
  --budget 0 \
  --seeds 1 \
  --coda-grid "$GRID" \
  --output-dir outputs/coda

python method/coda_reproduction.py \
  --data-dir "$DATA_DIR" \
  --task "$TASK" \
  --method ensv \
  --budget 0 \
  --seeds 1 \
  --output-dir outputs/coda

METHODS="ensv coda" bash scripts/evaluate.sh \
  outputs/coda \
  "$TASK" \
  "${TASK}_unsupervised_selection"
