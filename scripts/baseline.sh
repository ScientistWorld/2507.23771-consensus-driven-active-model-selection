#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user

TASK="${1:-glue_rte}"
BUDGET="${2:-100}"
SEEDS="${3:-5}"
DATA_DIR="${DATA_DIR:-data/downloads/modelselector}"

for METHOD in random uncertainty; do
  python method/coda_reproduction.py \
    --data-dir "$DATA_DIR" \
    --task "$TASK" \
    --method "$METHOD" \
    --budget "$BUDGET" \
    --seeds "$SEEDS" \
    --output-dir outputs/coda
done

METHODS="random uncertainty" bash scripts/evaluate.sh outputs/coda "$TASK" "${TASK}_active_selection"
