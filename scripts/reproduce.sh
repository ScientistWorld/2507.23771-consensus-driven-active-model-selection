#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user

TASK="${1:-glue_rte}"
BUDGET="${2:-100}"
SEEDS="${3:-5}"
GRID="${4:-128}"

if [ "$TASK" = "all" ]; then
  python data/prepare_modelselector.py --tasks glue_rte glue_sst2
  bash scripts/baseline.sh glue_rte 100 5
  bash scripts/method.sh glue_rte 100 5 128
  MERGE_EXISTING=1 bash scripts/baseline.sh glue_sst2 100 2
  MERGE_EXISTING=1 bash scripts/method.sh glue_sst2 100 2 64
  exit 0
fi

python data/prepare_modelselector.py --tasks "$TASK"
bash scripts/baseline.sh "$TASK" "$BUDGET" "$SEEDS"
bash scripts/method.sh "$TASK" "$BUDGET" "$SEEDS" "$GRID"
