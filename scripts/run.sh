#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

python data/prepare_modelselector.py --tasks glue_rte glue_sst2 glue_mrpc

# The previous corrective CPU job completed all five RTE CODA seeds and SST2
# seed 0 before timing out during SST2 seed 1. Resume only the missing/stale
# corrected traces so this job fits the one-hour CPU-node limit.
rm -f outputs/coda/glue_sst2/coda/seed_1.json
python method/coda_reproduction.py \
  --data-dir data/downloads/modelselector \
  --task glue_sst2 \
  --method coda \
  --budget 100 \
  --seeds 2 \
  --start-seed 1 \
  --coda-grid 64 \
  --output-dir outputs/coda

# MRPC traces were produced before the corrected CODA acquisition change, so
# regenerate them. A 64-point quadrature grid preserves the CODA computation
# while fitting the available CPU wall-clock budget.
rm -f outputs/coda/glue_mrpc/coda/seed_*.json
python method/coda_reproduction.py \
  --data-dir data/downloads/modelselector \
  --task glue_mrpc \
  --method coda \
  --budget 100 \
  --seeds 5 \
  --coda-grid 64 \
  --output-dir outputs/coda

bash scripts/unsupervised.sh glue_rte 128
bash scripts/unsupervised.sh glue_sst2 64

bash scripts/evaluate.sh /home/user/outputs/coda
bash scripts/evaluate_train.sh /home/user/outputs/coda
bash scripts/evaluate_test.sh /home/user/outputs/coda
python validate.py --compare
