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

# Corrected CODA traces are already present for RTE (5 seeds), SST2 (2 seeds),
# and MRPC (1 seed). This job is intentionally evaluation-only so it finishes
# inside the CPU scheduler limit and materializes the refreshed main and
# active-selection train/test score files.
bash scripts/unsupervised.sh glue_rte 128
bash scripts/unsupervised.sh glue_sst2 64

bash scripts/evaluate.sh /home/user/outputs/coda
bash scripts/evaluate_train.sh /home/user/outputs/coda
bash scripts/evaluate_test.sh /home/user/outputs/coda
python validate.py --compare
