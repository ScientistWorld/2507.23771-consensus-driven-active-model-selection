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

python data/prepare_modelselector.py --tasks glue_mrpc

# Add another paper-reported active-selection task without rerunning the
# already completed RTE/SST2 jobs.
MERGE_EXISTING=1 bash scripts/baseline.sh glue_mrpc 100 5
MERGE_EXISTING=1 bash scripts/method.sh glue_mrpc 100 5 128

python validate.py --compare
