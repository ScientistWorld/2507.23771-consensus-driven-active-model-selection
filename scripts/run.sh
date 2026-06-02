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

python data/prepare_modelselector.py --tasks glue_rte glue_sst2

# Add Table 5 no-label model-selection experiments without rerunning the
# already completed 100-label active-selection jobs.
MERGE_EXISTING=1 bash scripts/unsupervised.sh glue_rte 128
MERGE_EXISTING=1 bash scripts/unsupervised.sh glue_sst2 64

python validate.py --compare
