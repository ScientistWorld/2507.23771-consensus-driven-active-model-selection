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

# Finish the MRPC expansion after the first CPU job timed out with the
# baselines and CODA seeds 0-2 already written.
START_SEED=3 SKIP_EXISTING=1 MERGE_EXISTING=1 bash scripts/method.sh glue_mrpc 100 5 128

python validate.py --compare
