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

# Additional GLUE setting after the RTE core claim run.  Two seeds and a
# smaller integration grid keep the CPU job under the scheduler timeout while
# preserving CODA's active-selection procedure.
MERGE_EXISTING=1 bash scripts/baseline.sh glue_sst2 100 2
MERGE_EXISTING=1 bash scripts/method.sh glue_sst2 100 2 64
