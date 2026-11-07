#!/bin/bash
# Reproduce the paper's results end-to-end on the visible slice.
#
# Registers the baselines and CODA, then scores them on every experiment
# (GLUE RTE/SST2/MRPC/CoLA active selection with a 100-label budget, and
# zero-label selection on RTE/SST2), five seeds each. Writes
# scoring/scores.json and scoring/scores_train.json. Runs on one CPU core in
# about half an hour; no GPU and no network needed.

set -euo pipefail

cd /home/user

SKIP_EVAL=1 bash scripts/baseline.sh
SKIP_EVAL=1 bash scripts/method.sh
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
