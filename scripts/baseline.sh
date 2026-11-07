#!/bin/bash
# Run the paper's baselines through the evaluator.
#
# Registers random sampling and uncertainty sampling (both select by
# empirical risk on the queried labels) and the zero-label ensemble
# validation baseline (EnsV), then scores all registered selectors with
# scripts/evaluate.sh. Set SKIP_EVAL=1 to only register.
#
# Usage: bash scripts/baseline.sh

set -euo pipefail

cd /home/user

bash scripts/register_selector.sh random python baseline/selectors.py --acquisition random
bash scripts/register_selector.sh uncertainty python baseline/selectors.py --acquisition uncertainty
bash scripts/register_selector.sh ensv python baseline/selectors.py --acquisition ensv

if [ "${SKIP_EVAL:-0}" != "1" ]; then
  bash scripts/evaluate.sh
fi
