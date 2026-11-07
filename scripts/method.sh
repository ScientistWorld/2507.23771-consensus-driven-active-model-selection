#!/bin/bash
# Run the paper's proposed method (CODA) through the evaluator.
#
# Registers the CODA selector (method/coda_reproduction.py, paper defaults
# alpha=0.1, T=0.5, eta=0.01) and scores all registered selectors with
# scripts/evaluate.sh. Set SKIP_EVAL=1 to only register.
#
# Usage: bash scripts/method.sh

set -euo pipefail

cd /home/user

bash scripts/register_selector.sh coda python method/coda_reproduction.py

if [ "${SKIP_EVAL:-0}" != "1" ]; then
  bash scripts/evaluate.sh
fi
