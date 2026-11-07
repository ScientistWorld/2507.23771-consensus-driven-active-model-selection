#!/bin/bash
# Hardening job (CPU). Reproduces the reference end to end and reruns the
# variants and shortcut attempts through the hidden test.
#
# 1. check the vectorized CODA against the original per-model loop
# 2. register the reference selectors (CODA + random, uncertainty, EnsV)
# 3. hidden test with no arguments -> scoring/scores_test.json
# 4. visible slice -> scoring/scores_train.json; scores.json <- reference run
# 5. variants and shortcuts through scripts/evaluate_test.sh
#    -> redteam/hardening/<case>/result.json, redteam/hardening/summary.json
# Steps leave markers in tmp/run_markers so a killed job resumes.

set -euo pipefail

cd /home/user
export PYTHONUNBUFFERED=1
MARK=tmp/run_markers
mkdir -p "$MARK" logs

step() {
  local name="$1"
  shift
  if [ -e "$MARK/$name" ]; then
    echo "== skip $name (done)"
    return
  fi
  echo "== $name $(date -u +%H:%M:%S)"
  "$@"
  touch "$MARK/$name"
}

step check_fast_coda python redteam/hardening/check_fast_coda.py
bash scripts/download.sh
rm -rf outputs/selectors
SKIP_EVAL=1 bash scripts/baseline.sh
SKIP_EVAL=1 bash scripts/method.sh
step evaluate_test bash scripts/evaluate_test.sh
step evaluate_train bash scripts/evaluate_train.sh
# scoring/scores.json records the reference reproduction on every
# experiment, i.e. the held-out evaluation (RTE, MRPC and CoLA exist only
# there). scripts/evaluate.sh, the scientist-facing command, rewrites it
# from the visible slice.
step evaluate_main python -c "import json; d = json.load(open('scoring/scores_test.json')); d['slice'] = 'reference_reproduction'; json.dump(d, open('scoring/scores.json', 'w'), indent=2)"
step cases python redteam/hardening/run_cases.py
echo "== validate $(date -u +%H:%M:%S)"
python validate.py || echo "validate.py reported issues (see above)"
python validate.py --post-design || echo "post-design check reported issues (see above)"
