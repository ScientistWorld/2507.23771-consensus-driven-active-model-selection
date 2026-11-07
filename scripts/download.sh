#!/bin/bash
# Download all data and models needed for this environment.
#
# Estimated total download size: 0 MB. Python packages (numpy, scipy) are
# installed by environment/setup.sh.
#
# The benchmark inputs are small files derived from the public ModelSelector
# release (https://github.com/RobustML-Lab/model-selector, resources/datasets/
# glue/{rte,sst2,mrpc,cola}/{predictions,oracle}.npy) and are tracked in git:
# data/benchmarks/<task>_train.json holds the visible half of each task
# (candidate predictions + labels). The held-out half is built and kept by
# the hidden evaluator. The full-pool upstream arrays are deliberately NOT
# fetched into this workspace: they contain the held-out labels.
#
# This script is idempotent and verifies the tracked benchmark files.

set -euo pipefail

cd /home/user

python3 - <<'PY'
import hashlib
from pathlib import Path

expected = {
    "glue_rte_train.json": "7fa84c73359e35a2aee510f208efd2d037cb924f8665bd03c99317315f50c05b",
    "glue_sst2_train.json": "f02c4b137710dbe3eab87a6aba4fa29fe8fd25c00ff71c246535faff5ee88555",
    "glue_mrpc_train.json": "e0b86bb32287322d261922deac10013fbcf5d8ba69515729c78c89662269eddd",
    "glue_cola_train.json": "edfa637732f719c00e1084d4f929ac8fa127300cf880cd9e009cc081ff9146f4",
}
for name, digest in expected.items():
    path = Path("data/benchmarks") / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f"{path}: checksum mismatch ({actual}); restore it with git checkout")
    print(f"ok {path}")
PY
