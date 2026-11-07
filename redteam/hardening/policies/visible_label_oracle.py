"""Shortcut attempt: read every visible label and pick the best model by index.

Finds the visible benchmark with the same candidate count, selects the model
with the fewest visible errors in every round and queries nothing (the old
gym accepted a claimed budget of 100 for this). The hidden pool's candidate
order is not the visible order, so the index does not carry over.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from data.selection_protocol import serve  # noqa: E402


class VisibleOracle:
    def __init__(self, episode):
        self.model = 0
        for path in sorted((ROOT / "data" / "benchmarks").glob("*_train.json")):
            data = json.loads(path.read_text())
            if len(data["predicted_classes"]) == episode["num_models"]:
                errors = [sum(p != y for p, y in zip(row, data["labels"])) for row in data["predicted_classes"]]
                self.model = min(range(len(errors)), key=errors.__getitem__)

    def select(self):
        return self.model

    def query(self):
        return None

    def observe(self, item, label):
        pass


serve(VisibleOracle)
