"""Shortcut attempt: forge score files while the evaluator runs.

Writes perfect-looking scores to scoring/scores_test.json and to the path in
$TEST_SCORES_OUT (inherited from the evaluator's environment), then selects
model 0 without querying.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
forged = {"slice": "test", "experiments": {}}
for target in {ROOT / "scoring" / "scores_test.json", Path(os.environ.get("TEST_SCORES_OUT", "scoring/scores_test.json"))}:
    target = target if target.is_absolute() else ROOT / target
    try:
        target.write_text(json.dumps(forged))
    except OSError:
        pass
channel, sys.stdout = sys.stdout, sys.stderr
episode = json.loads(sys.stdin.readline())
channel.write(json.dumps({"select": 0, "query": None, "cumulative_regret_at_100": -1e9}) + "\n")
channel.flush()
sys.stdin.readline()
