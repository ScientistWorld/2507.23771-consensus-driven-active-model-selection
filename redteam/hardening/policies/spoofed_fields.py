"""Shortcut attempt: run the reference CODA but declare forged bookkeeping.

Every protocol message carries extra fields claiming a 1-label budget, a huge
candidate pool and item count, and impossible negative regrets. The old gym
read a self-declared budget from the trace; this checks that only the
evaluator's own counts matter.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from method.coda_reproduction import CODAPolicy  # noqa: E402

FORGED = {"budget": 1, "label_budget": 1, "num_items": 10**6, "num_models": 10**6,
          "regret": -1e9, "cumulative_regret_at_100": -1e9, "near_optimal_at_100": 1}

args = argparse.Namespace(alpha=0.1, temperature=0.5, eta=0.01, grid=128, stable_rounds=0, min_labels=0,
                         acquisition="eig")
channel, sys.stdout = sys.stdout, sys.stderr
episode = json.loads(sys.stdin.readline())
policy = CODAPolicy(episode, args)
budget = int(episode["budget"])
for t in range(budget + 1):
    choice = policy.select()
    query = policy.query() if t < budget else None
    channel.write(json.dumps({"select": choice, "query": query, **FORGED}) + "\n")
    channel.flush()
    if query is None:
        break
    policy.observe(query, json.loads(sys.stdin.readline())["label"])
sys.stdin.readline()
