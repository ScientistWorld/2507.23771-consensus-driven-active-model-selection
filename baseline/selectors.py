"""Baseline selectors from the paper's comparison, as oracle-protocol policies.

random       query a uniformly random unlabeled item; select the model with
             the lowest empirical risk on the labels seen so far
uncertainty  query the unlabeled item whose mean committee prediction has the
             highest entropy; select by empirical risk
ensv         zero-label ensemble validation: select the model that agrees
             most with the committee's majority vote; never queries

As in the paper's released code, the empirical-risk baselines break ties
(in risk and in entropy) uniformly at random with the episode seed, so
before any label they select a random model. EnsV ties go to the lowest
index. Standard library only.

Usage: python baseline/selectors.py --acquisition {random,uncertainty,ensv}
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.selection_protocol import serve  # noqa: E402


def _argmax(values) -> int:
    best = 0
    for idx, value in enumerate(values):
        if value > values[best]:
            best = idx
    return best


def _random_argmax(values, rng: random.Random) -> int:
    top = max(values)
    ties = [idx for idx, value in enumerate(values) if value == top]
    return ties[0] if len(ties) == 1 else rng.choice(ties)


class EmpiricalRiskSelector:
    def __init__(self, episode: dict, acquisition: str):
        self.preds = episode["predicted_classes"]
        self.num_models = len(self.preds)
        self.num_items = len(self.preds[0])
        self.num_classes = int(episode["num_classes"])
        self.acquisition = acquisition
        self.rng = random.Random(int(episode["seed"]))
        self.unlabeled = list(range(self.num_items))
        self.correct = [0] * self.num_models
        if acquisition == "uncertainty":
            self.entropy = []
            for item in range(self.num_items):
                votes = [0] * self.num_classes
                for row in self.preds:
                    votes[row[item]] += 1
                probs = [v / self.num_models for v in votes]
                self.entropy.append(-sum(p * math.log(p + 1e-8) for p in probs))

    def select(self) -> int:
        return _random_argmax(self.correct, self.rng)

    def query(self) -> int | None:
        if not self.unlabeled:
            return None
        if self.acquisition == "random":
            return self.rng.choice(self.unlabeled)
        local = _random_argmax([self.entropy[item] for item in self.unlabeled], self.rng)
        return self.unlabeled[local]

    def observe(self, item: int, label: int) -> None:
        self.unlabeled.remove(item)
        for h, row in enumerate(self.preds):
            if row[item] == label:
                self.correct[h] += 1


class EnsembleValidationSelector:
    def __init__(self, episode: dict):
        preds = episode["predicted_classes"]
        num_classes = int(episode["num_classes"])
        consensus = []
        for item in range(len(preds[0])):
            votes = [0] * num_classes
            for row in preds:
                votes[row[item]] += 1
            consensus.append(_argmax(votes))
        agreement = [sum(p == c for p, c in zip(row, consensus)) for row in preds]
        self.choice = _argmax(agreement)

    def select(self) -> int:
        return self.choice

    def query(self) -> int | None:
        return None

    def observe(self, item: int, label: int) -> None:
        raise RuntimeError("ensemble validation does not query labels")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition", choices=["random", "uncertainty", "ensv"], required=True)
    args = parser.parse_args()
    if args.acquisition == "ensv":
        serve(EnsembleValidationSelector)
    else:
        serve(lambda episode: EmpiricalRiskSelector(episode, args.acquisition))


if __name__ == "__main__":
    main()
