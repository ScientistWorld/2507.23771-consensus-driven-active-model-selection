"""Hidden test evaluator for active model selection.

Every registered selector (``outputs/selectors/<name>.json`` holding
``{"command": [...]}``) is run as a subprocess against a label oracle that
only this evaluator holds. The selector receives the candidate predictions
for the held-out half of the item pool, and in every round must commit the model it
currently selects before it is told the label of the item it queries next.
The evaluator records those selections itself, counts the labels it reveals,
and recomputes every metric from the held-out labels. Nothing the selector
writes besides its per-round ``select``/``query`` choices is used.

Any protocol violation (crash, timeout, malformed message, out-of-range or
repeated query, querying past the budget) fails that selector on that
experiment: it is scored as if it had chosen the worst candidate in every
round and spent the whole budget. Standard library only; never imports
method code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCHMARK_DIR = HERE / "benchmarks"
BENCHMARK_SHA256 = {
    "glue_rte": "dd42565b52a38e93b9b5c5674c63229337d49a2384351d56cb62ce05b66513a1",
    "glue_sst2": "b2319b8d7c35fbadf6a34656eb1625c9309b3bf0ecbbd3e29c7171516eadc2db",
    "glue_qnli": "83a8596ad7bcb7df83a969e855a0a45eef4c444cf1c3a8eb4f74e2d8ddf138c9",
    "glue_cola": "159c25fce8e20c17bd22d923808e9ba62d7aaa4585259402299ea2bd86121737",
}
SEEDS = (0, 1, 2, 3, 4)
ACTIVE_BUDGET = 100
NEAR_OPTIMAL_PP = 1.0
EPISODE_TIMEOUT_SEC = 1800
EXPERIMENTS = {
    "glue_sst2_active_selection": ("glue_sst2", ACTIVE_BUDGET),
    "glue_qnli_active_selection": ("glue_qnli", ACTIVE_BUDGET),
    "glue_cola_active_selection": ("glue_cola", ACTIVE_BUDGET),
    "glue_rte_unsupervised_selection": ("glue_rte", 0),
    "glue_sst2_unsupervised_selection": ("glue_sst2", 0),
}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
LABEL_KEY = "coda-hidden-oracle-v1"


def label_offset(task: str, position: int, num_classes: int) -> int:
    digest = hashlib.sha256(f"{LABEL_KEY}|{task}|{position}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % num_classes


class Violation(Exception):
    """The selector broke the oracle protocol."""


@dataclass
class Benchmark:
    task: str
    num_classes: int
    predicted_classes: list[list[int]]
    labels: list[int]
    errors: list[int]

    @property
    def num_models(self) -> int:
        return len(self.predicted_classes)

    @property
    def num_items(self) -> int:
        return len(self.labels)

    def regret(self, model: int) -> float:
        return 100.0 * (self.errors[model] - min(self.errors)) / self.num_items


def load_benchmark(task: str) -> Benchmark:
    path = BENCHMARK_DIR / f"{task}_test.json"
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != BENCHMARK_SHA256[task]:
        raise SystemExit(f"{path} does not match its recorded checksum ({digest})")
    data = json.loads(raw)
    preds = data["predicted_classes"]
    num_classes = int(data["num_classes"])
    labels = [
        (code - label_offset(task, pos, num_classes)) % num_classes
        for pos, code in enumerate(data["label_code"])
    ]
    if not preds or any(len(row) != len(labels) for row in preds):
        raise SystemExit(f"{path}: malformed prediction matrix")
    errors = [sum(1 for p, y in zip(row, labels) if p != y) for row in preds]
    return Benchmark(task, num_classes, preds, labels, errors)


def discover_selectors(selectors_dir: Path) -> dict[str, object]:
    selectors: dict[str, object] = {}
    if not selectors_dir.is_dir():
        return selectors
    for path in sorted(selectors_dir.glob("*.json")):
        name = path.stem
        if not NAME_RE.match(name):
            print(f"skip selector manifest with unusable name: {path}", file=sys.stderr)
            continue
        try:
            spec = json.loads(path.read_text())
            command = spec["command"]
            if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
                raise ValueError("'command' must be a non-empty list of strings")
            selectors[name] = command
        except Exception as exc:  # broken manifest: the method is on disk and fails
            selectors[name] = Violation(f"invalid manifest {path}: {exc}")
    return selectors


def _reject_constant(token: str):
    raise ValueError(f"non-finite number {token}")


def _parse(line: str) -> dict:
    try:
        msg = json.loads(line, parse_constant=_reject_constant)
    except ValueError as exc:
        raise Violation(f"malformed message {line[:120]!r}: {exc}") from None
    if not isinstance(msg, dict):
        raise Violation(f"message is not an object: {line[:120]!r}")
    return msg


def _index(value, bound: int, what: str) -> int:
    if type(value) is not int or not 0 <= value < bound:
        raise Violation(f"{what} must be an integer in [0, {bound}), got {value!r}")
    return value


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(var, "1")
    return env


def _resolve(command: list[str]) -> list[str]:
    if command[0] in ("python", "python3"):
        return [sys.executable, *command[1:]]
    return list(command)


def run_episode(command: list[str], bench: Benchmark, budget: int, seed: int, log_path: Path):
    """Drive one selector episode. Returns (selections, labels_revealed)."""
    episode = {
        "protocol": 1,
        "num_models": bench.num_models,
        "num_items": bench.num_items,
        "num_classes": bench.num_classes,
        "budget": budget,
        "seed": seed,
        "predicted_classes": bench.predicted_classes,
    }
    deadline = time.monotonic() + EPISODE_TIMEOUT_SEC
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        try:
            proc = subprocess.Popen(
                _resolve(command), cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=log, text=True, bufsize=1, env=_child_env(),
            )
        except OSError as exc:
            raise Violation(f"cannot start selector: {exc}") from None
        lines: queue.Queue = queue.Queue()

        def pump() -> None:
            try:
                for line in proc.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        threading.Thread(target=pump, daemon=True).start()

        def send(obj: dict) -> None:
            try:
                proc.stdin.write(json.dumps(obj) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                raise Violation("selector closed its input early") from None

        def receive(round_index: int) -> dict:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Violation(f"timeout after {EPISODE_TIMEOUT_SEC}s")
            try:
                line = lines.get(timeout=remaining)
            except queue.Empty:
                raise Violation(f"timeout after {EPISODE_TIMEOUT_SEC}s") from None
            if line is None:
                raise Violation(f"selector exited before answering round {round_index}")
            return _parse(line)

        selections: list[int] = []
        queried: set[int] = set()
        try:
            send(episode)
            for round_index in range(budget + 1):
                msg = receive(round_index)
                selections.append(_index(msg.get("select"), bench.num_models, "select"))
                query = msg.get("query")
                if query is None:
                    break
                if round_index == budget:
                    raise Violation(f"query beyond the {budget}-label budget")
                item = _index(query, bench.num_items, "query")
                if item in queried:
                    raise Violation(f"item {item} queried twice")
                queried.add(item)
                send({"label": bench.labels[item]})
            send({"type": "end"})
            try:
                proc.stdin.close()
            except OSError:
                pass
            try:
                code = proc.wait(timeout=max(1.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                raise Violation("selector did not exit after the episode ended") from None
            if code != 0:
                raise Violation(f"selector exited with status {code}")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    return selections, len(queried)


def episode_metrics(bench: Benchmark, budget: int, selections: list[int], labels_revealed: int) -> dict[str, float]:
    rounds = selections + [selections[-1]] * (budget + 1 - len(selections))
    regret = [bench.regret(model) for model in rounds]
    if budget == 0:
        return {"initial_regret": regret[0]}
    return {
        "initial_regret": regret[0],
        "cumulative_regret_at_25": math.fsum(regret[1:26]),
        "cumulative_regret_at_50": math.fsum(regret[1:51]),
        "cumulative_regret_at_100": math.fsum(regret[1:101]),
        "instant_regret_at_50": regret[50],
        "instant_regret_at_100": regret[100],
        "near_optimal_at_100": 1.0 if regret[100] <= NEAR_OPTIMAL_PP else 0.0,
        "label_budget": float(labels_revealed),
    }


def failed_metrics(bench: Benchmark, budget: int) -> dict[str, float]:
    worst = max(range(bench.num_models), key=bench.errors.__getitem__)
    return episode_metrics(bench, budget, [worst] * (budget + 1), budget)


def evaluate(selectors_dir: Path, log_dir: Path) -> tuple[dict, dict, list]:
    selectors = discover_selectors(selectors_dir)
    if not selectors:
        raise SystemExit(f"no selector manifests found in {selectors_dir}")
    benchmarks = {task: load_benchmark(task) for task, _ in EXPERIMENTS.values()}
    experiments: dict[str, dict] = {}
    failures: dict[str, dict[str, list[str]]] = {}
    episodes: list[dict] = []
    for experiment, (task, budget) in EXPERIMENTS.items():
        bench = benchmarks[task]
        results = {}
        for name, command in selectors.items():
            rows, problems = [], []
            if isinstance(command, Violation):
                problems.append(str(command))
            else:
                for seed in SEEDS:
                    start = time.monotonic()
                    log_path = log_dir / name / f"{experiment}_seed{seed}.log"
                    try:
                        selections, revealed = run_episode(command, bench, budget, seed, log_path)
                    except Violation as exc:
                        problems.append(f"seed {seed}: {exc}")
                        break
                    metrics = episode_metrics(bench, budget, selections, revealed)
                    rows.append(metrics)
                    # Per-round selections are deliberately not logged: they would
                    # reveal which hidden candidate index is best.
                    episodes.append({
                        "experiment": experiment, "method": name, "seed": seed,
                        "labels_revealed": revealed, "metrics": metrics,
                        "seconds": round(time.monotonic() - start, 2),
                    })
            if problems:
                failures.setdefault(experiment, {})[name] = problems
                results[name] = failed_metrics(bench, budget)
            else:
                results[name] = {key: math.fsum(r[key] for r in rows) / len(rows) for key in rows[0]}
            print(f"[test] {experiment} {name}: " + json.dumps(results[name]), flush=True)
        experiments[experiment] = {"results": results}
    return experiments, failures, episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selectors-dir", default=os.environ.get("SELECTORS_DIR", "outputs/selectors"))
    parser.add_argument("--output", default="scoring/scores_test.json")
    parser.add_argument("--episodes-output", default="logs/eval_test/episodes.json")
    parser.add_argument("--log-dir", default="logs/eval_test")
    args = parser.parse_args()

    experiments, failures, episodes = evaluate(ROOT / args.selectors_dir, ROOT / args.log_dir)
    scores = {"slice": "test", "experiments": experiments, "failures": failures}
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2) + "\n")
    episodes_out = ROOT / args.episodes_output
    episodes_out.parent.mkdir(parents=True, exist_ok=True)
    episodes_out.write_text(json.dumps(episodes) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
