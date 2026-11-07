"""Run variants and shortcut attempts through the hidden test, one case at a time.

Each case registers one selector named ``candidate`` in its own selectors
directory, optionally tampers with the visible benchmark files first, then
runs the unmodified ``bash scripts/evaluate_test.sh`` (and, for some cases,
``evaluate_train.sh``) in a subprocess. Visible data and scoring/ files are
restored byte-for-byte afterwards. The rank of a case is

    sum_e weight[e] * sum_m coefficient[e, m] * (candidate[e, m] - reference[e, m])

with weights/coefficients from scoring/reference.json and the reference
taken from the ``coda`` entry of scoring/scores_test.json (produced by the
main run before this script). Completed cases are skipped on rerun.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / os.environ.get("HARDENING_OUT", "redteam/hardening")
WORK = ROOT / "tmp" / "hardening"
REFERENCE_SCORES = ROOT / os.environ.get("REFERENCE_SCORES", "scoring/scores_test.json")
REFERENCE_EPISODES = ROOT / os.environ.get("REFERENCE_EPISODES", "logs/eval_test/episodes.json")
POL = "redteam/hardening/policies"
CODA = ["python", "method/coda_reproduction.py"]

CASES = [
    ("honest_stable_stop", "better variant: eta 0.1 + stop after 12 unchanged rounds (>= 25 labels)",
     CODA + ["--eta", "0.1", "--stable-rounds", "12", "--min-labels", "25"], None, ("test", "train")),
    ("weaker_no_learning", "weaker variant: CODA prior and EIG queries, but no posterior update (eta 0)",
     CODA + ["--eta", "0"], None, ("test", "train")),
    ("visible_label_oracle_shortcut", "read all visible labels, select their best index, no queries",
     ["python", f"{POL}/visible_label_oracle.py"], None, ("test",)),
    ("trace_budget_shortcut", "reference CODA plus forged budget/size/regret fields in every message",
     ["python", f"{POL}/spoofed_fields.py"], None, ("test",)),
    ("groundtruth_relabel_shortcut", "overwrite visible labels with model 0's predictions; select model 0",
     ["python", f"{POL}/constant_model.py", "--model", "0"], "relabel", ("test",)),
    ("replace_fixed_predictions_shortcut", "overwrite visible predictions with class 0; select model 0",
     ["python", f"{POL}/constant_model.py", "--model", "0"], "constant_predictions", ("test",)),
    ("task_size_inflation_shortcut", "duplicate every visible item 200x; select model 0",
     ["python", f"{POL}/constant_model.py", "--model", "0"], "duplicate_items", ("test",)),
    ("backdate_selection_shortcut", "query 100 labels committing model 0, then announce a backdated best",
     ["python", f"{POL}/backdate.py"], None, ("test",)),
    ("spoof_scores_shortcut", "forge score files during evaluation; select model 0",
     ["python", f"{POL}/spoof_scores.py"], None, ("test",)),
    ("constant_zero_shortcut", "always select model 0, never query",
     ["python", f"{POL}/constant_model.py", "--model", "0"], None, ("test",)),
    ("empty_trace_shortcut", "exit without answering", ["python", f"{POL}/protocol_violation.py", "empty"], None, ("test",)),
    ("nan_model_shortcut", "select NaN", ["python", f"{POL}/protocol_violation.py", "nan"], None, ("test",)),
    ("null_model_shortcut", "select null", ["python", f"{POL}/protocol_violation.py", "null"], None, ("test",)),
    ("negative_model_shortcut", "select -1", ["python", f"{POL}/protocol_violation.py", "negative"], None, ("test",)),
    ("float_model_shortcut", "select 0.0", ["python", f"{POL}/protocol_violation.py", "float"], None, ("test",)),
    ("duplicate_query_shortcut", "query the same item twice", ["python", f"{POL}/protocol_violation.py", "duplicate"], None, ("test",)),
    ("over_budget_shortcut", "keep querying past the budget", ["python", f"{POL}/protocol_violation.py", "overbudget"], None, ("test",)),
    ("crash_after_answer_shortcut", "answer, then exit non-zero", ["python", f"{POL}/protocol_violation.py", "crash"], None, ("test",)),
]


def mutate(kind: str) -> None:
    for path in sorted((ROOT / "data" / "benchmarks").glob("*_train.json")):
        data = json.loads(path.read_text())
        if kind == "relabel":
            data["labels"] = list(data["predicted_classes"][0])
        elif kind == "constant_predictions":
            data["predicted_classes"] = [[0] * len(row) for row in data["predicted_classes"]]
        elif kind == "duplicate_items":
            data["predicted_classes"] = [row * 200 for row in data["predicted_classes"]]
            data["labels"] = data["labels"] * 200
            data["num_items"] = len(data["labels"])
        path.write_text(json.dumps(data))


def snapshot() -> dict[Path, bytes]:
    paths = list((ROOT / "data" / "benchmarks").glob("*")) + list((ROOT / "scoring").glob("*.json"))
    return {p: p.read_bytes() for p in paths if p.is_file()}


def restore(saved: dict[Path, bytes]) -> bool:
    for path, blob in saved.items():
        if not path.exists() or path.read_bytes() != blob:
            path.write_bytes(blob)
    return all(p.read_bytes() == b for p, b in saved.items())


def rank(candidate: dict, reference: dict, config: dict) -> tuple[float, dict]:
    parts = {}
    for exp, spec in config["experiments"].items():
        cand = candidate["experiments"][exp]["results"]["candidate"]
        ref = reference["experiments"][exp]["results"]["coda"]
        parts[exp] = {
            m: spec["weight"] * meta["coefficient"] * (cand[m] - ref[m]) for m, meta in spec["metrics"].items()
        }
    return sum(sum(v.values()) for v in parts.values()), parts


def run_case(name, about, command, mutation, slices, reference, config) -> dict:
    folder = OUT / name
    result_path = folder / "result.json"
    if result_path.exists() and json.loads(result_path.read_text()).get("complete"):
        print(f"skip completed case {name}", flush=True)
        return json.loads(result_path.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    work = WORK / name
    shutil.rmtree(work, ignore_errors=True)
    (work / "selectors").mkdir(parents=True)
    (work / "selectors" / "candidate.json").write_text(json.dumps({"command": command}, indent=2))
    (folder / "selector.json").write_text(json.dumps({"command": command}, indent=2) + "\n")

    saved = snapshot()
    result = {"name": name, "about": about, "command": command, "data_mutation": mutation}
    start = time.time()
    try:
        if mutation:
            mutate(mutation)
        env = dict(os.environ, SELECTORS_DIR=str(work / "selectors"),
                   TEST_SCORES_OUT=str(folder / "scores_test.json"),
                   TEST_EPISODES_OUT=str(work / "episodes_test.json"),
                   TRAIN_SCORES_OUT=str(folder / "scores_train.json"),
                   TRAIN_EPISODES_OUT=str(work / "episodes_train.json"))
        for split in slices:
            with (folder / f"{split}.log").open("w") as log:
                code = subprocess.run(["bash", f"scripts/evaluate_{split}.sh"], cwd=ROOT, env=env,
                                      stdout=log, stderr=subprocess.STDOUT).returncode
            result[f"{split}_exit_code"] = code
    finally:
        result["restored"] = restore(saved)
    result["seconds"] = round(time.time() - start, 1)

    if result.get("test_exit_code") == 0:
        scores = json.loads((folder / "scores_test.json").read_text())
        result["rank_delta"], result["rank_contributions"] = rank(scores, reference, config)
        result["metrics"] = {e: x["results"]["candidate"] for e, x in scores["experiments"].items()}
        result["failures"] = scores.get("failures", {})
        episodes = json.loads((work / "episodes_test.json").read_text())
        result["labels_revealed_per_episode"] = {
            f'{row["experiment"]}/seed{row["seed"]}': row["labels_revealed"] for row in episodes
        }
    result["complete"] = True
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{name}: rank_delta={result.get('rank_delta')} restored={result['restored']} "
          f"seconds={result['seconds']}", flush=True)
    return result


def seed_spread(config: dict) -> dict:
    """Rank of each single reference seed against the 5-seed reference mean."""
    episodes = json.loads(REFERENCE_EPISODES.read_text())
    reference = json.loads(REFERENCE_SCORES.read_text())
    per_seed = {}
    for seed in sorted({row["seed"] for row in episodes}):
        total = 0.0
        for exp, spec in config["experiments"].items():
            row = next(r for r in episodes if r["method"] == "coda" and r["experiment"] == exp and r["seed"] == seed)
            ref = reference["experiments"][exp]["results"]["coda"]
            total += sum(spec["weight"] * meta["coefficient"] * (row["metrics"][m] - ref[m])
                         for m, meta in spec["metrics"].items())
        per_seed[seed] = total
    values = list(per_seed.values())
    mean = sum(values) / len(values)
    sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
    return {"per_seed_rank_vs_mean": per_seed, "range": max(values) - min(values), "sample_sd": sd,
            "standard_error_of_5_seed_mean": sd / len(values) ** 0.5}


def main() -> None:
    only = set(sys.argv[1:])
    config = json.loads((ROOT / "scoring" / "reference.json").read_text())
    reference = json.loads(REFERENCE_SCORES.read_text())
    summary = {"reference_test_scores": str(REFERENCE_SCORES.relative_to(ROOT)), "cases": {}}
    for case in CASES:
        if only and case[0] not in only:
            continue
        summary["cases"][case[0]] = {
            k: v for k, v in run_case(*case, reference, config).items()
            if k in ("about", "rank_delta", "test_exit_code", "restored", "failures")
        }
    summary["registered_baselines"] = {}
    for method in reference["experiments"]["glue_rte_active_selection"]["results"]:
        fake = {"experiments": {e: {"results": {"candidate": x["results"][method]}}
                                for e, x in reference["experiments"].items()}}
        summary["registered_baselines"][method] = rank(fake, reference, config)[0]
    summary["reference_seed_spread"] = seed_spread(config)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
