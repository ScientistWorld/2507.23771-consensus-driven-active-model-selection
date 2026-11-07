"""Build the visible (train) and hidden (test) benchmark halves.

One-off provenance script; the hidden test never runs it. It fetches the
public ModelSelector prediction/label arrays (QNLI cut to a fixed 1200-item
sample) and splits each task's items into two halves balanced in class mix
and per-model error. For CoLA only the held-out half is kept. It writes

  data/benchmarks/<task>_train.json   visible half: predictions + labels
  eval/test/benchmarks/<task>_test.json  held-out pool: predictions + keyed
                                         label codes; candidate order and
                                         item order shuffled with a fixed
                                         secret seed

Only the hidden half's labels ever exist in eval/test/. The raw downloads go
to a temporary directory that is deleted afterwards. Standard library only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import struct
import tempfile
import urllib.request
from array import array
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_URL = "https://raw.githubusercontent.com/RobustML-Lab/model-selector/main/resources/datasets"
# task -> (upstream path, number of classes, item subsample size or None, mode)
# mode "split":       visible half in data/benchmarks, held-out half in eval/test
# mode "hidden_half": same balanced split, but only the held-out half is kept
#                     (scored by the hidden test only; no visible part)
TASKS = {
    "glue_rte": ("glue/rte", 2, None, "split"),
    "glue_sst2": ("glue/sst2", 2, None, "split"),
    "glue_qnli": ("glue/qnli", 2, 1200, "split"),
    "glue_cola": ("glue/cola", 2, None, "hidden_half"),
}
SUBSAMPLE_SEED = 20250731  # fixed; large pools are cut to a uniform item sample
SPLIT_SEED = 250723771
SWAP_ITERATIONS = 30
PAIR_LIMIT = 20000
HIDDEN_ORDER_SEED = 0x5EC12E7C0DA  # fixed; candidate/item order of the hidden half
LABEL_KEY = "coda-hidden-oracle-v1"  # keyed encoding of hidden labels (see evaluate.py)


def label_offset(task: str, position: int, num_classes: int) -> int:
    digest = hashlib.sha256(f"{LABEL_KEY}|{task}|{position}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % num_classes


def read_npy(raw: bytes) -> tuple[list[float], tuple[int, ...]]:
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("not an .npy file")
    major = raw[6]
    if major == 1:
        header_len = struct.unpack("<H", raw[8:10])[0]
        start = 10
    else:
        header_len = struct.unpack("<I", raw[8:12])[0]
        start = 12
    header = ast.literal_eval(raw[start:start + header_len].decode("latin1"))
    if header["descr"] != "<f8" or header["fortran_order"]:
        raise ValueError(f"unsupported npy layout: {header}")
    values = array("d")
    values.frombytes(raw[start + header_len:])
    return list(values), tuple(header["shape"])


def fetch(url: str, dest: Path) -> bytes:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetch {url}", flush=True)
    urllib.request.urlretrieve(url, dest)
    return dest.read_bytes()


def jitter(count: int) -> list[float]:
    mask = (1 << 64) - 1
    out = []
    for idx in range(count):
        x = (idx + SPLIT_SEED) & mask
        x ^= x >> 33
        x = (x * 0xFF51AFD7ED558CCD) & mask
        x ^= x >> 33
        x = (x * 0xC4CEB9FE1A85EC53) & mask
        x ^= x >> 33
        out.append((x & ((1 << 53) - 1)) / float(1 << 53))
    return out


def pairs(train_items: list[int], test_items: list[int], limit: int):
    count = len(train_items) * len(test_items)
    if count <= limit:
        for a in train_items:
            for b in test_items:
                yield a, b
        return
    step = max(1, count // limit)
    offset = SPLIT_SEED % count
    for pos in range(limit):
        flat = (offset + pos * step) % count
        yield train_items[flat // len(test_items)], test_items[flat % len(test_items)]


def balanced_split(item_preds: list[list[int]], labels: list[int]) -> list[bool]:
    """Alternate items sorted by (label, difficulty, error pattern), then swap
    same-label items to equalize every candidate's error rate across halves."""
    n_items = len(labels)
    n_models = len(item_preds[0])
    wrong = [[int(p != labels[i]) for p in item_preds[i]] for i in range(n_items)]
    bits = [sum(w << h for h, w in enumerate(row)) for row in wrong]
    item_error = [sum(row) / n_models for row in wrong]
    signature = [tuple(1 - w for w in row) for row in wrong]
    jit = jitter(n_items)
    order = sorted(range(n_items), key=lambda i: (labels[i], item_error[i], signature[i], jit[i]))
    train = [False] * n_items
    for pos, item in enumerate(order):
        if pos % 2 == 0:
            train[item] = True

    n_train = sum(train)
    n_test = n_items - n_train
    k = 1.0 / n_train + 1.0 / n_test
    train_sum = [0] * n_models
    test_sum = [0] * n_models
    for i in range(n_items):
        target = train_sum if train[i] else test_sum
        for h in range(n_models):
            target[h] += wrong[i][h]
    diff = [train_sum[h] / n_train - test_sum[h] / n_test for h in range(n_models)]
    objective = sum(d * d for d in diff) / n_models
    label_values = sorted(set(labels))

    for _ in range(SWAP_ITERATIONS):
        weighted = [sum(diff[h] for h in range(n_models) if wrong[i][h]) for i in range(n_items)]
        best = None
        best_objective = objective
        for label in label_values:
            train_items = [i for i in range(n_items) if train[i] and labels[i] == label]
            test_items = [i for i in range(n_items) if not train[i] and labels[i] == label]
            if not train_items or not test_items:
                continue
            for a, b in pairs(train_items, test_items, PAIR_LIMIT):
                new_objective = (
                    objective
                    + 2.0 * k * (weighted[b] - weighted[a]) / n_models
                    + k * k * (bits[a] ^ bits[b]).bit_count() / n_models
                )
                if new_objective < best_objective - 1e-12:
                    best_objective = new_objective
                    best = (a, b)
        if best is None:
            break
        a, b = best
        train[a] = False
        train[b] = True
        diff = [diff[h] + k * (wrong[b][h] - wrong[a][h]) for h in range(n_models)]
        objective = sum(d * d for d in diff) / n_models
    return train


def write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":")) + "\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def build_task(task: str, rel: str, num_classes: int, subsample: int | None, mode: str, workdir: Path) -> dict:
    pred_raw = fetch(f"{BASE_URL}/{rel}/predictions.npy", workdir / task / "predictions.npy")
    label_raw = fetch(f"{BASE_URL}/{rel}/oracle.npy", workdir / task / "oracle.npy")
    pred_values, pred_shape = read_npy(pred_raw)
    label_values, label_shape = read_npy(label_raw)
    n_items, n_models = pred_shape
    if label_shape != (n_items,):
        raise ValueError((pred_shape, label_shape))
    keep = list(range(n_items))
    if subsample is not None and subsample < n_items:
        keep = sorted(random.Random(SUBSAMPLE_SEED + sum(map(ord, task))).sample(keep, subsample))
    item_preds = [[int(pred_values[i * n_models + h]) for h in range(n_models)] for i in keep]
    labels = [int(label_values[i]) for i in keep]
    n_items = len(keep)
    for row, y in zip(item_preds, labels):
        if any(p < 0 or p >= num_classes for p in row) or not 0 <= y < num_classes:
            raise ValueError(f"{task}: value outside {num_classes} classes")
    train = balanced_split(item_preds, labels)

    train_items = [i for i in range(n_items) if train[i]]
    test_items = [i for i in range(n_items) if not train[i]]
    visible = {
        "task": task,
        "slice": "train",
        "num_classes": num_classes,
        "num_models": n_models,
        "num_items": len(train_items),
        "predicted_classes": [[item_preds[i][h] for i in train_items] for h in range(n_models)],
        "labels": [labels[i] for i in train_items],
    }

    rng = random.Random(HIDDEN_ORDER_SEED + sum(map(ord, task)))
    model_order = list(range(n_models))
    rng.shuffle(model_order)
    item_order = list(test_items)
    rng.shuffle(item_order)
    hidden = {
        "task": task,
        "slice": "test",
        "num_classes": num_classes,
        "num_models": n_models,
        "num_items": len(item_order),
        "predicted_classes": [[item_preds[i][h] for i in item_order] for h in model_order],
        "label_code": [
            (labels[i] + label_offset(task, pos, num_classes)) % num_classes
            for pos, i in enumerate(item_order)
        ],
    }

    train_sha = write_json(ROOT / "data" / "benchmarks" / f"{task}_train.json", visible) if mode == "split" else None
    test_sha = write_json(HERE / "benchmarks" / f"{task}_test.json", hidden)

    def err_rates(items):
        return [sum(item_preds[i][h] != labels[i] for i in items) / len(items) for h in range(n_models)]

    te = err_rates(test_items)
    msg = f"{task}: mode={mode} models={n_models} items={n_items} train={len(train_items)} test={len(test_items)}"
    if train_items:
        tr = err_rates(train_items)
        msg += f" max|err gap|={max(abs(a - b) for a, b in zip(tr, te)):.4f}"
    print(msg, flush=True)
    return {
        "source_predictions": f"{BASE_URL}/{rel}/predictions.npy",
        "source_labels": f"{BASE_URL}/{rel}/oracle.npy",
        "source_predictions_sha256": hashlib.sha256(pred_raw).hexdigest(),
        "source_labels_sha256": hashlib.sha256(label_raw).hexdigest(),
        "num_classes": num_classes,
        "item_subsample": subsample,
        "mode": mode,
        "train_file_sha256": train_sha,
        "test_file_sha256": test_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    args = parser.parse_args()
    provenance = {}
    with tempfile.TemporaryDirectory() as tmp:
        for task in args.tasks:
            provenance[task] = build_task(task, *TASKS[task], Path(tmp))
    (HERE / "benchmarks" / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
