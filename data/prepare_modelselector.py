"""Download and convert ModelSelector raw prediction files.

The ModelSelector release stores predictions as class IDs with shape (N, H).
CODA's active model selection formulation uses prediction vectors with shape
(H, N, C), so this script converts IDs to one-hot vectors and writes a compact
portable .npz file.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np


TASKS = {
    "glue_rte": ("glue/rte", 2),
    "glue_sst2": ("glue/sst2", 2),
    "glue_mrpc": ("glue/mrpc", 2),
    "cifar10_5592": ("cifar10_5592", 10),
    "cifar10_4070": ("cifar10_4070", 10),
    "pacs": ("pacs", 7),
}

BASE = "https://raw.githubusercontent.com/RobustML-Lab/model-selector/main/resources/datasets"


def fetch(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetch {url}")
    urllib.request.urlretrieve(url, path)


def convert_task(task: str, output_dir: Path) -> None:
    if task not in TASKS:
        raise KeyError(f"Unknown task {task}; known tasks: {sorted(TASKS)}")
    rel, num_classes = TASKS[task]
    raw_dir = output_dir / "modelselector_raw" / task
    pred_path = raw_dir / "predictions.npy"
    label_path = raw_dir / "oracle.npy"
    fetch(f"{BASE}/{rel}/predictions.npy", pred_path)
    fetch(f"{BASE}/{rel}/oracle.npy", label_path)

    class_ids = np.load(pred_path)
    labels = np.load(label_path).reshape(-1).astype(np.int64)
    if class_ids.shape[0] != labels.shape[0]:
        raise ValueError((class_ids.shape, labels.shape))
    class_ids = class_ids.astype(np.int64)
    predictions = np.eye(num_classes, dtype=np.float32)[class_ids]
    predictions = np.transpose(predictions, (1, 0, 2))
    out_path = output_dir / "modelselector" / f"{task}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, predictions=predictions, labels=labels)
    print(f"wrote {out_path} {predictions.shape} labels={labels.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/downloads")
    parser.add_argument("--tasks", nargs="+", default=["glue_rte", "glue_sst2"])
    args = parser.parse_args()
    out = Path(args.output_dir)
    for task in args.tasks:
        convert_task(task, out)


if __name__ == "__main__":
    main()
