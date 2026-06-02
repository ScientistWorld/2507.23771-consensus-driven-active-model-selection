"""Utilities for active model selection prediction benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PredictionBenchmark:
    """Container for a model-selection benchmark.

    predictions has shape (num_models, num_items, num_classes) and contains
    post-softmax scores or one-hot class predictions. labels has shape
    (num_items,) and contains integer ground-truth labels.
    """

    name: str
    predictions: np.ndarray
    labels: np.ndarray

    @property
    def pred_classes(self) -> np.ndarray:
        return self.predictions.argmax(axis=-1)

    @property
    def true_losses(self) -> np.ndarray:
        return 1.0 - (self.pred_classes == self.labels[None, :]).mean(axis=1)

    @property
    def true_accuracies(self) -> np.ndarray:
        return 1.0 - self.true_losses


def _load_pt(path: Path):
    import torch

    value = torch.load(path, map_location="cpu")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return value


def load_prediction_benchmark(data_dir: str | Path, task: str) -> PredictionBenchmark:
    """Load benchmark predictions for one task.

    Preferred portable format is ``<task>.npz`` with ``predictions`` and
    ``labels`` arrays. The loader also supports upstream release ``.pt`` files
    when PyTorch is available.
    """

    data_dir = Path(data_dir)
    npz_path = data_dir / f"{task}.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        predictions = np.asarray(data["predictions"], dtype=np.float64)
        labels = np.asarray(data["labels"], dtype=np.int64)
        return _validate(task, predictions, labels)

    pred_path = data_dir / f"{task}.pt"
    label_path = data_dir / f"{task}_labels.pt"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction tensor: {pred_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label tensor: {label_path}")

    predictions = np.asarray(_load_pt(pred_path), dtype=np.float64)
    labels = np.asarray(_load_pt(label_path), dtype=np.int64)
    return _validate(task, predictions, labels)


def _validate(task: str, predictions: np.ndarray, labels: np.ndarray) -> PredictionBenchmark:
    if predictions.ndim != 3:
        raise ValueError(f"Expected predictions shape (H,N,C), got {predictions.shape}")
    if labels.ndim != 1 or labels.shape[0] != predictions.shape[1]:
        raise ValueError(
            f"Expected labels shape ({predictions.shape[1]},), got {labels.shape}"
        )
    row_sums = predictions.sum(axis=-1, keepdims=True)
    predictions = predictions / np.clip(row_sums, 1e-12, None)
    return PredictionBenchmark(task, predictions, labels)
