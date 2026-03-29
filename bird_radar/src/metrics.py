from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from config import CLASSES

EPS = 1e-8


def sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def one_hot_labels(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(labels), num_classes), dtype=np.float32)
    out[np.arange(len(labels)), labels.astype(int)] = 1.0
    return out


def per_class_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    scores: dict[str, float] = {}
    for i, name in enumerate(CLASSES[: y_true.shape[1]]):
        yt = y_true[:, i]
        ys = y_score[:, i]
        if np.sum(yt) == 0:
            scores[name] = 0.0
            continue
        try:
            scores[name] = float(average_precision_score(yt, ys))
        except ValueError:
            scores[name] = 0.0
    return scores


def macro_map_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    scores = per_class_average_precision(y_true, y_score)
    return float(np.mean(list(scores.values()))) if scores else 0.0


def save_scores_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

