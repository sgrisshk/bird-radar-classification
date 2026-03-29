from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score

EPS = 1e-12


def per_class_ap(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, float]:
    out: dict[int, float] = {}
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        if int(np.sum(yt)) == 0:
            out[i] = 0.0
        else:
            out[i] = float(average_precision_score(yt, yp))
    return out


def macro_map(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pcs = per_class_ap(y_true, y_pred)
    return float(np.mean(list(pcs.values()))) if pcs else 0.0


def _normalize_weights_array(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    s = float(np.sum(w))
    if s <= EPS:
        w = np.full_like(w, 1.0 / max(len(w), 1), dtype=np.float64)
    else:
        w = w / s
    return w


def apply_blend(predictions: Any, weights: Any) -> np.ndarray:
    if isinstance(predictions, dict):
        names = list(predictions.keys())
        arr = np.stack([np.asarray(predictions[n], dtype=np.float64) for n in names], axis=0)
        if isinstance(weights, dict):
            w = np.asarray([float(weights.get(n, 0.0)) for n in names], dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
    else:
        arr = np.stack([np.asarray(p, dtype=np.float64) for p in predictions], axis=0)
        w = np.asarray(weights, dtype=np.float64)

    w = _normalize_weights_array(w)
    out = np.tensordot(w, arr, axes=(0, 0))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def optimize_blend_weights(
    y_true: np.ndarray,
    oof_predictions: dict[str, np.ndarray],
) -> dict[str, float]:
    names = list(oof_predictions.keys())
    if not names:
        raise ValueError("oof_predictions is empty")
    if len(names) == 1:
        return {names[0]: 1.0}

    pred_list = [np.asarray(oof_predictions[n], dtype=np.float64) for n in names]

    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z)
        e = np.exp(z)
        return e / (np.sum(e) + EPS)

    def objective(z: np.ndarray) -> float:
        w = _softmax(z)
        pred = apply_blend(pred_list, w)
        return -macro_map(y_true, pred)

    z0 = np.zeros(len(names), dtype=np.float64)
    result = minimize(objective, z0, method="L-BFGS-B")
    if result.success:
        w_opt = _softmax(result.x)
    else:
        w_opt = np.full(len(names), 1.0 / len(names), dtype=np.float64)

    return {name: float(w) for name, w in zip(names, w_opt)}
