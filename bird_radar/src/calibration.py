from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score

EPS = 1e-8


def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    z = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def apply_temperature_scaling(probs: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
    logits = probs_to_logits(probs)
    t = np.clip(temperatures.reshape(1, -1), 0.05, 20.0)
    return logits_to_probs(logits / t)


def _fit_single_class_temperature(y_true: np.ndarray, p: np.ndarray) -> float:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 1.0

    logits = probs_to_logits(p)

    def objective(temp: float) -> float:
        temp = float(np.clip(temp, 0.05, 20.0))
        p_cal = logits_to_probs(logits / temp)
        try:
            return -float(average_precision_score(y_true, p_cal))
        except ValueError:
            return 0.0

    res = minimize_scalar(objective, bounds=(0.05, 20.0), method="bounded", options={"xatol": 1e-3})
    return float(res.x if res.success else 1.0)


def fit_temperature_scaling(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    temps = np.ones(probs.shape[1], dtype=np.float32)
    for c in range(probs.shape[1]):
        temps[c] = _fit_single_class_temperature(y_true[:, c].astype(np.float32), probs[:, c].astype(np.float32))
    return temps

