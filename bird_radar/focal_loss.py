from __future__ import annotations

from typing import Callable

import numpy as np

EPS = 1e-12


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def dynamic_focal_gamma(pos_ratio: float) -> float:
    if pos_ratio < 0.05:
        return 2.0
    if pos_ratio < 0.10:
        return 1.5
    return 1.0


def compute_scale_pos_weight(y_binary: np.ndarray) -> float:
    y = np.asarray(y_binary)
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos <= 0.0:
        return 1.0
    return max(neg / (pos + EPS), 1.0)


def build_class_adaptive_weights(y_binary: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_binary)
    n = max(len(y), 1)
    pos_ratio = float(np.sum(y == 1) / n)
    gamma = float(dynamic_focal_gamma(pos_ratio))
    scale_pos_weight = float(compute_scale_pos_weight(y))
    return {
        "pos_ratio": pos_ratio,
        "gamma": gamma,
        "scale_pos_weight": scale_pos_weight,
    }


def make_focal_surrogate_objective(gamma: float, alpha: float = 0.25) -> Callable:
    gamma = float(max(gamma, 0.0))
    alpha = float(np.clip(alpha, 1e-6, 1.0 - 1e-6))

    def _objective(preds: np.ndarray, train_data) -> tuple[np.ndarray, np.ndarray]:
        y = train_data.get_label().astype(np.float64)
        p = sigmoid(preds.astype(np.float64))
        pt = y * p + (1.0 - y) * (1.0 - p)
        mod = np.power(np.maximum(1.0 - pt, EPS), gamma)
        alpha_t = y * alpha + (1.0 - y) * (1.0 - alpha)
        grad = alpha_t * mod * (p - y)
        hess = alpha_t * mod * np.maximum(p * (1.0 - p), EPS)
        return grad.astype(np.float64), hess.astype(np.float64)

    return _objective
