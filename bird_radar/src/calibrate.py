from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.scoreboard import average_precision_binary, macro_map, per_class_ap


def _probs_to_logits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _logits_to_probs(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def apply_temperature_scaling(probs: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
    logits = _probs_to_logits(np.asarray(probs, dtype=np.float64))
    t = np.asarray(temperatures, dtype=np.float64).reshape(1, -1)
    t = np.clip(t, 0.05, 20.0)
    return _logits_to_probs(logits / t).astype(np.float32)


def fit_temperature_scaling_grid(
    y_true: np.ndarray,
    probs: np.ndarray,
    steps: int = 31,
    t_min: float = 0.5,
    t_max: float = 3.0,
) -> np.ndarray:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    logits = _probs_to_logits(probs)
    grid = np.linspace(t_min, t_max, num=steps, dtype=np.float64)
    temps = np.ones(probs.shape[1], dtype=np.float32)
    for c in range(probs.shape[1]):
        yc = y_true[:, c].astype(np.int8)
        if yc.sum() == 0 or yc.sum() == len(yc):
            temps[c] = 1.0
            continue
        best_t = 1.0
        best_ap = -1.0
        zc = logits[:, c]
        for t in grid:
            pc = _logits_to_probs(zc / t)
            ap = average_precision_binary(yc, pc)
            if ap > best_ap:
                best_ap = ap
                best_t = float(t)
        temps[c] = best_t
    return temps


def calibrate_and_score(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    raw_macro = macro_map(y_true, probs)
    temps = fit_temperature_scaling_grid(y_true, probs)
    cal = apply_temperature_scaling(probs, temps)
    cal_macro = macro_map(y_true, cal)
    best_probs = cal if cal_macro >= raw_macro else np.asarray(probs, dtype=np.float32)
    best_macro = max(raw_macro, cal_macro)
    return {
        "temperatures": temps.astype(np.float32),
        "raw_macro_map": float(raw_macro),
        "calibrated_macro_map": float(cal_macro),
        "selected_macro_map": float(best_macro),
        "selected_probs": best_probs.astype(np.float32),
        "selected_per_class_ap": per_class_ap(y_true, best_probs),
        "selected": "calibrated" if cal_macro >= raw_macro else "raw",
    }


def save_calibration(out_dir: str | Path, result: dict[str, Any]) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "calibration.json"
    payload = {k: v for k, v in result.items() if k not in {"selected_probs"}}
    if "temperatures" in payload:
        payload["temperatures"] = [float(x) for x in np.asarray(payload["temperatures"]).ravel()]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    return str(path)

