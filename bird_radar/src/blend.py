from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.scoreboard import macro_map, per_class_ap


def blend_predictions(preds: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    arr = np.stack([np.asarray(p, dtype=np.float64) for p in preds], axis=0)
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.clip(w.sum(), 1e-12, None)
    out = np.tensordot(w, arr, axes=(0, 0))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def optimize_two_model_weight(y_true: np.ndarray, p0: np.ndarray, p1: np.ndarray, steps: int = 401) -> dict[str, Any]:
    best = {"w0": 0.5, "w1": 0.5, "macro_map": -1.0}
    grid = np.linspace(0.0, 1.0, num=steps, dtype=np.float64)
    for w1 in grid:
        w0 = 1.0 - w1
        p = (w0 * p0 + w1 * p1).astype(np.float32)
        s = macro_map(y_true, p)
        if s > best["macro_map"]:
            best = {"w0": float(w0), "w1": float(w1), "macro_map": float(s)}
    best_pred = (best["w0"] * p0 + best["w1"] * p1).astype(np.float32)
    best["per_class_ap"] = per_class_ap(y_true, best_pred)
    return best


def save_blend_artifact(
    out_dir: str | Path,
    oof: np.ndarray,
    test: np.ndarray,
    meta: dict[str, Any],
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_path = out_dir / "oof_blend.npy"
    test_path = out_dir / "test_blend.npy"
    meta_path = out_dir / "blend_meta.json"
    np.save(oof_path, oof.astype(np.float32))
    np.save(test_path, test.astype(np.float32))
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=True)
    return {"oof_path": str(oof_path), "test_path": str(test_path), "meta_path": str(meta_path)}

