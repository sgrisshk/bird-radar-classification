from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import CLASSES


def choose_group_column(df: pd.DataFrame) -> str:
    for col in ["primary_observation_id", "observation_id"]:
        if col in df.columns and df[col].nunique() < len(df):
            return col
    return "observation_id" if "observation_id" in df.columns else "track_id"


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def average_predictions(preds: Iterable[np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    preds = [np.asarray(p, dtype=np.float32) for p in preds]
    if not preds:
        raise ValueError("No predictions provided")
    stacked = np.stack(preds, axis=0)
    if weights is None:
        return stacked.mean(axis=0)
    w = np.asarray(weights, dtype=np.float32)
    w = w / (w.sum() + 1e-8)
    return np.tensordot(w, stacked, axes=(0, 0))


def make_submission(track_ids: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"track_id": track_ids.astype(int)})
    for i, cls in enumerate(CLASSES):
        df[cls] = probs[:, i].astype(float)
    return df

