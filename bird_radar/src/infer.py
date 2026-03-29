from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLASS_COLUMNS = [
    "Clutter",
    "Cormorants",
    "Pigeons",
    "Ducks",
    "Geese",
    "Gulls",
    "Birds of Prey",
    "Waders",
    "Songbirds",
]


def make_submission_df(test_csv: str | Path, probs: np.ndarray) -> pd.DataFrame:
    test_df = pd.read_csv(test_csv, usecols=["track_id"])
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape[0] != len(test_df):
        raise ValueError(f"Prediction row mismatch: expected {len(test_df)}, got {probs.shape[0]}")
    if probs.shape[1] != len(CLASS_COLUMNS):
        raise ValueError(f"Prediction col mismatch: expected {len(CLASS_COLUMNS)}, got {probs.shape[1]}")
    sub = pd.DataFrame({"track_id": test_df["track_id"].to_numpy()})
    clipped = np.clip(probs, 0.0, 1.0)
    for i, c in enumerate(CLASS_COLUMNS):
        sub[c] = clipped[:, i].astype(float)
    return sub


def write_submission_csv(
    test_csv: str | Path,
    probs: np.ndarray,
    out_csv: str | Path,
    sample_submission_csv: str | Path | None = None,
) -> str:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub = make_submission_df(test_csv, probs)
    if sample_submission_csv is not None and Path(sample_submission_csv).exists():
        sample_cols = pd.read_csv(sample_submission_csv, nrows=1).columns.tolist()
        sub = sub[sample_cols]
    sub.to_csv(out_csv, index=False)
    return str(out_csv)

