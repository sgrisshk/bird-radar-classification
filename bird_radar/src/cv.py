from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _to_datetime_ns(series: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"timestamp parsing failed for {bad} rows")
    return ts.astype("int64").to_numpy()


def _assert_group_disjoint(groups: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    g_train = set(groups[train_idx].tolist())
    g_val = set(groups[val_idx].tolist())
    overlap = g_train.intersection(g_val)
    if overlap:
        raise RuntimeError(f"group leakage detected: {len(overlap)} overlapping groups")


def make_forward_temporal_group_folds(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp_start_radar_utc",
    group_col: str = "observation_id",
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if timestamp_col not in df.columns:
        raise ValueError(f"missing timestamp column: {timestamp_col}")
    if group_col not in df.columns:
        raise ValueError(f"missing group column: {group_col}")

    n = len(df)
    if n == 0:
        return []

    ts_ns = _to_datetime_ns(df[timestamp_col])
    groups = df[group_col].to_numpy()
    order = np.argsort(ts_ns, kind="mergesort")

    # n_splits folds need n_splits+1 windows.
    windows = np.array_split(order, n_splits + 1)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        train_cand = np.concatenate(windows[: k + 1]) if (k + 1) > 0 else np.array([], dtype=np.int64)
        val_cand = windows[k + 1].astype(np.int64)
        if len(train_cand) == 0 or len(val_cand) == 0:
            continue

        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        train_mask[train_cand] = True
        val_mask[val_cand] = True
        candidate_mask = train_mask | val_mask

        # Resolve conflicts only inside train/val candidate region.
        group_to_indices: dict[int | str, list[int]] = defaultdict(list)
        for i in np.where(candidate_mask)[0]:
            g = groups[i]
            group_to_indices[g].append(i)

        for _, idxs in group_to_indices.items():
            idxs_np = np.asarray(idxs, dtype=np.int64)
            in_train = int(np.sum(train_mask[idxs_np]))
            in_val = int(np.sum(val_mask[idxs_np]))
            if in_train == 0 or in_val == 0:
                continue
            send_to_val = in_val >= in_train
            train_mask[idxs_np] = not send_to_val
            val_mask[idxs_np] = send_to_val

        # Exclude all rows outside candidate windows (future windows k+2:).
        train_mask &= candidate_mask
        val_mask &= candidate_mask

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        _assert_group_disjoint(groups=groups, train_idx=train_idx, val_idx=val_idx)
        folds.append((train_idx, val_idx))

    if len(folds) == 0:
        raise RuntimeError("unable to construct temporal group folds")
    return folds


def make_temporal_holdout_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp_start_radar_utc",
    group_col: str = "observation_id",
    holdout_quantile: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    if not (0.0 < holdout_quantile < 1.0):
        raise ValueError("holdout_quantile must be in (0,1)")
    if timestamp_col not in df.columns:
        raise ValueError(f"missing timestamp column: {timestamp_col}")
    if group_col not in df.columns:
        raise ValueError(f"missing group column: {group_col}")

    ts = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"timestamp parsing failed for {bad} rows")
    groups = df[group_col].to_numpy()

    cutoff = ts.quantile(holdout_quantile)
    val_mask = (ts >= cutoff).to_numpy()

    group_to_indices: dict[int | str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        group_to_indices[g].append(i)

    for _, idxs in group_to_indices.items():
        idxs_np = np.asarray(idxs, dtype=np.int64)
        in_val = int(np.sum(val_mask[idxs_np]))
        in_train = int(len(idxs_np) - in_val)
        if in_val == 0 or in_train == 0:
            continue
        send_to_val = in_val >= in_train
        val_mask[idxs_np] = send_to_val

    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("invalid temporal holdout split (empty train or val)")

    _assert_group_disjoint(groups=groups, train_idx=train_idx, val_idx=val_idx)
    return train_idx, val_idx, cutoff


def make_temporal_holdout_split_with_cutoff(
    df: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    timestamp_col: str = "timestamp_start_radar_utc",
    group_col: str = "observation_id",
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    if timestamp_col not in df.columns:
        raise ValueError(f"missing timestamp column: {timestamp_col}")
    if group_col not in df.columns:
        raise ValueError(f"missing group column: {group_col}")

    ts = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"timestamp parsing failed for {bad} rows")
    groups = df[group_col].to_numpy()

    cutoff_ts = pd.to_datetime(cutoff, utc=True)
    if pd.isna(cutoff_ts):
        raise ValueError(f"invalid cutoff value: {cutoff}")

    val_mask = (ts >= cutoff_ts).to_numpy()

    group_to_indices: dict[int | str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        group_to_indices[g].append(i)

    for _, idxs in group_to_indices.items():
        idxs_np = np.asarray(idxs, dtype=np.int64)
        in_val = int(np.sum(val_mask[idxs_np]))
        in_train = int(len(idxs_np) - in_val)
        if in_val == 0 or in_train == 0:
            continue
        send_to_val = in_val >= in_train
        val_mask[idxs_np] = send_to_val

    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("invalid temporal holdout split (empty train or val)")

    _assert_group_disjoint(groups=groups, train_idx=train_idx, val_idx=val_idx)
    return train_idx, val_idx, cutoff_ts
