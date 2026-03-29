from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import CLASSES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--test-csv",
        type=str,
        default="/Users/sgrisshk/Desktop/AI-task/test.csv",
    )
    p.add_argument(
        "--base-submission",
        type=str,
        default="/Users/sgrisshk/Desktop/AI-task/bird_radar/artifacts/temporal_run_fix2_rcs/sub_temporal_best.csv",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="/Users/sgrisshk/Desktop/AI-task/bird_radar/artifacts/temporal_postprocess_primary",
    )
    p.add_argument("--group-col", type=str, default="primary_observation_id")
    p.add_argument("--gamma", type=float, default=1.2)
    return p.parse_args()


def smooth_by_group(pred: np.ndarray, group_ids: np.ndarray, mode: str) -> np.ndarray:
    if mode not in {"mean", "max"}:
        raise ValueError(f"unsupported mode: {mode}")
    df = pd.DataFrame(pred.astype(np.float64))
    df["_g"] = group_ids
    if mode == "mean":
        agg = df.groupby("_g", sort=False).mean()
    else:
        agg = df.groupby("_g", sort=False).max()
    out = (
        df[["_g"]]
        .merge(agg, left_on="_g", right_index=True, how="left")
        .drop(columns=["_g"])
        .to_numpy(dtype=np.float32)
    )
    return np.clip(out, 0.0, 1.0)


def power_calibrate(pred: np.ndarray, gamma: float) -> np.ndarray:
    p = np.clip(pred.astype(np.float64), 1e-6, 1.0 - 1e-6)
    out = np.power(p, float(gamma))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def save_submission(path: Path, track_ids: np.ndarray, pred: np.ndarray) -> None:
    sub = pd.DataFrame(pred.astype(np.float32), columns=CLASSES)
    sub.insert(0, "track_id", track_ids.astype(np.int64))
    sub.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    test_csv = Path(args.test_csv).resolve()
    base_sub_csv = Path(args.base_submission).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(test_csv, usecols=["track_id", args.group_col])
    sub_df = pd.read_csv(base_sub_csv)

    required = {"track_id", *CLASSES}
    missing = [c for c in required if c not in sub_df.columns]
    if missing:
        raise ValueError(f"base submission missing columns: {missing}")
    if args.group_col not in test_df.columns:
        raise ValueError(f"test.csv missing group column: {args.group_col}")

    merged = sub_df[["track_id", *CLASSES]].merge(
        test_df,
        on="track_id",
        how="left",
        validate="one_to_one",
    )
    if merged[args.group_col].isna().any():
        bad = int(merged[args.group_col].isna().sum())
        raise ValueError(f"group ids missing for {bad} rows after merge")

    track_ids = merged["track_id"].to_numpy(dtype=np.int64)
    groups = merged[args.group_col].to_numpy()
    pred = merged[CLASSES].to_numpy(dtype=np.float32)

    pred_mean = smooth_by_group(pred, groups, mode="mean")
    pred_mean_power = power_calibrate(pred_mean, gamma=args.gamma)
    pred_max = smooth_by_group(pred, groups, mode="max")

    p_mean = out_dir / "sub_primary_mean.csv"
    p_mean_pow = out_dir / f"sub_primary_mean_power_g{str(args.gamma).replace('.', 'p')}.csv"
    p_max = out_dir / "sub_primary_max.csv"

    save_submission(p_mean, track_ids, pred_mean)
    save_submission(p_mean_pow, track_ids, pred_mean_power)
    save_submission(p_max, track_ids, pred_max)

    report = {
        "base_submission": str(base_sub_csv),
        "test_csv": str(test_csv),
        "group_col": args.group_col,
        "gamma": float(args.gamma),
        "outputs": {
            "primary_mean": str(p_mean),
            "primary_mean_power": str(p_mean_pow),
            "primary_max": str(p_max),
        },
        "prediction_means": {
            "base": {cls: float(np.mean(pred[:, i])) for i, cls in enumerate(CLASSES)},
            "primary_mean": {cls: float(np.mean(pred_mean[:, i])) for i, cls in enumerate(CLASSES)},
            "primary_mean_power": {cls: float(np.mean(pred_mean_power[:, i])) for i, cls in enumerate(CLASSES)},
            "primary_max": {cls: float(np.mean(pred_max[:, i])) for i, cls in enumerate(CLASSES)},
        },
    }
    (out_dir / "postprocess_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(str(p_mean), flush=True)
    print(str(p_mean_pow), flush=True)
    print(str(p_max), flush=True)


if __name__ == "__main__":
    main()
