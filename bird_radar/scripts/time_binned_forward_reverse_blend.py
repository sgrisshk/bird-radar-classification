#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds


def _macro_map(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    vals: list[float] = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        vals.append(0.0 if yt.sum() <= 0 else float(average_precision_score(yt, yp)))
    return float(np.mean(vals))


def _parse_grid(s: str) -> np.ndarray:
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    arr = np.array(vals, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("weights-grid is empty")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("weights-grid values must be in [0,1]")
    return arr


def _onehot(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.zeros((len(y_idx), n_classes), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    return y


def _evaluate_fold_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    teacher_ref: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, float, list[dict[str, float | int]]]:
    fold_reports: list[dict[str, float | int]] = []
    for fold_id, (_, va_idx) in enumerate(folds):
        yt = y[va_idx]
        pr = pred[va_idx]
        te = teacher_ref[va_idx]
        covered = (pr.sum(axis=1) > 0.0)
        if int(covered.sum()) == 0:
            fold_reports.append(
                {
                    "fold": int(fold_id),
                    "n_val": int(len(va_idx)),
                    "n_cov": 0,
                    "teacher_macro": 0.0,
                    "pred_macro": 0.0,
                    "fold_delta": 0.0,
                }
            )
            continue
        yt_c = yt[covered]
        pr_c = pr[covered]
        te_c = te[covered]
        te_s = _macro_map(yt_c, te_c)
        pr_s = _macro_map(yt_c, pr_c)
        fold_reports.append(
            {
                "fold": int(fold_id),
                "n_val": int(len(va_idx)),
                "n_cov": int(covered.sum()),
                "teacher_macro": float(te_s),
                "pred_macro": float(pr_s),
                "fold_delta": float(pr_s - te_s),
            }
        )

    valid = [r for r in fold_reports if int(r["n_cov"]) > 0]
    if not valid:
        return 0.0, 0.0, 0.0, fold_reports

    teacher_mean = float(np.mean([float(r["teacher_macro"]) for r in valid]))
    pred_mean = float(np.mean([float(r["pred_macro"]) for r in valid]))
    min_fold_delta = float(np.min([float(r["fold_delta"]) for r in valid]))
    return teacher_mean, pred_mean, min_fold_delta, fold_reports


def _build_bins(ts: pd.Series, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    ts_parsed = pd.to_datetime(ts, utc=True, errors="coerce")
    if ts_parsed.isna().any():
        raise ValueError("timestamp contains NaT")
    ts_vals = ts_parsed.astype("int64").to_numpy(dtype=np.int64)
    # Quantile bins with duplicate handling.
    q = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(ts_vals, q)
    edges = np.unique(edges.astype(np.int64))
    if len(edges) < 2:
        raise ValueError("failed to build bins from timestamp")
    # Bin index in [0, n_effective_bins-1].
    b = np.digitize(ts_vals, edges[1:-1], right=False).astype(np.int64)
    return b, edges


def _assign_bins(ts: pd.Series, edges: np.ndarray) -> np.ndarray:
    ts_parsed = pd.to_datetime(ts, utc=True, errors="coerce")
    if ts_parsed.isna().any():
        raise ValueError("timestamp contains NaT")
    ts_vals = ts_parsed.astype("int64").to_numpy(dtype=np.int64)
    b = np.digitize(ts_vals, edges[1:-1], right=False).astype(np.int64)
    return b


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Worst-aware time-binned forward/reverse teacher blending.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--forward-oof-csv", required=True)
    p.add_argument("--reverse-oof-csv", required=True)
    p.add_argument("--forward-test-csv", required=True)
    p.add_argument("--reverse-test-csv", required=True)
    p.add_argument("--reference-oof-csv", default="")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-time-bins", type=int, default=8)
    p.add_argument("--weights-grid", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--min-fold-delta", type=float, default=-0.002)
    p.add_argument("--smooth-neighbors", type=float, default=0.0)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    w_grid = _parse_grid(args.weights_grid)

    train_df = pd.read_csv(
        args.train_csv,
        usecols=["track_id", "bird_group", "timestamp_start_radar_utc", "observation_id"],
    )
    test_df = pd.read_csv(args.test_csv, usecols=["track_id", "timestamp_start_radar_utc"])

    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = _onehot(y_idx, len(CLASSES))
    train_track_ids = train_df["track_id"].to_numpy(dtype=np.int64)

    f_oof_df = pd.read_csv(args.forward_oof_csv, usecols=["track_id", *CLASSES]).set_index("track_id")
    r_oof_df = pd.read_csv(args.reverse_oof_csv, usecols=["track_id", *CLASSES]).set_index("track_id")
    f_oof = f_oof_df.loc[train_track_ids, CLASSES].to_numpy(dtype=np.float32)
    r_oof = r_oof_df.loc[train_track_ids, CLASSES].to_numpy(dtype=np.float32)

    if args.reference_oof_csv.strip():
        ref_oof_df = pd.read_csv(args.reference_oof_csv, usecols=["track_id", *CLASSES]).set_index("track_id")
        teacher_ref = ref_oof_df.loc[train_track_ids, CLASSES].to_numpy(dtype=np.float32)
    else:
        teacher_ref = f_oof.copy()

    cv_df = train_df.rename(columns={"timestamp_start_radar_utc": "_cv_ts", "observation_id": "_cv_group"})
    folds = make_forward_temporal_group_folds(
        cv_df,
        timestamp_col="_cv_ts",
        group_col="_cv_group",
        n_splits=int(args.n_splits),
    )

    bin_ids_train, edges = _build_bins(train_df["timestamp_start_radar_utc"], int(args.n_time_bins))
    n_bins_eff = int(bin_ids_train.max()) + 1

    # Start from global best against teacher_ref without constraints.
    best_global_w = 0.5
    best_global_gain = -1e9
    for wr in w_grid:
        pred = np.clip((1.0 - wr) * f_oof + wr * r_oof, 0.0, 1.0)
        t_mean, p_mean, _, _ = _evaluate_fold_metrics(y, pred, teacher_ref, folds)
        gain = p_mean - t_mean
        if gain > best_global_gain:
            best_global_gain = gain
            best_global_w = float(wr)

    wr_bins = np.full(n_bins_eff, best_global_w, dtype=np.float32)

    # Coordinate ascent with hard worst-fold constraint.
    for _ in range(max(1, int(args.passes))):
        for b in range(n_bins_eff):
            mask = bin_ids_train == b
            best_wr = float(wr_bins[b])
            best_score = -1e18
            for wr in w_grid:
                cand = wr_bins.copy()
                cand[b] = float(wr)
                pred = np.clip((1.0 - cand[bin_ids_train, None]) * f_oof + cand[bin_ids_train, None] * r_oof, 0.0, 1.0)
                t_mean, p_mean, min_delta, _ = _evaluate_fold_metrics(y, pred, teacher_ref, folds)
                gain = p_mean - t_mean
                if min_delta < float(args.min_fold_delta):
                    continue
                score = gain
                if score > best_score:
                    best_score = score
                    best_wr = float(wr)
            wr_bins[b] = best_wr

        if float(args.smooth_neighbors) > 0.0 and n_bins_eff > 2:
            s = float(args.smooth_neighbors)
            sm = wr_bins.copy()
            for b in range(1, n_bins_eff - 1):
                neigh = 0.5 * (wr_bins[b - 1] + wr_bins[b + 1])
                sm[b] = np.clip((1.0 - s) * wr_bins[b] + s * neigh, 0.0, 1.0)
            wr_bins = sm.astype(np.float32)

    pred_oof = np.clip((1.0 - wr_bins[bin_ids_train, None]) * f_oof + wr_bins[bin_ids_train, None] * r_oof, 0.0, 1.0)
    teacher_mean, pred_mean, min_fold_delta, fold_reports = _evaluate_fold_metrics(y, pred_oof, teacher_ref, folds)
    gain = pred_mean - teacher_mean

    # Build test submission.
    f_test_df = pd.read_csv(args.forward_test_csv)
    r_test_df = pd.read_csv(args.reverse_test_csv)
    if "track_id" not in f_test_df.columns or "track_id" not in r_test_df.columns:
        raise ValueError("test csv must contain track_id")
    f_test_df = f_test_df[["track_id", *CLASSES]].copy()
    r_test_df = r_test_df[["track_id", *CLASSES]].copy()
    merged = f_test_df.merge(r_test_df, on="track_id", suffixes=("_f", "_r"), how="inner")
    if len(merged) != len(f_test_df) or len(merged) != len(r_test_df):
        raise ValueError("track_id mismatch between forward/reverse test csv")

    test_bins = _assign_bins(test_df.set_index("track_id").loc[merged["track_id"].to_numpy(np.int64)]["timestamp_start_radar_utc"], edges)
    wr_test = wr_bins[test_bins].astype(np.float32)

    out_sub = pd.DataFrame({"track_id": merged["track_id"].to_numpy(np.int64)})
    for c in CLASSES:
        pf = merged[f"{c}_f"].to_numpy(dtype=np.float32)
        pr = merged[f"{c}_r"].to_numpy(dtype=np.float32)
        p = (1.0 - wr_test) * pf + wr_test * pr
        out_sub[c] = np.clip(p, 0.0, 1.0)

    sub_path = out_dir / "sub_time_binned_forward_reverse.csv"
    out_sub.to_csv(sub_path, index=False)

    oof_path = out_dir / "oof_time_binned_forward_reverse.csv"
    oof_df = pd.DataFrame({"track_id": train_track_ids})
    for j, c in enumerate(CLASSES):
        oof_df[c] = pred_oof[:, j]
    oof_df.to_csv(oof_path, index=False)

    weights_path = out_dir / "time_bin_weights.csv"
    pd.DataFrame(
        {
            "bin_id": np.arange(n_bins_eff, dtype=np.int64),
            "edge_left": edges[:-1],
            "edge_right": edges[1:],
            "w_reverse": wr_bins,
            "w_forward": 1.0 - wr_bins,
            "n_train": np.bincount(bin_ids_train, minlength=n_bins_eff),
        }
    ).to_csv(weights_path, index=False)

    report = {
        "train_csv": str(Path(args.train_csv).resolve()),
        "test_csv": str(Path(args.test_csv).resolve()),
        "forward_oof_csv": str(Path(args.forward_oof_csv).resolve()),
        "reverse_oof_csv": str(Path(args.reverse_oof_csv).resolve()),
        "reference_oof_csv": str(Path(args.reference_oof_csv).resolve()) if args.reference_oof_csv.strip() else "",
        "n_bins_effective": int(n_bins_eff),
        "weights_grid": [float(x) for x in w_grid.tolist()],
        "wr_bins": [float(x) for x in wr_bins.tolist()],
        "teacher_macro_mean": float(teacher_mean),
        "pred_macro_mean": float(pred_mean),
        "gain_macro_mean": float(gain),
        "min_fold_delta": float(min_fold_delta),
        "constraint_min_fold_delta": float(args.min_fold_delta),
        "constraint_passed": bool(min_fold_delta >= float(args.min_fold_delta)),
        "folds": fold_reports,
        "output_submission": str(sub_path),
        "output_oof": str(oof_path),
        "output_weights": str(weights_path),
    }
    report_path = out_dir / "time_binned_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(
        f"teacher_mean={teacher_mean:.6f} pred_mean={pred_mean:.6f} gain={gain:+.6f} "
        f"min_fold_delta={min_fold_delta:+.6f} passed={min_fold_delta >= float(args.min_fold_delta)}",
        flush=True,
    )
    print(str(sub_path), flush=True)
    print(str(report_path), flush=True)


if __name__ == "__main__":
    main()
