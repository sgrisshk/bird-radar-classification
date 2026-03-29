#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.feature_engineering import build_feature_frame, compute_monthly_track_centers
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache
from train_temporal_lgbm import (
    _apply_blacklist,
    _blacklist_patterns,
    _time_weight_from_timestamps,
    compute_drift_ks,
    macro_map,
    per_class_ap,
    train_classifier,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate leave-one-month-out CV on months [1,4,9,10].")
    p.add_argument("--data-dir", type=str, default="/Users/sgrisshk/Desktop/AI-task")
    p.add_argument("--cache-dir", type=str, default="/Users/sgrisshk/Desktop/AI-task/bird_radar/artifacts/cache")
    p.add_argument("--output-json", type=str, default="bird_radar/artifacts/proposal/lomo_month_cv_report.json")
    p.add_argument("--output-csv", type=str, default="bird_radar/artifacts/proposal/lomo_month_cv_grid.csv")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-estimators", type=int, default=5000)
    p.add_argument("--drop-k-grid", type=str, default="10")
    p.add_argument("--configs", type=str, default="base,reg,aggr,deep,wide")
    p.add_argument("--training-objective", type=str, default="ovr", choices=["ovr", "multiclass"])
    p.add_argument("--use-time-weights", action="store_true", default=True)
    p.add_argument("--no-time-weights", dest="use_time_weights", action="store_false")
    p.add_argument("--blacklist-mode", type=str, default="none", choices=["none", "A", "B"])
    p.add_argument(
        "--extra-blacklist",
        type=str,
        default=(
            "observer_pos_has_z,observer_pos_z,observer_pos_z_minus_mean_alt,observer_pos_z_minus_last_alt,"
            "hours_since_sunrise,hour_utc,frac_of_day,day_length_hours,is_daytime"
        ),
    )
    p.add_argument(
        "--spatial-blacklist",
        type=str,
        default=(
            "track_center_lat,track_center_lon,dist_to_,harbor_bearing_,north_of_harbor,east_of_harbor,"
            "lat_deviation_from_month_center,lon_deviation_from_month_center,dist_from_month_center,"
            "month_center_bearing_,track_span_lat_deg,track_span_lon_deg"
        ),
    )
    p.add_argument("--monotone-safe-bio", action="store_true", default=False)
    p.add_argument("--multiclass-oversample-rare", action="store_true", default=False)
    p.add_argument("--multiclass-corm-factor", type=float, default=10.0)
    p.add_argument("--multiclass-wader-factor", type=float, default=5.0)
    p.add_argument("--multiclass-oversample-noise", type=float, default=0.02)
    return p.parse_args()


def load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict[str, Any]]:
    cache = load_track_cache(path)
    if cache is not None:
        return cache
    built = build_track_cache(df)
    save_track_cache(built, path)
    return built


def candidate_cfgs(names: list[str]) -> list[dict[str, Any]]:
    all_cfgs = [
        {"name": "base", "lr": 0.03, "leaves": 63, "ff": 0.75, "mc": 20, "l1": 0.05, "l2": 0.2},
        {"name": "reg", "lr": 0.02, "leaves": 47, "ff": 0.70, "mc": 30, "l1": 0.10, "l2": 0.40},
        {"name": "aggr", "lr": 0.03, "leaves": 95, "ff": 0.85, "mc": 10, "l1": 0.00, "l2": 0.05},
        {"name": "deep", "lr": 0.02, "leaves": 127, "ff": 0.70, "mc": 15, "l1": 0.05, "l2": 0.10},
        {"name": "wide", "lr": 0.01, "leaves": 63, "ff": 0.60, "mc": 40, "l1": 0.20, "l2": 0.50},
    ]
    name_set = {x.strip() for x in names if x.strip()}
    return [c for c in all_cfgs if c["name"] in name_set]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    train_cache = load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")

    monthly_centers = compute_monthly_track_centers(train_df, track_cache=train_cache)
    feat_train = build_feature_frame(train_df, track_cache=train_cache, monthly_centers=monthly_centers)
    feat_test = build_feature_frame(test_df, track_cache=test_cache, monthly_centers=monthly_centers)

    train_df["ts"] = pd.to_datetime(train_df["timestamp_start_radar_utc"], errors="coerce", utc=True)
    train_df["month"] = train_df["ts"].dt.month.astype("Int64")
    feat_train = feat_train.merge(
        train_df[["track_id", "bird_group", "ts", "month"]],
        on="track_id",
        how="left",
    )

    y_idx = feat_train["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = np.zeros((len(feat_train), len(CLASSES)), dtype=np.int32)
    y[np.arange(len(feat_train)), y_idx] = 1

    feature_cols = [
        c for c in feat_train.columns if c not in {"track_id", "observation_id", "primary_observation_id", "bird_group", "ts", "month"}
    ]
    patterns = _blacklist_patterns(args.blacklist_mode)
    patterns.extend([x.strip() for x in args.extra_blacklist.split(",") if x.strip()])
    patterns.extend([x.strip() for x in args.spatial_blacklist.split(",") if x.strip()])
    feature_cols, dropped = _apply_blacklist(feature_cols, patterns)

    drift = compute_drift_ks(feat_train, feat_test, feature_cols)
    drop_k_values = [int(x.strip()) for x in args.drop_k_grid.split(",") if x.strip()]
    cfgs = candidate_cfgs(args.configs.split(","))
    if len(cfgs) == 0:
        raise ValueError("No configs selected")

    # Leave-one-month-out folds on observed train months.
    folds = [
        {"name": "m1_holdout", "train": [4, 9, 10], "val": [1]},
        {"name": "m4_holdout", "train": [1, 9, 10], "val": [4]},
        {"name": "m9_holdout", "train": [1, 4, 10], "val": [9]},
        {"name": "m10_holdout", "train": [1, 4, 9], "val": [10]},
    ]

    rows: list[dict[str, Any]] = []
    for drop_k in drop_k_values:
        drop_set = {name for name, _ in drift[:drop_k]}
        cols = [c for c in feature_cols if c not in drop_set]
        for cfg in cfgs:
            fold_scores: list[float] = []
            fold_sizes: list[int] = []
            fold_class_ap: dict[str, list[float]] = {c: [] for c in CLASSES}
            for fi, fold in enumerate(folds):
                tr_mask = feat_train["month"].isin(fold["train"]).to_numpy()
                va_mask = feat_train["month"].isin(fold["val"]).to_numpy()
                tr_idx = np.where(tr_mask)[0]
                va_idx = np.where(va_mask)[0]
                if len(tr_idx) == 0 or len(va_idx) == 0:
                    continue

                Xtr = feat_train.iloc[tr_idx][cols].astype(np.float32)
                Xva = feat_train.iloc[va_idx][cols].astype(np.float32)
                ytr = y[tr_idx]
                yva = y[va_idx]
                wtr = _time_weight_from_timestamps(feat_train.iloc[tr_idx]["ts"], enabled=args.use_time_weights)

                pred_va = train_classifier(
                    X_train=Xtr,
                    y_train=ytr,
                    X_valid=Xva,
                    y_valid=yva,
                    cfg=cfg,
                    seed=int(args.seed + drop_k * 97 + fi * 131),
                    n_estimators=int(args.n_estimators),
                    sample_weight=wtr,
                    training_objective=args.training_objective,
                    monotone_safe_bio=bool(args.monotone_safe_bio),
                    multiclass_oversample_rare=bool(args.multiclass_oversample_rare),
                    multiclass_corm_factor=float(args.multiclass_corm_factor),
                    multiclass_wader_factor=float(args.multiclass_wader_factor),
                    multiclass_oversample_noise=float(args.multiclass_oversample_noise),
                )
                score = float(macro_map(yva, pred_va))
                fold_scores.append(score)
                fold_sizes.append(int(len(va_idx)))
                ap_map = per_class_ap(yva, pred_va)
                for c in CLASSES:
                    fold_class_ap[c].append(float(ap_map.get(c, np.nan)))

                print(
                    f"drop_k={drop_k} cfg={cfg['name']} fold={fold['name']} "
                    f"macro={score:.6f} train={len(tr_idx)} valid={len(va_idx)}",
                    flush=True,
                )

            if len(fold_scores) == 0:
                continue
            unweighted_mean = float(np.mean(fold_scores))
            weighted_mean = float(np.average(np.array(fold_scores), weights=np.array(fold_sizes)))
            row: dict[str, Any] = {
                "drop_k": int(drop_k),
                "cfg": cfg["name"],
                "n_features": int(len(cols)),
                "lomo_macro_mean": unweighted_mean,
                "lomo_macro_weighted_mean": weighted_mean,
                "fold_m1": float(fold_scores[0]) if len(fold_scores) > 0 else np.nan,
                "fold_m4": float(fold_scores[1]) if len(fold_scores) > 1 else np.nan,
                "fold_m9": float(fold_scores[2]) if len(fold_scores) > 2 else np.nan,
                "fold_m10": float(fold_scores[3]) if len(fold_scores) > 3 else np.nan,
            }
            for c in CLASSES:
                vals = [v for v in fold_class_ap[c] if np.isfinite(v)]
                row[f"class_ap_mean__{c}"] = float(np.mean(vals)) if vals else np.nan
            rows.append(row)

    if len(rows) == 0:
        raise RuntimeError("No LOMO results produced")

    grid = pd.DataFrame(rows).sort_values("lomo_macro_mean", ascending=False).reset_index(drop=True)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out_csv, index=False)

    best = grid.iloc[0].to_dict()
    out_json = Path(args.output_json)
    out = {
        "seed": int(args.seed),
        "n_estimators": int(args.n_estimators),
        "training_objective": args.training_objective,
        "blacklist_mode": args.blacklist_mode,
        "patterns_applied": patterns,
        "blacklist_dropped_count": int(len(dropped)),
        "grid_rows": int(len(grid)),
        "best": best,
        "grid_csv": str(out_csv.resolve()),
        "fold_definition": folds,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=== LOMO MONTH CV REPORT ===", flush=True)
    print(f"best_cfg={best['cfg']} drop_k={int(best['drop_k'])} n_features={int(best['n_features'])}", flush=True)
    print(
        f"best_lomo_mean={float(best['lomo_macro_mean']):.6f} "
        f"best_lomo_weighted={float(best['lomo_macro_weighted_mean']):.6f}",
        flush=True,
    )
    print(f"saved_grid={out_csv}", flush=True)
    print(f"saved_report={out_json}", flush=True)


if __name__ == "__main__":
    main()
