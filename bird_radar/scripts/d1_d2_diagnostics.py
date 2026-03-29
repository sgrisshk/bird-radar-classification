from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds, make_temporal_holdout_split
from src.feature_engineering import build_feature_frame
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache
from train_temporal_lgbm import macro_map, per_class_ap, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="/Users/sgrisshk/Desktop/AI-task")
    p.add_argument("--cache-dir", type=str, default="/Users/sgrisshk/Desktop/AI-task/bird_radar/artifacts/cache")
    p.add_argument("--output-dir", type=str, default="/Users/sgrisshk/Desktop/AI-task/bird_radar/artifacts/d1_d2_diag")
    p.add_argument(
        "--config-report",
        type=str,
        default="/Users/sgrisshk/Desktop/AI-task/bird_radar/artifacts/temporal_run_fix2_rcs/report.json",
    )
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-estimators", type=int, default=6000)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--holdout-quantile", type=float, default=0.8)
    return p.parse_args()


def load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict[str, Any]]:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


def one_hot(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.zeros((len(y_idx), n_classes), dtype=np.int32)
    y[np.arange(len(y_idx)), y_idx] = 1
    return y


def aggregate_by_primary(
    primary_ids: np.ndarray,
    y_idx: np.ndarray,
    pred: np.ndarray,
    mode: str = "mean",
) -> dict[str, Any]:
    if mode not in {"mean", "max"}:
        raise ValueError("mode must be mean|max")

    df = pd.DataFrame({"primary_observation_id": primary_ids, "y_idx": y_idx})
    for i, cls in enumerate(CLASSES):
        df[f"p_{i}"] = pred[:, i]

    rows_y: list[int] = []
    rows_p: list[np.ndarray] = []
    ambiguous_groups = 0
    total_groups = 0

    for _, g in df.groupby("primary_observation_id", sort=False):
        total_groups += 1
        labels = g["y_idx"].unique()
        if len(labels) != 1:
            ambiguous_groups += 1
            continue
        rows_y.append(int(labels[0]))
        arr = g[[f"p_{i}" for i in range(len(CLASSES))]].to_numpy(dtype=np.float32)
        if mode == "mean":
            rows_p.append(arr.mean(axis=0))
        else:
            rows_p.append(arr.max(axis=0))

    if not rows_y:
        return {
            "macro_map": 0.0,
            "per_class_ap": {c: 0.0 for c in CLASSES},
            "n_groups_total": int(total_groups),
            "n_groups_used": 0,
            "n_groups_ambiguous": int(ambiguous_groups),
        }

    y_idx_agg = np.asarray(rows_y, dtype=np.int64)
    y_agg = one_hot(y_idx_agg, n_classes=len(CLASSES))
    p_agg = np.asarray(rows_p, dtype=np.float32)
    return {
        "macro_map": float(macro_map(y_agg, p_agg)),
        "per_class_ap": per_class_ap(y_agg, p_agg),
        "n_groups_total": int(total_groups),
        "n_groups_used": int(len(rows_y)),
        "n_groups_ambiguous": int(ambiguous_groups),
    }


def compute_sample_weights(ts_series: pd.Series) -> np.ndarray:
    ts_num = ts_series.astype("int64").to_numpy(dtype=np.float64)
    ts_norm = (ts_num - ts_num.min()) / (ts_num.max() - ts_num.min() + 1e-9)
    return (0.7 + 0.6 * ts_norm).astype(np.float32)


def train_ovr_safe(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    cfg: dict[str, float | int | str],
    seed: int,
    n_estimators: int,
    sample_weight: np.ndarray | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    preds = np.zeros((len(X_valid), len(CLASSES)), dtype=np.float32)
    fallback_events: list[dict[str, Any]] = []

    for ci in range(len(CLASSES)):
        ytr = y_train[:, ci].astype(np.int32)
        yva = y_valid[:, ci].astype(np.int32)
        uniq = np.unique(ytr)
        if uniq.size < 2:
            prior = float(np.mean(ytr))
            preds[:, ci] = prior
            fallback_events.append(
                {
                    "class": CLASSES[ci],
                    "reason": "single_class_train",
                    "prior": prior,
                }
            )
            continue

        pos = max(int(ytr.sum()), 1)
        neg = max(int(len(ytr) - ytr.sum()), 1)
        model = lgb.LGBMClassifier(
            objective="binary",
            metric="average_precision",
            n_estimators=n_estimators,
            learning_rate=float(cfg["lr"]),
            num_leaves=int(cfg["leaves"]),
            colsample_bytree=float(cfg["ff"]),
            subsample=0.8,
            subsample_freq=1,
            reg_alpha=float(cfg["l1"]),
            reg_lambda=float(cfg["l2"]),
            min_child_samples=int(cfg["mc"]),
            scale_pos_weight=float(neg / pos),
            random_state=seed + ci,
            n_jobs=-1,
            verbosity=-1,
        )
        if np.unique(yva).size >= 2:
            model.fit(
                X_train,
                ytr,
                sample_weight=sample_weight,
                eval_set=[(X_valid, yva)],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
            )
        else:
            model.fit(
                X_train,
                ytr,
                sample_weight=sample_weight,
                callbacks=[lgb.log_evaluation(0)],
            )
        preds[:, ci] = model.predict_proba(X_valid)[:, 1]

    return preds, fallback_events


def forward_oof_for_group(
    feat_train: pd.DataFrame,
    y: np.ndarray,
    y_idx: np.ndarray,
    feature_cols: list[str],
    group_col: str,
    cfg: dict[str, float | int | str],
    seed: int,
    n_estimators: int,
    n_splits: int,
) -> dict[str, Any]:
    cv_df = pd.DataFrame(
        {
            "_cv_ts": pd.to_datetime(feat_train["ts"], errors="coerce", utc=True),
            "_cv_group": feat_train[group_col].astype(np.int64),
        }
    )
    folds = make_forward_temporal_group_folds(
        cv_df,
        timestamp_col="_cv_ts",
        group_col="_cv_group",
        n_splits=n_splits,
    )

    oof = np.zeros((len(feat_train), len(CLASSES)), dtype=np.float32)
    covered = np.zeros(len(feat_train), dtype=bool)
    fold_scores: list[float] = []

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        Xtr = feat_train.iloc[tr_idx][feature_cols].astype(np.float32)
        Xva = feat_train.iloc[va_idx][feature_cols].astype(np.float32)
        ytr = y[tr_idx]
        yva = y[va_idx]
        sw = compute_sample_weights(feat_train.iloc[tr_idx]["ts"])
        pred, fallback_events = train_ovr_safe(
            X_train=Xtr,
            y_train=ytr,
            X_valid=Xva,
            y_valid=yva,
            cfg=cfg,
            seed=seed + fold_id * 17,
            n_estimators=n_estimators,
            sample_weight=sw,
        )
        oof[va_idx] = pred
        covered[va_idx] = True
        fold_scores.append(float(macro_map(yva, pred)))
        _ = fallback_events

    y_cov = y[covered]
    oof_cov = oof[covered]
    y_idx_cov = y_idx[covered]
    primary_cov = feat_train.loc[covered, "primary_observation_id"].to_numpy(dtype=np.int64)

    d1_mean = aggregate_by_primary(primary_cov, y_idx_cov, oof_cov, mode="mean")
    d1_max = aggregate_by_primary(primary_cov, y_idx_cov, oof_cov, mode="max")

    return {
        "group_col": group_col,
        "n_folds": int(len(folds)),
        "n_covered_rows": int(covered.sum()),
        "n_total_rows": int(len(covered)),
        "coverage_ratio": float(covered.mean()),
        "fold_scores": [float(x) for x in fold_scores],
        "fold_mean": float(np.mean(fold_scores)) if fold_scores else 0.0,
        "fold_std": float(np.std(fold_scores)) if fold_scores else 0.0,
        "fold_spread": float(np.max(fold_scores) - np.min(fold_scores)) if fold_scores else 0.0,
        "oof_track_macro_map": float(macro_map(y_cov, oof_cov)) if len(y_cov) else 0.0,
        "oof_track_per_class_ap": per_class_ap(y_cov, oof_cov) if len(y_cov) else {c: 0.0 for c in CLASSES},
        "d1_primary_mean": d1_mean,
        "d1_primary_max": d1_max,
    }


def temporal_holdout_for_group(
    feat_train: pd.DataFrame,
    y: np.ndarray,
    y_idx: np.ndarray,
    feature_cols: list[str],
    group_col: str,
    cfg: dict[str, float | int | str],
    seed: int,
    n_estimators: int,
    holdout_quantile: float,
) -> dict[str, Any]:
    split_df = pd.DataFrame(
        {
            "_cv_ts": pd.to_datetime(feat_train["ts"], errors="coerce", utc=True),
            "_cv_group": feat_train[group_col].astype(np.int64),
        }
    )
    tr_idx, va_idx, cutoff = make_temporal_holdout_split(
        split_df,
        timestamp_col="_cv_ts",
        group_col="_cv_group",
        holdout_quantile=holdout_quantile,
    )

    Xtr = feat_train.iloc[tr_idx][feature_cols].astype(np.float32)
    Xva = feat_train.iloc[va_idx][feature_cols].astype(np.float32)
    ytr = y[tr_idx]
    yva = y[va_idx]
    sw = compute_sample_weights(feat_train.iloc[tr_idx]["ts"])
    pred, fallback_events = train_ovr_safe(
        X_train=Xtr,
        y_train=ytr,
        X_valid=Xva,
        y_valid=yva,
        cfg=cfg,
        seed=seed,
        n_estimators=n_estimators,
        sample_weight=sw,
    )

    primary_va = feat_train.iloc[va_idx]["primary_observation_id"].to_numpy(dtype=np.int64)
    y_idx_va = y_idx[va_idx]
    d1_mean = aggregate_by_primary(primary_va, y_idx_va, pred, mode="mean")
    d1_max = aggregate_by_primary(primary_va, y_idx_va, pred, mode="max")

    return {
        "group_col": group_col,
        "cutoff": str(cutoff),
        "train_size": int(len(tr_idx)),
        "valid_size": int(len(va_idx)),
        "track_macro_map": float(macro_map(yva, pred)),
        "track_per_class_ap": per_class_ap(yva, pred),
        "d1_primary_mean": d1_mean,
        "d1_primary_max": d1_max,
        "fallback_events": fallback_events,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg_report = json.loads(Path(args.config_report).read_text(encoding="utf-8"))
    cfg = cfg_report["best_config"]

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    train_cache = load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")
    _ = test_cache

    feat_train = build_feature_frame(train_df, track_cache=train_cache)
    feat_train = feat_train.merge(
        train_df[["track_id", "bird_group", "observation_id", "primary_observation_id", "timestamp_start_radar_utc"]],
        on="track_id",
        how="left",
        suffixes=("", "_src"),
    ).copy()
    feat_train["ts"] = pd.to_datetime(feat_train["timestamp_start_radar_utc"], errors="coerce", utc=True)
    feat_train["observation_id"] = feat_train["observation_id"].astype(np.int64)
    feat_train["primary_observation_id"] = feat_train["primary_observation_id"].astype(np.int64)
    feat_train["track_id"] = feat_train["track_id"].astype(np.int64)

    reserved = {
        "track_id",
        "observation_id",
        "primary_observation_id",
        "bird_group",
        "timestamp_start_radar_utc",
        "ts",
        "primary_observation_id_src",
        "observation_id_src",
        "primary_observation_id_x",
        "primary_observation_id_y",
    }
    feature_cols = [c for c in feat_train.columns if c not in reserved]

    y_idx = feat_train["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = one_hot(y_idx, n_classes=len(CLASSES))

    holdout_obs = temporal_holdout_for_group(
        feat_train=feat_train,
        y=y,
        y_idx=y_idx,
        feature_cols=feature_cols,
        group_col="observation_id",
        cfg=cfg,
        seed=args.seed,
        n_estimators=args.n_estimators,
        holdout_quantile=args.holdout_quantile,
    )

    d2_forward_obs = forward_oof_for_group(
        feat_train=feat_train,
        y=y,
        y_idx=y_idx,
        feature_cols=feature_cols,
        group_col="observation_id",
        cfg=cfg,
        seed=args.seed,
        n_estimators=args.n_estimators,
        n_splits=args.n_splits,
    )
    d2_forward_primary = forward_oof_for_group(
        feat_train=feat_train,
        y=y,
        y_idx=y_idx,
        feature_cols=feature_cols,
        group_col="primary_observation_id",
        cfg=cfg,
        seed=args.seed + 101,
        n_estimators=args.n_estimators,
        n_splits=args.n_splits,
    )
    d2_forward_track = forward_oof_for_group(
        feat_train=feat_train,
        y=y,
        y_idx=y_idx,
        feature_cols=feature_cols,
        group_col="track_id",
        cfg=cfg,
        seed=args.seed + 202,
        n_estimators=args.n_estimators,
        n_splits=args.n_splits,
    )

    report = {
        "config_report": str(Path(args.config_report).resolve()),
        "best_config_used": cfg,
        "n_estimators": int(args.n_estimators),
        "n_splits": int(args.n_splits),
        "holdout_quantile": float(args.holdout_quantile),
        "n_features": int(len(feature_cols)),
        "holdout_observation_group": holdout_obs,
        "d2_forward_observation_group": d2_forward_obs,
        "d2_forward_primary_group": d2_forward_primary,
        "d2_forward_track_group": d2_forward_track,
    }

    out_path = out_dir / "d1_d2_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}", flush=True)
    print(
        "D2 forward track-mAP | obs={:.6f} primary={:.6f} track={:.6f}".format(
            d2_forward_obs["oof_track_macro_map"],
            d2_forward_primary["oof_track_macro_map"],
            d2_forward_track["oof_track_macro_map"],
        ),
        flush=True,
    )
    print(
        "D1 primary-mean on obs-holdout: {:.6f}".format(holdout_obs["d1_primary_mean"]["macro_map"]),
        flush=True,
    )


if __name__ == "__main__":
    main()
