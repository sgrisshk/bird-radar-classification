#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache
from src.redesign.features import build_tabular_frame, get_feature_columns
from src.redesign.utils import dump_json, macro_map, per_class_ap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost OVR model on tabular temporal features (forward CV).")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--sample-submission", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default="bird_radar/artifacts/cache")

    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--time-col", default="timestamp_start_radar_utc")
    p.add_argument("--group-col", default="observation_id")
    p.add_argument("--id-col", default="track_id")
    p.add_argument("--label-col", default="bird_group")

    p.add_argument("--min-class-count", type=int, default=5)
    p.add_argument(
        "--invalid-class-policy",
        type=str,
        default="fallback_teacher",
        choices=["fallback_prior", "fallback_teacher"],
    )
    p.add_argument("--teacher-oof-csv", type=str, default="")
    p.add_argument("--teacher-test-csv", type=str, default="")

    p.add_argument("--n-estimators", type=int, default=5000)
    p.add_argument("--learning-rate", type=float, default=0.02)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--reg-alpha", type=float, default=0.5)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--early-stopping-rounds", type=int, default=300)
    p.add_argument("--tree-method", type=str, default="hist")
    p.add_argument("--n-jobs", type=int, default=-1)

    p.add_argument(
        "--time-weight-mode",
        type=str,
        default="fold0_boost",
        choices=["none", "fold0_boost", "linear_time"],
    )
    p.add_argument("--fold0-boost", type=float, default=0.7)
    p.add_argument("--time-weight-strength", type=float, default=0.6)
    return p.parse_args()


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict[str, Any]]:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


def _align_probs_from_csv(csv_path: str, ids: np.ndarray, id_col: str) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=[id_col, *CLASSES])
    mp = {int(r[id_col]): np.asarray([r[c] for c in CLASSES], dtype=np.float32) for _, r in df.iterrows()}
    missing = [int(t) for t in ids if int(t) not in mp]
    if missing:
        raise ValueError(f"{csv_path} missing {len(missing)} ids; first={missing[:10]}")
    return np.stack([mp[int(t)] for t in ids], axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = out_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    sample_sub = pd.read_csv(args.sample_submission)
    id_col = str(args.id_col)
    label_col = str(args.label_col)
    time_col = str(args.time_col)
    group_col = str(args.group_col)

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")

    tab_train = build_tabular_frame(train_df, train_cache)
    tab_test = build_tabular_frame(test_df, test_cache)
    feature_cols = get_feature_columns(tab_train)

    train_ids = train_df[id_col].to_numpy(dtype=np.int64)
    test_ids = test_df[id_col].to_numpy(dtype=np.int64)
    np.save(out_dir / "train_track_ids.npy", train_ids)
    np.save(out_dir / "test_track_ids.npy", test_ids)

    x_train = tab_train[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=np.float32)
    x_test = tab_test[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=np.float32)

    # Normalize exactly once for stable scale across OvR heads.
    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    x_train = (x_train - mu) / sd
    x_test = (x_test - mu) / sd

    y_idx = train_df[label_col].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = np.zeros((len(y_idx), len(CLASSES)), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    np.save(out_dir / "oof_targets.npy", y)

    global_priors = np.clip(y.mean(axis=0), 1e-5, 1.0 - 1e-5).astype(np.float32)
    teacher_oof: np.ndarray | None = None
    teacher_test: np.ndarray | None = None
    if str(args.invalid_class_policy) == "fallback_teacher":
        if not str(args.teacher_oof_csv).strip() or not str(args.teacher_test_csv).strip():
            raise ValueError("fallback_teacher requires --teacher-oof-csv and --teacher-test-csv")
        teacher_oof = _align_probs_from_csv(str(args.teacher_oof_csv), train_ids, id_col=id_col)
        teacher_test = _align_probs_from_csv(str(args.teacher_test_csv), test_ids, id_col=id_col)

    cv_df = pd.DataFrame({"_cv_ts": train_df[time_col], "_cv_group": train_df[group_col].astype(np.int64)})
    folds = make_forward_temporal_group_folds(
        cv_df,
        timestamp_col="_cv_ts",
        group_col="_cv_group",
        n_splits=int(args.n_splits),
    )
    if len(folds) <= 0:
        raise RuntimeError("no folds were generated")

    train_weights = np.ones((len(train_df),), dtype=np.float32)
    if str(args.time_weight_mode) == "fold0_boost":
        fold0_val_idx = folds[0][1]
        train_weights[fold0_val_idx] += max(0.0, float(args.fold0_boost))
        print(
            f"[xgb] time_weight_mode=fold0_boost fold0_n={len(fold0_val_idx)} boost={float(args.fold0_boost):.4f} "
            f"w_mean={np.mean(train_weights):.4f}",
            flush=True,
        )
    elif str(args.time_weight_mode) == "linear_time":
        ts = pd.to_datetime(train_df[time_col], errors="coerce", utc=True)
        if ts.isna().any():
            raise ValueError(f"timestamp parsing failed for {int(ts.isna().sum())} rows")
        ns = ts.astype("int64").to_numpy(dtype=np.int64)
        t_min = int(ns.min())
        t_max = int(ns.max())
        denom = float(max(1, t_max - t_min))
        t_norm = (ns.astype(np.float64) - float(t_min)) / denom
        strength = max(0.0, float(args.time_weight_strength))
        train_weights = (1.0 + strength * (1.0 - t_norm)).astype(np.float32)
        print(
            f"[xgb] time_weight_mode=linear_time strength={strength:.4f} w_mean={np.mean(train_weights):.4f}",
            flush=True,
        )

    n_train = len(train_df)
    n_test = len(test_df)
    n_classes = len(CLASSES)
    oof = np.zeros((n_train, n_classes), dtype=np.float32)
    test_accum = np.zeros((n_test, n_classes), dtype=np.float32)
    fold_scores: list[float] = []
    fallback_events: list[dict[str, Any]] = []

    min_count = max(1, int(args.min_class_count))
    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        tr_idx = np.asarray(tr_idx, dtype=np.int64)
        va_idx = np.asarray(va_idx, dtype=np.int64)
        fold_test = np.zeros((n_test, n_classes), dtype=np.float32)

        for c, cls in enumerate(CLASSES):
            y_tr = y[tr_idx, c]
            y_va = y[va_idx, c]
            pos = int(np.sum(y_tr))
            neg = int(len(y_tr) - pos)

            invalid = (pos < min_count) or (neg < min_count) or (np.unique(y_tr).size < 2)
            if invalid:
                if str(args.invalid_class_policy) == "fallback_teacher":
                    assert teacher_oof is not None and teacher_test is not None
                    oof[va_idx, c] = teacher_oof[va_idx, c]
                    fold_test[:, c] = teacher_test[:, c]
                    fallback_events.append(
                        {
                            "fold": int(fold_id),
                            "class": str(cls),
                            "reason": "fallback_teacher",
                            "n_pos_train": int(pos),
                            "n_neg_train": int(neg),
                        }
                    )
                else:
                    prior = float(global_priors[c])
                    oof[va_idx, c] = prior
                    fold_test[:, c] = prior
                    fallback_events.append(
                        {
                            "fold": int(fold_id),
                            "class": str(cls),
                            "reason": "fallback_prior",
                            "prior": float(prior),
                            "n_pos_train": int(pos),
                            "n_neg_train": int(neg),
                        }
                    )
                continue

            scale_pos_weight = float((neg + 1.0) / (pos + 1.0))
            model = xgb.XGBClassifier(
                n_estimators=int(args.n_estimators),
                learning_rate=float(args.learning_rate),
                max_depth=int(args.max_depth),
                subsample=float(args.subsample),
                colsample_bytree=float(args.colsample_bytree),
                reg_alpha=float(args.reg_alpha),
                reg_lambda=float(args.reg_lambda),
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                early_stopping_rounds=int(args.early_stopping_rounds),
                tree_method=str(args.tree_method),
                random_state=int(args.seed) + 97 * int(fold_id) + 13 * int(c),
                n_jobs=int(args.n_jobs),
                verbosity=0,
            )
            model.fit(
                x_train[tr_idx],
                y_tr,
                sample_weight=train_weights[tr_idx],
                eval_set=[(x_train[va_idx], y_va)],
                verbose=False,
            )
            oof[va_idx, c] = model.predict_proba(x_train[va_idx])[:, 1].astype(np.float32)
            fold_test[:, c] = model.predict_proba(x_test)[:, 1].astype(np.float32)

        test_accum += fold_test
        fold_score = float(macro_map(y[va_idx], oof[va_idx]))
        fold_scores.append(fold_score)
        print(f"[fold] {fold_id + 1}/{len(folds)} macro_map={fold_score:.6f} n_val={len(va_idx)}", flush=True)

    test_accum /= float(len(folds))
    macro_covered = float(macro_map(y, oof))
    per_covered = per_class_ap(y, oof)

    np.save(art_dir / "xgb_ovr_oof.npy", oof.astype(np.float32))
    np.save(art_dir / "xgb_ovr_test.npy", test_accum.astype(np.float32))

    sub = sample_sub.copy()
    sub = sub[[c for c in sub.columns if c == id_col or c in CLASSES]].copy()
    sub[id_col] = test_ids
    sub[CLASSES] = np.clip(test_accum, 0.0, 1.0)
    sub.to_csv(out_dir / "submission_xgb_ovr.csv", index=False)
    sub.to_csv(out_dir / "submission.csv", index=False)

    # oof csv for external validators
    oof_df = pd.DataFrame(oof, columns=CLASSES)
    oof_df.insert(0, id_col, train_ids)
    oof_df.to_csv(out_dir / "oof_preds.csv", index=False)

    summary = {
        "project_root": str(PROJECT_ROOT.resolve()),
        "output_dir": str(out_dir),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "features_total": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "seed": int(args.seed),
        "n_splits": int(len(folds)),
        "params": {
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "max_depth": int(args.max_depth),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "reg_alpha": float(args.reg_alpha),
            "reg_lambda": float(args.reg_lambda),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "tree_method": str(args.tree_method),
        },
        "time_weight_mode": str(args.time_weight_mode),
        "fallback_events": fallback_events,
        "models": {
            f"xgb_ovr_seed{int(args.seed)}": {
                "type": "tabular",
                "oof_path": str((art_dir / "xgb_ovr_oof.npy").resolve()),
                "test_path": str((art_dir / "xgb_ovr_test.npy").resolve()),
                "macro_map": float(macro_covered),
                "per_class_ap": per_covered,
                "fold_scores": [float(x) for x in fold_scores],
                "fold_mean": float(np.mean(fold_scores)),
                "worst_fold": float(np.min(fold_scores)),
            }
        },
        "track_ids_test_path": str((out_dir / "test_track_ids.npy").resolve()),
        "track_ids_train_path": str((out_dir / "train_track_ids.npy").resolve()),
    }
    dump_json(out_dir / "oof_summary.json", summary)

    print("=== XGB OVR COMPLETE ===", flush=True)
    print(f"output_dir: {out_dir}", flush=True)
    print(f"macro_covered={macro_covered:.6f} fold_mean={np.mean(fold_scores):.6f}", flush=True)


if __name__ == "__main__":
    main()

