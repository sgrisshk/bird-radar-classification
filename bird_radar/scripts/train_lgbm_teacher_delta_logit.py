#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache
from src.redesign.features import build_tabular_frame, get_feature_columns
from src.redesign.utils import dump_json, macro_map, per_class_ap

EPS = 1e-8


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-class LGBM delta-logit teacher corrector (forward CV).")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--sample-submission", required=True)
    p.add_argument("--teacher-oof-csv", required=True)
    p.add_argument("--teacher-test-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default="bird_radar/artifacts/cache")

    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--start-fold", type=int, default=1)
    p.add_argument("--max-folds", type=int, default=0)
    p.add_argument("--time-col", default="timestamp_start_radar_utc")
    p.add_argument("--group-col", default="observation_id")

    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--feature-fraction", type=float, default=0.8)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--bagging-freq", type=int, default=1)
    p.add_argument("--min-data-in-leaf", type=int, default=120)
    p.add_argument("--lambda-l2", type=float, default=3.0)
    p.add_argument("--max-depth", type=int, default=-1)
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--early-stopping-rounds", type=int, default=100)
    p.add_argument("--num-boost-round", type=int, default=2500)

    p.add_argument("--target-eps", type=float, default=1e-4, help="Clip for y in logit(y).")
    p.add_argument(
        "--target-mode",
        choices=["logit_delta", "prob_delta"],
        default="logit_delta",
        help="Target definition for residual learning.",
    )
    p.add_argument(
        "--delta-target-clip",
        type=float,
        default=0.0,
        help="Optional clip for delta target before training. <=0 disables clipping.",
    )
    p.add_argument("--delta-clip", type=float, default=0.5, help="Clip applied to predicted delta at inference.")
    p.add_argument(
        "--feature-mode",
        choices=["rich_only", "rich_plus_teacher_prob", "rich_plus_teacher_meta"],
        default="rich_only",
        help="Feature set used for delta regressor.",
    )
    p.add_argument(
        "--uncertainty-weight-mode",
        choices=["none", "binary_maxprob", "linear_maxprob"],
        default="none",
        help="Sample weighting based on teacher confidence.",
    )
    p.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.65,
        help="Threshold for binary_maxprob mode.",
    )
    p.add_argument(
        "--uncertainty-k",
        type=float,
        default=4.0,
        help="Strength multiplier for uncertainty-based weights.",
    )
    return p.parse_args()


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict[str, Any]]:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


def _align_probs_from_csv(csv_path: str, ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=["track_id", *CLASSES])
    mp = {int(r["track_id"]): np.asarray([r[c] for c in CLASSES], dtype=np.float32) for _, r in df.iterrows()}
    missing = [int(t) for t in ids if int(t) not in mp]
    if missing:
        raise ValueError(f"{csv_path} missing {len(missing)} ids; first={missing[:10]}")
    return np.stack([mp[int(t)] for t in ids], axis=0).astype(np.float32)


def _one_hot_targets(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.zeros((len(y_idx), n_classes), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    return y


def _logit(p: np.ndarray) -> np.ndarray:
    x = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (train_x - mu) / sd, (test_x - mu) / sd


def _teacher_meta(teacher_probs: np.ndarray) -> pd.DataFrame:
    sort_probs = np.sort(teacher_probs, axis=1)
    top1 = sort_probs[:, -1]
    top2 = sort_probs[:, -2]
    margin = top1 - top2
    entropy = -np.sum(
        np.clip(teacher_probs, 1e-12, 1.0) * np.log(np.clip(teacher_probs, 1e-12, 1.0)),
        axis=1,
    ) / np.log(float(teacher_probs.shape[1]))
    top1_class = np.argmax(teacher_probs, axis=1).astype(np.float32)
    return pd.DataFrame(
        {
            "teacher_top1_prob": top1.astype(np.float32),
            "teacher_top2_prob": top2.astype(np.float32),
            "teacher_margin": margin.astype(np.float32),
            "teacher_entropy": entropy.astype(np.float32),
            "teacher_top1_class_id": top1_class,
        }
    )


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = out_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    sample_sub = pd.read_csv(args.sample_submission)

    train_ids = train_df["track_id"].to_numpy(dtype=np.int64)
    test_ids = test_df["track_id"].to_numpy(dtype=np.int64)
    np.save(out_dir / "train_track_ids.npy", train_ids)
    np.save(out_dir / "test_track_ids.npy", test_ids)

    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = _one_hot_targets(y_idx, len(CLASSES))
    np.save(out_dir / "oof_targets.npy", y)

    teacher_oof = _align_probs_from_csv(args.teacher_oof_csv, train_ids)
    teacher_test = _align_probs_from_csv(args.teacher_test_csv, test_ids)
    np.save(art_dir / "teacher_oof.npy", teacher_oof)
    np.save(art_dir / "teacher_test.npy", teacher_test)

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")

    tab_train = build_tabular_frame(train_df, train_cache)
    tab_test = build_tabular_frame(test_df, test_cache)
    base_cols = get_feature_columns(tab_train)

    meta_train = _teacher_meta(teacher_oof)
    meta_test = _teacher_meta(teacher_test)
    rich_train = tab_train[base_cols].to_numpy(dtype=np.float32)
    rich_test = tab_test[base_cols].to_numpy(dtype=np.float32)
    rich_train, rich_test = _zscore(rich_train, rich_test)

    meta_cols = meta_train.columns.tolist()
    meta_train_np = meta_train.to_numpy(dtype=np.float32)
    meta_test_np = meta_test.to_numpy(dtype=np.float32)
    meta_train_np, meta_test_np = _zscore(meta_train_np, meta_test_np)

    teacher_prob_train_np = teacher_oof.astype(np.float32)
    teacher_prob_test_np = teacher_test.astype(np.float32)

    cv_df = pd.DataFrame(
        {
            "_cv_ts": train_df[str(args.time_col)],
            "_cv_group": train_df[str(args.group_col)].astype(np.int64),
        }
    )
    folds = make_forward_temporal_group_folds(cv_df, timestamp_col="_cv_ts", group_col="_cv_group", n_splits=int(args.n_splits))
    if int(args.start_fold) > 0:
        s = min(int(args.start_fold), len(folds))
        folds = folds[s:]
        print(f"[delta] starting from fold index {s} (skipped {s} earliest folds)", flush=True)
    if int(args.max_folds) > 0 and len(folds) > int(args.max_folds):
        folds = folds[: int(args.max_folds)]
        print(f"[delta] limiting folds to first {int(args.max_folds)}", flush=True)
    if len(folds) == 0:
        raise RuntimeError("no folds selected")

    teacher_logits_oof = _logit(teacher_oof)
    teacher_logits_test = _logit(teacher_test)
    y_soft = np.clip(y, float(args.target_eps), 1.0 - float(args.target_eps))
    if str(args.target_mode) == "logit_delta":
        y_logits = _logit(y_soft)
        delta_target = y_logits - teacher_logits_oof
    else:
        delta_target = y_soft - teacher_oof
    delta_target_raw = delta_target.copy()
    clip_frac = 0.0
    if float(args.delta_target_clip) > 0:
        clip_val = float(args.delta_target_clip)
        clip_frac = float(np.mean(np.abs(delta_target) > clip_val))
        delta_target = np.clip(delta_target, -clip_val, clip_val)

    delta_abs = np.abs(delta_target_raw)
    print(
        "[delta][target] "
        f"mean={float(delta_target_raw.mean()):.6f} "
        f"std={float(delta_target_raw.std()):.6f} "
        f"p95_abs={float(np.percentile(delta_abs, 95)): .6f} "
        f"teacher_prob_std={float(teacher_oof.std()):.6f} "
        f"target_clip={float(args.delta_target_clip):.4f} "
        f"clip_frac={clip_frac:.4f}",
        flush=True,
    )

    delta_oof = np.zeros_like(teacher_oof, dtype=np.float32)
    delta_test_accum = np.zeros_like(teacher_test, dtype=np.float32)
    fold_scores: list[float] = []

    teacher_max_prob = teacher_oof.max(axis=1).astype(np.float32)
    sample_weight = np.ones(len(train_df), dtype=np.float32)
    if str(args.uncertainty_weight_mode) == "binary_maxprob":
        sample_weight += float(args.uncertainty_k) * (teacher_max_prob < float(args.uncertainty_threshold)).astype(np.float32)
    elif str(args.uncertainty_weight_mode) == "linear_maxprob":
        sample_weight += float(args.uncertainty_k) * (1.0 - teacher_max_prob)
    sample_weight = np.clip(sample_weight, 1.0, 1e6).astype(np.float32)
    print(
        "[delta][weights] "
        f"mode={args.uncertainty_weight_mode} "
        f"thr={float(args.uncertainty_threshold):.3f} "
        f"k={float(args.uncertainty_k):.3f} "
        f"w_mean={float(sample_weight.mean()):.6f} "
        f"w_p95={float(np.percentile(sample_weight, 95)): .6f} "
        f"w_max={float(sample_weight.max()):.6f}",
        flush=True,
    )

    feat_cols = list(base_cols)
    if str(args.feature_mode) == "rich_plus_teacher_prob":
        feat_cols = feat_cols + ["teacher_prob_this_class"]
    elif str(args.feature_mode) == "rich_plus_teacher_meta":
        feat_cols = feat_cols + meta_cols

    def _make_features_for_class(class_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if str(args.feature_mode) == "rich_only":
            return rich_train, rich_test
        if str(args.feature_mode) == "rich_plus_teacher_prob":
            p_train = teacher_prob_train_np[:, [class_idx]]
            p_test = teacher_prob_test_np[:, [class_idx]]
            p_train, p_test = _zscore(p_train, p_test)
            x_tr = np.concatenate([rich_train, p_train], axis=1, dtype=np.float32)
            x_te = np.concatenate([rich_test, p_test], axis=1, dtype=np.float32)
            return x_tr, x_te
        x_tr = np.concatenate([rich_train, meta_train_np], axis=1, dtype=np.float32)
        x_te = np.concatenate([rich_test, meta_test_np], axis=1, dtype=np.float32)
        return x_tr, x_te

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        fold_delta_test = np.zeros_like(teacher_test, dtype=np.float32)
        for c, cls in enumerate(CLASSES):
            x_train_c, x_test_c = _make_features_for_class(c)
            dtrain = lgb.Dataset(
                x_train_c[tr_idx],
                label=delta_target[tr_idx, c],
                weight=sample_weight[tr_idx],
                free_raw_data=False,
            )
            dvalid = lgb.Dataset(
                x_train_c[va_idx],
                label=delta_target[va_idx, c],
                reference=dtrain,
                free_raw_data=False,
            )
            params = {
                "objective": "regression",
                "metric": "l2",
                "learning_rate": float(args.learning_rate),
                "num_leaves": int(args.num_leaves),
                "feature_fraction": float(args.feature_fraction),
                "bagging_fraction": float(args.bagging_fraction),
                "bagging_freq": int(args.bagging_freq),
                "min_data_in_leaf": int(args.min_data_in_leaf),
                "lambda_l2": float(args.lambda_l2),
                "max_depth": int(args.max_depth),
                "seed": int(args.seed) + 97 * fold_id + 13 * c,
                "verbosity": -1,
            }
            if int(args.num_threads) > 0:
                params["num_threads"] = int(args.num_threads)

            booster = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=int(args.num_boost_round),
                valid_sets=[dvalid],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=int(args.early_stopping_rounds), verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            delta_oof[va_idx, c] = booster.predict(
                x_train_c[va_idx], num_iteration=booster.best_iteration
            ).astype(np.float32)
            fold_delta_test[:, c] = booster.predict(x_test_c, num_iteration=booster.best_iteration).astype(np.float32)

        delta_test_accum += fold_delta_test / float(len(folds))
        if str(args.target_mode) == "logit_delta":
            pred_val = _sigmoid(
                teacher_logits_oof[va_idx] + np.clip(delta_oof[va_idx], -float(args.delta_clip), float(args.delta_clip))
            )
        else:
            pred_val = np.clip(
                teacher_oof[va_idx] + np.clip(delta_oof[va_idx], -float(args.delta_clip), float(args.delta_clip)),
                float(args.target_eps),
                1.0 - float(args.target_eps),
            )
        score = float(macro_map(y[va_idx], pred_val))
        fold_scores.append(score)
        print(f"[fold] {fold_id+1}/{len(folds)} macro_map={score:.6f} n_val={len(va_idx)}", flush=True)

    if str(args.target_mode) == "logit_delta":
        pred_oof = _sigmoid(teacher_logits_oof + np.clip(delta_oof, -float(args.delta_clip), float(args.delta_clip))).astype(np.float32)
        pred_test = _sigmoid(teacher_logits_test + np.clip(delta_test_accum, -float(args.delta_clip), float(args.delta_clip))).astype(
            np.float32
        )
    else:
        pred_oof = np.clip(
            teacher_oof + np.clip(delta_oof, -float(args.delta_clip), float(args.delta_clip)),
            float(args.target_eps),
            1.0 - float(args.target_eps),
        ).astype(np.float32)
        pred_test = np.clip(
            teacher_test + np.clip(delta_test_accum, -float(args.delta_clip), float(args.delta_clip)),
            float(args.target_eps),
            1.0 - float(args.target_eps),
        ).astype(np.float32)

    np.save(art_dir / "lgbm_teacher_delta_oof.npy", pred_oof)
    np.save(art_dir / "lgbm_teacher_delta_test.npy", pred_test)
    np.save(art_dir / "lgbm_teacher_delta_raw_oof.npy", delta_oof)
    np.save(art_dir / "lgbm_teacher_delta_raw_test.npy", delta_test_accum)

    sub = sample_sub.copy()
    if "track_id" not in sub.columns:
        raise ValueError("sample submission must contain 'track_id'")
    sub = sub[[c for c in sub.columns if c == "track_id" or c in CLASSES]].copy()
    sub["track_id"] = test_ids
    sub[CLASSES] = pred_test
    sub.to_csv(out_dir / "submission_lgbm_teacher_delta.csv", index=False)

    model_name = f"lgbm_teacher_delta_seed{int(args.seed)}"
    summary = {
        "project_root": str(PROJECT_ROOT.resolve()),
        "output_dir": str(out_dir),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "features_total": int(len(feat_cols)),
        "feature_columns": feat_cols,
        "feature_mode": str(args.feature_mode),
        "target_mode": str(args.target_mode),
        "delta_clip": float(args.delta_clip),
        "delta_target_clip": float(args.delta_target_clip),
        "target_eps": float(args.target_eps),
        "target_stats": {
            "delta_mean": float(delta_target_raw.mean()),
            "delta_std": float(delta_target_raw.std()),
            "delta_p95_abs": float(np.percentile(np.abs(delta_target_raw), 95)),
            "teacher_prob_std": float(teacher_oof.std()),
            "target_clip_fraction": float(clip_frac),
        },
        "uncertainty_weighting": {
            "mode": str(args.uncertainty_weight_mode),
            "threshold": float(args.uncertainty_threshold),
            "k": float(args.uncertainty_k),
            "weight_mean": float(sample_weight.mean()),
            "weight_p95": float(np.percentile(sample_weight, 95)),
            "weight_max": float(sample_weight.max()),
        },
        "fold_count": int(len(folds)),
        "models": {
            model_name: {
                "type": "tabular_delta",
                "oof_path": str((art_dir / "lgbm_teacher_delta_oof.npy").resolve()),
                "test_path": str((art_dir / "lgbm_teacher_delta_test.npy").resolve()),
                "macro_map": float(macro_map(y, pred_oof)),
                "per_class_ap": per_class_ap(y, pred_oof),
                "fold_scores": [float(x) for x in fold_scores],
                "worst_fold": float(np.min(fold_scores)) if fold_scores else 0.0,
            }
        },
        "track_ids_test_path": str((out_dir / "test_track_ids.npy").resolve()),
        "track_ids_train_path": str((out_dir / "train_track_ids.npy").resolve()),
    }
    dump_json(out_dir / "oof_summary.json", summary)

    print("=== LGBM TEACHER DELTA COMPLETE ===", flush=True)
    print(f"output_dir: {out_dir}", flush=True)
    print(f"oof_summary: {out_dir / 'oof_summary.json'}", flush=True)
    print(
        f"macro_oof={float(macro_map(y, pred_oof)):.6f} "
        f"fold_mean={float(np.mean(fold_scores)):.6f} fold_worst={float(np.min(fold_scores)):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
