#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASSES
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache
from src.redesign.features import build_tabular_frame, get_feature_columns


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train adversarial train-vs-test classifier and export train sample weights.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default="bird_radar/artifacts/cache")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--feature-fraction", type=float, default=0.8)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--bagging-freq", type=int, default=1)
    p.add_argument("--min-data-in-leaf", type=int, default=100)
    p.add_argument("--lambda-l2", type=float, default=1.0)
    p.add_argument("--num-boost-round", type=int, default=1200)
    p.add_argument("--early-stopping-rounds", type=int, default=100)
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--weight-clip-min", type=float, default=0.25)
    p.add_argument("--weight-clip-max", type=float, default=4.0)
    p.add_argument("--psi-bins", type=int, default=10)
    p.add_argument("--teacher-oof-csv", default="")
    p.add_argument("--teacher-test-csv", default="")
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


def _teacher_meta_matrix(probs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    p = np.clip(probs.astype(np.float32), 1e-6, 1.0 - 1e-6)
    logits = np.log(p / (1.0 - p))
    sort_p = np.sort(p, axis=1)
    top1 = sort_p[:, -1]
    top2 = sort_p[:, -2]
    margin = top1 - top2
    entropy = -np.sum(p * np.log(p), axis=1)
    top1_cls = np.argmax(p, axis=1).astype(np.float32)
    cols = ["teacher_top1_prob", "teacher_top2_prob", "teacher_margin", "teacher_entropy", "teacher_top1_class_id"]
    mats: list[np.ndarray] = [top1, top2, margin, entropy, top1_cls]
    for j, cls in enumerate(CLASSES):
        cols.extend([f"teacher_p_{cls}", f"teacher_logit_{cls}"])
        mats.extend([p[:, j], logits[:, j]])
    return np.column_stack(mats).astype(np.float32), cols


def _feature_psi(train_col: np.ndarray, test_col: np.ndarray, n_bins: int, eps: float = 1e-6) -> float:
    a = np.asarray(train_col, dtype=np.float64)
    b = np.asarray(test_col, dtype=np.float64)
    all_v = np.concatenate([a, b], axis=0)
    if not np.isfinite(all_v).any():
        return 0.0
    q = np.linspace(0.0, 1.0, int(max(3, n_bins + 1)))
    bins = np.quantile(all_v, q)
    bins = np.unique(bins)
    if len(bins) < 3:
        return 0.0
    # Expand edges for np.histogram rightmost handling.
    bins[0] = bins[0] - 1e-9
    bins[-1] = bins[-1] + 1e-9
    ha, _ = np.histogram(a, bins=bins)
    hb, _ = np.histogram(b, bins=bins)
    pa = ha.astype(np.float64)
    pb = hb.astype(np.float64)
    pa = pa / max(1.0, pa.sum())
    pb = pb / max(1.0, pb.sum())
    pa = np.clip(pa, eps, None)
    pb = np.clip(pb, eps, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = out_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    train_ids = train_df["track_id"].to_numpy(dtype=np.int64)
    test_ids = test_df["track_id"].to_numpy(dtype=np.int64)

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")

    tab_train = build_tabular_frame(train_df, train_cache)
    tab_test = build_tabular_frame(test_df, test_cache)
    feature_cols = get_feature_columns(tab_train)

    x_train = tab_train[feature_cols].to_numpy(dtype=np.float32)
    x_test = tab_test[feature_cols].to_numpy(dtype=np.float32)
    model_feature_cols = list(feature_cols)
    teacher_meta_info: dict[str, Any] = {"enabled": False}

    if str(args.teacher_oof_csv).strip() or str(args.teacher_test_csv).strip():
        if not str(args.teacher_oof_csv).strip() or not str(args.teacher_test_csv).strip():
            raise ValueError("teacher meta requires both --teacher-oof-csv and --teacher-test-csv")
        teacher_train = _align_probs_from_csv(str(args.teacher_oof_csv), train_ids)
        teacher_test = _align_probs_from_csv(str(args.teacher_test_csv), test_ids)
        xm_train, meta_cols = _teacher_meta_matrix(teacher_train)
        xm_test, _ = _teacher_meta_matrix(teacher_test)
        x_train = np.concatenate([x_train, xm_train], axis=1).astype(np.float32)
        x_test = np.concatenate([x_test, xm_test], axis=1).astype(np.float32)
        model_feature_cols.extend(meta_cols)
        teacher_meta_info = {
            "enabled": True,
            "n_meta_features": int(len(meta_cols)),
            "source_oof": str(Path(args.teacher_oof_csv).resolve()),
            "source_test": str(Path(args.teacher_test_csv).resolve()),
        }

    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    x_train = (x_train - mu) / sd
    x_test = (x_test - mu) / sd

    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate(
        [
            np.zeros((len(x_train),), dtype=np.int32),
            np.ones((len(x_test),), dtype=np.int32),
        ],
        axis=0,
    )
    is_train = np.arange(len(x_all)) < len(x_train)

    oof = np.zeros((len(x_all),), dtype=np.float32)
    skf = StratifiedKFold(n_splits=int(args.n_splits), shuffle=True, random_state=int(args.seed))
    fold_auc: list[float] = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_all, y_all)):
        dtrain = lgb.Dataset(x_all[tr_idx], label=y_all[tr_idx], free_raw_data=False)
        dval = lgb.Dataset(x_all[va_idx], label=y_all[va_idx], reference=dtrain, free_raw_data=False)
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "feature_fraction": float(args.feature_fraction),
            "bagging_fraction": float(args.bagging_fraction),
            "bagging_freq": int(args.bagging_freq),
            "min_data_in_leaf": int(args.min_data_in_leaf),
            "lambda_l2": float(args.lambda_l2),
            "seed": int(args.seed) + 101 * fold,
            "verbosity": -1,
            "num_threads": int(args.num_threads),
        }
        booster = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=int(args.num_boost_round),
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(int(args.early_stopping_rounds), verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        pred = booster.predict(x_all[va_idx], num_iteration=booster.best_iteration)
        oof[va_idx] = pred.astype(np.float32)
        auc = float(roc_auc_score(y_all[va_idx], pred))
        fold_auc.append(auc)
        print(f"[adv] fold={fold} auc={auc:.6f} n_val={len(va_idx)}", flush=True)

    auc_all = float(roc_auc_score(y_all, oof))
    pred_train = oof[is_train].astype(np.float32)
    pred_test = oof[~is_train].astype(np.float32)

    eps = 1e-6
    odds = pred_train / np.clip(1.0 - pred_train, eps, None)
    odds = np.clip(odds, float(args.weight_clip_min), float(args.weight_clip_max))
    weights = odds / max(eps, float(np.mean(odds)))
    weights = np.clip(weights, float(args.weight_clip_min), float(args.weight_clip_max)).astype(np.float32)

    psi_rows: list[dict[str, float | str]] = []
    for j, name in enumerate(model_feature_cols):
        psi = _feature_psi(x_train[:, j], x_test[:, j], n_bins=int(args.psi_bins))
        psi_rows.append({"feature": name, "psi": float(psi)})
    psi_rows.sort(key=lambda x: float(x["psi"]), reverse=True)

    np.save(art_dir / "adversarial_weights_train.npy", weights)
    np.save(art_dir / "adversarial_pred_train.npy", pred_train)
    np.save(art_dir / "adversarial_pred_test.npy", pred_test)
    np.save(out_dir / "train_track_ids.npy", train_ids)
    np.save(out_dir / "test_track_ids.npy", test_ids)

    report = {
        "train_csv": str(Path(args.train_csv).resolve()),
        "test_csv": str(Path(args.test_csv).resolve()),
        "n_train": int(len(train_ids)),
        "n_test": int(len(test_ids)),
        "n_features": int(x_train.shape[1]),
        "teacher_meta": teacher_meta_info,
        "cv": {
            "n_splits": int(args.n_splits),
            "fold_auc": fold_auc,
            "auc_mean": float(np.mean(fold_auc)) if fold_auc else 0.0,
            "auc_oof_all": auc_all,
        },
        "weighting": {
            "formula": "w = clip(p/(1-p), [clip_min,clip_max]); normalize mean(w)=1",
            "clip_min": float(args.weight_clip_min),
            "clip_max": float(args.weight_clip_max),
            "summary": {
                "mean": float(np.mean(weights)),
                "std": float(np.std(weights)),
                "min": float(np.min(weights)),
                "max": float(np.max(weights)),
                "p50": float(np.quantile(weights, 0.50)),
                "p90": float(np.quantile(weights, 0.90)),
                "p95": float(np.quantile(weights, 0.95)),
                "p99": float(np.quantile(weights, 0.99)),
            },
        },
        "psi_top20": psi_rows[:20],
        "outputs": {
            "weights_train_npy": str((art_dir / "adversarial_weights_train.npy").resolve()),
            "pred_train_npy": str((art_dir / "adversarial_pred_train.npy").resolve()),
            "pred_test_npy": str((art_dir / "adversarial_pred_test.npy").resolve()),
            "train_track_ids_npy": str((out_dir / "train_track_ids.npy").resolve()),
            "test_track_ids_npy": str((out_dir / "test_track_ids.npy").resolve()),
        },
    }
    report_path = out_dir / "adversarial_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print("=== ADVERSARIAL WEIGHTS COMPLETE ===", flush=True)
    print(f"output_dir: {out_dir}", flush=True)
    print(f"auc_oof_all={auc_all:.6f} weights_mean={float(np.mean(weights)):.6f}", flush=True)
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
