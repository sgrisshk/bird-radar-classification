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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache
from src.redesign.features import build_tabular_frame, get_feature_columns


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train fold-regime detector (e.g. fold2-like) and hard-gate teacher_FR vs forward.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--cache-dir", default="bird_radar/artifacts/cache")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--teacher-fr-oof-csv", required=True)
    p.add_argument("--teacher-fr-test-csv", required=True)
    p.add_argument("--forward-oof-csv", required=True)
    p.add_argument("--forward-test-csv", required=True)

    p.add_argument("--time-col", default="timestamp_start_radar_utc")
    p.add_argument("--group-col", default="observation_id")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--target-fold-index", type=int, default=2)
    p.add_argument(
        "--detector-cv-mode",
        choices=["forward", "stratified"],
        default="stratified",
        help="CV for detector OOF. forward can be degenerate for is_foldK targets.",
    )

    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--feature-fraction", type=float, default=0.8)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--bagging-freq", type=int, default=1)
    p.add_argument("--min-data-in-leaf", type=int, default=120)
    p.add_argument("--lambda-l2", type=float, default=1.0)
    p.add_argument("--max-depth", type=int, default=-1)
    p.add_argument("--num-boost-round", type=int, default=3000)
    p.add_argument("--early-stopping-rounds", type=int, default=120)
    p.add_argument("--num-threads", type=int, default=0)

    p.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--gate-direction", choices=["high", "low"], default="high", help="high: override when p>=t, low: override when p<=t")
    p.add_argument("--min-fold-delta", type=float, default=-0.002)
    p.add_argument("--max-target-fold-drop", type=float, default=-0.002)
    p.add_argument("--min-override", type=float, default=0.10)
    p.add_argument("--max-override", type=float, default=0.30)
    return p.parse_args()


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict[str, Any]]:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


def _one_hot(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.zeros((len(y_idx), n_classes), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    return y


def _align_probs_from_csv(csv_path: str, ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=["track_id", *CLASSES])
    mp = {int(r["track_id"]): np.asarray([r[c] for c in CLASSES], dtype=np.float32) for _, r in df.iterrows()}
    missing = [int(t) for t in ids if int(t) not in mp]
    if missing:
        raise ValueError(f"{csv_path} missing {len(missing)} ids; first={missing[:10]}")
    return np.stack([mp[int(t)] for t in ids], axis=0).astype(np.float32)


def _teacher_meta_features(teacher_probs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    p = np.clip(teacher_probs, 1e-6, 1.0 - 1e-6).astype(np.float32)
    logits = np.log(p / (1.0 - p))
    sort_probs = np.sort(p, axis=1)
    top1 = sort_probs[:, -1]
    top2 = sort_probs[:, -2]
    margin = top1 - top2
    top1_class = np.argmax(p, axis=1).astype(np.float32)
    entropy = -np.sum(p * np.log(p), axis=1)
    cols = ["teacher_top1_prob", "teacher_top2_prob", "teacher_margin", "teacher_entropy", "teacher_top1_class_id"]
    arrs = [top1, top2, margin, entropy, top1_class]
    for j, c in enumerate(CLASSES):
        cols.append(f"teacher_p_{c}")
        arrs.append(p[:, j])
        cols.append(f"teacher_logit_{c}")
        arrs.append(logits[:, j])
    x = np.column_stack(arrs).astype(np.float32)
    return x, cols


def _macro_map(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    vals: list[float] = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        vals.append(0.0 if yt.sum() <= 0 else float(average_precision_score(yt, yp)))
    return float(np.mean(vals))


def _eval_blend(
    y: np.ndarray,
    teacher_ref: np.ndarray,
    pred: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, float, dict[int, float], list[dict[str, float | int]]]:
    fold_reports: list[dict[str, float | int]] = []
    for fid, (_, va_idx) in enumerate(folds):
        yt = y[va_idx]
        te = teacher_ref[va_idx]
        pr = pred[va_idx]
        covered = (pr.sum(axis=1) > 0.0)
        if int(covered.sum()) == 0:
            fold_reports.append(
                {
                    "fold": int(fid),
                    "n_cov": 0,
                    "teacher_macro": 0.0,
                    "pred_macro": 0.0,
                    "fold_delta": 0.0,
                }
            )
            continue
        yt_c = yt[covered]
        te_c = te[covered]
        pr_c = pr[covered]
        t = _macro_map(yt_c, te_c)
        p = _macro_map(yt_c, pr_c)
        fold_reports.append(
            {
                "fold": int(fid),
                "n_cov": int(covered.sum()),
                "teacher_macro": float(t),
                "pred_macro": float(p),
                "fold_delta": float(p - t),
            }
        )
    valid = [r for r in fold_reports if int(r["n_cov"]) > 0]
    teacher_mean = float(np.mean([float(r["teacher_macro"]) for r in valid])) if valid else 0.0
    pred_mean = float(np.mean([float(r["pred_macro"]) for r in valid])) if valid else 0.0
    min_delta = float(np.min([float(r["fold_delta"]) for r in valid])) if valid else 0.0
    per_fold_delta = {int(r["fold"]): float(r["fold_delta"]) for r in valid}
    return teacher_mean, pred_mean, min_delta, per_fold_delta, fold_reports


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    train_ids = train_df["track_id"].to_numpy(dtype=np.int64)
    test_ids = test_df["track_id"].to_numpy(dtype=np.int64)
    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = _one_hot(y_idx, len(CLASSES))

    teacher_fr_oof = _align_probs_from_csv(args.teacher_fr_oof_csv, train_ids)
    teacher_fr_test = _align_probs_from_csv(args.teacher_fr_test_csv, test_ids)
    forward_oof = _align_probs_from_csv(args.forward_oof_csv, train_ids)
    forward_test = _align_probs_from_csv(args.forward_test_csv, test_ids)

    cv_df = pd.DataFrame(
        {
            "_cv_ts": train_df[str(args.time_col)],
            "_cv_group": train_df[str(args.group_col)].astype(np.int64),
        }
    )
    folds = make_forward_temporal_group_folds(
        cv_df,
        timestamp_col="_cv_ts",
        group_col="_cv_group",
        n_splits=int(args.n_splits),
    )
    if len(folds) == 0:
        raise RuntimeError("no folds built")
    target_fold = int(args.target_fold_index)
    if target_fold < 0 or target_fold >= len(folds):
        raise ValueError(f"target-fold-index {target_fold} out of range [0,{len(folds)-1}]")

    fold_id = np.full((len(train_df),), -1, dtype=np.int64)
    for fid, (_, va_idx) in enumerate(folds):
        fold_id[np.asarray(va_idx, dtype=np.int64)] = int(fid)
    y_regime = (fold_id == target_fold).astype(np.float32)

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")
    tab_train = build_tabular_frame(train_df, train_cache)
    tab_test = build_tabular_frame(test_df, test_cache)
    base_cols = get_feature_columns(tab_train)
    x_train_base = tab_train[base_cols].to_numpy(dtype=np.float32)
    x_test_base = tab_test[base_cols].to_numpy(dtype=np.float32)
    x_train_meta, meta_cols = _teacher_meta_features(teacher_fr_oof)
    x_test_meta, _ = _teacher_meta_features(teacher_fr_test)
    x_train = np.concatenate([x_train_base, x_train_meta], axis=1).astype(np.float32)
    x_test = np.concatenate([x_test_base, x_test_meta], axis=1).astype(np.float32)
    feat_cols = base_cols + meta_cols

    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    x_train = (x_train - mu) / sd
    x_test = (x_test - mu) / sd

    det_oof = np.zeros((len(train_df),), dtype=np.float32)
    det_test_accum = np.zeros((len(test_df),), dtype=np.float64)
    fold_auc: list[float] = []
    fold_cov: list[dict[str, int | float]] = []

    if str(args.detector_cv_mode) == "forward":
        det_folds = [(np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64)) for tr, va in folds]
    else:
        skf = StratifiedKFold(n_splits=int(args.n_splits), shuffle=True, random_state=int(args.seed))
        det_folds = []
        for tr_idx, va_idx in skf.split(x_train, y_regime.astype(np.int64)):
            det_folds.append((np.asarray(tr_idx, dtype=np.int64), np.asarray(va_idx, dtype=np.int64)))

    for fid, (tr_idx, va_idx) in enumerate(det_folds):
        tr_idx = np.asarray(tr_idx, dtype=np.int64)
        va_idx = np.asarray(va_idx, dtype=np.int64)
        y_tr = y_regime[tr_idx]
        y_va = y_regime[va_idx]

        if float(y_tr.sum()) <= 0.0 or float((1.0 - y_tr).sum()) <= 0.0:
            det_oof[va_idx] = float(y_tr.mean()) if len(y_tr) > 0 else 0.0
            det_test_accum += float(y_tr.mean()) if len(y_tr) > 0 else 0.0
            fold_auc.append(float("nan"))
            fold_cov.append({"fold": int(fid), "n_tr": int(len(tr_idx)), "n_va": int(len(va_idx)), "pos_tr": int(y_tr.sum()), "fallback": 1})
            continue

        dtr = lgb.Dataset(x_train[tr_idx], label=y_tr)
        dva = lgb.Dataset(x_train[va_idx], label=y_va, reference=dtr)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "feature_fraction": float(args.feature_fraction),
            "bagging_fraction": float(args.bagging_fraction),
            "bagging_freq": int(args.bagging_freq),
            "min_data_in_leaf": int(args.min_data_in_leaf),
            "lambda_l2": float(args.lambda_l2),
            "max_depth": int(args.max_depth),
            "verbosity": -1,
            "seed": int(args.seed) + 1009 * int(fid),
        }
        if int(args.num_threads) > 0:
            params["num_threads"] = int(args.num_threads)

        booster = lgb.train(
            params=params,
            train_set=dtr,
            num_boost_round=int(args.num_boost_round),
            valid_sets=[dva],
            callbacks=[
                lgb.early_stopping(int(args.early_stopping_rounds), verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        p_va = booster.predict(x_train[va_idx], num_iteration=booster.best_iteration)
        p_te = booster.predict(x_test, num_iteration=booster.best_iteration)
        det_oof[va_idx] = p_va.astype(np.float32)
        det_test_accum += p_te
        try:
            auc = float(roc_auc_score(y_va, p_va))
        except Exception:
            auc = float("nan")
        fold_auc.append(auc)
        fold_cov.append(
            {
                "fold": int(fid),
                "n_tr": int(len(tr_idx)),
                "n_va": int(len(va_idx)),
                "pos_tr": int(y_tr.sum()),
                "pos_va": int(y_va.sum()),
                "fallback": 0,
            }
        )

    det_test = (det_test_accum / float(len(det_folds))).astype(np.float32)
    np.save(out_dir / "detector_oof.npy", det_oof)
    np.save(out_dir / "detector_test.npy", det_test)
    np.save(out_dir / "train_track_ids.npy", train_ids)
    np.save(out_dir / "test_track_ids.npy", test_ids)

    threshold_grid = [float(x.strip()) for x in str(args.thresholds).split(",") if x.strip()]
    if len(threshold_grid) == 0:
        raise ValueError("thresholds grid empty")

    scan_rows: list[dict[str, float]] = []
    feasible: list[dict[str, float]] = []
    for thr in threshold_grid:
        if str(args.gate_direction) == "low":
            mask = det_oof <= float(thr)
        else:
            mask = det_oof >= float(thr)
        pred = np.where(mask[:, None], forward_oof, teacher_fr_oof).astype(np.float32)
        teacher_mean, pred_mean, min_delta, per_fold_delta, _ = _eval_blend(y, teacher_fr_oof, pred, folds)
        gain = pred_mean - teacher_mean
        target_fold_delta = float(per_fold_delta.get(target_fold, 0.0))
        override_frac = float(mask.mean())
        row = {
            "threshold": float(thr),
            "override_frac": override_frac,
            "teacher_mean": teacher_mean,
            "pred_mean": pred_mean,
            "gain": gain,
            "min_fold_delta": min_delta,
            "target_fold_delta": target_fold_delta,
        }
        scan_rows.append(row)
        if (
            min_delta >= float(args.min_fold_delta)
            and target_fold_delta >= float(args.max_target_fold_drop)
            and override_frac >= float(args.min_override)
            and override_frac <= float(args.max_override)
        ):
            feasible.append(row)

    scan_df = pd.DataFrame(scan_rows).sort_values(["gain", "min_fold_delta"], ascending=[False, False]).reset_index(drop=True)
    scan_df.to_csv(out_dir / "threshold_scan.csv", index=False)

    if feasible:
        best = sorted(feasible, key=lambda r: (r["gain"], r["min_fold_delta"]), reverse=True)[0]
        status = "GO"
    else:
        best = scan_rows[np.argmax([r["gain"] for r in scan_rows])]
        status = "NO_FEASIBLE"

    best_thr = float(best["threshold"])
    if str(args.gate_direction) == "low":
        mask_test = det_test <= best_thr
    else:
        mask_test = det_test >= best_thr
    pred_test = np.where(mask_test[:, None], forward_test, teacher_fr_test).astype(np.float32)
    sub_df = pd.DataFrame({"track_id": test_ids})
    for j, c in enumerate(CLASSES):
        sub_df[c] = pred_test[:, j]
    sub_path = out_dir / f"sub_fold_regime_gate_t{best_thr:.2f}.csv"
    sub_df.to_csv(sub_path, index=False)

    # OOF of best config
    if str(args.gate_direction) == "low":
        mask_oof = det_oof <= best_thr
    else:
        mask_oof = det_oof >= best_thr
    pred_oof_best = np.where(mask_oof[:, None], forward_oof, teacher_fr_oof).astype(np.float32)
    oof_df = pd.DataFrame({"track_id": train_ids})
    for j, c in enumerate(CLASSES):
        oof_df[c] = pred_oof_best[:, j]
    oof_path = out_dir / f"oof_fold_regime_gate_t{best_thr:.2f}.csv"
    oof_df.to_csv(oof_path, index=False)

    # Detailed fold report for best.
    teacher_mean, pred_mean, min_delta, per_fold_delta, fold_reports = _eval_blend(y, teacher_fr_oof, pred_oof_best, folds)
    try:
        det_auc = float(roc_auc_score(y_regime, det_oof))
    except Exception:
        det_auc = float("nan")

    report = {
        "status": status,
        "target_fold_index": int(target_fold),
        "detector_auc_oof": det_auc,
        "detector_fold_auc": fold_auc,
        "detector_fold_coverage": fold_cov,
        "detector_cv_mode": str(args.detector_cv_mode),
        "best": best,
        "n_scan": int(len(scan_rows)),
        "n_feasible": int(len(feasible)),
        "constraints": {
            "min_fold_delta": float(args.min_fold_delta),
            "target_fold_delta": float(args.max_target_fold_drop),
            "min_override": float(args.min_override),
            "max_override": float(args.max_override),
        },
        "gate_direction": str(args.gate_direction),
        "best_fold_reports": fold_reports,
        "best_teacher_mean": float(teacher_mean),
        "best_pred_mean": float(pred_mean),
        "best_gain": float(pred_mean - teacher_mean),
        "best_min_fold_delta": float(min_delta),
        "best_target_fold_delta": float(per_fold_delta.get(target_fold, 0.0)),
        "best_override_frac_oof": float(mask_oof.mean()),
        "feature_count": int(x_train.shape[1]),
        "feature_columns": feat_cols,
        "output_submission": str(sub_path),
        "output_oof": str(oof_path),
    }
    (out_dir / "fold_regime_gate_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(
        f"status={status} det_auc={det_auc:.6f} n_feasible={len(feasible)}/{len(scan_rows)} "
        f"best_gain={float(pred_mean - teacher_mean):+.6f} best_min_fold_delta={float(min_delta):+.6f} "
        f"best_target_fold_delta={float(per_fold_delta.get(target_fold, 0.0)):+.6f} "
        f"override_frac={float(mask_oof.mean()):.4f}",
        flush=True,
    )
    print(str(sub_path), flush=True)
    print(str(out_dir / "fold_regime_gate_report.json"), flush=True)


if __name__ == "__main__":
    main()
