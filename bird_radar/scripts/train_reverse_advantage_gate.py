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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache
from src.redesign.features import build_tabular_frame, get_feature_columns


EPS = 1e-6


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train reverse advantage model and build worst-aware gated blends.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--cache-dir", default="bird_radar/artifacts/cache")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--teacher-fr-oof-csv", required=True)
    p.add_argument("--teacher-fr-test-csv", required=True)
    p.add_argument("--forward-oof-csv", required=True)
    p.add_argument("--reverse-oof-csv", required=True)
    p.add_argument("--forward-test-csv", required=True)
    p.add_argument("--reverse-test-csv", required=True)

    p.add_argument("--time-col", default="timestamp_start_radar_utc")
    p.add_argument("--group-col", default="observation_id")
    p.add_argument("--n-splits", type=int, default=5)

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

    p.add_argument("--k-grid", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40")
    p.add_argument("--w-high-grid", default="0.55,0.60,0.65,0.70,0.75,0.80,0.85")
    p.add_argument("--w-low-grid", default="0.00,0.05,0.10,0.15")
    p.add_argument("--min-fold-delta", type=float, default=-0.002)
    p.add_argument("--target-fold-index", type=int, default=2)
    p.add_argument("--max-target-fold-drop", type=float, default=-0.002)
    p.add_argument("--min-override", type=float, default=0.05)
    p.add_argument("--max-override", type=float, default=0.50)
    return p.parse_args()


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict[str, Any]]:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


def _parse_grid(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    arr = np.array(vals, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("grid is empty")
    return arr


def _one_hot(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.zeros((len(y_idx), n_classes), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    return y


def _align_probs_from_csv(csv_path: str, ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=["track_id", *CLASSES]).set_index("track_id")
    return df.loc[ids, CLASSES].to_numpy(dtype=np.float32)


def _align_test_pair(forward_csv: str, reverse_csv: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = pd.read_csv(forward_csv, usecols=["track_id", *CLASSES])
    r = pd.read_csv(reverse_csv, usecols=["track_id", *CLASSES])
    m = f.merge(r, on="track_id", suffixes=("_f", "_r"), how="inner")
    if len(m) != len(f) or len(m) != len(r):
        raise ValueError("track_id mismatch between forward/reverse test csv")
    ids = m["track_id"].to_numpy(dtype=np.int64)
    pf = m[[f"{c}_f" for c in CLASSES]].to_numpy(dtype=np.float32)
    pr = m[[f"{c}_r" for c in CLASSES]].to_numpy(dtype=np.float32)
    return ids, pf, pr


def _macro_map(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    vals: list[float] = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        vals.append(0.0 if yt.sum() <= 0 else float(average_precision_score(yt, yp)))
    return float(np.mean(vals))


def _fold_eval(
    y: np.ndarray,
    baseline: np.ndarray,
    pred: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, float, dict[int, float], list[dict[str, float | int]]]:
    rows: list[dict[str, float | int]] = []
    for fid, (_, va_idx) in enumerate(folds):
        yt = y[va_idx]
        pb = baseline[va_idx]
        pr = pred[va_idx]
        covered = (pr.sum(axis=1) > 0.0)
        if int(covered.sum()) == 0:
            rows.append(
                {
                    "fold": int(fid),
                    "n_cov": 0,
                    "baseline_macro": 0.0,
                    "pred_macro": 0.0,
                    "fold_delta": 0.0,
                }
            )
            continue
        yt = yt[covered]
        pb = pb[covered]
        pr = pr[covered]
        b = _macro_map(yt, pb)
        p = _macro_map(yt, pr)
        rows.append(
            {
                "fold": int(fid),
                "n_cov": int(covered.sum()),
                "baseline_macro": float(b),
                "pred_macro": float(p),
                "fold_delta": float(p - b),
            }
        )
    valid = [r for r in rows if int(r["n_cov"]) > 0]
    if not valid:
        return 0.0, 0.0, 0.0, {}, rows
    b_mean = float(np.mean([float(r["baseline_macro"]) for r in valid]))
    p_mean = float(np.mean([float(r["pred_macro"]) for r in valid]))
    min_delta = float(np.min([float(r["fold_delta"]) for r in valid]))
    per_fold = {int(r["fold"]): float(r["fold_delta"]) for r in valid}
    return b_mean, p_mean, min_delta, per_fold, rows


def _bce_per_row(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return loss.mean(axis=1).astype(np.float32)


def _meta_features(pf: np.ndarray, pr: np.ndarray) -> tuple[np.ndarray, list[str]]:
    cols: list[str] = []
    mats: list[np.ndarray] = []

    diff = pr - pf
    abs_diff = np.abs(diff)
    maxf = np.max(pf, axis=1)
    maxr = np.max(pr, axis=1)
    sort_f = np.sort(pf, axis=1)
    sort_r = np.sort(pr, axis=1)
    margin_f = sort_f[:, -1] - sort_f[:, -2]
    margin_r = sort_r[:, -1] - sort_r[:, -2]
    ent_f = -np.sum(np.clip(pf, EPS, 1.0) * np.log(np.clip(pf, EPS, 1.0)), axis=1)
    ent_r = -np.sum(np.clip(pr, EPS, 1.0) * np.log(np.clip(pr, EPS, 1.0)), axis=1)

    summary = {
        "max_f": maxf,
        "max_r": maxr,
        "max_diff": maxr - maxf,
        "margin_f": margin_f,
        "margin_r": margin_r,
        "margin_diff": margin_r - margin_f,
        "ent_f": ent_f,
        "ent_r": ent_r,
        "ent_diff": ent_r - ent_f,
        "l1_mean": abs_diff.mean(axis=1),
        "l1_max": abs_diff.max(axis=1),
        "signed_mean": diff.mean(axis=1),
    }
    for k, v in summary.items():
        cols.append(k)
        mats.append(v.astype(np.float32))

    for j, c in enumerate(CLASSES):
        cols.extend([f"pf_{c}", f"pr_{c}", f"diff_{c}", f"absdiff_{c}"])
        mats.extend(
            [
                pf[:, j].astype(np.float32),
                pr[:, j].astype(np.float32),
                diff[:, j].astype(np.float32),
                abs_diff[:, j].astype(np.float32),
            ]
        )
    x = np.column_stack(mats).astype(np.float32)
    return x, cols


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    train_ids = train_df["track_id"].to_numpy(dtype=np.int64)

    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = _one_hot(y_idx, len(CLASSES))

    p_fr = _align_probs_from_csv(args.teacher_fr_oof_csv, train_ids)
    p_f = _align_probs_from_csv(args.forward_oof_csv, train_ids)
    p_r = _align_probs_from_csv(args.reverse_oof_csv, train_ids)

    loss_f = _bce_per_row(y, p_f)
    loss_r = _bce_per_row(y, p_r)
    adv_target = (loss_f - loss_r).astype(np.float32)  # >0 means reverse better
    adv_cls = (adv_target > 0.0).astype(np.int32)

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")
    tab_train = build_tabular_frame(train_df, train_cache)
    tab_test = build_tabular_frame(test_df, test_cache)
    tab_cols = get_feature_columns(tab_train)
    x_tab_train = tab_train[tab_cols].to_numpy(dtype=np.float32)
    x_tab_test = tab_test[tab_cols].to_numpy(dtype=np.float32)

    x_meta_train, meta_cols = _meta_features(p_f, p_r)
    test_ids, p_f_test, p_r_test = _align_test_pair(args.forward_test_csv, args.reverse_test_csv)
    p_fr_test = _align_probs_from_csv(args.teacher_fr_test_csv, test_ids)
    x_meta_test, _ = _meta_features(p_f_test, p_r_test)

    x_train = np.concatenate([x_tab_train, x_meta_train], axis=1).astype(np.float32)
    x_test = np.concatenate([x_tab_test, x_meta_test], axis=1).astype(np.float32)
    feat_cols = tab_cols + meta_cols

    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    x_train = (x_train - mu) / sd
    x_test = (x_test - mu) / sd

    cv_df = pd.DataFrame(
        {
            "_cv_ts": train_df[str(args.time_col)],
            "_cv_group": train_df[str(args.group_col)].astype(np.int64),
        }
    )
    folds = make_forward_temporal_group_folds(cv_df, timestamp_col="_cv_ts", group_col="_cv_group", n_splits=int(args.n_splits))

    adv_oof = np.zeros((len(train_df),), dtype=np.float32)
    adv_test_acc = np.zeros((len(test_df),), dtype=np.float64)
    fold_rmse: list[float] = []
    for fid, (tr_idx, va_idx) in enumerate(folds):
        tr_idx = np.asarray(tr_idx, dtype=np.int64)
        va_idx = np.asarray(va_idx, dtype=np.int64)
        dtr = lgb.Dataset(x_train[tr_idx], label=adv_target[tr_idx])
        dva = lgb.Dataset(x_train[va_idx], label=adv_target[va_idx], reference=dtr)
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
            "verbosity": -1,
            "seed": int(args.seed) + 701 * int(fid),
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
        p_va = booster.predict(x_train[va_idx], num_iteration=booster.best_iteration).astype(np.float32)
        p_te = booster.predict(x_test, num_iteration=booster.best_iteration).astype(np.float32)
        adv_oof[va_idx] = p_va
        adv_test_acc += p_te
        rmse = float(np.sqrt(np.mean((p_va - adv_target[va_idx]) ** 2)))
        fold_rmse.append(rmse)
    adv_test = (adv_test_acc / float(len(folds))).astype(np.float32)

    # Diagnostic AUC over sign(target)
    try:
        auc = float(roc_auc_score(adv_cls, adv_oof))
    except Exception:
        auc = float("nan")

    np.save(out_dir / "advantage_oof.npy", adv_oof)
    np.save(out_dir / "advantage_test.npy", adv_test)
    np.save(out_dir / "train_track_ids.npy", train_ids)
    np.save(out_dir / "test_track_ids.npy", test_ids)

    k_grid = _parse_grid(args.k_grid)
    w_hi_grid = _parse_grid(args.w_high_grid)
    w_lo_grid = _parse_grid(args.w_low_grid)
    target_fold = int(args.target_fold_index)

    rows: list[dict[str, float]] = []
    feasible: list[dict[str, float]] = []
    for k in k_grid:
        kf = float(k)
        if not (0.0 < kf < 1.0):
            continue
        thr = float(np.quantile(adv_oof, 1.0 - kf))
        gate = adv_oof >= thr  # top-k predicted reverse-advantage
        override_frac = float(gate.mean())
        for w_hi in w_hi_grid:
            for w_lo in w_lo_grid:
                if float(w_hi) < float(w_lo):
                    continue
                pred = np.clip((1.0 - float(w_lo)) * p_f + float(w_lo) * p_r, 0.0, 1.0)
                pred_gate = np.clip((1.0 - float(w_hi)) * p_f + float(w_hi) * p_r, 0.0, 1.0)
                pred[gate] = pred_gate[gate]
                b_mean, p_mean, min_delta, per_fold, _ = _fold_eval(y, p_fr, pred, folds)
                gain = p_mean - b_mean
                target_fold_delta = float(per_fold.get(target_fold, 0.0))
                row = {
                    "k_frac": kf,
                    "thr": thr,
                    "w_high": float(w_hi),
                    "w_low": float(w_lo),
                    "override_frac": override_frac,
                    "baseline_mean": b_mean,
                    "pred_mean": p_mean,
                    "gain": gain,
                    "min_fold_delta": min_delta,
                    "target_fold_delta": target_fold_delta,
                }
                rows.append(row)
                ok = (
                    (min_delta >= float(args.min_fold_delta))
                    and (target_fold_delta >= float(args.max_target_fold_drop))
                    and (override_frac >= float(args.min_override))
                    and (override_frac <= float(args.max_override))
                )
                if ok:
                    feasible.append(row)

    scan_df = pd.DataFrame(rows).sort_values(["gain", "min_fold_delta"], ascending=[False, False]).reset_index(drop=True)
    scan_df.to_csv(out_dir / "scan.csv", index=False)

    if feasible:
        best = sorted(feasible, key=lambda r: (r["gain"], r["min_fold_delta"]), reverse=True)[0]
        status = "GO"
    else:
        best = rows[np.argmax([r["gain"] for r in rows])]
        status = "NO_FEASIBLE"

    kf = float(best["k_frac"])
    thr = float(best["thr"])
    w_hi = float(best["w_high"])
    w_lo = float(best["w_low"])
    gate_oof = adv_oof >= thr

    pred_oof = np.clip((1.0 - w_lo) * p_f + w_lo * p_r, 0.0, 1.0)
    pred_oof_gate = np.clip((1.0 - w_hi) * p_f + w_hi * p_r, 0.0, 1.0)
    pred_oof[gate_oof] = pred_oof_gate[gate_oof]

    gate_test = adv_test >= thr
    pred_test = np.clip((1.0 - w_lo) * p_f_test + w_lo * p_r_test, 0.0, 1.0)
    pred_test_gate = np.clip((1.0 - w_hi) * p_f_test + w_hi * p_r_test, 0.0, 1.0)
    pred_test[gate_test] = pred_test_gate[gate_test]

    oof_df = pd.DataFrame({"track_id": train_ids})
    for j, c in enumerate(CLASSES):
        oof_df[c] = pred_oof[:, j]
    oof_path = out_dir / "oof_best.csv"
    oof_df.to_csv(oof_path, index=False)

    sub_df = pd.DataFrame({"track_id": test_ids})
    for j, c in enumerate(CLASSES):
        sub_df[c] = pred_test[:, j]
    sub_path = out_dir / "sub_best.csv"
    sub_df.to_csv(sub_path, index=False)

    b_mean, p_mean, min_delta, per_fold, fold_rows = _fold_eval(y, p_fr, pred_oof, folds)
    report = {
        "status": status,
        "advantage_auc_sign": auc,
        "adv_fold_rmse": fold_rmse,
        "adv_target_mean": float(adv_target.mean()),
        "adv_target_std": float(adv_target.std()),
        "n_scan": int(len(rows)),
        "n_feasible": int(len(feasible)),
        "best": best,
        "constraints": {
            "min_fold_delta": float(args.min_fold_delta),
            "target_fold_index": int(target_fold),
            "max_target_fold_drop": float(args.max_target_fold_drop),
            "min_override": float(args.min_override),
            "max_override": float(args.max_override),
        },
        "best_baseline_mean": float(b_mean),
        "best_pred_mean": float(p_mean),
        "best_gain": float(p_mean - b_mean),
        "best_min_fold_delta": float(min_delta),
        "best_target_fold_delta": float(per_fold.get(target_fold, 0.0)),
        "best_override_frac_oof": float(gate_oof.mean()),
        "best_override_frac_test": float(gate_test.mean()),
        "feature_count": int(x_train.shape[1]),
        "feature_columns": feat_cols,
        "output_oof": str(oof_path),
        "output_submission": str(sub_path),
        "scan_csv": str(out_dir / "scan.csv"),
        "fold_rows": fold_rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(
        f"status={status} adv_auc={auc:.6f} n_feasible={len(feasible)}/{len(rows)} "
        f"gain={float(p_mean - b_mean):+.6f} min_fold_delta={float(min_delta):+.6f} "
        f"target_fold_delta={float(per_fold.get(target_fold, 0.0)):+.6f} "
        f"override_oof={float(gate_oof.mean()):.4f}",
        flush=True,
    )
    print(str(sub_path), flush=True)
    print(str(out_dir / "report.json"), flush=True)


if __name__ == "__main__":
    main()
