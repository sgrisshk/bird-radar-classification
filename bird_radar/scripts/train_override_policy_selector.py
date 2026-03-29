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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache
from src.redesign.features import build_tabular_frame, get_feature_columns

EPS = 1e-6


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train OOF policy selector for selective override.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--teacher-oof-csv", required=True)
    p.add_argument("--teacher-test-csv", required=True)
    p.add_argument("--spec-oof-npy", required=True)
    p.add_argument("--spec-track-ids-npy", required=True)
    p.add_argument("--spec-test-csv", required=True)
    p.add_argument("--spec2-oof-npy", default="")
    p.add_argument("--spec2-track-ids-npy", default="")
    p.add_argument("--spec2-test-csv", default="")
    p.add_argument("--bank-mode", choices=["single", "mean", "best_proxy_track", "best_proxy_class"], default="single")
    p.add_argument("--alpha-target", type=float, default=0.45, help="Override alpha used to build selector training target.")
    p.add_argument("--target-mode", choices=["regression", "binary"], default="regression")
    p.add_argument(
        "--target-weight-abs-gain-scale",
        type=float,
        default=0.0,
        help="Extra sample reweight factor: w *= (1 + scale * abs(target_gain)).",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default="bird_radar/artifacts/cache")
    p.add_argument("--use-tabular", action="store_true", help="Append rich tabular temporal features.")
    p.add_argument("--adversarial-pred-train-npy", default="")
    p.add_argument("--adversarial-pred-test-npy", default="")
    p.add_argument("--sample-weights-npy", default="", help="Optional sample weights for selector training/eval.")
    p.add_argument("--time-col", default="timestamp_start_radar_utc")
    p.add_argument("--group-col", default="observation_id")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--feature-fraction", type=float, default=0.8)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--bagging-freq", type=int, default=1)
    p.add_argument("--min-data-in-leaf", type=int, default=64)
    p.add_argument("--lambda-l2", type=float, default=2.0)
    p.add_argument("--num-boost-round", type=int, default=2000)
    p.add_argument("--early-stopping-rounds", type=int, default=100)
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--q-grid", default="0.02,0.03,0.05,0.06,0.07,0.08,0.10")
    return p.parse_args()


def _parse_grid(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("q-grid is empty")
    return arr


def _align_probs_from_csv(csv_path: str, ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=["track_id", *CLASSES]).set_index("track_id")
    return df.loc[ids, CLASSES].to_numpy(dtype=np.float32)


def _align_specialist_oof(npy_path: str, ids_path: str, target_ids: np.ndarray) -> np.ndarray:
    arr = np.load(npy_path).astype(np.float32)
    ids = np.load(ids_path).astype(np.int64)
    pos = {int(t): i for i, t in enumerate(ids.tolist())}
    missing = [int(t) for t in target_ids if int(t) not in pos]
    if missing:
        raise ValueError(f"specialist oof missing {len(missing)} ids; first={missing[:10]}")
    return np.stack([arr[pos[int(t)]] for t in target_ids], axis=0).astype(np.float32)


def _safe_logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, EPS, 1.0 - EPS)
    return np.log(pc / (1.0 - pc)).astype(np.float32)


def _one_hot(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.zeros((len(y_idx), n_classes), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    return y


def _row_bce(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return (-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))).mean(axis=1).astype(np.float32)


def _make_bank_probs(
    teacher: np.ndarray,
    spec1: np.ndarray,
    spec2: np.ndarray | None,
    mode: str,
) -> np.ndarray:
    if spec2 is None or str(mode) == "single":
        return spec1.astype(np.float32)
    if str(mode) == "mean":
        return np.clip(0.5 * spec1 + 0.5 * spec2, 0.0, 1.0).astype(np.float32)
    p_t = np.clip(teacher, EPS, 1.0 - EPS)
    entropy = (-np.sum(p_t * np.log(p_t), axis=1, keepdims=True)).astype(np.float32)
    if str(mode) == "best_proxy_track":
        s1 = np.sum(np.abs(spec1 - teacher) * entropy, axis=1)
        s2 = np.sum(np.abs(spec2 - teacher) * entropy, axis=1)
        use1 = s1 >= s2
        out = spec2.copy()
        out[use1] = spec1[use1]
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    if str(mode) == "best_proxy_class":
        s1 = np.abs(spec1 - teacher) * entropy
        s2 = np.abs(spec2 - teacher) * entropy
        use1 = s1 >= s2
        out = np.where(use1, spec1, spec2)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"unknown bank mode: {mode}")


def _topk_mask(score: np.ndarray, q_frac: float) -> np.ndarray:
    n = int(score.shape[0])
    k = int(np.round(float(q_frac) * float(n)))
    k = max(1, min(n, k))
    order = np.argsort(score, kind="mergesort")
    sel = order[-k:]
    m = np.zeros((n,), dtype=bool)
    m[sel] = True
    return m


def _macro_map(y_true: np.ndarray, y_prob: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    vals = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        vals.append(0.0 if yt.sum() <= 0 else float(average_precision_score(yt, yp, sample_weight=sample_weight)))
    return float(np.mean(vals))


def _make_meta_features(
    teacher: np.ndarray,
    bank: np.ndarray,
    spec1: np.ndarray,
    spec2: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    pt = np.clip(teacher, EPS, 1.0 - EPS)
    pb = np.clip(bank, EPS, 1.0 - EPS)
    st = np.sort(pt, axis=1)
    sb = np.sort(pb, axis=1)
    t_top1, t_top2 = st[:, -1], st[:, -2]
    b_top1, b_top2 = sb[:, -1], sb[:, -2]
    t_margin = t_top1 - t_top2
    b_margin = b_top1 - b_top2
    t_entropy = -np.sum(pt * np.log(pt), axis=1)
    b_entropy = -np.sum(pb * np.log(pb), axis=1)
    d = bank - teacher
    ad = np.abs(d)
    lt = _safe_logit(teacher)
    lb = _safe_logit(bank)
    ld = np.abs(lb - lt)
    feats = [
        t_top1,
        t_top2,
        t_margin,
        t_entropy,
        b_top1,
        b_top2,
        b_margin,
        b_entropy,
        np.mean(ad, axis=1),
        np.max(ad, axis=1),
        np.quantile(ad, 0.90, axis=1),
        np.mean(ld, axis=1),
        np.max(ld, axis=1),
        np.quantile(ld, 0.90, axis=1),
    ]
    cols = [
        "teacher_top1",
        "teacher_top2",
        "teacher_margin",
        "teacher_entropy",
        "bank_top1",
        "bank_top2",
        "bank_margin",
        "bank_entropy",
        "abs_delta_mean",
        "abs_delta_max",
        "abs_delta_p90",
        "abs_logit_delta_mean",
        "abs_logit_delta_max",
        "abs_logit_delta_p90",
    ]
    for j, c in enumerate(CLASSES):
        feats.extend([pt[:, j], pb[:, j], d[:, j], ad[:, j], ld[:, j]])
        cols.extend([f"teacher_p_{c}", f"bank_p_{c}", f"delta_{c}", f"abs_delta_{c}", f"abs_logit_delta_{c}"])
    if spec2 is not None:
        s12 = np.abs(spec1 - spec2)
        feats.extend([np.mean(s12, axis=1), np.max(s12, axis=1), np.quantile(s12, 0.9, axis=1)])
        cols.extend(["abs_spec12_mean", "abs_spec12_max", "abs_spec12_p90"])
    x = np.column_stack(feats).astype(np.float32)
    return x, cols


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict[str, Any]]:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


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
    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = _one_hot(y_idx, len(CLASSES))

    teacher_oof = _align_probs_from_csv(args.teacher_oof_csv, train_ids)
    teacher_test = _align_probs_from_csv(args.teacher_test_csv, test_ids)
    spec1_oof = _align_specialist_oof(args.spec_oof_npy, args.spec_track_ids_npy, train_ids)
    spec1_test = _align_probs_from_csv(args.spec_test_csv, test_ids)
    if str(args.spec2_oof_npy).strip():
        if not str(args.spec2_track_ids_npy).strip() or not str(args.spec2_test_csv).strip():
            raise ValueError("spec2 requires --spec2-track-ids-npy and --spec2-test-csv")
        spec2_oof = _align_specialist_oof(args.spec2_oof_npy, args.spec2_track_ids_npy, train_ids)
        spec2_test = _align_probs_from_csv(args.spec2_test_csv, test_ids)
    else:
        spec2_oof = None
        spec2_test = None

    bank_oof = _make_bank_probs(teacher_oof, spec1_oof, spec2_oof, str(args.bank_mode))
    bank_test = _make_bank_probs(teacher_test, spec1_test, spec2_test, str(args.bank_mode))

    alpha = float(args.alpha_target)
    override_oof = np.clip((1.0 - alpha) * teacher_oof + alpha * bank_oof, 0.0, 1.0).astype(np.float32)
    override_test = np.clip((1.0 - alpha) * teacher_test + alpha * bank_test, 0.0, 1.0).astype(np.float32)

    teacher_row_loss = _row_bce(y, teacher_oof)
    override_row_loss = _row_bce(y, override_oof)
    target_gain = (teacher_row_loss - override_row_loss).astype(np.float32)
    target_binary = (target_gain > 0.0).astype(np.int32)

    x_meta_train, meta_cols = _make_meta_features(teacher_oof, bank_oof, spec1_oof, spec2_oof)
    x_meta_test, _ = _make_meta_features(teacher_test, bank_test, spec1_test, spec2_test)
    x_train_parts = [x_meta_train]
    x_test_parts = [x_meta_test]
    feature_cols = list(meta_cols)

    if str(args.adversarial_pred_train_npy).strip() and str(args.adversarial_pred_test_npy).strip():
        adv_train = np.asarray(np.load(args.adversarial_pred_train_npy), dtype=np.float32).reshape(-1, 1)
        adv_test = np.asarray(np.load(args.adversarial_pred_test_npy), dtype=np.float32).reshape(-1, 1)
        if len(adv_train) != len(train_df) or len(adv_test) != len(test_df):
            raise ValueError("adversarial pred length mismatch")
        x_train_parts.append(adv_train)
        x_test_parts.append(adv_test)
        feature_cols.append("adversarial_pred_testlike")

    if bool(args.use_tabular):
        cache_dir = Path(args.cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
        test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")
        tab_train = build_tabular_frame(train_df, train_cache)
        tab_test = build_tabular_frame(test_df, test_cache)
        tab_cols = get_feature_columns(tab_train)
        x_train_parts.append(tab_train[tab_cols].to_numpy(dtype=np.float32))
        x_test_parts.append(tab_test[tab_cols].to_numpy(dtype=np.float32))
        feature_cols.extend([f"tab_{c}" for c in tab_cols])

    x_train = np.concatenate(x_train_parts, axis=1).astype(np.float32)
    x_test = np.concatenate(x_test_parts, axis=1).astype(np.float32)

    sample_weights: np.ndarray | None = None
    if str(args.sample_weights_npy).strip():
        sample_weights = np.asarray(np.load(args.sample_weights_npy), dtype=np.float32).reshape(-1)
        if len(sample_weights) != len(train_df):
            raise ValueError("sample-weights-npy length mismatch")
        sample_weights = np.clip(sample_weights, 1e-8, None).astype(np.float32)

    cv_df = pd.DataFrame(
        {
            "_cv_ts": train_df[str(args.time_col)],
            "_cv_group": train_df[str(args.group_col)].astype(np.int64),
        }
    )
    folds = make_forward_temporal_group_folds(cv_df, timestamp_col="_cv_ts", group_col="_cv_group", n_splits=int(args.n_splits))

    train_weights = sample_weights.copy() if sample_weights is not None else np.ones((len(train_df),), dtype=np.float32)
    if float(args.target_weight_abs_gain_scale) > 0.0:
        train_weights = train_weights * (1.0 + float(args.target_weight_abs_gain_scale) * np.abs(target_gain))

    oof_score = np.zeros((len(train_df),), dtype=np.float32)
    test_score = np.zeros((len(test_df),), dtype=np.float32)
    used_folds = 0
    fold_rows: list[dict[str, float | int]] = []

    for fid, (tr_idx, va_idx) in enumerate(folds):
        tr = np.asarray(tr_idx, dtype=np.int64)
        va = np.asarray(va_idx, dtype=np.int64)
        if len(tr) == 0 or len(va) == 0:
            continue
        fold_weight_tr = train_weights[tr]
        fold_weight_va = train_weights[va]
        label_tr = target_gain[tr] if str(args.target_mode) == "regression" else target_binary[tr]
        label_va = target_gain[va] if str(args.target_mode) == "regression" else target_binary[va]

        dtrain = lgb.Dataset(
            x_train[tr],
            label=label_tr,
            weight=fold_weight_tr,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            x_train[va],
            label=label_va,
            weight=fold_weight_va,
            reference=dtrain,
            free_raw_data=False,
        )
        if str(args.target_mode) == "binary":
            objective = "binary"
            metric = "auc"
        else:
            objective = "regression_l2"
            metric = "l2"
        params = {
            "objective": objective,
            "metric": metric,
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "feature_fraction": float(args.feature_fraction),
            "bagging_fraction": float(args.bagging_fraction),
            "bagging_freq": int(args.bagging_freq),
            "min_data_in_leaf": int(args.min_data_in_leaf),
            "lambda_l2": float(args.lambda_l2),
            "seed": int(args.seed) + 101 * fid,
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
        p_va = booster.predict(x_train[va], num_iteration=booster.best_iteration).astype(np.float32)
        p_te = booster.predict(x_test, num_iteration=booster.best_iteration).astype(np.float32)
        oof_score[va] = p_va
        test_score += p_te
        used_folds += 1

        corr = float(np.corrcoef(target_gain[va], p_va)[0, 1]) if len(va) > 3 else 0.0
        fold_rows.append(
            {
                "fold": int(fid),
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "target_gain_mean_val": float(np.mean(target_gain[va])),
                "pred_score_mean_val": float(np.mean(p_va)),
                "corr_val": corr,
            }
        )
        print(f"[policy] fold={fid} corr={corr:.6f} n_val={len(va)}", flush=True)

    if used_folds <= 0:
        raise RuntimeError("no valid folds used for policy selector")
    test_score /= float(used_folds)

    # OOF diagnostics for top-q override using policy score.
    q_grid = _parse_grid(args.q_grid)
    eval_rows: list[dict[str, float]] = []
    for q in q_grid.tolist():
        m = _topk_mask(oof_score, float(q))
        pred = teacher_oof.copy()
        pred[m] = override_oof[m]
        sw = sample_weights if sample_weights is not None else None
        base = _macro_map(y, teacher_oof, sample_weight=sw)
        now = _macro_map(y, pred, sample_weight=sw)
        eval_rows.append(
            {
                "q_frac": float(q),
                "override_frac_oof": float(m.mean()),
                "weighted_teacher_mean": float(base),
                "weighted_pred_mean": float(now),
                "weighted_gain": float(now - base),
            }
        )

    best_eval = sorted(eval_rows, key=lambda r: r["weighted_gain"], reverse=True)[0]
    corr_all = float(np.corrcoef(target_gain, oof_score)[0, 1])
    auc_binary = float(roc_auc_score(target_binary, oof_score)) if np.unique(target_binary).size > 1 else 0.5

    np.save(art_dir / "policy_selector_score_oof.npy", oof_score.astype(np.float32))
    np.save(art_dir / "policy_selector_score_test.npy", test_score.astype(np.float32))
    np.save(out_dir / "train_track_ids.npy", train_ids)
    np.save(out_dir / "test_track_ids.npy", test_ids)

    report = {
        "status": "OK",
        "bank_mode": str(args.bank_mode),
        "alpha_target": float(alpha),
        "target_mode": str(args.target_mode),
        "target_weight_abs_gain_scale": float(args.target_weight_abs_gain_scale),
        "used_folds": int(used_folds),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": int(x_train.shape[1]),
        "feature_cols_count": int(len(feature_cols)),
        "sample_weighted_training": bool(sample_weights is not None),
        "metrics": {
            "corr_target_vs_oof_score": float(corr_all),
            "auc_binary_gain_positive": float(auc_binary),
            "target_gain_mean": float(np.mean(target_gain)),
            "target_gain_p90": float(np.quantile(target_gain, 0.90)),
            "target_gain_p10": float(np.quantile(target_gain, 0.10)),
        },
        "fold_rows": fold_rows,
        "q_eval_rows": eval_rows,
        "best_q_eval": best_eval,
        "outputs": {
            "selector_oof_npy": str((art_dir / "policy_selector_score_oof.npy").resolve()),
            "selector_test_npy": str((art_dir / "policy_selector_score_test.npy").resolve()),
            "train_track_ids_npy": str((out_dir / "train_track_ids.npy").resolve()),
            "test_track_ids_npy": str((out_dir / "test_track_ids.npy").resolve()),
        },
    }
    (out_dir / "policy_selector_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print("=== POLICY SELECTOR COMPLETE ===", flush=True)
    print(f"output_dir: {out_dir}", flush=True)
    print(f"corr={corr_all:.6f} auc={auc_binary:.6f} best_q_gain={best_eval['weighted_gain']:+.6f}", flush=True)
    print(str((out_dir / "policy_selector_report.json").resolve()), flush=True)


if __name__ == "__main__":
    main()
