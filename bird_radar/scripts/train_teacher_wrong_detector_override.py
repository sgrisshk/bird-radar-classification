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
from sklearn.metrics import average_precision_score

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
    p = argparse.ArgumentParser(description="Train teacher-wrong detector and build selective teacher->alt overrides.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--cache-dir", default="bird_radar/artifacts/cache")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--teacher-oof-csv", required=True)
    p.add_argument("--teacher-test-csv", required=True)
    p.add_argument("--alt-oof-npy", required=True)
    p.add_argument("--alt-track-ids-npy", required=True)
    p.add_argument("--alt-test-csv", default="")
    p.add_argument("--alt-test-npy", default="", help="Optional alternative to --alt-test-csv.")
    p.add_argument("--alt-test-ids-npy", default="", help="Required with --alt-test-npy when ids differ.")

    p.add_argument("--time-col", default="timestamp_start_radar_utc")
    p.add_argument("--group-col", default="observation_id")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--start-fold", type=int, default=1, help="Skip earliest folds for stability.")
    p.add_argument("--max-folds", type=int, default=0, help="0 means use all remaining folds.")

    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--feature-fraction", type=float, default=0.8)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--bagging-freq", type=int, default=1)
    p.add_argument("--min-data-in-leaf", type=int, default=150)
    p.add_argument("--lambda-l2", type=float, default=1.0)
    p.add_argument("--max-depth", type=int, default=-1)
    p.add_argument("--num-boost-round", type=int, default=2500)
    p.add_argument("--early-stopping-rounds", type=int, default=100)
    p.add_argument("--num-threads", type=int, default=0)

    p.add_argument("--t-pos", type=float, default=0.20)
    p.add_argument("--t-neg", type=float, default=0.80)
    p.add_argument("--q-thresholds", default="0.70,0.60,0.50")
    p.add_argument("--override-classes", default="", help="Comma-separated classes eligible for override. Empty=all.")
    p.add_argument("--override-ap-threshold", type=float, default=-1.0, help="If >=0, auto-select override classes with AP(alt)-AP(teacher) >= threshold.")
    p.add_argument("--min-override", type=float, default=0.05)
    p.add_argument("--max-override", type=float, default=0.15)
    p.add_argument("--max-worst-drop", type=float, default=0.005)
    return p.parse_args()


def _logit(x: np.ndarray) -> np.ndarray:
    p = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


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


def _align_alt_oof(oof_npy: str, ids_npy: str, target_ids: np.ndarray) -> np.ndarray:
    arr = np.load(oof_npy).astype(np.float32)
    ids = np.load(ids_npy).astype(np.int64)
    if len(arr) != len(ids):
        raise ValueError("alt oof and track ids length mismatch")
    if arr.shape[1] != len(CLASSES):
        raise ValueError("alt oof classes mismatch")
    pos = {int(t): i for i, t in enumerate(ids.tolist())}
    missing = [int(t) for t in target_ids if int(t) not in pos]
    if missing:
        raise ValueError(f"alt oof missing {len(missing)} ids; first={missing[:10]}")
    return np.stack([arr[pos[int(t)]] for t in target_ids], axis=0).astype(np.float32)


def _align_alt_test_from_npy(test_npy: str, test_ids_npy: str, target_ids: np.ndarray) -> np.ndarray:
    arr = np.load(test_npy).astype(np.float32)
    ids = np.load(test_ids_npy).astype(np.int64)
    if len(arr) != len(ids):
        raise ValueError("alt test npy and test ids length mismatch")
    if arr.shape[1] != len(CLASSES):
        raise ValueError("alt test npy classes mismatch")
    pos = {int(t): i for i, t in enumerate(ids.tolist())}
    missing = [int(t) for t in target_ids if int(t) not in pos]
    if missing:
        raise ValueError(f"alt test npy missing {len(missing)} ids; first={missing[:10]}")
    return np.stack([arr[pos[int(t)]] for t in target_ids], axis=0).astype(np.float32)


def _one_hot(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.zeros((len(y_idx), n_classes), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    return y


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (np.sum(ez, axis=1, keepdims=True) + EPS)


def _teacher_meta_features(teacher_probs: np.ndarray) -> dict[str, np.ndarray]:
    logits = _logit(teacher_probs)
    sort_probs = np.sort(teacher_probs, axis=1)
    top1 = sort_probs[:, -1]
    top2 = sort_probs[:, -2]
    margin = top1 - top2
    top1_class = np.argmax(teacher_probs, axis=1).astype(np.float32)
    soft = _softmax_rows(teacher_probs)
    entropy = -np.sum(soft * np.log(np.clip(soft, 1e-8, 1.0)), axis=1)
    out: dict[str, np.ndarray] = {
        "teacher_top1_prob": top1.astype(np.float32),
        "teacher_top2_prob": top2.astype(np.float32),
        "teacher_margin": margin.astype(np.float32),
        "teacher_entropy": entropy.astype(np.float32),
        "teacher_top1_class_id": top1_class,
    }
    for j, cls in enumerate(CLASSES):
        p = teacher_probs[:, j]
        l = logits[:, j]
        order = np.argsort(p, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(len(p), dtype=np.float32)
        if len(p) > 1:
            ranks /= float(len(p) - 1)
        out[f"teacher_p_{cls}"] = p.astype(np.float32)
        out[f"teacher_logit_{cls}"] = l.astype(np.float32)
        out[f"teacher_rank_{cls}"] = ranks.astype(np.float32)
    return out


def _build_detector_matrix(tab_df: pd.DataFrame, teacher_probs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    feat_cols = get_feature_columns(tab_df)
    x_tab = tab_df[feat_cols].to_numpy(dtype=np.float32)
    m = _teacher_meta_features(teacher_probs)
    meta_cols = sorted(list(m.keys()))
    x_meta = np.column_stack([m[c] for c in meta_cols]).astype(np.float32)
    x = np.concatenate([x_tab, x_meta], axis=1).astype(np.float32)
    cols = feat_cols + meta_cols
    return x, cols


def _safe_macro_map(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(macro_map(y_true, y_prob))


def _make_override_probs(
    teacher: np.ndarray,
    alt: np.ndarray,
    det: np.ndarray,
    q_thr: float,
    class_enabled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (det > float(q_thr)) & class_enabled.reshape(1, -1)
    out = np.where(mask, alt, teacher).astype(np.float32)
    return out, mask


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
    alt_oof = _align_alt_oof(args.alt_oof_npy, args.alt_track_ids_npy, train_ids)
    if str(args.alt_test_csv).strip():
        alt_test = _align_probs_from_csv(args.alt_test_csv, test_ids)
    else:
        if not str(args.alt_test_npy).strip():
            raise ValueError("provide either --alt-test-csv or --alt-test-npy")
        alt_test_ids_npy = str(args.alt_test_ids_npy).strip()
        if not alt_test_ids_npy:
            raise ValueError("--alt-test-ids-npy is required with --alt-test-npy")
        alt_test = _align_alt_test_from_npy(args.alt_test_npy, alt_test_ids_npy, test_ids)

    wrong = (
        ((y == 1.0) & (teacher_oof < float(args.t_pos)))
        | ((y == 0.0) & (teacher_oof > float(args.t_neg)))
    ).astype(np.float32)

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")
    tab_train = build_tabular_frame(train_df, train_cache)
    tab_test = build_tabular_frame(test_df, test_cache)

    x_train, detector_feature_cols = _build_detector_matrix(tab_train, teacher_oof)
    x_test, _ = _build_detector_matrix(tab_test, teacher_test)
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
    folds = make_forward_temporal_group_folds(
        cv_df,
        timestamp_col="_cv_ts",
        group_col="_cv_group",
        n_splits=int(args.n_splits),
    )
    if int(args.start_fold) > 0:
        start = min(int(args.start_fold), len(folds))
        folds = folds[start:]
        print(f"[detector] starting from fold index {start} (skipped {start} earliest folds)", flush=True)
    if int(args.max_folds) > 0 and len(folds) > int(args.max_folds):
        folds = folds[: int(args.max_folds)]
        print(f"[detector] limiting folds to first {int(args.max_folds)}", flush=True)
    if len(folds) == 0:
        raise RuntimeError("no folds selected for detector")

    det_oof = np.zeros((len(train_df), len(CLASSES)), dtype=np.float32)
    det_test_accum = np.zeros((len(test_df), len(CLASSES)), dtype=np.float32)
    fallback_events: list[dict[str, Any]] = []

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        fold_det_test = np.zeros((len(test_df), len(CLASSES)), dtype=np.float32)
        for c, cls in enumerate(CLASSES):
            y_tr = wrong[tr_idx, c]
            if np.unique(y_tr).size < 2:
                prior = float(np.mean(y_tr))
                det_oof[va_idx, c] = prior
                fold_det_test[:, c] = prior
                fallback_events.append(
                    {
                        "fold": int(fold_id),
                        "class": str(cls),
                        "reason": "single_class_train",
                        "prior": prior,
                    }
                )
                continue

            pos = float(y_tr.sum())
            neg = float(len(y_tr) - y_tr.sum())
            scale_pos_weight = float((neg + 1.0) / (pos + 1.0))

            dtrain = lgb.Dataset(x_train[tr_idx], label=y_tr, free_raw_data=False)
            dvalid = lgb.Dataset(x_train[va_idx], label=wrong[va_idx, c], reference=dtrain, free_raw_data=False)
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
                "scale_pos_weight": scale_pos_weight,
                "seed": int(args.seed) + 101 * fold_id + 17 * c,
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
            det_oof[va_idx, c] = booster.predict(x_train[va_idx], num_iteration=booster.best_iteration).astype(np.float32)
            fold_det_test[:, c] = booster.predict(x_test, num_iteration=booster.best_iteration).astype(np.float32)
        det_test_accum += fold_det_test / len(folds)

    np.save(art_dir / "teacher_wrong_detector_oof.npy", det_oof)
    np.save(art_dir / "teacher_wrong_detector_test.npy", det_test_accum)
    np.save(out_dir / "train_track_ids.npy", train_ids)
    np.save(out_dir / "test_track_ids.npy", test_ids)
    np.save(art_dir / "teacher_probs_oof.npy", teacher_oof)
    np.save(art_dir / "alt_probs_oof.npy", alt_oof)

    teacher_fold_scores: list[float] = []
    for _, va_idx in folds:
        teacher_fold_scores.append(_safe_macro_map(y[va_idx], teacher_oof[va_idx]))
    teacher_mean = float(np.mean(teacher_fold_scores))
    teacher_worst = float(np.min(teacher_fold_scores))

    q_grid = [float(x.strip()) for x in str(args.q_thresholds).split(",") if x.strip()]
    if not q_grid:
        raise ValueError("empty q-thresholds")

    teacher_cls_ap = {
        cls: float(average_precision_score(y[:, j], teacher_oof[:, j])) if float(y[:, j].sum()) > 0.0 else 0.0
        for j, cls in enumerate(CLASSES)
    }
    alt_cls_ap = {
        cls: float(average_precision_score(y[:, j], alt_oof[:, j])) if float(y[:, j].sum()) > 0.0 else 0.0
        for j, cls in enumerate(CLASSES)
    }
    ap_delta = {cls: float(alt_cls_ap[cls] - teacher_cls_ap[cls]) for cls in CLASSES}

    manual_set = {x.strip() for x in str(args.override_classes).split(",") if x.strip()}
    if float(args.override_ap_threshold) >= 0.0:
        auto_set = {cls for cls in CLASSES if ap_delta[cls] >= float(args.override_ap_threshold)}
    else:
        auto_set = set(CLASSES)

    if manual_set:
        unknown = sorted(list(manual_set - set(CLASSES)))
        if unknown:
            raise ValueError(f"unknown override classes: {unknown}")
        selected_override = sorted(list(manual_set.intersection(auto_set)))
    else:
        selected_override = sorted(list(auto_set))

    class_enabled = np.array([cls in set(selected_override) for cls in CLASSES], dtype=bool)
    if not np.any(class_enabled):
        raise ValueError("no classes selected for override")
    print(f"[override-classes] {selected_override}", flush=True)

    candidates: list[dict[str, Any]] = []
    for q in q_grid:
        blend_oof, mask_oof = _make_override_probs(teacher_oof, alt_oof, det_oof, q, class_enabled)
        override_rate = float(mask_oof.mean())
        per_class_override = {cls: float(mask_oof[:, j].mean()) for j, cls in enumerate(CLASSES)}
        fold_scores: list[float] = []
        for _, va_idx in folds:
            fold_scores.append(_safe_macro_map(y[va_idx], blend_oof[va_idx]))
        mean_score = float(np.mean(fold_scores))
        worst = float(np.min(fold_scores))
        gain = float(mean_score - teacher_mean)
        worst_delta = float(worst - teacher_worst)
        feasible = (
            (override_rate >= float(args.min_override))
            and (override_rate <= float(args.max_override))
            and (worst_delta >= -float(args.max_worst_drop))
        )

        blend_test, _ = _make_override_probs(teacher_test, alt_test, det_test_accum, q, class_enabled)
        sub = pd.DataFrame({"track_id": test_ids})
        for j, cls in enumerate(CLASSES):
            sub[cls] = np.clip(blend_test[:, j], 0.0, 1.0)
        sub_path = out_dir / f"submission_selective_override_q{int(round(q * 100)):02d}.csv"
        sub.to_csv(sub_path, index=False)

        candidates.append(
            {
                "q_threshold": float(q),
                "feasible": bool(feasible),
                "override_rate": override_rate,
                "per_class_override": per_class_override,
                "teacher_mean": teacher_mean,
                "teacher_worst": teacher_worst,
                "blend_mean": mean_score,
                "blend_worst": worst,
                "gain_vs_teacher": gain,
                "worst_delta": worst_delta,
                "fold_scores": [float(x) for x in fold_scores],
                "oof_macro": float(_safe_macro_map(y, blend_oof)),
                "oof_per_class_ap": per_class_ap(y, blend_oof),
                "submission_csv": str(sub_path.resolve()),
            }
        )
        print(
            f"[q={q:.2f}] override={override_rate:.4f} "
            f"mean={mean_score:.6f} gain={gain:+.6f} worst_delta={worst_delta:+.6f} feasible={feasible}",
            flush=True,
        )

    feasible_rows = [r for r in candidates if bool(r["feasible"])]
    if feasible_rows:
        best = max(feasible_rows, key=lambda r: float(r["gain_vs_teacher"]))
        selection_reason = "best_feasible_gain"
    else:
        best = max(candidates, key=lambda r: float(r["gain_vs_teacher"]))
        selection_reason = "best_gain_no_feasible"

    best_q = float(best["q_threshold"])
    best_test, _ = _make_override_probs(teacher_test, alt_test, det_test_accum, best_q, class_enabled)
    best_sub = pd.DataFrame({"track_id": test_ids})
    for j, cls in enumerate(CLASSES):
        best_sub[cls] = np.clip(best_test[:, j], 0.0, 1.0)
    best_sub_path = out_dir / "submission_selective_override_best.csv"
    best_sub.to_csv(best_sub_path, index=False)

    report = {
        "output_dir": str(out_dir),
        "settings": {
            "t_pos": float(args.t_pos),
            "t_neg": float(args.t_neg),
            "q_thresholds": q_grid,
            "min_override": float(args.min_override),
            "max_override": float(args.max_override),
            "max_worst_drop": float(args.max_worst_drop),
            "n_splits": int(args.n_splits),
            "start_fold": int(args.start_fold),
            "max_folds": int(args.max_folds),
        },
        "detector": {
            "n_features": int(x_train.shape[1]),
            "feature_columns": detector_feature_cols,
            "fallback_events": fallback_events,
            "oof_path": str((art_dir / "teacher_wrong_detector_oof.npy").resolve()),
            "test_path": str((art_dir / "teacher_wrong_detector_test.npy").resolve()),
        },
        "override_class_selection": {
            "override_classes": selected_override,
            "override_ap_threshold": float(args.override_ap_threshold),
            "manual_override_classes": sorted(list(manual_set)),
            "teacher_per_class_ap": teacher_cls_ap,
            "alt_per_class_ap": alt_cls_ap,
            "ap_delta_alt_minus_teacher": ap_delta,
        },
        "teacher_baseline": {
            "fold_scores": [float(x) for x in teacher_fold_scores],
            "mean": teacher_mean,
            "worst": teacher_worst,
            "oof_macro": float(_safe_macro_map(y, teacher_oof)),
        },
        "candidates": candidates,
        "selected": {
            "reason": selection_reason,
            "q_threshold": best_q,
            "submission_csv": str(best_sub_path.resolve()),
            "gain_vs_teacher": float(best["gain_vs_teacher"]),
            "worst_delta": float(best["worst_delta"]),
            "override_rate": float(best["override_rate"]),
        },
    }
    dump_json(out_dir / "teacher_wrong_detector_report.json", report)

    print("=== TEACHER WRONG DETECTOR COMPLETE ===", flush=True)
    print(f"output_dir: {out_dir}", flush=True)
    print(f"report: {out_dir / 'teacher_wrong_detector_report.json'}", flush=True)
    print(f"best_submission: {best_sub_path}", flush=True)


if __name__ == "__main__":
    main()
