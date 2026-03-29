#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds
from src.feature_engineering import build_feature_frame
from src.metrics import macro_map_score, per_class_average_precision
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CatBoost multiclass on trajectory tabular features (forward CV).")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--sample-submission", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default="bird_radar/artifacts/cache")

    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--time-col", type=str, default="timestamp_start_radar_utc")
    p.add_argument("--group-col", type=str, default="observation_id")
    p.add_argument("--id-col", type=str, default="track_id")

    p.add_argument("--iterations", type=int, default=4000)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--l2-leaf-reg", type=float, default=3.0)
    p.add_argument("--random-strength", type=float, default=1.0)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--rsm", type=float, default=0.8, help="Feature sampling per tree.")
    p.add_argument("--od-wait", type=int, default=200)
    p.add_argument("--thread-count", type=int, default=-1)

    p.add_argument("--sample-weights-npy", type=str, default="")
    p.add_argument("--teacher-oof-csv", type=str, default="")
    return p.parse_args()


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict[str, Any]]:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


def _align_teacher_probs(csv_path: str, ids: np.ndarray, id_col: str) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=[id_col, *CLASSES])
    mp = {str(int(r[id_col])): np.asarray([r[c] for c in CLASSES], dtype=np.float32) for _, r in df.iterrows()}
    miss = [str(int(t)) for t in ids.tolist() if str(int(t)) not in mp]
    if miss:
        raise ValueError(f"{csv_path} missing {len(miss)} ids; first={miss[:10]}")
    return np.stack([mp[str(int(t))] for t in ids.tolist()], axis=0).astype(np.float32)


def _safe_corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size == 0:
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    return c if np.isfinite(c) else 0.0


def _teacher_diag(y_true: np.ndarray, teacher: np.ndarray, model: np.ndarray) -> dict[str, float]:
    teacher = np.clip(teacher.astype(np.float32), 1e-6, 1.0 - 1e-6)
    model = np.clip(model.astype(np.float32), 1e-6, 1.0 - 1e-6)
    teacher_macro = float(macro_map_score(y_true, teacher))
    model_macro = float(macro_map_score(y_true, model))
    best = teacher_macro
    best_w = 1.0
    for w in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
        blend = np.clip(w * teacher + (1.0 - w) * model, 0.0, 1.0)
        m = float(macro_map_score(y_true, blend))
        if m > best:
            best = m
            best_w = float(w)
    return {
        "teacher_macro": teacher_macro,
        "model_macro": model_macro,
        "corr_with_teacher": float(_safe_corr_flat(teacher, model)),
        "best_blend_macro": best,
        "best_blend_w_teacher": best_w,
        "best_blend_gain": float(best - teacher_macro),
    }


def _save_per_class_delta_vs_teacher(
    y_true: np.ndarray,
    teacher_prob: np.ndarray,
    model_prob: np.ndarray,
    out_csv: Path,
) -> None:
    ap_teacher = per_class_average_precision(y_true, teacher_prob)
    ap_model = per_class_average_precision(y_true, model_prob)
    rows: list[dict[str, Any]] = []
    for c in CLASSES:
        at = float(ap_teacher.get(c, 0.0))
        am = float(ap_model.get(c, 0.0))
        rows.append(
            {
                "class": c,
                "ap_teacher": at,
                "ap_model": am,
                "delta_model_vs_teacher": float(am - at),
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir).resolve()
    art_dir = out_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    sample_sub = pd.read_csv(args.sample_submission)

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")

    feat_train = build_feature_frame(train_df, train_cache)
    feat_test = build_feature_frame(test_df, test_cache)

    id_col = str(args.id_col)
    time_col = str(args.time_col)
    group_col = str(args.group_col)

    feature_cols = [
        c
        for c in feat_train.columns
        if c
        not in {
            id_col,
            "observation_id",
            "primary_observation_id",
        }
    ]

    train_ids = train_df[id_col].to_numpy(dtype=np.int64)
    test_ids = test_df[id_col].to_numpy(dtype=np.int64)
    np.save(out_dir / "train_track_ids.npy", train_ids)
    np.save(out_dir / "test_track_ids.npy", test_ids)

    x_train = feat_train[feature_cols].to_numpy(dtype=np.float32)
    x_test = feat_test[feature_cols].to_numpy(dtype=np.float32)
    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y_onehot = np.zeros((len(y_idx), len(CLASSES)), dtype=np.float32)
    y_onehot[np.arange(len(y_idx)), y_idx] = 1.0

    sample_weights: np.ndarray | None = None
    if str(args.sample_weights_npy).strip():
        sw = np.asarray(np.load(str(args.sample_weights_npy)), dtype=np.float32).reshape(-1)
        if len(sw) != len(train_ids):
            raise ValueError(
                f"sample_weights length mismatch: got {len(sw)}, expected {len(train_ids)}"
            )
        sample_weights = sw

    cv_df = train_df[[time_col, group_col]].copy()
    cv_df.columns = ["_cv_ts", "_cv_group"]
    folds = make_forward_temporal_group_folds(
        cv_df,
        timestamp_col="_cv_ts",
        group_col="_cv_group",
        n_splits=int(args.n_splits),
    )
    if len(folds) == 0:
        raise RuntimeError("no folds created")

    oof = np.zeros((len(train_ids), len(CLASSES)), dtype=np.float32)
    covered_mask = np.zeros((len(train_ids),), dtype=bool)
    test_accum = np.zeros((len(test_ids), len(CLASSES)), dtype=np.float32)
    fold_scores: list[float] = []

    class_priors = np.clip(y_onehot.mean(axis=0), 1e-6, 1.0 - 1e-6).astype(np.float32)

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        print(f"[fold] {fold_id + 1}/{len(folds)} train={len(tr_idx)} val={len(va_idx)}", flush=True)
        x_tr = x_train[tr_idx]
        y_tr = y_idx[tr_idx]
        x_va = x_train[va_idx]
        sw_tr = sample_weights[tr_idx] if sample_weights is not None else None
        va_prob = np.zeros((len(va_idx), len(CLASSES)), dtype=np.float32)
        te_prob = np.zeros((len(test_ids), len(CLASSES)), dtype=np.float32)

        # OVR is robust when a temporal fold misses some classes.
        for ci, cname in enumerate(CLASSES):
            y_tr_bin = (y_tr == ci).astype(np.int32)
            y_va_bin = (y_idx[va_idx] == ci).astype(np.int32)
            if int(np.min(y_tr_bin)) == int(np.max(y_tr_bin)):
                prior = float(class_priors[ci])
                va_prob[:, ci] = prior
                te_prob[:, ci] = prior
                continue

            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                iterations=int(args.iterations),
                depth=int(args.depth),
                learning_rate=float(args.learning_rate),
                l2_leaf_reg=float(args.l2_leaf_reg),
                random_strength=float(args.random_strength),
                bootstrap_type="Bernoulli",
                subsample=float(args.subsample),
                rsm=float(args.rsm),
                auto_class_weights="Balanced",
                random_seed=int(args.seed + fold_id * 101 + ci),
                od_type="Iter",
                od_wait=int(args.od_wait),
                use_best_model=True,
                thread_count=int(args.thread_count),
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(
                x_tr,
                y_tr_bin,
                sample_weight=sw_tr,
                eval_set=(x_va, y_va_bin),
                verbose=False,
            )
            va_prob[:, ci] = model.predict_proba(x_va)[:, 1].astype(np.float32)
            te_prob[:, ci] = model.predict_proba(x_test)[:, 1].astype(np.float32)

        oof[va_idx] = np.clip(va_prob, 0.0, 1.0)
        covered_mask[va_idx] = True
        test_accum += np.clip(te_prob, 0.0, 1.0) / float(len(folds))
        fold_scores.append(float(macro_map_score(y_onehot[va_idx], oof[va_idx])))

    oof_path = art_dir / "catboost_ovr_oof.npy"
    test_path = art_dir / "catboost_ovr_test.npy"
    np.save(oof_path, oof.astype(np.float32))
    np.save(test_path, test_accum.astype(np.float32))

    covered_idx = np.where(covered_mask)[0].astype(np.int64)
    uncovered_idx = np.where(~covered_mask)[0].astype(np.int64)
    np.save(out_dir / "oof_covered_idx.npy", covered_idx)
    if len(covered_idx) == 0:
        raise RuntimeError("no covered validation rows")

    macro_raw_full = float(macro_map_score(y_onehot, oof))
    macro_covered = float(macro_map_score(y_onehot[covered_idx], oof[covered_idx]))
    per_class_covered = per_class_average_precision(y_onehot[covered_idx], oof[covered_idx])

    teacher_diag: dict[str, float] = {}
    oof_full_filled = oof.copy()
    macro_full_filled_teacher = 0.0
    if str(args.teacher_oof_csv).strip():
        teacher_probs = _align_teacher_probs(str(args.teacher_oof_csv), train_ids, id_col=id_col)
        oof_full_filled[uncovered_idx] = teacher_probs[uncovered_idx]
        macro_full_filled_teacher = float(macro_map_score(y_onehot, oof_full_filled))
        teacher_diag = _teacher_diag(
            y_true=y_onehot[covered_idx],
            teacher=teacher_probs[covered_idx],
            model=oof[covered_idx],
        )
        _save_per_class_delta_vs_teacher(
            y_true=y_onehot[covered_idx],
            teacher_prob=teacher_probs[covered_idx],
            model_prob=oof[covered_idx],
            out_csv=out_dir / "diagnostics" / "per_class_ap_delta_vs_teacher.csv",
        )

    np.save(out_dir / "oof_forward_cv_complete.npy", oof_full_filled.astype(np.float32))

    sub_ids = sample_sub[id_col].to_numpy(dtype=np.int64)
    if not np.array_equal(sub_ids, test_ids):
        pos = {int(tid): i for i, tid in enumerate(test_ids.tolist())}
        miss = [int(tid) for tid in sub_ids.tolist() if int(tid) not in pos]
        if miss:
            raise ValueError(f"sample submission ids missing in test predictions; first={miss[:10]}")
        take = np.asarray([pos[int(t)] for t in sub_ids.tolist()], dtype=np.int64)
        pred_sub = test_accum[take]
    else:
        pred_sub = test_accum

    sub = pd.DataFrame({id_col: sub_ids})
    for i, cls in enumerate(CLASSES):
        sub[cls] = np.clip(pred_sub[:, i], 0.0, 1.0).astype(np.float32)
    sub_path = out_dir / "submission_catboost_ovr.csv"
    sub.to_csv(sub_path, index=False)

    summary = {
        "output_dir": str(out_dir),
        "seed": int(args.seed),
        "n_splits": int(len(folds)),
        "n_train": int(len(train_ids)),
        "n_test": int(len(test_ids)),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "iterations": int(args.iterations),
        "depth": int(args.depth),
        "learning_rate": float(args.learning_rate),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_strength": float(args.random_strength),
        "subsample": float(args.subsample),
        "rsm": float(args.rsm),
        "od_wait": int(args.od_wait),
        "sample_weights_npy": str(args.sample_weights_npy),
        "macro_map_raw_full": float(macro_raw_full),
        "macro_map_covered": float(macro_covered),
        "macro_map_full_filled_teacher": float(macro_full_filled_teacher),
        "covered_ratio": float(len(covered_idx) / len(train_ids)),
        "n_covered": int(len(covered_idx)),
        "n_uncovered": int(len(uncovered_idx)),
        "per_class_ap_covered": {k: float(v) for k, v in per_class_covered.items()},
        "fold_scores": [float(v) for v in fold_scores],
        "fold_mean": float(np.mean(fold_scores) if fold_scores else 0.0),
        "fold_worst": float(np.min(fold_scores) if fold_scores else 0.0),
        "teacher_diag": teacher_diag,
        "models": {
            "catboost_ovr": {
                "type": "catboost_multiclass",
                "oof_path": str(oof_path.resolve()),
                "test_path": str(test_path.resolve()),
            }
        },
        "track_ids_train_path": str((out_dir / "train_track_ids.npy").resolve()),
        "track_ids_test_path": str((out_dir / "test_track_ids.npy").resolve()),
        "submission_path": str(sub_path.resolve()),
    }
    dump_json(out_dir / "oof_summary.json", summary)

    print("=== CATBOOST TRAJECTORY TRAIN COMPLETE ===", flush=True)
    print(f"output_dir={out_dir}", flush=True)
    print(f"oof_summary={out_dir / 'oof_summary.json'}", flush=True)
    print(
        f"macro_covered={summary['macro_map_covered']:.6f} "
        f"full_filled={summary['macro_map_full_filled_teacher']:.6f} "
        f"fold_mean={summary['fold_mean']:.6f} fold_worst={summary['fold_worst']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
