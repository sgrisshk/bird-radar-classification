#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASSES, CLASS_TO_INDEX
from src.cv import make_forward_temporal_group_folds
from src.feature_engineering import build_feature_frame
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache
import train_temporal_lgbm as ttl


SIZE_POSSIBLE_CLASSES = {
    "Small bird": ["Gulls", "Birds of Prey", "Songbirds", "Waders", "Pigeons", "Cormorants"],
    "Medium bird": ["Gulls", "Waders", "Ducks", "Clutter", "Geese", "Cormorants"],
    "Large bird": ["Gulls", "Clutter", "Cormorants", "Geese", "Ducks", "Pigeons"],
    "Flock": ["Gulls", "Songbirds", "Geese", "Pigeons", "Waders", "Ducks"],
}
SIZE_SEED_OFFSET = {name: i for i, name in enumerate(SIZE_POSSIBLE_CLASSES.keys())}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Size-stratified LightGBM OvR with forward CV routing.")
    p.add_argument("--data-dir", type=str, default="/Users/sgrisshk/Desktop/AI-task")
    p.add_argument("--cache-dir", type=str, default="/Users/sgrisshk/Desktop/AI-task/bird_radar/artifacts/cache")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-estimators", type=int, default=4000)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--extra-blacklist", type=str, default="")
    return p.parse_args()


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


def _one_hot_from_labels(labels: np.ndarray) -> np.ndarray:
    y = np.zeros((len(labels), len(CLASSES)), dtype=np.int32)
    for i, lbl in enumerate(labels):
        y[i, CLASS_TO_INDEX[lbl]] = 1
    return y


def _normalize_rows(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, None)
    s = p.sum(axis=1, keepdims=True)
    zero_rows = s[:, 0] <= 1e-12
    if np.any(zero_rows):
        p = p.copy()
        p[zero_rows] = 1.0 / len(CLASSES)
        s = p.sum(axis=1, keepdims=True)
    return (p / s).astype(np.float32)


def _macro_ap(labels: np.ndarray, probs: np.ndarray, mask: np.ndarray) -> float:
    y = labels[mask]
    p = probs[mask]
    vals: list[float] = []
    for i, cls in enumerate(CLASSES):
        yt = (y == cls).astype(np.int32)
        if yt.sum() > 0:
            vals.append(float(average_precision_score(yt, p[:, i])))
    return float(np.mean(vals)) if vals else 0.0


def _apply_possible_mask(
    probs: np.ndarray,
    possible_classes: list[str],
    fallback_prior: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(probs, dtype=np.float32)
    idx = np.array([CLASS_TO_INDEX[c] for c in possible_classes], dtype=np.int64)
    out[:, idx] = probs[:, idx]
    row_sum = out.sum(axis=1, keepdims=True)
    zero_rows = (row_sum[:, 0] <= 1e-12)
    if np.any(zero_rows):
        out[zero_rows] = fallback_prior
    return _normalize_rows(out)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = out_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    train_df["ts"] = pd.to_datetime(train_df["timestamp_start_radar_utc"], errors="coerce", utc=True)

    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")

    feat_train = build_feature_frame(train_df, track_cache=train_cache)
    feat_test = build_feature_frame(test_df, track_cache=test_cache)
    feat_train = feat_train.merge(train_df[["track_id", "bird_group", "radar_bird_size", "ts"]], on="track_id", how="left")
    feat_test = feat_test.merge(test_df[["track_id", "radar_bird_size"]], on="track_id", how="left")
    if "observation_id" in feat_train.columns:
        feat_train["_cv_group"] = feat_train["observation_id"].astype(np.int64)
    else:
        feat_train["_cv_group"] = feat_train["track_id"].astype(np.int64)
    feat_train["_cv_ts"] = pd.to_datetime(feat_train["ts"], errors="coerce", utc=True)

    feature_cols = [
        c for c in feat_train.columns
        if c not in {"track_id", "observation_id", "primary_observation_id", "bird_group", "radar_bird_size", "ts", "_cv_group", "_cv_ts"}
    ]

    blacklist_patterns = ttl._blacklist_patterns("none")
    if args.extra_blacklist.strip():
        blacklist_patterns.extend([x.strip() for x in args.extra_blacklist.split(",") if x.strip()])
    feature_cols, blacklist_dropped = ttl._apply_blacklist(feature_cols, blacklist_patterns)

    cfg = {"name": "reg", "lr": 0.02, "leaves": 47, "ff": 0.70, "mc": 30, "l1": 0.10, "l2": 0.40}
    folds = make_forward_temporal_group_folds(
        feat_train[["_cv_ts", "_cv_group"]].copy(),
        timestamp_col="_cv_ts",
        group_col="_cv_group",
        n_splits=max(int(args.n_splits), 2),
    )
    if not folds:
        raise RuntimeError("No forward folds were constructed.")

    labels_all = feat_train["bird_group"].to_numpy()
    y_all = _one_hot_from_labels(labels_all)
    size_all = feat_train["radar_bird_size"].to_numpy()
    test_size = feat_test["radar_bird_size"].to_numpy()
    X_test_all = feat_test[feature_cols].astype(np.float32)

    oof = np.zeros((len(feat_train), len(CLASSES)), dtype=np.float32)
    test_fold_preds: list[np.ndarray] = []
    fold_reports: list[dict[str, object]] = []

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        pred_val = np.zeros((len(va_idx), len(CLASSES)), dtype=np.float32)
        pred_test = np.zeros((len(feat_test), len(CLASSES)), dtype=np.float32)
        va_sizes = size_all[va_idx]
        fold_counts = pd.Series(labels_all[tr_idx]).value_counts()
        fold_prior = np.zeros((len(CLASSES),), dtype=np.float32)
        for cls, cnt in fold_counts.items():
            fold_prior[CLASS_TO_INDEX[cls]] = float(cnt)
        if fold_prior.sum() > 0:
            fold_prior = fold_prior / fold_prior.sum()
        else:
            fold_prior[:] = 1.0 / len(CLASSES)

        for size_name, possible in SIZE_POSSIBLE_CLASSES.items():
            tr_size_mask = size_all[tr_idx] == size_name
            va_size_mask = va_sizes == size_name
            te_size_mask = test_size == size_name
            if not np.any(va_size_mask) and not np.any(te_size_mask):
                continue

            tr_idx_size = tr_idx[tr_size_mask]
            if len(tr_idx_size) == 0:
                if np.any(va_size_mask):
                    base = np.tile(fold_prior, (int(np.sum(va_size_mask)), 1))
                    pred_val[va_size_mask] = _apply_possible_mask(base, possible, fold_prior)
                if np.any(te_size_mask):
                    base = np.tile(fold_prior, (int(np.sum(te_size_mask)), 1))
                    pred_test[te_size_mask] = _apply_possible_mask(base, possible, fold_prior)
                continue

            # Restrict training rows to the size-specific allowed classes.
            tr_lbl = labels_all[tr_idx_size]
            tr_allowed_mask = np.isin(tr_lbl, possible)
            tr_idx_size = tr_idx_size[tr_allowed_mask]
            if len(tr_idx_size) == 0:
                if np.any(va_size_mask):
                    base = np.tile(fold_prior, (int(np.sum(va_size_mask)), 1))
                    pred_val[va_size_mask] = _apply_possible_mask(base, possible, fold_prior)
                if np.any(te_size_mask):
                    base = np.tile(fold_prior, (int(np.sum(te_size_mask)), 1))
                    pred_test[te_size_mask] = _apply_possible_mask(base, possible, fold_prior)
                continue

            Xtr = feat_train.iloc[tr_idx_size][feature_cols].astype(np.float32)
            ytr = y_all[tr_idx_size]

            # Fallback prior for impossible/empty predictions.
            prior = np.zeros((len(CLASSES),), dtype=np.float32)
            counts = pd.Series(labels_all[tr_idx_size]).value_counts()
            for cls, cnt in counts.items():
                prior[CLASS_TO_INDEX[cls]] = float(cnt)
            if prior.sum() > 0:
                prior = prior / prior.sum()
            else:
                idx = [CLASS_TO_INDEX[c] for c in possible]
                prior[idx] = 1.0 / max(len(idx), 1)

            if np.any(va_size_mask):
                Xva = feat_train.iloc[va_idx[va_size_mask]][feature_cols].astype(np.float32)
                yva = y_all[va_idx[va_size_mask]]
                pv = ttl.train_ovr(
                    X_train=Xtr,
                    y_train=ytr,
                    X_valid=Xva,
                    y_valid=yva,
                    cfg=cfg,
                    seed=args.seed + fold_id * 1009 + SIZE_SEED_OFFSET[size_name] * 17,
                    n_estimators=int(args.n_estimators),
                    sample_weight=None,
                )
                pv = _apply_possible_mask(pv, possible, prior)
                pred_val[va_size_mask] = pv

            if np.any(te_size_mask):
                Xte = X_test_all.iloc[te_size_mask]
                pt = ttl.train_ovr_full(
                    X_train=Xtr,
                    y_train=ytr,
                    X_test=Xte,
                    cfg=cfg,
                    seed=args.seed + fold_id * 1009 + SIZE_SEED_OFFSET[size_name] * 17,
                    n_estimators=int(args.n_estimators),
                    sample_weight=None,
                )
                pt = _apply_possible_mask(pt, possible, prior)
                pred_test[te_size_mask] = pt

        if np.any(pred_val.sum(axis=1) <= 1e-12):
            zero_mask = pred_val.sum(axis=1) <= 1e-12
            pred_val[zero_mask] = np.tile(fold_prior, (int(np.sum(zero_mask)), 1))
        if np.any(pred_test.sum(axis=1) <= 1e-12):
            zero_mask_t = pred_test.sum(axis=1) <= 1e-12
            pred_test[zero_mask_t] = np.tile(fold_prior, (int(np.sum(zero_mask_t)), 1))

        oof[va_idx] = pred_val
        test_fold_preds.append(pred_test)
        fold_macro = _macro_ap(labels_all, oof, np.isin(np.arange(len(labels_all)), va_idx))
        fold_reports.append(
            {
                "fold": int(fold_id),
                "train_size": int(len(tr_idx)),
                "valid_size": int(len(va_idx)),
                "macro_map_valid": float(fold_macro),
            }
        )
        print(f"size-strat fold={fold_id} macro_valid={fold_macro:.6f}", flush=True)

    test_pred = np.mean(np.stack(test_fold_preds, axis=0), axis=0).astype(np.float32)
    oof = _normalize_rows(oof)
    test_pred = _normalize_rows(test_pred)

    # Metrics compared in the same protocol as current strong w80 reference.
    ref_train_ids = np.load(PROJECT_ROOT / "artifacts/tcn_trajectory_forward_2stream_distill_full_seed777_cpu/train_track_ids.npy", allow_pickle=True)
    ref_train = pd.read_csv(data_dir / "train.csv").set_index("track_id").loc[ref_train_ids].reset_index()
    ref_labels = ref_train["bird_group"].to_numpy()
    ref_months = pd.to_datetime(ref_train["timestamp_start_radar_utc"]).dt.month.to_numpy()
    teacher_oof = pd.read_csv(PROJECT_ROOT / "artifacts/oof_csv_for_per_class/oof_teacher_fr_complete.csv")
    covered_mask = np.array([tid in set(teacher_oof["track_id"].values) for tid in ref_train_ids])
    month_stress_mask = ref_months >= 9

    # Reorder OOF to reference train ids.
    train_id_to_idx = {int(tid): i for i, tid in enumerate(feat_train["track_id"].to_numpy(dtype=np.int64))}
    oof_ref = np.array([oof[train_id_to_idx[int(tid)]] if int(tid) in train_id_to_idx else np.zeros((len(CLASSES),), dtype=np.float32) for tid in ref_train_ids])

    def macro_ref(p: np.ndarray, mask: np.ndarray) -> float:
        y = ref_labels[mask]
        pp = p[mask]
        vals: list[float] = []
        for ci, cls in enumerate(CLASSES):
            yt = (y == cls).astype(np.int32)
            if yt.sum() > 0:
                vals.append(float(average_precision_score(yt, pp[:, ci])))
        return float(np.mean(vals)) if vals else 0.0

    covered_macro = macro_ref(oof_ref, covered_mask)
    month_stress_macro = macro_ref(oof_ref, month_stress_mask)

    # Save artifacts
    np.save(art_dir / "train_track_ids.npy", feat_train["track_id"].to_numpy(dtype=np.int64))
    np.save(art_dir / "test_track_ids.npy", feat_test["track_id"].to_numpy(dtype=np.int64))
    np.save(art_dir / "oof_forward_cv_complete.npy", oof.astype(np.float32))
    np.save(art_dir / "test_forward_cv_mean.npy", test_pred.astype(np.float32))

    test_out = pd.DataFrame(test_pred, columns=CLASSES)
    test_out.insert(0, "track_id", feat_test["track_id"].to_numpy(dtype=np.int64))
    test_out["bird_group"] = np.array(CLASSES)[np.argmax(test_pred, axis=1)]
    test_out.to_csv(out_dir / "sub_size_stratified_ovr.csv", index=False)

    report = {
        "model": "size_stratified_ovr_forward",
        "seed": int(args.seed),
        "n_splits": int(args.n_splits),
        "n_estimators": int(args.n_estimators),
        "cfg": cfg,
        "size_possible_classes": SIZE_POSSIBLE_CLASSES,
        "n_features": int(len(feature_cols)),
        "blacklist_dropped_features": blacklist_dropped,
        "fold_reports": fold_reports,
        "oof_macro_ap_covered_ref_protocol": float(covered_macro),
        "oof_macro_ap_month_stress_ref_protocol": float(month_stress_macro),
        "outputs": {
            "oof_npy": str((art_dir / "oof_forward_cv_complete.npy").resolve()),
            "test_npy": str((art_dir / "test_forward_cv_mean.npy").resolve()),
            "submission_csv": str((out_dir / "sub_size_stratified_ovr.csv").resolve()),
        },
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("=== SIZE-STRATIFIED REPORT ===", flush=True)
    print(f"OOF covered (ref protocol): {covered_macro:.6f}", flush=True)
    print(f"OOF month_stress (ref protocol): {month_stress_macro:.6f}", flush=True)
    print(f"Saved submission: {out_dir / 'sub_size_stratified_ovr.csv'}", flush=True)


if __name__ == "__main__":
    main()
