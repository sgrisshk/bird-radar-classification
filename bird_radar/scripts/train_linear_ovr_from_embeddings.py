#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds
from src.redesign.utils import dump_json, macro_map, per_class_ap


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train linear OVR (Standardize + PCA + Logistic) on embeddings.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--sample-submission", required=True)
    p.add_argument("--emb-oof-npy", required=True)
    p.add_argument("--emb-test-npy", required=True)
    p.add_argument("--train-track-ids-npy", required=True)
    p.add_argument("--test-track-ids-npy", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--start-fold", type=int, default=1)
    p.add_argument("--time-col", default="timestamp_start_radar_utc")
    p.add_argument("--group-col", default="observation_id")

    p.add_argument("--pca-dim", type=int, default=128)
    p.add_argument("--clip-value", type=float, default=5.0)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=3000)
    p.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    p.add_argument("--min-class-count", type=int, default=5)
    p.add_argument(
        "--invalid-class-policy",
        choices=["fallback_prior", "skip_class", "skip_fold"],
        default="skip_class",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = out_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    sample_sub = pd.read_csv(args.sample_submission)

    emb_train = np.load(args.emb_oof_npy).astype(np.float32)
    emb_test = np.load(args.emb_test_npy).astype(np.float32)
    train_ids = np.load(args.train_track_ids_npy).astype(np.int64)
    test_ids = np.load(args.test_track_ids_npy).astype(np.int64)

    if emb_train.shape[0] != len(train_ids):
        raise ValueError(f"emb-oof rows mismatch: {emb_train.shape[0]} vs ids={len(train_ids)}")
    if emb_test.shape[0] != len(test_ids):
        raise ValueError(f"emb-test rows mismatch: {emb_test.shape[0]} vs ids={len(test_ids)}")
    if emb_train.shape[1] != emb_test.shape[1]:
        raise ValueError("embedding dim mismatch between train/test")

    row_map = dict(zip(train_df["track_id"].astype(np.int64).tolist(), np.arange(len(train_df), dtype=np.int64).tolist()))
    rows = np.asarray([row_map[int(t)] for t in train_ids], dtype=np.int64)

    y_idx_all = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y_idx = y_idx_all[rows]
    y = np.zeros((len(train_ids), len(CLASSES)), dtype=np.float32)
    y[np.arange(len(train_ids)), y_idx] = 1.0

    cv_df = pd.DataFrame(
        {
            "_cv_ts": train_df[str(args.time_col)].iloc[rows].to_numpy(),
            "_cv_group": train_df[str(args.group_col)].iloc[rows].astype(np.int64).to_numpy(),
        }
    )
    folds_full = make_forward_temporal_group_folds(
        cv_df, timestamp_col="_cv_ts", group_col="_cv_group", n_splits=int(args.n_splits)
    )
    fold_id_by_row = np.full(len(train_ids), -1, dtype=np.int64)
    for k, (_, va_idx_k) in enumerate(folds_full):
        fold_id_by_row[np.asarray(va_idx_k, dtype=np.int64)] = int(k)
    folds = list(folds_full)
    if int(args.start_fold) > 0:
        s = min(int(args.start_fold), len(folds))
        folds = folds[s:]
        print(f"[linear-embed] starting from fold index {s} (skipped {s} earliest folds)", flush=True)
    if len(folds) == 0:
        raise RuntimeError("no folds selected")

    oof = np.zeros((len(train_ids), len(CLASSES)), dtype=np.float32)
    test_accum = np.zeros((len(test_ids), len(CLASSES)), dtype=np.float32)
    fold_scores: list[float] = []
    fold_reports: list[dict[str, Any]] = []
    skipped_folds: list[int] = []

    pca_dim = int(max(2, min(args.pca_dim, emb_train.shape[1])))
    class_weight = None if str(args.class_weight) == "none" else "balanced"
    effective_folds = 0

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        tr_idx = np.asarray(tr_idx, dtype=np.int64)
        va_idx = np.asarray(va_idx, dtype=np.int64)
        if int(args.start_fold) > 0:
            tr_idx = tr_idx[fold_id_by_row[tr_idx] >= int(args.start_fold)]

        invalid_classes: list[str] = []
        for c, cls in enumerate(CLASSES):
            pos = int(y[tr_idx, c].sum())
            neg = int(len(tr_idx) - pos)
            if min(pos, neg) < int(args.min_class_count):
                invalid_classes.append(cls)

        if invalid_classes and str(args.invalid_class_policy) == "skip_fold":
            skipped_folds.append(fold_id)
            fold_reports.append(
                {
                    "fold_id": int(fold_id),
                    "n_val": int(len(va_idx)),
                    "skipped": True,
                    "invalid_classes": invalid_classes,
                }
            )
            print(f"[fold] {fold_id+1}/{len(folds)} skipped invalid_classes={len(invalid_classes)}", flush=True)
            continue

        xtr = emb_train[tr_idx].astype(np.float32)
        xva = emb_train[va_idx].astype(np.float32)
        xte = emb_test.astype(np.float32)

        fold_test = np.zeros((len(test_ids), len(CLASSES)), dtype=np.float32)
        for c, cls in enumerate(CLASSES):
            yt = y[tr_idx, c]
            pos = int(yt.sum())
            neg = int(len(tr_idx) - pos)
            if min(pos, neg) < int(args.min_class_count):
                prior = float(np.mean(yt)) if len(yt) > 0 else 0.5
                oof[va_idx, c] = prior
                fold_test[:, c] = prior
                continue

            clf = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("pca", PCA(n_components=pca_dim, random_state=int(args.seed))),
                    (
                        "logreg",
                        LogisticRegression(
                            solver="saga",
                            penalty="l2",
                            C=float(args.c),
                            class_weight=class_weight,
                            max_iter=int(args.max_iter),
                            random_state=int(args.seed) + 97 * fold_id + 13 * c,
                        ),
                    ),
                ]
            )
            clf.fit(xtr, yt)
            pva = clf.predict_proba(xva)[:, 1].astype(np.float32)
            pte = clf.predict_proba(xte)[:, 1].astype(np.float32)
            if float(args.clip_value) > 0:
                lo, hi = float(1e-6), float(1.0 - 1e-6)
                pva = np.clip(pva, lo, hi)
                pte = np.clip(pte, lo, hi)
            oof[va_idx, c] = pva
            fold_test[:, c] = pte

        effective_folds += 1
        test_accum += fold_test
        fold_macro = float(macro_map(y[va_idx], oof[va_idx]))
        fold_scores.append(fold_macro)
        fold_reports.append(
            {
                "fold_id": int(fold_id),
                "n_val": int(len(va_idx)),
                "skipped": False,
                "invalid_classes": invalid_classes,
                "macro_map": fold_macro,
            }
        )
        print(
            f"[fold] {fold_id+1}/{len(folds)} macro_map={fold_macro:.6f} n_val={len(va_idx)} invalid_classes={len(invalid_classes)}",
            flush=True,
        )

    if effective_folds <= 0:
        raise RuntimeError("all folds skipped")
    test_accum /= float(effective_folds)

    macro_all = float(macro_map(y, oof))
    per_all = per_class_ap(y, oof)

    np.save(out_dir / "train_track_ids.npy", train_ids)
    np.save(out_dir / "test_track_ids.npy", test_ids)
    np.save(out_dir / "oof_targets.npy", y)
    np.save(art_dir / "linear_embed_ovr_oof.npy", oof)
    np.save(art_dir / "linear_embed_ovr_test.npy", test_accum)

    sub = sample_sub.copy()
    if "track_id" not in sub.columns:
        raise ValueError("sample submission must contain track_id")
    sub = sub[[c for c in sub.columns if c == "track_id" or c in CLASSES]].copy()
    sub["track_id"] = test_ids
    sub[CLASSES] = test_accum
    sub.to_csv(out_dir / "submission_linear_embed_ovr.csv", index=False)

    summary = {
        "project_root": str(PROJECT_ROOT.resolve()),
        "output_dir": str(out_dir),
        "n_train": int(len(train_ids)),
        "n_test": int(len(test_ids)),
        "embedding_dim": int(emb_train.shape[1]),
        "start_fold": int(args.start_fold),
        "model": {
            "type": "standardize_pca_logreg_ovr",
            "pca_dim": int(pca_dim),
            "c": float(args.c),
            "class_weight": str(args.class_weight),
            "max_iter": int(args.max_iter),
        },
        "invalid_class_handling": {
            "policy": str(args.invalid_class_policy),
            "min_class_count": int(args.min_class_count),
            "requested_folds": int(len(folds)),
            "effective_folds": int(effective_folds),
            "skipped_folds": skipped_folds,
        },
        "models": {
            f"linear_embed_ovr_seed{int(args.seed)}": {
                "type": "tabular_embedding_linear",
                "oof_path": str((art_dir / "linear_embed_ovr_oof.npy").resolve()),
                "test_path": str((art_dir / "linear_embed_ovr_test.npy").resolve()),
                "macro_map": macro_all,
                "per_class_ap": per_all,
                "fold_scores": [float(x) for x in fold_scores],
                "worst_fold": float(np.min(fold_scores)) if fold_scores else 0.0,
                "fold_reports": fold_reports,
            }
        },
    }
    dump_json(out_dir / "oof_summary.json", summary)

    print("=== LINEAR EMBED OVR COMPLETE ===", flush=True)
    print(f"output_dir: {out_dir}", flush=True)
    print(f"oof_summary: {out_dir / 'oof_summary.json'}", flush=True)
    print(
        f"macro_oof={macro_all:.6f} fold_mean={float(np.mean(fold_scores)):.6f} fold_worst={float(np.min(fold_scores)):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

