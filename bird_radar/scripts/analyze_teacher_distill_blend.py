#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds


def _macro_map(y_true: np.ndarray, y_prob: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    vals: list[float] = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        if yt.sum() <= 0:
            vals.append(0.0)
        else:
            vals.append(float(average_precision_score(yt, yp, sample_weight=sample_weight)))
    return float(np.mean(vals))


def _onehot(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.zeros((len(y_idx), n_classes), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    return y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze teacher vs seq-distill blend value by temporal folds.")
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--teacher-oof-csv", type=str, required=True)
    p.add_argument("--seq-oof-npy", type=str, required=True)
    p.add_argument("--seq-track-ids-npy", type=str, default="")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--max-folds", type=int, default=0)
    p.add_argument("--weights", type=str, default="0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50")
    p.add_argument("--sample-weights-npy", type=str, default="")
    p.add_argument("--output-json", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train_csv, usecols=["track_id", "bird_group", "timestamp_start_radar_utc", "observation_id"])
    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = _onehot(y_idx, len(CLASSES))
    track_ids = train_df["track_id"].to_numpy(dtype=np.int64)

    teacher_df = pd.read_csv(args.teacher_oof_csv, usecols=["track_id", *CLASSES])
    teacher_map = {int(r["track_id"]): np.asarray([r[c] for c in CLASSES], dtype=np.float32) for _, r in teacher_df.iterrows()}
    missing_teacher = [int(t) for t in track_ids if int(t) not in teacher_map]
    if missing_teacher:
        raise ValueError(f"teacher_oof_csv missing {len(missing_teacher)} train ids; first={missing_teacher[:10]}")
    teacher = np.stack([teacher_map[int(t)] for t in track_ids], axis=0).astype(np.float32)

    sample_weights: np.ndarray | None = None
    if str(args.sample_weights_npy).strip():
        sample_weights = np.asarray(np.load(args.sample_weights_npy), dtype=np.float32).reshape(-1)
        if len(sample_weights) != len(track_ids):
            raise ValueError(
                f"sample_weights length mismatch: {len(sample_weights)} vs train {len(track_ids)}"
            )
        sample_weights = np.clip(sample_weights, 1e-8, None).astype(np.float32)

    seq = np.load(args.seq_oof_npy).astype(np.float32)
    seq_track_ids_path = str(args.seq_track_ids_npy).strip()
    if seq_track_ids_path:
        seq_ids = np.load(seq_track_ids_path).astype(np.int64)
        if len(seq_ids) != len(track_ids):
            raise ValueError(f"seq_track_ids length mismatch: {len(seq_ids)} vs train {len(track_ids)}")
        if not np.array_equal(seq_ids, track_ids):
            # realign to train order
            pos = {int(t): i for i, t in enumerate(seq_ids)}
            missing_seq = [int(t) for t in track_ids if int(t) not in pos]
            if missing_seq:
                raise ValueError(f"seq_track_ids missing {len(missing_seq)} train ids; first={missing_seq[:10]}")
            seq = np.stack([seq[pos[int(t)]] for t in track_ids], axis=0).astype(np.float32)
    else:
        if len(seq) != len(track_ids):
            raise ValueError(
                "seq-track-ids-npy not provided and seq_oof length does not match train length "
                f"({len(seq)} vs {len(track_ids)})"
            )

    cv_df = train_df.rename(columns={"timestamp_start_radar_utc": "_cv_ts", "observation_id": "_cv_group"})
    folds = make_forward_temporal_group_folds(cv_df, timestamp_col="_cv_ts", group_col="_cv_group", n_splits=int(args.n_splits))
    if args.max_folds > 0:
        folds = folds[: int(args.max_folds)]

    w_grid = np.array([float(x.strip()) for x in args.weights.split(",") if x.strip()], dtype=np.float32)
    if len(w_grid) == 0:
        raise ValueError("weights grid is empty")

    fold_reports: list[dict[str, float | int]] = []
    for fold_id, (_, va_idx) in enumerate(folds):
        yt = y[va_idx]
        te = teacher[va_idx]
        sq = seq[va_idx]
        sw = sample_weights[va_idx] if sample_weights is not None else None
        covered = (sq.sum(axis=1) > 0.0)
        n_val = int(len(va_idx))
        n_cov = int(covered.sum())
        fold_weight = float(sw[covered].sum()) if sw is not None else float(n_cov)
        if n_cov == 0:
            fold_reports.append(
                {
                    "fold": int(fold_id),
                    "n_val": n_val,
                    "n_covered": 0,
                    "teacher_macro": 0.0,
                    "seq_macro": 0.0,
                    "best_blend_macro": 0.0,
                    "best_w_teacher": 1.0,
                    "corr_seq_teacher": 0.0,
                    "fold_weight": 0.0,
                }
            )
            continue

        yt_c = yt[covered]
        te_c = te[covered]
        sq_c = sq[covered]
        sw_c = sw[covered] if sw is not None else None
        teacher_macro = _macro_map(yt_c, te_c, sample_weight=sw_c)
        seq_macro = _macro_map(yt_c, sq_c, sample_weight=sw_c)
        corr = float(np.corrcoef(te_c.reshape(-1), sq_c.reshape(-1))[0, 1])

        best_macro = -1.0
        best_w = 1.0
        for w in w_grid:
            p = np.clip(w * te_c + (1.0 - w) * sq_c, 0.0, 1.0)
            s = _macro_map(yt_c, p, sample_weight=sw_c)
            if s > best_macro:
                best_macro = s
                best_w = float(w)

        fold_reports.append(
            {
                "fold": int(fold_id),
                "n_val": n_val,
                "n_covered": n_cov,
                "teacher_macro": float(teacher_macro),
                "seq_macro": float(seq_macro),
                "best_blend_macro": float(best_macro),
                "best_w_teacher": float(best_w),
                "corr_seq_teacher": float(corr),
                "fold_weight": float(fold_weight),
            }
        )

    valid = [r for r in fold_reports if int(r["n_covered"]) > 0]
    if valid:
        if sample_weights is not None:
            vw = np.asarray([float(r["fold_weight"]) for r in valid], dtype=np.float64)
            teacher_mean = float(np.average([float(r["teacher_macro"]) for r in valid], weights=vw))
            seq_mean = float(np.average([float(r["seq_macro"]) for r in valid], weights=vw))
            blend_mean = float(np.average([float(r["best_blend_macro"]) for r in valid], weights=vw))
        else:
            teacher_mean = float(np.mean([float(r["teacher_macro"]) for r in valid]))
            seq_mean = float(np.mean([float(r["seq_macro"]) for r in valid]))
            blend_mean = float(np.mean([float(r["best_blend_macro"]) for r in valid]))
        corr_mean = float(np.mean([float(r["corr_seq_teacher"]) for r in valid]))
    else:
        teacher_mean = seq_mean = blend_mean = corr_mean = 0.0

    out = {
        "n_rows": int(len(track_ids)),
        "n_folds": int(len(folds)),
        "sample_weights_npy": str(Path(args.sample_weights_npy).resolve()) if str(args.sample_weights_npy).strip() else "",
        "sample_weighted_metrics": bool(sample_weights is not None),
        "weights_grid": [float(x) for x in w_grid.tolist()],
        "teacher_macro_mean": teacher_mean,
        "seq_macro_mean": seq_mean,
        "best_blend_macro_mean": blend_mean,
        "corr_seq_teacher_mean": corr_mean,
        "folds": fold_reports,
    }

    print(
        f"teacher_mean={teacher_mean:.6f} seq_mean={seq_mean:.6f} "
        f"best_blend_mean={blend_mean:.6f} corr_mean={corr_mean:.6f}",
        flush=True,
    )
    for r in fold_reports:
        print(
            f"fold={int(r['fold'])} n_cov={int(r['n_covered'])}/{int(r['n_val'])} "
            f"teacher={float(r['teacher_macro']):.6f} seq={float(r['seq_macro']):.6f} "
            f"blend={float(r['best_blend_macro']):.6f} w={float(r['best_w_teacher']):.2f} "
            f"corr={float(r['corr_seq_teacher']):.6f}",
            flush=True,
        )

    if args.output_json.strip():
        out_path = Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
        print(str(out_path), flush=True)


if __name__ == "__main__":
    main()
