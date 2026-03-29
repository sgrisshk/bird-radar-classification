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


def _rank01(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(x), dtype=np.float32)
    if len(x) > 1:
        ranks /= float(len(x) - 1)
    return ranks


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-select classwise blend classes from OOF AP deltas.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--teacher-oof-csv", required=True)
    p.add_argument("--seq-oof-npy", default=None)
    p.add_argument("--new-oof-npy", default=None, help="Alias for --seq-oof-npy.")
    p.add_argument("--seq-track-ids-npy", required=True)
    p.add_argument("--ap-threshold", type=float, default=0.01)
    p.add_argument("--base-sub-csv", default=None)
    p.add_argument("--teacher-test-csv", default=None, help="Alias for --base-sub-csv.")
    p.add_argument("--seq-sub-csv", default=None)
    p.add_argument("--new-test-csv", default=None, help="Alias for --seq-sub-csv.")
    p.add_argument("--w-base", type=float, default=None)
    p.add_argument("--alpha-teacher", type=float, default=None, help="Alias for --w-base.")
    p.add_argument("--w-seq", type=float, default=None)
    p.add_argument("--alpha-new", type=float, default=None, help="Alias for --w-seq.")
    p.add_argument("--mode", choices=["prob_mean", "rank_mean"], default="prob_mean")
    p.add_argument("--sample-weights-npy", default=None)
    p.add_argument("--output-csv", default=None)
    p.add_argument("--out-csv", default=None, help="Alias for --output-csv.")
    p.add_argument("--output-json", default=None)
    p.add_argument("--out-json", default=None, help="Alias for --output-json.")
    args = p.parse_args()

    args.seq_oof_npy = args.seq_oof_npy or args.new_oof_npy
    args.base_sub_csv = args.base_sub_csv or args.teacher_test_csv
    args.seq_sub_csv = args.seq_sub_csv or args.new_test_csv
    args.output_csv = args.output_csv or args.out_csv
    args.output_json = args.output_json or args.out_json

    args.w_base = 0.8 if args.w_base is None and args.alpha_teacher is None else (
        args.w_base if args.w_base is not None else args.alpha_teacher
    )
    args.w_seq = 0.2 if args.w_seq is None and args.alpha_new is None else (
        args.w_seq if args.w_seq is not None else args.alpha_new
    )

    missing = []
    for key in ("seq_oof_npy", "base_sub_csv", "seq_sub_csv", "output_csv", "output_json"):
        if getattr(args, key) in (None, ""):
            missing.append(key)
    if missing:
        p.error(f"missing required arguments after alias resolution: {', '.join(missing)}")
    return args


def main() -> None:
    args = _parse_args()

    train = pd.read_csv(args.train_csv, usecols=["track_id", "bird_group"])
    y_idx = train["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = np.zeros((len(train), len(CLASSES)), dtype=np.float32)
    y[np.arange(len(train)), y_idx] = 1.0
    idx_map = dict(zip(train["track_id"].astype(np.int64).tolist(), np.arange(len(train), dtype=np.int64).tolist()))

    seq_track_ids = np.load(args.seq_track_ids_npy).astype(np.int64)
    seq_oof = np.load(args.seq_oof_npy).astype(np.float32)
    if seq_oof.shape[0] != len(seq_track_ids) or seq_oof.shape[1] != len(CLASSES):
        raise ValueError("seq_oof shape mismatch")

    rows = np.array([idx_map[int(t)] for t in seq_track_ids], dtype=np.int64)
    y_aligned = y[rows]
    sample_weights: np.ndarray | None = None
    if args.sample_weights_npy not in (None, ""):
        sw = np.asarray(np.load(args.sample_weights_npy), dtype=np.float32).reshape(-1)
        if len(sw) != len(train):
            raise ValueError(f"sample_weights length mismatch: {len(sw)} vs train {len(train)}")
        sample_weights = np.clip(sw[rows], 1e-8, None).astype(np.float32)

    teacher_df = pd.read_csv(args.teacher_oof_csv, usecols=["track_id", *CLASSES]).set_index("track_id")
    teacher_oof = teacher_df.loc[seq_track_ids, CLASSES].to_numpy(dtype=np.float32)

    class_report: list[dict[str, float | str]] = []
    selected: list[str] = []
    for j, c in enumerate(CLASSES):
        yt = y_aligned[:, j]
        ap_teacher = float(average_precision_score(yt, teacher_oof[:, j], sample_weight=sample_weights))
        ap_seq = float(average_precision_score(yt, seq_oof[:, j], sample_weight=sample_weights))
        delta = float(ap_seq - ap_teacher)
        class_report.append(
            {
                "class": c,
                "ap_teacher": ap_teacher,
                "ap_seq": ap_seq,
                "ap_delta": delta,
            }
        )
        if delta >= float(args.ap_threshold):
            selected.append(c)

    base_sub = pd.read_csv(args.base_sub_csv)
    seq_sub = pd.read_csv(args.seq_sub_csv)
    if "track_id" not in base_sub.columns or "track_id" not in seq_sub.columns:
        raise ValueError("submission CSVs must contain track_id")
    merged = base_sub.merge(seq_sub, on="track_id", suffixes=("_base", "_seq"), how="inner")
    if len(merged) != len(base_sub) or len(merged) != len(seq_sub):
        raise ValueError("track_id mismatch between submissions")

    out = pd.DataFrame({"track_id": merged["track_id"].to_numpy(dtype=np.int64)})
    for c in CLASSES:
        xb = merged[f"{c}_base"].to_numpy(dtype=np.float32)
        xs = merged[f"{c}_seq"].to_numpy(dtype=np.float32)
        if c in selected:
            if args.mode == "rank_mean":
                yhat = float(args.w_base) * _rank01(xb) + float(args.w_seq) * _rank01(xs)
            else:
                yhat = float(args.w_base) * xb + float(args.w_seq) * xs
            out[c] = np.clip(yhat, 0.0, 1.0)
        else:
            out[c] = xb

    out_csv = Path(args.output_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    report = {
        "train_csv": str(Path(args.train_csv).resolve()),
        "teacher_oof_csv": str(Path(args.teacher_oof_csv).resolve()),
        "seq_oof_npy": str(Path(args.seq_oof_npy).resolve()),
        "seq_track_ids_npy": str(Path(args.seq_track_ids_npy).resolve()),
        "base_sub_csv": str(Path(args.base_sub_csv).resolve()),
        "seq_sub_csv": str(Path(args.seq_sub_csv).resolve()),
        "mode": str(args.mode),
        "w_base": float(args.w_base),
        "w_seq": float(args.w_seq),
        "ap_threshold": float(args.ap_threshold),
        "sample_weights_npy": str(Path(args.sample_weights_npy).resolve()) if args.sample_weights_npy not in (None, "") else "",
        "sample_weighted_ap": bool(sample_weights is not None),
        "selected_classes": selected,
        "n_selected": int(len(selected)),
        "class_report": sorted(class_report, key=lambda x: float(x["ap_delta"]), reverse=True),
        "output_csv": str(out_csv),
    }
    out_json = Path(args.output_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(str(out_csv), flush=True)
    print(f"selected_classes={selected}", flush=True)


if __name__ == "__main__":
    main()
