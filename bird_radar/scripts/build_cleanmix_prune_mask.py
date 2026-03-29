#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build hard-prune mask from consensus noise weights for CleanMix.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--weights-parquet", required=True)
    p.add_argument("--max-global-drop-ratio", type=float, default=0.03)
    p.add_argument("--max-class-drop-ratio", type=float, default=0.08)
    p.add_argument("--weight-threshold", type=float, default=0.999)
    p.add_argument("--output-mask-npy", required=True)
    p.add_argument("--output-json", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_mask = Path(args.output_mask_npy).resolve()
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    out_json = (
        Path(args.output_json).resolve()
        if str(args.output_json).strip()
        else out_mask.with_suffix(".json")
    )

    train = pd.read_csv(args.train_csv, usecols=["track_id", "bird_group"])
    w = pd.read_parquet(args.weights_parquet)[["track_id", "sample_weight"]]
    df = train.merge(w, on="track_id", how="left")
    df["sample_weight"] = df["sample_weight"].fillna(1.0).astype(np.float32)

    n = len(df)
    candidate = df["sample_weight"].to_numpy() < float(args.weight_threshold)
    keep = np.ones((n,), dtype=bool)

    # Per-class cap.
    for _, idx in df.groupby("bird_group").groups.items():
        idx_arr = np.asarray(list(idx), dtype=np.int64)
        cand_idx = idx_arr[candidate[idx_arr]]
        max_drop_cls = int(np.floor(len(idx_arr) * float(args.max_class_drop_ratio)))
        if max_drop_cls <= 0 or len(cand_idx) <= 0:
            continue
        order = cand_idx[np.argsort(df.iloc[cand_idx]["sample_weight"].to_numpy())]
        drop_idx = order[:max_drop_cls]
        keep[drop_idx] = False

    # Global cap.
    max_drop_global = int(np.floor(n * float(args.max_global_drop_ratio)))
    dropped = np.where(~keep)[0]
    if len(dropped) > max_drop_global >= 0:
        # Keep only the globally most suspicious among already selected.
        order = dropped[np.argsort(df.iloc[dropped]["sample_weight"].to_numpy())]
        final_drop = set(order[:max_drop_global].tolist())
        keep = np.array([i not in final_drop for i in range(n)], dtype=bool)

    np.save(out_mask, keep.astype(np.uint8))

    dropped_idx = np.where(~keep)[0]
    per_class_drop = (
        df.iloc[dropped_idx]["bird_group"].value_counts().sort_index().to_dict()
        if len(dropped_idx) > 0
        else {}
    )
    per_class_total = df["bird_group"].value_counts().sort_index().to_dict()
    per_class_ratio = {
        k: (float(per_class_drop.get(k, 0)) / float(v) if v else 0.0)
        for k, v in per_class_total.items()
    }
    report = {
        "train_csv": str(Path(args.train_csv).resolve()),
        "weights_parquet": str(Path(args.weights_parquet).resolve()),
        "max_global_drop_ratio": float(args.max_global_drop_ratio),
        "max_class_drop_ratio": float(args.max_class_drop_ratio),
        "weight_threshold": float(args.weight_threshold),
        "n_rows": int(n),
        "n_drop": int(len(dropped_idx)),
        "drop_ratio": float(len(dropped_idx) / max(1, n)),
        "per_class_drop": per_class_drop,
        "per_class_drop_ratio": per_class_ratio,
        "output_mask_npy": str(out_mask),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_mask_npy": str(out_mask), "n_drop": int(len(dropped_idx)), "drop_ratio": report["drop_ratio"]}))


if __name__ == "__main__":
    main()

