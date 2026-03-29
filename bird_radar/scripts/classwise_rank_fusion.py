#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _rank01(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(x), dtype=np.float32)
    if len(x) > 1:
        ranks /= float(len(x) - 1)
    return ranks


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Class-wise fusion for submission CSVs.")
    p.add_argument("--a-csv", required=True, help="Base submission CSV (kept for untouched classes).")
    p.add_argument("--b-csv", required=True, help="Second submission CSV.")
    p.add_argument("--classes", required=True, help="Comma-separated class names to fuse.")
    p.add_argument("--w-a", type=float, default=0.8, help="Weight for A in fusion.")
    p.add_argument("--w-b", type=float, default=0.2, help="Weight for B in fusion.")
    p.add_argument(
        "--mode",
        choices=["rank_mean", "prob_mean"],
        default="rank_mean",
        help="Fusion mode for selected classes.",
    )
    p.add_argument("--output-csv", required=True, help="Output CSV path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    df_a = pd.read_csv(args.a_csv)
    df_b = pd.read_csv(args.b_csv)

    if "track_id" not in df_a.columns or "track_id" not in df_b.columns:
        raise ValueError("Both CSVs must contain 'track_id' column.")

    class_cols_a = [c for c in df_a.columns if c != "track_id"]
    class_cols_b = [c for c in df_b.columns if c != "track_id"]
    if class_cols_a != class_cols_b:
        raise ValueError("Class columns mismatch between inputs.")

    merged = df_a.merge(df_b, on="track_id", suffixes=("_a", "_b"), how="inner")
    if len(merged) != len(df_a) or len(merged) != len(df_b):
        raise ValueError("track_id mismatch between inputs.")

    classes = [c.strip() for c in str(args.classes).split(",") if c.strip()]
    missing = [c for c in classes if c not in class_cols_a]
    if missing:
        raise ValueError(f"Unknown classes in --classes: {missing}")

    out = pd.DataFrame({"track_id": merged["track_id"].to_numpy()})
    for c in class_cols_a:
        xa = merged[f"{c}_a"].to_numpy(dtype=np.float32)
        xb = merged[f"{c}_b"].to_numpy(dtype=np.float32)
        if c in classes:
            if args.mode == "rank_mean":
                ra = _rank01(xa)
                rb = _rank01(xb)
                y = args.w_a * ra + args.w_b * rb
            else:
                y = args.w_a * xa + args.w_b * xb
            out[c] = np.clip(y, 0.0, 1.0)
        else:
            out[c] = xa

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(str(out_path.resolve()))
    print(f"rows={len(out)} classes_fused={classes} mode={args.mode}")


if __name__ == "__main__":
    main()

