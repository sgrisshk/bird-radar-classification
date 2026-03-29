#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASSES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build high-confidence pseudo labels from submission probabilities.")
    p.add_argument("--submission-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--id-col", default="track_id")
    p.add_argument("--prob-threshold", type=float, default=0.97)
    p.add_argument("--max-per-class", type=int, default=200)
    p.add_argument("--min-per-class", type=int, default=0)
    p.add_argument("--sort-by", choices=["prob", "margin"], default="prob")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sub = pd.read_csv(args.submission_csv)

    req = {str(args.id_col), *CLASSES}
    miss = [c for c in req if c not in sub.columns]
    if miss:
        raise ValueError(f"submission missing columns: {miss}")

    probs = np.clip(sub[CLASSES].to_numpy(dtype=np.float64), 0.0, 1.0)
    top_idx = np.argmax(probs, axis=1)
    top_prob = probs[np.arange(len(probs)), top_idx]

    # Margin between top1/top2 can stabilize filtering in highly-confusable regions.
    part = np.partition(probs, kth=probs.shape[1] - 2, axis=1)
    top2_prob = part[:, -2]
    margin = top_prob - top2_prob

    keep = top_prob >= float(args.prob_threshold)
    df = pd.DataFrame(
        {
            str(args.id_col): sub[str(args.id_col)].to_numpy(dtype=np.int64),
            "bird_group": [CLASSES[int(i)] for i in top_idx],
            "pseudo_prob": top_prob.astype(np.float32),
            "pseudo_margin": margin.astype(np.float32),
        }
    )
    for c in CLASSES:
        df[c] = probs[:, CLASSES.index(c)].astype(np.float32)
    df = df.loc[keep].copy()

    pieces: list[pd.DataFrame] = []
    max_per_class = int(max(1, args.max_per_class))
    min_per_class = int(max(0, args.min_per_class))
    sort_col = "pseudo_prob" if str(args.sort_by) == "prob" else "pseudo_margin"
    for cls in CLASSES:
        part_cls = df[df["bird_group"] == cls].sort_values(sort_col, ascending=False)
        if len(part_cls) == 0:
            continue
        take_n = min(max_per_class, len(part_cls))
        if min_per_class > 0:
            take_n = max(min_per_class, take_n)
            take_n = min(take_n, len(part_cls))
        pieces.append(part_cls.head(take_n))
    out = pd.concat(pieces, axis=0, ignore_index=True) if pieces else df.iloc[:0].copy()

    out_csv = Path(args.output_csv).resolve()
    out_json = Path(args.output_json).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    counts = out["bird_group"].value_counts().to_dict() if len(out) else {}
    report = {
        "submission_csv": str(Path(args.submission_csv).resolve()),
        "output_csv": str(out_csv),
        "n_input": int(len(sub)),
        "n_after_threshold": int(keep.sum()),
        "n_output": int(len(out)),
        "prob_threshold": float(args.prob_threshold),
        "max_per_class": int(max_per_class),
        "min_per_class": int(min_per_class),
        "sort_by": str(args.sort_by),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "pseudo_prob_stats": {
            "min": float(out["pseudo_prob"].min()) if len(out) else 0.0,
            "max": float(out["pseudo_prob"].max()) if len(out) else 0.0,
            "mean": float(out["pseudo_prob"].mean()) if len(out) else 0.0,
            "p95": float(out["pseudo_prob"].quantile(0.95)) if len(out) else 0.0,
        },
    }
    out_json.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(str(out_csv), flush=True)
    print(f"n_output={len(out)}", flush=True)


if __name__ == "__main__":
    main()
