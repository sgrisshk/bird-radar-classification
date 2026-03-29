from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASSES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pseudo labels from a teacher submission.")
    p.add_argument("--teacher-submission", type=str, required=True)
    p.add_argument("--output-csv", type=str, required=True)
    p.add_argument("--min-prob", type=float, default=0.95)
    p.add_argument("--topk-per-class", type=int, default=0, help="0 = disabled")
    p.add_argument("--max-total-ratio", type=float, default=0.15, help="0 = disabled")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sub_path = Path(args.teacher_submission).resolve()
    out_path = Path(args.output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sub_path)
    req = ["track_id", *CLASSES]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"teacher submission is missing columns: {missing}")

    probs = np.clip(df[CLASSES].to_numpy(dtype=np.float64), 0.0, 1.0)
    top_idx = np.argmax(probs, axis=1)
    top_prob = probs[np.arange(len(df)), top_idx]
    top_cls = np.array(CLASSES, dtype=object)[top_idx]

    out = pd.DataFrame(
        {
            "track_id": df["track_id"].to_numpy(),
            "bird_group": top_cls,
            "pseudo_prob": top_prob.astype(np.float32),
            "pseudo_class_idx": top_idx.astype(np.int32),
        }
    )
    for c in CLASSES:
        out[c] = df[c].to_numpy(dtype=np.float32)

    out = out[out["pseudo_prob"] >= float(args.min_prob)].copy()
    if len(out) == 0:
        raise RuntimeError("No pseudo labels after min-prob filter; lower --min-prob")

    if int(args.topk_per_class) > 0:
        chunks: list[pd.DataFrame] = []
        for c in CLASSES:
            part = out[out["bird_group"] == c].copy()
            if len(part) == 0:
                continue
            part = part.sort_values("pseudo_prob", ascending=False).head(int(args.topk_per_class))
            chunks.append(part)
        if chunks:
            out = pd.concat(chunks, axis=0, ignore_index=True)
        else:
            out = out.iloc[0:0].copy()

    if float(args.max_total_ratio) > 0.0 and len(out) > 0:
        n_cap = int(np.floor(float(args.max_total_ratio) * len(df)))
        n_cap = max(n_cap, 1)
        if len(out) > n_cap:
            # Keep highest-confidence rows; tie-break with deterministic shuffle.
            out = out.sample(frac=1.0, random_state=args.seed).sort_values("pseudo_prob", ascending=False).head(n_cap)

    out = out.sort_values("pseudo_prob", ascending=False).reset_index(drop=True)
    out.to_csv(out_path, index=False)

    by_cls = out["bird_group"].value_counts().to_dict()
    print(f"[OK] pseudo labels saved: {out_path}", flush=True)
    print(f"rows={len(out)} min_prob={args.min_prob} topk_per_class={args.topk_per_class} max_total_ratio={args.max_total_ratio}", flush=True)
    print(f"pseudo_prob_mean={float(out['pseudo_prob'].mean()):.6f} min={float(out['pseudo_prob'].min()):.6f} max={float(out['pseudo_prob'].max()):.6f}", flush=True)
    print(f"class_counts={by_cls}", flush=True)


if __name__ == "__main__":
    main()
