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

from config import CLASS_TO_INDEX, CLASSES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build train sample weights from consensus OOF disagreement.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--oof-csvs", required=True, help="Comma-separated OOF CSV paths (must contain track_id + class cols).")
    p.add_argument("--focus-classes", default="Cormorants,Waders,Ducks")
    p.add_argument("--low-true-q", type=float, default=0.10)
    p.add_argument("--high-alt-q", type=float, default=0.90)
    p.add_argument("--downweight", type=float, default=0.30)
    p.add_argument("--output-parquet", required=True)
    p.add_argument("--output-json", default="")
    return p.parse_args()


def _parse_list(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _read_aligned_oof(path: str, track_ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path, usecols=["track_id", *CLASSES])
    mp = {int(r["track_id"]): np.asarray([r[c] for c in CLASSES], dtype=np.float32) for _, r in df.iterrows()}
    missing = [int(t) for t in track_ids if int(t) not in mp]
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} track_ids, first={missing[:10]}")
    return np.stack([mp[int(t)] for t in track_ids], axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    out_parquet = Path(args.output_parquet).resolve()
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.output_json).resolve() if str(args.output_json).strip() else out_parquet.with_suffix(".json")

    train = pd.read_csv(args.train_csv)
    if "track_id" not in train.columns or "bird_group" not in train.columns:
        raise ValueError("train.csv must contain track_id and bird_group")

    track_ids = train["track_id"].to_numpy(dtype=np.int64)
    y_idx = train["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    if np.isnan(y_idx).any():
        raise ValueError("found unknown class labels in train.csv")

    oof_paths = _parse_list(args.oof_csvs)
    if len(oof_paths) < 2:
        raise ValueError("need at least 2 OOF CSVs for consensus")
    probs_list = [_read_aligned_oof(p, track_ids) for p in oof_paths]
    probs = np.stack(probs_list, axis=0)  # [m, n, c]

    n_models, n_rows, n_classes = probs.shape
    rows = np.arange(n_rows, dtype=np.int64)
    true_prob_each = np.take_along_axis(probs, y_idx[None, :, None], axis=2).squeeze(2)  # [m, n]
    mean_true = true_prob_each.mean(axis=0)

    tmp = probs.copy()
    tmp[np.arange(n_models)[:, None], rows[None, :], y_idx[None, :]] = -np.inf
    alt_max_each = tmp.max(axis=2)  # [m, n]
    mean_alt = alt_max_each.mean(axis=0)

    weights = np.ones((n_rows,), dtype=np.float32)
    flagged = np.zeros((n_rows,), dtype=bool)

    focus_classes = _parse_list(args.focus_classes)
    per_class_stats: dict[str, dict[str, float | int]] = {}
    for cls in focus_classes:
        if cls not in CLASS_TO_INDEX:
            raise ValueError(f"unknown focus class: {cls}")
        ci = int(CLASS_TO_INDEX[cls])
        mask = y_idx == ci
        n_cls = int(mask.sum())
        if n_cls == 0:
            per_class_stats[cls] = {"n": 0, "q_true": float("nan"), "q_alt": float("nan"), "flagged": 0}
            continue

        q_true = float(np.quantile(mean_true[mask], float(args.low_true_q)))
        q_alt = float(np.quantile(mean_alt[mask], float(args.high_alt_q)))
        bad = mask & (mean_true <= q_true) & (mean_alt >= q_alt)
        flagged |= bad

        per_class_stats[cls] = {
            "n": n_cls,
            "q_true": q_true,
            "q_alt": q_alt,
            "flagged": int(bad.sum()),
            "flagged_ratio": float(bad.mean()),
        }

    downweight = float(args.downweight)
    if not (0.0 < downweight <= 1.0):
        raise ValueError("--downweight must be in (0,1]")
    weights[flagged] = downweight

    out_df = pd.DataFrame(
        {
            "track_id": track_ids.astype(np.int64),
            "sample_weight": weights.astype(np.float32),
        }
    )
    out_df.to_parquet(out_parquet, index=False)

    # Human-readable shortlist of most suspicious tracks.
    susp_idx = np.where(flagged)[0]
    susp_order = susp_idx[np.argsort(mean_true[susp_idx])] if len(susp_idx) > 0 else np.array([], dtype=np.int64)
    top = []
    for i in susp_order[:30]:
        top.append(
            {
                "track_id": int(track_ids[i]),
                "bird_group": str(train.iloc[i]["bird_group"]),
                "mean_true_prob": float(mean_true[i]),
                "mean_alt_prob": float(mean_alt[i]),
            }
        )

    report = {
        "train_csv": str(Path(args.train_csv).resolve()),
        "oof_csvs": [str(Path(p).resolve()) for p in oof_paths],
        "focus_classes": focus_classes,
        "low_true_q": float(args.low_true_q),
        "high_alt_q": float(args.high_alt_q),
        "downweight": downweight,
        "n_rows": int(n_rows),
        "n_models": int(n_models),
        "flagged_total": int(flagged.sum()),
        "flagged_ratio": float(flagged.mean()),
        "weights_summary": {
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "min": float(weights.min()),
            "max": float(weights.max()),
        },
        "per_class_stats": per_class_stats,
        "top_suspicious_tracks": top,
        "output_parquet": str(out_parquet),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_parquet": str(out_parquet), "output_json": str(out_json), "flagged_total": int(flagged.sum())}))


if __name__ == "__main__":
    main()
