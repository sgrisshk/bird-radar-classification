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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build group-CVaR sample weights from OOF probs.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--oof-npy", required=True)
    p.add_argument("--track-ids-npy", required=True)
    p.add_argument("--groups", default="month,radar_bird_size")
    p.add_argument("--focus-classes", default="Cormorants,Waders,Ducks")
    p.add_argument("--eta", type=float, default=0.8)
    p.add_argument("--tail-quantile", type=float, default=0.75)
    p.add_argument("--clip-min", type=float, default=0.4)
    p.add_argument("--clip-max", type=float, default=1.8)
    p.add_argument("--output-parquet", required=True)
    p.add_argument("--output-json", default="")
    return p.parse_args()


def _bce(y_bin: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-6, 1.0 - 1e-6)
    y = y_bin.astype(np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def main() -> None:
    args = _parse_args()
    train_path = Path(args.train_csv).resolve()
    oof_path = Path(args.oof_npy).resolve()
    ids_path = Path(args.track_ids_npy).resolve()
    out_path = Path(args.output_parquet).resolve()

    train = pd.read_csv(train_path)
    oof = np.asarray(np.load(oof_path), dtype=np.float64)
    track_ids = np.asarray(np.load(ids_path), dtype=np.int64)

    if oof.ndim != 2 or oof.shape[1] != len(CLASSES):
        raise ValueError(f"oof-npy shape must be [N,{len(CLASSES)}], got {oof.shape}")
    if oof.shape[0] != len(track_ids):
        raise ValueError(f"oof rows != track_ids rows: {oof.shape[0]} vs {len(track_ids)}")

    keys = [k.strip() for k in str(args.groups).split(",") if k.strip()]
    if not keys:
        raise ValueError("--groups must not be empty")
    focus_classes = [c.strip() for c in str(args.focus_classes).split(",") if c.strip()]
    if not focus_classes:
        raise ValueError("--focus-classes must not be empty")
    missing_cls = [c for c in focus_classes if c not in CLASSES]
    if missing_cls:
        raise ValueError(f"unknown classes in --focus-classes: {missing_cls}")

    merge_cols = ["track_id", "bird_group", "radar_bird_size", "timestamp_start_radar_utc"]
    missing_cols = [c for c in merge_cols if c not in train.columns]
    if missing_cols:
        raise ValueError(f"train-csv missing columns: {missing_cols}")

    oof_df = pd.DataFrame({"track_id": track_ids})
    for j, cls in enumerate(CLASSES):
        oof_df[cls] = oof[:, j].astype(np.float32)

    df = train[merge_cols].merge(oof_df, on="track_id", how="inner")
    if len(df) != len(train):
        raise ValueError(
            f"track alignment failed: merged={len(df)} train={len(train)} "
            f"(missing={len(train)-len(df)})"
        )

    df["month"] = pd.to_datetime(df["timestamp_start_radar_utc"], errors="coerce", utc=True).dt.month.fillna(-1).astype(np.int32)
    y_idx = df["bird_group"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy(dtype=np.int64)

    losses: list[np.ndarray] = []
    for cls in focus_classes:
        ci = CLASSES.index(cls)
        y_bin = (y_idx == ci).astype(np.int32)
        p = df[cls].to_numpy(dtype=np.float64)
        losses.append(_bce(y_bin, p))
    focus_loss = np.mean(np.vstack(losses), axis=0)
    df["focus_loss"] = focus_loss.astype(np.float64)

    for k in keys:
        if k not in df.columns:
            raise ValueError(f"group key not found: {k}")
    grp = df.groupby(keys, dropna=False)["focus_loss"].mean().reset_index().rename(columns={"focus_loss": "group_loss"})

    g_loss = grp["group_loss"].to_numpy(dtype=np.float64)
    tq = float(np.clip(args.tail_quantile, 0.5, 0.99))
    q_loss = float(np.quantile(g_loss, tq))
    tail_mask = g_loss >= q_loss
    cvar_tail_mean = float(np.mean(g_loss[tail_mask])) if np.any(tail_mask) else float(np.mean(g_loss))
    denom = max(cvar_tail_mean, 1e-6)

    # Upweight only worst-tail groups; keep others near 1.0.
    excess = np.maximum(g_loss - q_loss, 0.0)
    raw_group_w = 1.0 + float(args.eta) * (excess / denom)
    raw_group_w = np.clip(raw_group_w, float(args.clip_min), float(args.clip_max))
    grp["group_weight_raw"] = raw_group_w

    df = df.merge(grp[keys + ["group_weight_raw"]], on=keys, how="left")
    w = df["group_weight_raw"].to_numpy(dtype=np.float64)
    w = np.clip(w, float(args.clip_min), float(args.clip_max))
    w = w / max(np.mean(w), 1e-12)
    w = np.clip(w, float(args.clip_min), float(args.clip_max))
    df["sample_weight"] = w.astype(np.float32)

    out = df[["track_id", "sample_weight"]].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    grp_sorted = grp.sort_values("group_loss", ascending=False).copy()
    report = {
        "train_csv": str(train_path),
        "oof_npy": str(oof_path),
        "track_ids_npy": str(ids_path),
        "groups": keys,
        "focus_classes": focus_classes,
        "eta": float(args.eta),
        "tail_quantile": float(tq),
        "q_loss": float(q_loss),
        "cvar_tail_mean": float(cvar_tail_mean),
        "clip_min": float(args.clip_min),
        "clip_max": float(args.clip_max),
        "n_rows": int(len(df)),
        "weights_summary": {
            "mean": float(np.mean(w)),
            "std": float(np.std(w)),
            "min": float(np.min(w)),
            "max": float(np.max(w)),
            "p50": float(np.quantile(w, 0.5)),
            "p90": float(np.quantile(w, 0.9)),
            "p95": float(np.quantile(w, 0.95)),
            "p99": float(np.quantile(w, 0.99)),
        },
        "group_loss_top20": grp_sorted.head(20).to_dict(orient="records"),
        "group_loss_bottom20": grp_sorted.tail(20).to_dict(orient="records"),
        "output_parquet": str(out_path),
    }

    report_path = Path(args.output_json).resolve() if str(args.output_json).strip() else out_path.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps({"output_parquet": str(out_path), "output_json": str(report_path), "n_rows": len(out)}, ensure_ascii=True))


if __name__ == "__main__":
    main()

