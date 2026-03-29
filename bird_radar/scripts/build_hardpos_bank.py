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
    p = argparse.ArgumentParser(description="Build hard-positive bank from OOF predictions.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--oof-npy", required=True)
    p.add_argument("--track-ids-npy", required=True)
    p.add_argument("--focus-classes", default="Cormorants,Waders,Ducks,Pigeons,Geese")
    p.add_argument("--bottomq", type=float, default=0.20, help="Bottom quantile on positive p_c for hard positives.")
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def _parse_list(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    out_json = Path(args.output_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_csv, usecols=["track_id", "bird_group"])
    track_ids = np.asarray(np.load(args.track_ids_npy), dtype=np.int64)
    oof = np.asarray(np.load(args.oof_npy), dtype=np.float32)
    if oof.ndim != 2 or oof.shape[1] != len(CLASSES):
        raise ValueError(f"oof shape mismatch: {oof.shape}, expected (*,{len(CLASSES)})")
    if len(track_ids) != len(oof):
        raise ValueError(f"track_ids len {len(track_ids)} != oof rows {len(oof)}")

    truth_map = {int(r.track_id): str(r.bird_group) for r in train.itertuples(index=False)}
    y_name = np.array([truth_map[int(t)] for t in track_ids], dtype=object)
    y_idx = np.array([CLASS_TO_INDEX[n] for n in y_name], dtype=np.int64)

    bottomq = float(args.bottomq)
    if not (0.0 < bottomq < 1.0):
        raise ValueError("--bottomq must be in (0,1)")

    focus_classes = _parse_list(args.focus_classes)
    bank: dict[str, list[int]] = {}
    stats: dict[str, dict[str, float | int]] = {}

    for cls in focus_classes:
        if cls not in CLASS_TO_INDEX:
            raise ValueError(f"unknown class: {cls}")
        c = int(CLASS_TO_INDEX[cls])
        pos_mask = y_idx == c
        p_pos = oof[pos_mask, c]
        if len(p_pos) <= 0:
            bank[cls] = []
            stats[cls] = {"n_pos": 0, "threshold": float("nan"), "selected": 0}
            continue
        thr = float(np.quantile(p_pos, bottomq))
        sel = pos_mask & (oof[:, c] <= thr)
        ids = track_ids[sel].astype(np.int64).tolist()
        bank[cls] = ids
        stats[cls] = {
            "n_pos": int(pos_mask.sum()),
            "threshold": float(thr),
            "selected": int(len(ids)),
            "selected_ratio": float(len(ids) / max(1, int(pos_mask.sum()))),
        }

    report = {
        "train_csv": str(Path(args.train_csv).resolve()),
        "oof_npy": str(Path(args.oof_npy).resolve()),
        "track_ids_npy": str(Path(args.track_ids_npy).resolve()),
        "focus_classes": focus_classes,
        "bottomq": bottomq,
        "stats": stats,
        "bank": bank,
    }
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(out_json), "stats": stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()

