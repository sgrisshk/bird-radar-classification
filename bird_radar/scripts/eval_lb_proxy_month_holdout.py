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

from config import CLASSES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LB-proxy month holdout metrics from OOF probabilities.")
    p.add_argument("--train-csv", type=str, default="train.csv")
    p.add_argument("--oof-csv", type=str, required=True, help="CSV with track_id + class probability columns.")
    p.add_argument("--output-json", type=str, default="")
    return p.parse_args()


def macro_ap(y_true: np.ndarray, probs: np.ndarray) -> float:
    vals: list[float] = []
    for i in range(len(CLASSES)):
        yt = y_true[:, i]
        if yt.sum() > 0:
            vals.append(float(average_precision_score(yt, probs[:, i])))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_csv)
    train["month"] = pd.to_datetime(train["timestamp_start_radar_utc"]).dt.month

    oof = pd.read_csv(args.oof_csv)
    need = {"track_id", *CLASSES}
    miss = [c for c in need if c not in oof.columns]
    if miss:
        raise ValueError(f"oof-csv missing columns: {miss}")

    merged = train[["track_id", "bird_group", "month"]].merge(
        oof[["track_id", *CLASSES]], on="track_id", how="left"
    )
    merged[list(CLASSES)] = merged[list(CLASSES)].fillna(0.0)

    y_idx = merged["bird_group"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy(dtype=np.int64)
    y = np.zeros((len(merged), len(CLASSES)), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    p = merged[list(CLASSES)].to_numpy(dtype=np.float32)
    months = merged["month"].to_numpy(dtype=np.int32)

    m4 = months == 4
    m9 = months == 9
    m10 = months == 10
    mstress = months >= 9

    out = {
        "oof_csv": str(Path(args.oof_csv).resolve()),
        "n_rows": int(len(merged)),
        "metric_month4_macro_ap": macro_ap(y[m4], p[m4]) if np.any(m4) else None,
        "metric_month9_macro_ap": macro_ap(y[m9], p[m9]) if np.any(m9) else None,
        "metric_month10_macro_ap": macro_ap(y[m10], p[m10]) if np.any(m10) else None,
        "metric_monthstress_9plus_macro_ap": macro_ap(y[mstress], p[mstress]) if np.any(mstress) else None,
    }
    vals = [out["metric_month4_macro_ap"], out["metric_month9_macro_ap"]]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    out["lb_proxy_m4_m9_mean"] = float(np.mean(vals)) if vals else None

    print(json.dumps(out, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
