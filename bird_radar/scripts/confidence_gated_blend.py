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
    p = argparse.ArgumentParser(description="Confidence-gated blend of two submission CSVs.")
    p.add_argument("--base-csv", type=str, required=True)
    p.add_argument("--temporal-csv", type=str, required=True)
    p.add_argument("--output-csv", type=str, required=True)
    p.add_argument("--id-col", type=str, default="track_id")
    p.add_argument("--mode", type=str, default="three_zone", choices=["two_zone", "three_zone"])
    p.add_argument("--threshold", type=float, default=0.70, help="(two_zone) Base max-prob threshold.")
    p.add_argument("--threshold-low", type=float, default=0.55, help="(three_zone) low confidence threshold.")
    p.add_argument("--threshold-high", type=float, default=0.80, help="(three_zone) high confidence threshold.")
    p.add_argument("--alpha-high", type=float, default=0.99, help="Base weight for high confidence regime.")
    p.add_argument("--alpha-mid", type=float, default=0.95, help="(three_zone) Base weight for mid confidence regime.")
    p.add_argument("--alpha-low", type=float, default=0.85, help="Base weight for low confidence regime.")
    p.add_argument(
        "--margin-low",
        type=float,
        default=-1.0,
        help="Optional low-zone condition: margin < margin-low. Negative disables.",
    )
    p.add_argument(
        "--margin-high",
        type=float,
        default=-1.0,
        help="Optional high-zone condition: margin >= margin-high. Negative disables.",
    )
    p.add_argument(
        "--entropy-low",
        type=float,
        default=-1.0,
        help="Optional high-zone condition: entropy <= entropy-low. Negative disables.",
    )
    p.add_argument(
        "--entropy-high",
        type=float,
        default=-1.0,
        help="Optional low-zone condition: entropy > entropy-high. Negative disables.",
    )
    p.add_argument("--power-gamma", type=float, default=1.0, help="Optional post power-normalization gamma; 1.0 disables.")
    p.add_argument("--normalize-rows", action="store_true", default=False, help="Normalize each row to sum=1 after power.")
    return p.parse_args()


def _check(df: pd.DataFrame, name: str, id_col: str) -> None:
    req = [id_col, *CLASSES]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing columns: {miss}")


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.base_csv)
    temp = pd.read_csv(args.temporal_csv)
    _check(base, "base-csv", args.id_col)
    _check(temp, "temporal-csv", args.id_col)

    if len(base) != len(temp) or not np.array_equal(base[args.id_col].to_numpy(), temp[args.id_col].to_numpy()):
        raise RuntimeError("track_id mismatch between base and temporal")

    pb = np.clip(base[CLASSES].to_numpy(dtype=np.float32), 0.0, 1.0)
    pt = np.clip(temp[CLASSES].to_numpy(dtype=np.float32), 0.0, 1.0)
    conf = np.max(pb, axis=1)
    sorted_probs = np.sort(pb, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    ent = (-np.sum(np.clip(pb, 1e-12, 1.0) * np.log(np.clip(pb, 1e-12, 1.0)), axis=1) / np.log(float(pb.shape[1]))).astype(np.float32)

    if args.mode == "two_zone":
        alpha = np.where(conf >= float(args.threshold), float(args.alpha_high), float(args.alpha_low)).astype(np.float32)
        zone = np.where(conf >= float(args.threshold), 2, 0).astype(np.int32)  # 2=high, 0=low
    else:
        t_low = float(args.threshold_low)
        t_high = float(args.threshold_high)
        if t_low >= t_high:
            raise ValueError("threshold-low must be < threshold-high")
        alpha = np.full_like(conf, float(args.alpha_mid), dtype=np.float32)
        zone = np.full_like(conf, 1, dtype=np.int32)  # 1=mid
        low_mask = conf < t_low
        high_mask = conf >= t_high
        if float(args.margin_low) >= 0.0:
            low_mask = low_mask | (margin < float(args.margin_low))
        if float(args.entropy_high) >= 0.0:
            low_mask = low_mask | (ent > float(args.entropy_high))
        if float(args.margin_high) >= 0.0:
            high_mask = high_mask & (margin >= float(args.margin_high))
        if float(args.entropy_low) >= 0.0:
            high_mask = high_mask & (ent <= float(args.entropy_low))
        alpha[low_mask] = float(args.alpha_low)
        alpha[high_mask] = float(args.alpha_high)
        zone[low_mask] = 0
        zone[high_mask] = 2

    pred = np.clip(pb * alpha[:, None] + pt * (1.0 - alpha[:, None]), 0.0, 1.0)

    if float(args.power_gamma) != 1.0:
        pred = np.power(np.clip(pred, 1e-12, 1.0), float(args.power_gamma)).astype(np.float32)
    if args.normalize_rows:
        pred = pred / (np.sum(pred, axis=1, keepdims=True) + 1e-12)

    out = base[[args.id_col]].copy()
    out[CLASSES] = pred
    out_path = Path(args.output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    report = {
        "base_csv": str(Path(args.base_csv).resolve()),
        "temporal_csv": str(Path(args.temporal_csv).resolve()),
        "output_csv": str(out_path),
        "mode": str(args.mode),
        "threshold": float(args.threshold),
        "threshold_low": float(args.threshold_low),
        "threshold_high": float(args.threshold_high),
        "alpha_high": float(args.alpha_high),
        "alpha_mid": float(args.alpha_mid),
        "alpha_low": float(args.alpha_low),
        "margin_low": float(args.margin_low),
        "margin_high": float(args.margin_high),
        "entropy_low": float(args.entropy_low),
        "entropy_high": float(args.entropy_high),
        "power_gamma": float(args.power_gamma),
        "normalize_rows": bool(args.normalize_rows),
        "n_rows": int(len(out)),
        "gating": {
            "high_count": int(np.sum(zone == 2)),
            "mid_count": int(np.sum(zone == 1)),
            "low_count": int(np.sum(zone == 0)),
        },
        "mean_probs": {cls: float(np.mean(pred[:, i])) for i, cls in enumerate(CLASSES)},
    }
    (out_path.parent / "confidence_gated_blend_report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(str(out_path), flush=True)


if __name__ == "__main__":
    main()
