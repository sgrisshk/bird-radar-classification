from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import parse_ewkb_linestring_zm_hex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=str, default="/Users/sgrisshk/Desktop/AI-task/train.csv")
    parser.add_argument("--n", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_csv = Path(args.train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {train_csv}")

    df = pd.read_csv(train_csv, nrows=int(args.n))
    if len(df) == 0:
        raise RuntimeError("empty dataframe")

    non_constant = 0
    for row in df.itertuples(index=False):
        track_id = int(getattr(row, "track_id"))
        lon, lat, alt, rcs = parse_ewkb_linestring_zm_hex(getattr(row, "trajectory"), track_id=track_id)
        _ = lon, lat, alt

        rcs = np.asarray(rcs, dtype=np.float64)
        is_const = bool(np.nanmax(rcs) - np.nanmin(rcs) < 1e-9)
        if not is_const:
            non_constant += 1

        print(
            f"track_id={track_id} points={len(rcs)} "
            f"rcs_min={float(np.min(rcs)):.6f} rcs_max={float(np.max(rcs)):.6f} rcs_mean={float(np.mean(rcs)):.6f} "
            f"constant={is_const}"
        )

    if non_constant < 3:
        raise AssertionError(f"Expected >=3 non-constant RCS tracks in first {len(df)}, got {non_constant}")

    print(f"OK: non_constant_rcs_tracks={non_constant}/{len(df)}")


if __name__ == "__main__":
    main()
