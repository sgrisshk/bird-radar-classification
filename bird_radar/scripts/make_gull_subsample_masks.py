#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create train masks with Gull subsampling for bagging.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--keep-rates", type=str, default="0.25,0.35,0.50")
    p.add_argument("--n-masks-per-rate", type=int, default=2)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--target-class", type=str, default="Gulls")
    p.add_argument("--stratify-cols", type=str, default="month,radar_bird_size")
    p.add_argument("--timestamp-col", type=str, default="timestamp_start_radar_utc")
    return p.parse_args()


def _parse_rates(s: str) -> list[float]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        v = float(x)
        if not (0.0 < v <= 1.0):
            raise ValueError(f"keep-rate out of range (0,1]: {v}")
        vals.append(v)
    if not vals:
        raise ValueError("no valid keep-rates parsed")
    return vals


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_csv)
    if "bird_group" not in train.columns:
        raise ValueError("train.csv must contain bird_group")

    if "month" not in train.columns:
        ts = pd.to_datetime(train[args.timestamp_col], errors="coerce", utc=True)
        if ts.isna().any():
            raise ValueError(f"failed to parse some timestamps from {args.timestamp_col}")
        train["month"] = ts.dt.month.astype(int)

    rates = _parse_rates(args.keep_rates)
    strat_cols = [c.strip() for c in args.stratify_cols.split(",") if c.strip()]
    for c in strat_cols:
        if c not in train.columns:
            raise ValueError(f"stratify column not found: {c}")

    target_mask = train["bird_group"].eq(args.target_class).to_numpy()
    n = len(train)
    all_idx = np.arange(n, dtype=np.int64)
    target_idx = all_idx[target_mask]
    non_target_idx = all_idx[~target_mask]

    rows: list[dict[str, object]] = []
    train_target = train.loc[target_mask, strat_cols].copy()
    if len(train_target) == 0:
        raise ValueError(f"target class not found in train: {args.target_class}")

    for rate in rates:
        rate_tag = int(round(rate * 100))
        for m in range(int(args.n_masks_per_rate)):
            rng = np.random.RandomState(int(args.seed) + rate_tag * 100 + m)
            keep = np.zeros(n, dtype=bool)
            keep[non_target_idx] = True

            # Stratified subsample of target class.
            selected_target_parts: list[np.ndarray] = []
            for _, grp in train_target.groupby(strat_cols, dropna=False):
                idx_local = grp.index.to_numpy(dtype=np.int64)
                k = int(round(len(idx_local) * rate))
                if rate > 0 and len(idx_local) > 0:
                    k = max(1, min(len(idx_local), k))
                if k <= 0:
                    continue
                sel = rng.choice(idx_local, size=k, replace=False)
                selected_target_parts.append(np.asarray(sel, dtype=np.int64))

            selected_target = (
                np.concatenate(selected_target_parts, axis=0)
                if selected_target_parts
                else np.zeros((0,), dtype=np.int64)
            )
            keep[selected_target] = True

            name = f"mask_gulls_keep{rate_tag:02d}_m{m}.npy"
            path = out_dir / name
            np.save(path, keep.astype(np.bool_))

            rows.append(
                {
                    "mask_path": str(path),
                    "keep_rate_target": float(rate),
                    "mask_id": int(m),
                    "selected_total": int(keep.sum()),
                    "selected_ratio_total": float(keep.mean()),
                    "selected_target": int(keep[target_idx].sum()),
                    "target_total": int(len(target_idx)),
                    "selected_target_ratio": float(keep[target_idx].mean()),
                    "selected_non_target": int(keep[non_target_idx].sum()),
                    "non_target_total": int(len(non_target_idx)),
                }
            )

    manifest = pd.DataFrame(rows).sort_values(["keep_rate_target", "mask_id"])
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"wrote masks: {len(manifest)}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()

