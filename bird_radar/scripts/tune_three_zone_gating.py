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
from src.cv import make_forward_temporal_group_folds
from src.metrics import macro_map_score, one_hot_labels


def _parse_grid(s: str, cast=float) -> list[float]:
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def _check(df: pd.DataFrame, name: str, id_col: str) -> None:
    req = [id_col, *CLASSES]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing columns: {miss}")


def _blend_three_zone(
    pb: np.ndarray,
    pt: np.ndarray,
    t_low: float,
    t_high: float,
    a_low: float,
    a_mid: float,
    a_high: float,
) -> np.ndarray:
    conf = np.max(pb, axis=1)
    alpha = np.full((len(pb),), float(a_mid), dtype=np.float32)
    alpha[conf < float(t_low)] = float(a_low)
    alpha[conf >= float(t_high)] = float(a_high)
    return np.clip(pb * alpha[:, None] + pt * (1.0 - alpha[:, None]), 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune 3-zone confidence-gated blend on OOF.")
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--base-oof-csv", type=str, required=True)
    p.add_argument("--temporal-oof-csv", type=str, required=True)
    p.add_argument("--id-col", type=str, default="track_id")
    p.add_argument("--label-col", type=str, default="bird_group")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--max-folds", type=int, default=3)
    p.add_argument("--threshold-low-grid", type=str, default="0.50,0.55,0.60")
    p.add_argument("--threshold-high-grid", type=str, default="0.75,0.80,0.85,0.90")
    p.add_argument("--alpha-low-grid", type=str, default="0.80,0.85,0.90,0.93")
    p.add_argument("--alpha-mid-grid", type=str, default="0.94,0.96,0.97,0.98")
    p.add_argument("--alpha-high-grid", type=str, default="0.99,1.00")
    p.add_argument("--output-json", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(
        args.train_csv,
        usecols=[args.id_col, args.label_col, "timestamp_start_radar_utc", "observation_id"],
    )
    base = pd.read_csv(args.base_oof_csv)
    temp = pd.read_csv(args.temporal_oof_csv)
    _check(base, "base-oof-csv", args.id_col)
    _check(temp, "temporal-oof-csv", args.id_col)

    merged = (
        train[[args.id_col, args.label_col, "timestamp_start_radar_utc", "observation_id"]]
        .merge(base[[args.id_col, *CLASSES]], on=args.id_col, how="inner", suffixes=("", "_base"))
        .merge(temp[[args.id_col, *CLASSES]], on=args.id_col, how="inner", suffixes=("_base", "_temp"))
        .copy()
    )
    if len(merged) == 0:
        raise RuntimeError("No overlapping rows between train and OOF predictions")

    y_idx = merged[args.label_col].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = one_hot_labels(y_idx, len(CLASSES)).astype(np.float32)
    pb = merged[[f"{c}_base" for c in CLASSES]].to_numpy(dtype=np.float32)
    pt = merged[[f"{c}_temp" for c in CLASSES]].to_numpy(dtype=np.float32)

    cv_df = pd.DataFrame(
        {
            "_cv_ts": merged["timestamp_start_radar_utc"],
            "_cv_group": merged["observation_id"].astype(np.int64),
        }
    )
    folds = make_forward_temporal_group_folds(
        cv_df, timestamp_col="_cv_ts", group_col="_cv_group", n_splits=int(args.n_splits)
    )
    if int(args.max_folds) > 0:
        folds = folds[: int(args.max_folds)]

    tl_grid = _parse_grid(args.threshold_low_grid, float)
    th_grid = _parse_grid(args.threshold_high_grid, float)
    al_grid = _parse_grid(args.alpha_low_grid, float)
    am_grid = _parse_grid(args.alpha_mid_grid, float)
    ah_grid = _parse_grid(args.alpha_high_grid, float)

    teacher_fold = [float(macro_map_score(y[va], pb[va])) for _, va in folds]
    teacher_mean = float(np.mean(teacher_fold))

    best = {
        "score": -1.0,
        "params": None,
        "fold_scores": [],
    }
    n_eval = 0
    for t_low in tl_grid:
        for t_high in th_grid:
            if t_low >= t_high:
                continue
            for a_low in al_grid:
                for a_mid in am_grid:
                    for a_high in ah_grid:
                        if not (a_low <= a_mid <= a_high):
                            continue
                        n_eval += 1
                        fs: list[float] = []
                        for _, va in folds:
                            pred = _blend_three_zone(pb[va], pt[va], t_low, t_high, a_low, a_mid, a_high)
                            fs.append(float(macro_map_score(y[va], pred)))
                        m = float(np.mean(fs))
                        if m > float(best["score"]):
                            best["score"] = m
                            best["params"] = {
                                "threshold_low": float(t_low),
                                "threshold_high": float(t_high),
                                "alpha_low": float(a_low),
                                "alpha_mid": float(a_mid),
                                "alpha_high": float(a_high),
                            }
                            best["fold_scores"] = [float(v) for v in fs]

    out = {
        "n_rows": int(len(merged)),
        "n_eval": int(n_eval),
        "teacher_fold_scores": teacher_fold,
        "teacher_mean": teacher_mean,
        "best_mean": float(best["score"]),
        "best_gain_vs_teacher": float(float(best["score"]) - teacher_mean),
        "best_fold_scores": best["fold_scores"],
        "best_params": best["params"],
    }

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=True), flush=True)
    print(str(out_path), flush=True)


if __name__ == "__main__":
    main()
