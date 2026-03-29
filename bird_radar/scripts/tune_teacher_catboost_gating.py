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
from src.cv import make_forward_temporal_group_folds
from src.metrics import macro_map_score, one_hot_labels


def _parse_grid(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _entropy_rowwise(p: np.ndarray) -> np.ndarray:
    q = np.clip(p.astype(np.float64), 1e-12, 1.0)
    h = -np.sum(q * np.log(q), axis=1)
    h /= np.log(float(p.shape[1]))
    return h.astype(np.float32)


def _margin_rowwise(p: np.ndarray) -> np.ndarray:
    s = np.sort(p, axis=1)
    return (s[:, -1] - s[:, -2]).astype(np.float32)


def _zones_three_rule(
    teacher_prob: np.ndarray,
    t_low: float,
    t_high: float,
    margin_low: float,
    margin_high: float,
    entropy_low: float,
    entropy_high: float,
) -> np.ndarray:
    conf = np.max(teacher_prob, axis=1)
    margin = _margin_rowwise(teacher_prob)
    ent = _entropy_rowwise(teacher_prob)

    low = (conf < float(t_low)) | (margin < float(margin_low)) | (ent > float(entropy_high))
    high = (conf >= float(t_high)) & (margin >= float(margin_high)) & (ent <= float(entropy_low))
    zone = np.full((len(teacher_prob),), 1, dtype=np.int32)  # mid
    zone[low] = 0
    zone[high] = 2
    return zone


def _blend_by_zone(
    teacher_prob: np.ndarray,
    seq_prob: np.ndarray,
    zone: np.ndarray,
    alpha_low: float,
    alpha_mid: float,
    alpha_high: float,
) -> np.ndarray:
    alpha = np.full((len(zone),), float(alpha_mid), dtype=np.float32)
    alpha[zone == 0] = float(alpha_low)
    alpha[zone == 2] = float(alpha_high)
    return np.clip(teacher_prob * alpha[:, None] + seq_prob * (1.0 - alpha[:, None]), 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune robust 3-zone teacher-vs-catboost gating on forward OOF.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--teacher-oof-csv", required=True)
    p.add_argument("--seq-oof-npy", required=True)
    p.add_argument("--seq-track-ids-npy", required=True)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--max-folds", type=int, default=0)

    p.add_argument("--t-low-grid", type=str, default="0.45,0.50,0.55")
    p.add_argument("--t-high-grid", type=str, default="0.70,0.75,0.80")
    p.add_argument("--alpha-low-grid", type=str, default="0.80,0.85,0.90")
    p.add_argument("--alpha-mid-grid", type=str, default="0.95,0.98,0.99")
    p.add_argument("--alpha-high-grid", type=str, default="1.00")
    p.add_argument("--margin-low-grid", type=str, default="0.08,0.12")
    p.add_argument("--margin-high-grid", type=str, default="0.20,0.25")
    p.add_argument("--entropy-low-grid", type=str, default="0.52,0.58")
    p.add_argument("--entropy-high-grid", type=str, default="0.68,0.74")

    p.add_argument("--max-low-frac", type=float, default=0.25)
    p.add_argument("--min-worst-delta", type=float, default=-0.01)
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train = pd.read_csv(
        args.train_csv,
        usecols=["track_id", "bird_group", "timestamp_start_radar_utc", "observation_id"],
    )
    y_idx = train["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = one_hot_labels(y_idx, len(CLASSES)).astype(np.float32)
    train_ids = train["track_id"].to_numpy(dtype=np.int64)

    teacher_df = pd.read_csv(args.teacher_oof_csv, usecols=["track_id", *CLASSES]).set_index("track_id")
    teacher_prob = teacher_df.loc[train_ids, CLASSES].to_numpy(dtype=np.float32)

    seq_oof = np.load(args.seq_oof_npy).astype(np.float32)
    seq_ids = np.load(args.seq_track_ids_npy).astype(np.int64)
    if len(seq_ids) != len(train_ids):
        raise ValueError(f"seq ids length mismatch: {len(seq_ids)} vs train {len(train_ids)}")
    if not np.array_equal(seq_ids, train_ids):
        pos = {int(t): i for i, t in enumerate(seq_ids)}
        missing = [int(t) for t in train_ids if int(t) not in pos]
        if missing:
            raise ValueError(f"seq ids missing {len(missing)} train ids; first={missing[:10]}")
        seq_oof = np.stack([seq_oof[pos[int(t)]] for t in train_ids], axis=0).astype(np.float32)

    cv_df = pd.DataFrame(
        {
            "_cv_ts": train["timestamp_start_radar_utc"],
            "_cv_group": train["observation_id"].astype(np.int64),
        }
    )
    folds = make_forward_temporal_group_folds(
        cv_df,
        timestamp_col="_cv_ts",
        group_col="_cv_group",
        n_splits=int(args.n_splits),
    )
    if int(args.max_folds) > 0:
        folds = folds[: int(args.max_folds)]

    teacher_fold_scores = [float(macro_map_score(y[va], teacher_prob[va])) for _, va in folds]
    teacher_mean = float(np.mean(teacher_fold_scores))

    t_low_grid = _parse_grid(args.t_low_grid)
    t_high_grid = _parse_grid(args.t_high_grid)
    a_low_grid = _parse_grid(args.alpha_low_grid)
    a_mid_grid = _parse_grid(args.alpha_mid_grid)
    a_high_grid = _parse_grid(args.alpha_high_grid)
    m_low_grid = _parse_grid(args.margin_low_grid)
    m_high_grid = _parse_grid(args.margin_high_grid)
    e_low_grid = _parse_grid(args.entropy_low_grid)
    e_high_grid = _parse_grid(args.entropy_high_grid)

    best_any: dict[str, object] | None = None
    best_feasible: dict[str, object] | None = None
    n_eval = 0

    for t_low in t_low_grid:
        for t_high in t_high_grid:
            if t_low >= t_high:
                continue
            for m_low in m_low_grid:
                for m_high in m_high_grid:
                    if m_low >= m_high:
                        continue
                    for e_low in e_low_grid:
                        for e_high in e_high_grid:
                            if e_low >= e_high:
                                continue
                            zone_all = _zones_three_rule(
                                teacher_prob=teacher_prob,
                                t_low=t_low,
                                t_high=t_high,
                                margin_low=m_low,
                                margin_high=m_high,
                                entropy_low=e_low,
                                entropy_high=e_high,
                            )
                            low_frac = float(np.mean(zone_all == 0))
                            mid_frac = float(np.mean(zone_all == 1))
                            high_frac = float(np.mean(zone_all == 2))

                            for a_low in a_low_grid:
                                for a_mid in a_mid_grid:
                                    for a_high in a_high_grid:
                                        if not (a_low <= a_mid <= a_high):
                                            continue
                                        n_eval += 1
                                        fold_scores: list[float] = []
                                        for _, va in folds:
                                            pred = _blend_by_zone(
                                                teacher_prob=teacher_prob[va],
                                                seq_prob=seq_oof[va],
                                                zone=zone_all[va],
                                                alpha_low=a_low,
                                                alpha_mid=a_mid,
                                                alpha_high=a_high,
                                            )
                                            fold_scores.append(float(macro_map_score(y[va], pred)))

                                        mean_score = float(np.mean(fold_scores))
                                        deltas = [float(fold_scores[i] - teacher_fold_scores[i]) for i in range(len(folds))]
                                        worst_delta = float(np.min(deltas))
                                        row = {
                                            "params": {
                                                "threshold_low": float(t_low),
                                                "threshold_high": float(t_high),
                                                "margin_low": float(m_low),
                                                "margin_high": float(m_high),
                                                "entropy_low": float(e_low),
                                                "entropy_high": float(e_high),
                                                "alpha_low": float(a_low),
                                                "alpha_mid": float(a_mid),
                                                "alpha_high": float(a_high),
                                            },
                                            "mean_score": float(mean_score),
                                            "gain_vs_teacher": float(mean_score - teacher_mean),
                                            "fold_scores": [float(v) for v in fold_scores],
                                            "fold_deltas_vs_teacher": deltas,
                                            "worst_delta_vs_teacher": float(worst_delta),
                                            "zone_fraction": {
                                                "low": low_frac,
                                                "mid": mid_frac,
                                                "high": high_frac,
                                            },
                                        }

                                        if best_any is None or float(row["mean_score"]) > float(best_any["mean_score"]):
                                            best_any = row

                                        feasible = (
                                            low_frac <= float(args.max_low_frac)
                                            and worst_delta >= float(args.min_worst_delta)
                                        )
                                        if feasible and (
                                            best_feasible is None
                                            or float(row["mean_score"]) > float(best_feasible["mean_score"])
                                        ):
                                            best_feasible = row

    out = {
        "n_rows": int(len(train_ids)),
        "n_eval": int(n_eval),
        "teacher_fold_scores": [float(v) for v in teacher_fold_scores],
        "teacher_mean": float(teacher_mean),
        "constraints": {
            "max_low_frac": float(args.max_low_frac),
            "min_worst_delta": float(args.min_worst_delta),
        },
        "best_any": best_any,
        "best_feasible": best_feasible,
    }

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    if best_feasible is not None:
        b = best_feasible
        print(
            f"teacher_mean={teacher_mean:.6f} best_mean={float(b['mean_score']):.6f} "
            f"gain={float(b['gain_vs_teacher']):.6f} "
            f"worst_delta={float(b['worst_delta_vs_teacher']):.6f} "
            f"low_frac={float(b['zone_fraction']['low']):.4f}",
            flush=True,
        )
    elif best_any is not None:
        b = best_any
        print(
            f"teacher_mean={teacher_mean:.6f} best_any_mean={float(b['mean_score']):.6f} "
            f"gain={float(b['gain_vs_teacher']):.6f} "
            f"worst_delta={float(b['worst_delta_vs_teacher']):.6f} "
            f"low_frac={float(b['zone_fraction']['low']):.4f} (no feasible under constraints)",
            flush=True,
        )
    print(str(out_path), flush=True)


if __name__ == "__main__":
    main()
