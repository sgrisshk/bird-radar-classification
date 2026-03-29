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

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds


def _parse_grid(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    arr = np.array(vals, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("grid is empty")
    return arr


def _macro_map(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    vals: list[float] = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        vals.append(0.0 if yt.sum() <= 0 else float(average_precision_score(yt, yp)))
    return float(np.mean(vals))


def _align_oof(csv_path: str, track_ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=["track_id", *CLASSES]).set_index("track_id")
    return df.loc[track_ids, CLASSES].to_numpy(dtype=np.float32)


def _align_test_pair(forward_csv: str, reverse_csv: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = pd.read_csv(forward_csv, usecols=["track_id", *CLASSES])
    r = pd.read_csv(reverse_csv, usecols=["track_id", *CLASSES])
    m = f.merge(r, on="track_id", suffixes=("_f", "_r"), how="inner")
    if len(m) != len(f) or len(m) != len(r):
        raise ValueError("track_id mismatch between forward/reverse test csv")
    track_ids = m["track_id"].to_numpy(dtype=np.int64)
    pf = m[[f"{c}_f" for c in CLASSES]].to_numpy(dtype=np.float32)
    pr = m[[f"{c}_r" for c in CLASSES]].to_numpy(dtype=np.float32)
    return track_ids, pf, pr


def _fold_eval(
    y: np.ndarray,
    baseline: np.ndarray,
    pred: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, float, list[dict[str, float | int]]]:
    rows: list[dict[str, float | int]] = []
    for fid, (_, va_idx) in enumerate(folds):
        yt = y[va_idx]
        pb = baseline[va_idx]
        pr = pred[va_idx]
        covered = (pr.sum(axis=1) > 0.0)
        if int(covered.sum()) == 0:
            rows.append(
                {
                    "fold": int(fid),
                    "n_cov": 0,
                    "baseline_macro": 0.0,
                    "pred_macro": 0.0,
                    "fold_delta": 0.0,
                }
            )
            continue
        yt = yt[covered]
        pb = pb[covered]
        pr = pr[covered]
        b = _macro_map(yt, pb)
        p = _macro_map(yt, pr)
        rows.append(
            {
                "fold": int(fid),
                "n_cov": int(covered.sum()),
                "baseline_macro": float(b),
                "pred_macro": float(p),
                "fold_delta": float(p - b),
            }
        )
    valid = [r for r in rows if int(r["n_cov"]) > 0]
    if not valid:
        return 0.0, 0.0, 0.0, rows
    b_mean = float(np.mean([float(r["baseline_macro"]) for r in valid]))
    p_mean = float(np.mean([float(r["pred_macro"]) for r in valid]))
    min_delta = float(np.min([float(r["fold_delta"]) for r in valid]))
    return b_mean, p_mean, min_delta, rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agreement / asymmetry gated forward-reverse blending.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--forward-oof-csv", required=True)
    p.add_argument("--reverse-oof-csv", required=True)
    p.add_argument("--forward-test-csv", required=True)
    p.add_argument("--reverse-test-csv", required=True)
    p.add_argument("--reference-oof-csv", default="", help="Optional baseline for gains; default is forward-oof-csv.")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--time-col", default="timestamp_start_radar_utc")
    p.add_argument("--group-col", default="observation_id")
    p.add_argument("--n-splits", type=int, default=5)

    p.add_argument("--mode", choices=["agreement", "asymmetry"], default="agreement")
    p.add_argument("--metric", choices=["l1_mean", "sym_kl"], default="l1_mean")
    p.add_argument("--w-grid", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--t-grid", default="0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20")
    p.add_argument("--m-grid", default="0.00,0.02,0.04,0.06,0.08,0.10,0.12", help="Used only for mode=asymmetry.")

    p.add_argument("--min-fold-delta", type=float, default=-0.002)
    p.add_argument("--target-fold-index", type=int, default=-1, help="-1 disables target-fold constraint.")
    p.add_argument("--max-target-fold-drop", type=float, default=-0.002)
    p.add_argument("--min-override", type=float, default=0.0)
    p.add_argument("--max-override", type=float, default=1.0)
    return p.parse_args()


def _disagreement(pf: np.ndarray, pr: np.ndarray, metric: str) -> np.ndarray:
    if metric == "l1_mean":
        return np.mean(np.abs(pf - pr), axis=1).astype(np.float32)
    eps = 1e-6
    p = np.clip(pf, eps, 1.0)
    q = np.clip(pr, eps, 1.0)
    kl_pq = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    kl_qp = np.sum(q * (np.log(q) - np.log(p)), axis=1)
    return (kl_pq + kl_qp).astype(np.float32)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    w_grid = _parse_grid(args.w_grid)
    t_grid = _parse_grid(args.t_grid)
    m_grid = _parse_grid(args.m_grid) if str(args.mode) == "asymmetry" else np.array([0.0], dtype=np.float32)

    train_df = pd.read_csv(
        args.train_csv,
        usecols=["track_id", "bird_group", str(args.time_col), str(args.group_col)],
    )
    train_ids = train_df["track_id"].to_numpy(dtype=np.int64)
    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = np.zeros((len(train_df), len(CLASSES)), dtype=np.float32)
    y[np.arange(len(train_df)), y_idx] = 1.0

    cv_df = pd.DataFrame(
        {
            "_cv_ts": train_df[str(args.time_col)],
            "_cv_group": train_df[str(args.group_col)].astype(np.int64),
        }
    )
    folds = make_forward_temporal_group_folds(cv_df, timestamp_col="_cv_ts", group_col="_cv_group", n_splits=int(args.n_splits))

    p_f = _align_oof(args.forward_oof_csv, train_ids)
    p_r = _align_oof(args.reverse_oof_csv, train_ids)
    baseline = _align_oof(args.reference_oof_csv, train_ids) if str(args.reference_oof_csv).strip() else p_f
    d = _disagreement(p_f, p_r, str(args.metric))
    maxf = np.max(p_f, axis=1)
    maxr = np.max(p_r, axis=1)

    rows: list[dict[str, float]] = []
    feasible: list[dict[str, float]] = []
    target_fold = int(args.target_fold_index)
    for w in w_grid:
        mix = np.clip((1.0 - w) * p_f + w * p_r, 0.0, 1.0)
        for t in t_grid:
            for m in m_grid:
                if str(args.mode) == "agreement":
                    gate = d <= float(t)
                else:
                    gate = (d <= float(t)) & ((maxr - maxf) >= float(m))
                pred = p_f.copy()
                pred[gate] = mix[gate]
                b_mean, p_mean, min_delta, fold_rows = _fold_eval(y, baseline, pred, folds)
                gain = p_mean - b_mean
                fold_delta_target = 0.0
                if target_fold >= 0:
                    match = [r for r in fold_rows if int(r["fold"]) == target_fold and int(r["n_cov"]) > 0]
                    fold_delta_target = float(match[0]["fold_delta"]) if match else 0.0

                row = {
                    "w_reverse": float(w),
                    "t": float(t),
                    "m": float(m),
                    "override_frac": float(gate.mean()),
                    "baseline_mean": float(b_mean),
                    "pred_mean": float(p_mean),
                    "gain": float(gain),
                    "min_fold_delta": float(min_delta),
                    "target_fold_delta": float(fold_delta_target),
                }
                rows.append(row)

                ok = (
                    row["min_fold_delta"] >= float(args.min_fold_delta)
                    and row["override_frac"] >= float(args.min_override)
                    and row["override_frac"] <= float(args.max_override)
                )
                if target_fold >= 0:
                    ok = ok and (row["target_fold_delta"] >= float(args.max_target_fold_drop))
                if ok:
                    feasible.append(row)

    scan_df = pd.DataFrame(rows).sort_values(["gain", "min_fold_delta"], ascending=[False, False]).reset_index(drop=True)
    scan_path = out_dir / "scan.csv"
    scan_df.to_csv(scan_path, index=False)

    if feasible:
        best = sorted(feasible, key=lambda r: (r["gain"], r["min_fold_delta"]), reverse=True)[0]
        status = "GO"
    else:
        best = rows[np.argmax([r["gain"] for r in rows])]
        status = "NO_FEASIBLE"

    # Materialize best predictions.
    w = float(best["w_reverse"])
    t = float(best["t"])
    m = float(best["m"])
    mix_oof = np.clip((1.0 - w) * p_f + w * p_r, 0.0, 1.0)
    if str(args.mode) == "agreement":
        gate_oof = d <= t
    else:
        gate_oof = (d <= t) & ((maxr - maxf) >= m)
    pred_oof = p_f.copy()
    pred_oof[gate_oof] = mix_oof[gate_oof]

    oof_df = pd.DataFrame({"track_id": train_ids})
    for j, c in enumerate(CLASSES):
        oof_df[c] = pred_oof[:, j]
    oof_path = out_dir / "oof_best.csv"
    oof_df.to_csv(oof_path, index=False)

    test_ids, f_test, r_test = _align_test_pair(args.forward_test_csv, args.reverse_test_csv)
    mix_test = np.clip((1.0 - w) * f_test + w * r_test, 0.0, 1.0)
    d_test = _disagreement(f_test, r_test, str(args.metric))
    if str(args.mode) == "agreement":
        gate_test = d_test <= t
    else:
        gate_test = (d_test <= t) & ((np.max(r_test, axis=1) - np.max(f_test, axis=1)) >= m)
    pred_test = f_test.copy()
    pred_test[gate_test] = mix_test[gate_test]
    sub_df = pd.DataFrame({"track_id": test_ids})
    for j, c in enumerate(CLASSES):
        sub_df[c] = pred_test[:, j]
    sub_path = out_dir / "sub_best.csv"
    sub_df.to_csv(sub_path, index=False)

    # Fold report for best.
    _, _, _, best_fold_rows = _fold_eval(y, baseline, pred_oof, folds)
    report = {
        "status": status,
        "mode": str(args.mode),
        "metric": str(args.metric),
        "n_total": int(len(rows)),
        "n_feasible": int(len(feasible)),
        "best": best,
        "constraints": {
            "min_fold_delta": float(args.min_fold_delta),
            "target_fold_index": int(target_fold),
            "max_target_fold_drop": float(args.max_target_fold_drop),
            "min_override": float(args.min_override),
            "max_override": float(args.max_override),
        },
        "best_fold_rows": best_fold_rows,
        "scan_csv": str(scan_path),
        "output_oof": str(oof_path),
        "output_submission": str(sub_path),
    }
    rep_path = out_dir / "report.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(
        f"status={status} n_feasible={len(feasible)}/{len(rows)} "
        f"gain={float(best['gain']):+.6f} min_fold_delta={float(best['min_fold_delta']):+.6f} "
        f"target_fold_delta={float(best['target_fold_delta']):+.6f} override_frac={float(best['override_frac']):.4f}",
        flush=True,
    )
    print(str(sub_path), flush=True)
    print(str(rep_path), flush=True)


if __name__ == "__main__":
    main()
