#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds


def _one_hot(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.zeros((len(y_idx), n_classes), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    return y


def _macro_map(y_true: np.ndarray, y_prob: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    vals: list[float] = []
    for i in range(y_true.shape[1]):
        if float(y_true[:, i].sum()) <= 0.0:
            vals.append(0.0)
            continue
        vals.append(float(average_precision_score(y_true[:, i], y_prob[:, i], sample_weight=sample_weight)))
    return float(np.mean(vals))


def _align_from_csv(path: str | Path, ids_order: np.ndarray, id_col: str) -> np.ndarray:
    df = pd.read_csv(path, usecols=[id_col, *CLASSES])
    mp = {int(r[id_col]): np.asarray([r[c] for c in CLASSES], dtype=np.float32) for _, r in df.iterrows()}
    missing = [int(t) for t in ids_order.tolist() if int(t) not in mp]
    if missing:
        raise ValueError(f"{path} missing {len(missing)} ids; first={missing[:10]}")
    return np.stack([mp[int(t)] for t in ids_order.tolist()], axis=0).astype(np.float32)


def _align_from_npy(pred_path: str | Path, ids_path: str | Path, ids_order: np.ndarray) -> np.ndarray:
    pred = np.load(pred_path).astype(np.float32)
    ids = np.load(ids_path).astype(np.int64)
    if len(pred) != len(ids):
        raise ValueError(f"len(pred) != len(ids): {len(pred)} vs {len(ids)}")
    if np.array_equal(ids, ids_order):
        return pred
    pos = {int(t): i for i, t in enumerate(ids.tolist())}
    missing = [int(t) for t in ids_order.tolist() if int(t) not in pos]
    if missing:
        raise ValueError(f"{pred_path} ids missing {len(missing)} rows; first={missing[:10]}")
    return np.stack([pred[pos[int(t)]] for t in ids_order.tolist()], axis=0).astype(np.float32)


def _parse_float_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _topk_mask(score: np.ndarray, q: float) -> np.ndarray:
    n = int(score.shape[0])
    k = int(round(float(q) * n))
    if k <= 0:
        return np.zeros((n,), dtype=bool)
    k = min(k, n)
    idx = np.argpartition(score, -k)[-k:]
    mask = np.zeros((n,), dtype=bool)
    mask[idx] = True
    return mask


def _apply_corm_override(
    teacher: np.ndarray,
    new: np.ndarray,
    class_idx: int,
    q: float,
    alpha: float,
    p_teacher_min: float,
    p_teacher_max: float,
    p_new_min: float,
    score_center: float,
) -> tuple[np.ndarray, np.ndarray]:
    p_t = teacher[:, class_idx]
    p_n = new[:, class_idx]
    delta = p_n - p_t
    center_term = 1.0 - np.abs(p_t - float(score_center))
    center_term = np.clip(center_term, 0.0, None)
    score = delta * center_term
    cand = (
        (delta > 0.0)
        & (p_t >= float(p_teacher_min))
        & (p_t <= float(p_teacher_max))
        & (p_n >= float(p_new_min))
    )
    score = np.where(cand, score, -1e12).astype(np.float64)

    top = _topk_mask(score, q=float(q))
    mask = top & cand

    out = teacher.copy()
    out[mask, class_idx] = (1.0 - float(alpha)) * p_t[mask] + float(alpha) * p_n[mask]
    return out, mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cormorants-only selective override scan (teacher <-/-> new).")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--teacher-oof-csv", required=True)
    p.add_argument("--new-oof-npy", required=True)
    p.add_argument("--new-track-ids-npy", required=True)
    p.add_argument("--teacher-test-csv", required=True)
    p.add_argument("--new-test-csv", required=True)
    p.add_argument("--sample-weights-npy", default="")
    p.add_argument("--out-dir", required=True)

    p.add_argument("--id-col", default="track_id")
    p.add_argument("--time-col", default="timestamp_start_radar_utc")
    p.add_argument("--group-col", default="observation_id")
    p.add_argument("--class-name", default="Cormorants")
    p.add_argument("--n-splits", type=int, default=5)

    p.add_argument("--q-grid", default="0.005,0.01,0.02,0.03")
    p.add_argument("--alpha-grid", default="1.0,0.7,0.5")
    p.add_argument("--p-teacher-min", type=float, default=0.02)
    p.add_argument("--p-teacher-max-grid", default="0.25,0.35")
    p.add_argument("--p-new-min-grid", default="0.10,0.15,0.20")
    p.add_argument("--score-center", type=float, default=0.15)

    p.add_argument("--constraint-min-fold-delta", type=float, default=-0.002)
    p.add_argument("--constraint-min-fold-worst", type=float, default=0.20)
    p.add_argument("--constraint-min-corm-delta", type=float, default=-0.01)
    p.add_argument("--constraint-min-gain", type=float, default=0.03)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(args.class_name) not in CLASSES:
        raise ValueError(f"class-name must be one of {CLASSES}")
    class_idx = int(CLASSES.index(str(args.class_name)))

    train_df = pd.read_csv(
        args.train_csv,
        usecols=[str(args.id_col), "bird_group", str(args.time_col), str(args.group_col)],
    )
    train_ids = train_df[str(args.id_col)].to_numpy(dtype=np.int64)
    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = _one_hot(y_idx, len(CLASSES))

    teacher_oof = _align_from_csv(args.teacher_oof_csv, train_ids, id_col=str(args.id_col))
    new_oof = _align_from_npy(args.new_oof_npy, args.new_track_ids_npy, train_ids)

    w_train: np.ndarray | None = None
    if str(args.sample_weights_npy).strip():
        w_train = np.asarray(np.load(args.sample_weights_npy), dtype=np.float32).reshape(-1)
        if len(w_train) != len(train_ids):
            raise ValueError(f"sample weights len mismatch: {len(w_train)} vs {len(train_ids)}")
        w_train = np.clip(w_train, 1e-8, None)

    cv_df = train_df.rename(columns={str(args.time_col): "_cv_ts", str(args.group_col): "_cv_group"})
    folds = make_forward_temporal_group_folds(
        cv_df, timestamp_col="_cv_ts", group_col="_cv_group", n_splits=int(args.n_splits)
    )

    q_grid = _parse_float_grid(args.q_grid)
    alpha_grid = _parse_float_grid(args.alpha_grid)
    p_t_max_grid = _parse_float_grid(args.p_teacher_max_grid)
    p_n_min_grid = _parse_float_grid(args.p_new_min_grid)
    if not q_grid or not alpha_grid or not p_t_max_grid or not p_n_min_grid:
        raise ValueError("empty grid detected")

    # baselines
    teacher_fold_uw = []
    teacher_fold_aw = []
    fold_w_mass = []
    for _, va_idx in folds:
        yt = y[va_idx]
        te = teacher_oof[va_idx]
        sw = w_train[va_idx] if w_train is not None else None
        teacher_fold_uw.append(_macro_map(yt, te))
        teacher_fold_aw.append(_macro_map(yt, te, sample_weight=sw))
        fold_w_mass.append(float(sw.sum()) if sw is not None else float(len(va_idx)))
    teacher_mean_uw = float(np.mean(teacher_fold_uw))
    teacher_mean_aw = float(np.average(np.asarray(teacher_fold_aw), weights=np.asarray(fold_w_mass)))

    rows: list[dict[str, Any]] = []
    best_any: dict[str, Any] | None = None
    best_feasible: dict[str, Any] | None = None
    for q in q_grid:
        for alpha in alpha_grid:
            for p_t_max in p_t_max_grid:
                for p_n_min in p_n_min_grid:
                    pred_oof, use_mask = _apply_corm_override(
                        teacher=teacher_oof,
                        new=new_oof,
                        class_idx=class_idx,
                        q=float(q),
                        alpha=float(alpha),
                        p_teacher_min=float(args.p_teacher_min),
                        p_teacher_max=float(p_t_max),
                        p_new_min=float(p_n_min),
                        score_center=float(args.score_center),
                    )
                    fold_blend_uw = []
                    fold_blend_aw = []
                    fold_delta_uw = []
                    fold_delta_aw = []
                    for f_id, (_, va_idx) in enumerate(folds):
                        yt = y[va_idx]
                        pr = pred_oof[va_idx]
                        te = teacher_oof[va_idx]
                        sw = w_train[va_idx] if w_train is not None else None
                        b_uw = _macro_map(yt, pr)
                        t_uw = float(teacher_fold_uw[f_id])
                        b_aw = _macro_map(yt, pr, sample_weight=sw)
                        t_aw = float(teacher_fold_aw[f_id])
                        fold_blend_uw.append(b_uw)
                        fold_blend_aw.append(b_aw)
                        fold_delta_uw.append(b_uw - t_uw)
                        fold_delta_aw.append(b_aw - t_aw)

                    blend_mean_uw = float(np.mean(fold_blend_uw))
                    blend_mean_aw = float(np.average(np.asarray(fold_blend_aw), weights=np.asarray(fold_w_mass)))
                    gain_uw = float(blend_mean_uw - teacher_mean_uw)
                    gain_aw = float(blend_mean_aw - teacher_mean_aw)
                    min_fold_delta_uw = float(np.min(fold_delta_uw))
                    min_fold_delta_aw = float(np.min(fold_delta_aw))
                    fold_worst_uw = float(np.min(fold_blend_uw))
                    fold_worst_aw = float(np.min(fold_blend_aw))

                    c_t = teacher_oof[:, class_idx]
                    c_p = pred_oof[:, class_idx]
                    yc = y[:, class_idx]
                    c_ap_t = float(average_precision_score(yc, c_t))
                    c_ap_b = float(average_precision_score(yc, c_p))
                    c_delta = float(c_ap_b - c_ap_t)
                    if w_train is not None:
                        c_ap_t_w = float(average_precision_score(yc, c_t, sample_weight=w_train))
                        c_ap_b_w = float(average_precision_score(yc, c_p, sample_weight=w_train))
                    else:
                        c_ap_t_w = c_ap_t
                        c_ap_b_w = c_ap_b
                    c_delta_w = float(c_ap_b_w - c_ap_t_w)

                    rec = {
                        "q": float(q),
                        "alpha": float(alpha),
                        "p_teacher_min": float(args.p_teacher_min),
                        "p_teacher_max": float(p_t_max),
                        "p_new_min": float(p_n_min),
                        "score_center": float(args.score_center),
                        "override_frac_oof": float(use_mask.mean()),
                        "gain_uw": gain_uw,
                        "gain_adv": gain_aw,
                        "blend_mean_uw": blend_mean_uw,
                        "blend_mean_adv": blend_mean_aw,
                        "min_fold_delta_uw": min_fold_delta_uw,
                        "min_fold_delta_adv": min_fold_delta_aw,
                        "fold_worst_uw": fold_worst_uw,
                        "fold_worst_adv": fold_worst_aw,
                        "class_delta_ap": c_delta,
                        "class_delta_ap_weighted": c_delta_w,
                        # backward-compatible alias
                        "cormorants_delta_ap": c_delta,
                        "cormorants_delta_ap_weighted": c_delta_w,
                    }
                    rows.append(rec)

                    if best_any is None or rec["gain_uw"] > best_any["gain_uw"]:
                        best_any = rec
                    feasible = (
                        rec["min_fold_delta_uw"] >= float(args.constraint_min_fold_delta)
                        and rec["fold_worst_uw"] >= float(args.constraint_min_fold_worst)
                        and rec["class_delta_ap"] >= float(args.constraint_min_corm_delta)
                    )
                    if feasible and (best_feasible is None or rec["gain_uw"] > best_feasible["gain_uw"]):
                        best_feasible = rec

    scan_df = pd.DataFrame(rows).sort_values(["gain_uw", "min_fold_delta_uw"], ascending=[False, False]).reset_index(drop=True)
    scan_path = out_dir / "scan.csv"
    scan_df.to_csv(scan_path, index=False)

    chosen = best_feasible if best_feasible is not None else best_any
    if chosen is None:
        raise RuntimeError("scan produced no rows")

    # Build best OOF and test predictions
    best_oof, best_mask_oof = _apply_corm_override(
        teacher=teacher_oof,
        new=new_oof,
        class_idx=class_idx,
        q=float(chosen["q"]),
        alpha=float(chosen["alpha"]),
        p_teacher_min=float(chosen["p_teacher_min"]),
        p_teacher_max=float(chosen["p_teacher_max"]),
        p_new_min=float(chosen["p_new_min"]),
        score_center=float(chosen["score_center"]),
    )
    oof_best = pd.DataFrame({str(args.id_col): train_ids})
    for ci, c in enumerate(CLASSES):
        oof_best[c] = best_oof[:, ci].astype(np.float32)
    oof_best_path = out_dir / "oof_best.csv"
    oof_best.to_csv(oof_best_path, index=False)

    # test
    teacher_test_df = pd.read_csv(args.teacher_test_csv)
    test_ids = teacher_test_df[str(args.id_col)].to_numpy(dtype=np.int64)
    teacher_test = _align_from_csv(args.teacher_test_csv, test_ids, id_col=str(args.id_col))
    new_test = _align_from_csv(args.new_test_csv, test_ids, id_col=str(args.id_col))
    best_test, best_mask_test = _apply_corm_override(
        teacher=teacher_test,
        new=new_test,
        class_idx=class_idx,
        q=float(chosen["q"]),
        alpha=float(chosen["alpha"]),
        p_teacher_min=float(chosen["p_teacher_min"]),
        p_teacher_max=float(chosen["p_teacher_max"]),
        p_new_min=float(chosen["p_new_min"]),
        score_center=float(chosen["score_center"]),
    )
    sub_best = pd.DataFrame({str(args.id_col): test_ids})
    for ci, c in enumerate(CLASSES):
        sub_best[c] = best_test[:, ci].astype(np.float32)
    sub_best_path = out_dir / "sub_best.csv"
    sub_best.to_csv(sub_best_path, index=False)

    # report
    go_flag = (
        float(chosen["min_fold_delta_uw"]) >= float(args.constraint_min_fold_delta)
        and float(chosen["fold_worst_uw"]) >= float(args.constraint_min_fold_worst)
        and float(chosen["class_delta_ap"]) >= float(args.constraint_min_corm_delta)
        and float(chosen["gain_uw"]) >= float(args.constraint_min_gain)
    )
    report = {
        "class_name": str(args.class_name),
        "class_idx": int(class_idx),
        "teacher_mean_uw": float(teacher_mean_uw),
        "teacher_mean_adv": float(teacher_mean_aw),
        "constraints": {
            "min_fold_delta": float(args.constraint_min_fold_delta),
            "min_fold_worst": float(args.constraint_min_fold_worst),
            "min_corm_delta_ap": float(args.constraint_min_corm_delta),
            "min_gain": float(args.constraint_min_gain),
        },
        "best_any": best_any,
        "best_feasible": best_feasible,
        "chosen": chosen,
        "go": bool(go_flag),
        "best_override_frac_oof": float(best_mask_oof.mean()),
        "best_override_frac_test": float(best_mask_test.mean()),
        "scan_csv": str(scan_path.resolve()),
        "oof_best_csv": str(oof_best_path.resolve()),
        "sub_best_csv": str(sub_best_path.resolve()),
        "n_scan": int(len(rows)),
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(
        f"chosen gain_uw={float(chosen['gain_uw']):+.6f} "
        f"min_fold_delta_uw={float(chosen['min_fold_delta_uw']):+.6f} "
        f"fold_worst_uw={float(chosen['fold_worst_uw']):.6f} "
        f"class_delta_ap={float(chosen['class_delta_ap']):+.6f} "
        f"override_oof={float(best_mask_oof.mean()):.4f} "
        f"override_test={float(best_mask_test.mean()):.4f} "
        f"go={go_flag}",
        flush=True,
    )
    print(str(report_path.resolve()), flush=True)


if __name__ == "__main__":
    main()
