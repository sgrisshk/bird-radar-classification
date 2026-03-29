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
from src.redesign.utils import macro_map, per_class_ap


def _rank01(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(x), dtype=np.float32)
    if len(x) > 1:
        ranks /= float(len(x) - 1)
    return ranks


def _parse_grid(s: str) -> list[float]:
    s = (s or "").strip()
    if not s:
        return []
    vals: list[float] = []
    for part in s.split(","):
        x = float(part.strip())
        if x <= 0.0 or x >= 1.0:
            raise ValueError(f"weight must be in (0,1), got {x}")
        vals.append(x)
    return sorted(set(vals))


def _parse_classes_csv(s: str) -> set[str]:
    s = (s or "").strip()
    if not s:
        return set()
    out: set[str] = set()
    for part in s.split(","):
        cls = part.strip()
        if not cls:
            continue
        if cls not in CLASSES:
            raise ValueError(f"unknown class in --freeze-classes: {cls}")
        out.add(cls)
    return out


def _blend_col(teacher_col: np.ndarray, new_col: np.ndarray, w_new: float, mode: str) -> np.ndarray:
    if mode == "rank_mean":
        y = (1.0 - w_new) * _rank01(teacher_col) + w_new * _rank01(new_col)
        return np.clip(y, 0.0, 1.0).astype(np.float32)
    y = (1.0 - w_new) * teacher_col + w_new * new_col
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _macro_map_weighted(y_true: np.ndarray, y_prob: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    if sample_weight is None:
        return float(macro_map(y_true, y_prob))
    vals: list[float] = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_prob[:, j]
        if yt.sum() <= 0:
            vals.append(0.0)
        else:
            vals.append(float(average_precision_score(yt, yp, sample_weight=sample_weight)))
    return float(np.mean(vals))


def _per_class_ap_weighted(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    if sample_weight is None:
        return per_class_ap(y_true, y_prob)
    out: dict[str, float] = {}
    for j, cls in enumerate(CLASSES):
        yt = y_true[:, j]
        yp = y_prob[:, j]
        out[cls] = float(average_precision_score(yt, yp, sample_weight=sample_weight)) if yt.sum() > 0 else 0.0
    return out


def _align_probs_from_csv(csv_path: str, ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=["track_id", *CLASSES])
    mp = {int(r["track_id"]): np.asarray([r[c] for c in CLASSES], dtype=np.float32) for _, r in df.iterrows()}
    missing = [int(t) for t in ids if int(t) not in mp]
    if missing:
        raise ValueError(f"{csv_path} missing {len(missing)} ids; first={missing[:10]}")
    return np.stack([mp[int(t)] for t in ids], axis=0).astype(np.float32)


def _align_npy_probs_to_ids(probs: np.ndarray, src_ids: np.ndarray, target_ids: np.ndarray, label: str) -> np.ndarray:
    if probs.shape[0] != len(src_ids):
        raise ValueError(f"{label}: probs rows {probs.shape[0]} != src_ids {len(src_ids)}")
    row_map = {int(t): i for i, t in enumerate(src_ids.astype(np.int64).tolist())}
    miss = [int(t) for t in target_ids if int(t) not in row_map]
    if miss:
        raise ValueError(f"{label}: missing {len(miss)} ids; first={miss[:10]}")
    idx = np.asarray([row_map[int(t)] for t in target_ids], dtype=np.int64)
    return probs[idx].astype(np.float32)


def _make_fold_ids(
    train_df: pd.DataFrame, ids_order: np.ndarray, n_splits: int, time_col: str, group_col: str
) -> np.ndarray:
    cv_df = pd.DataFrame({"_cv_ts": train_df[str(time_col)], "_cv_group": train_df[str(group_col)].astype(np.int64)})
    folds = make_forward_temporal_group_folds(cv_df, timestamp_col="_cv_ts", group_col="_cv_group", n_splits=int(n_splits))
    fold_id_full = np.full(len(train_df), -1, dtype=np.int64)
    for k, (_, va_idx) in enumerate(folds):
        fold_id_full[np.asarray(va_idx, dtype=np.int64)] = int(k)

    row_map = dict(zip(train_df["track_id"].astype(np.int64).tolist(), np.arange(len(train_df), dtype=np.int64).tolist()))
    return np.asarray([fold_id_full[row_map[int(t)]] for t in ids_order], dtype=np.int64)


def _eval_metrics(
    y: np.ndarray,
    teacher: np.ndarray,
    pred: np.ndarray,
    fold_id: np.ndarray,
    start_fold: int,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    mask_valid = fold_id >= int(start_fold)
    yv = y[mask_valid]
    tv = teacher[mask_valid]
    pv = pred[mask_valid]
    wv = sample_weight[mask_valid] if sample_weight is not None else None
    fids = sorted(int(x) for x in np.unique(fold_id[mask_valid]))

    teacher_fold = []
    pred_fold = []
    fold_weight = []
    for f in fids:
        m = mask_valid & (fold_id == f)
        wm = sample_weight[m] if sample_weight is not None else None
        teacher_fold.append(float(_macro_map_weighted(y[m], teacher[m], sample_weight=wm)))
        pred_fold.append(float(_macro_map_weighted(y[m], pred[m], sample_weight=wm)))
        fold_weight.append(float(np.sum(wm)) if wm is not None else float(np.sum(m)))

    ap_teacher = _per_class_ap_weighted(yv, tv, sample_weight=wv)
    ap_pred = _per_class_ap_weighted(yv, pv, sample_weight=wv)
    ap_delta = {c: float(ap_pred[c] - ap_teacher[c]) for c in CLASSES}

    macro_teacher = float(_macro_map_weighted(yv, tv, sample_weight=wv))
    macro_pred = float(_macro_map_weighted(yv, pv, sample_weight=wv))
    if teacher_fold:
        if sample_weight is not None:
            fw = np.asarray(fold_weight, dtype=np.float64)
            fold_mean_teacher = float(np.average(teacher_fold, weights=fw))
            fold_mean_pred = float(np.average(pred_fold, weights=fw))
        else:
            fold_mean_teacher = float(np.mean(teacher_fold))
            fold_mean_pred = float(np.mean(pred_fold))
    else:
        fold_mean_teacher = 0.0
        fold_mean_pred = 0.0
    return {
        "macro_teacher": macro_teacher,
        "macro_pred": macro_pred,
        "macro_gain": float(macro_pred - macro_teacher),
        "fold_ids": fids,
        "teacher_fold_scores": teacher_fold,
        "pred_fold_scores": pred_fold,
        "fold_weights": fold_weight,
        "fold_mean_teacher": fold_mean_teacher,
        "fold_mean_pred": fold_mean_pred,
        "fold_mean_gain": float(fold_mean_pred - fold_mean_teacher),
        "min_fold_delta": float(np.min(np.asarray(pred_fold) - np.asarray(teacher_fold))) if teacher_fold else 0.0,
        "ap_delta": ap_delta,
        "n_valid": int(mask_valid.sum()),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Greedy constrained per-class weight search across multiple candidates.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--teacher-oof-csv", required=True)

    p.add_argument("--candidate-oof-npy", action="append", required=True)
    p.add_argument("--candidate-track-ids-npy", action="append", required=True)
    p.add_argument("--candidate-name", action="append", default=[])

    p.add_argument("--teacher-test-csv", default=None)
    p.add_argument("--candidate-test-csv", action="append", default=[])

    p.add_argument("--time-col", default="timestamp_start_radar_utc")
    p.add_argument("--group-col", default="observation_id")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--start-fold", type=int, default=1)

    p.add_argument("--mode", choices=["prob_mean", "rank_mean"], default="prob_mean")
    p.add_argument("--ap-threshold", type=float, default=0.0015)
    p.add_argument("--class-min-ap-gain", type=float, default=0.008)
    p.add_argument("--min-fold-delta", type=float, default=-0.002)
    p.add_argument("--max-change-frac", type=float, default=0.35)
    p.add_argument("--change-threshold", type=float, default=0.2)
    p.add_argument("--weights-grid", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    p.add_argument("--sample-weights-npy", default="")
    p.add_argument("--freeze-classes", default="")

    p.add_argument("--output-csv", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-oof-npy", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    w_grid = _parse_grid(args.weights_grid)
    if not w_grid:
        raise ValueError("weights-grid is empty")
    freeze_classes = _parse_classes_csv(args.freeze_classes)

    cand_oof_paths = [str(Path(p).resolve()) for p in (args.candidate_oof_npy or [])]
    cand_ids_paths = [str(Path(p).resolve()) for p in (args.candidate_track_ids_npy or [])]
    if len(cand_oof_paths) != len(cand_ids_paths):
        raise ValueError("--candidate-oof-npy and --candidate-track-ids-npy counts must match")
    n_cand = len(cand_oof_paths)
    if n_cand <= 0:
        raise ValueError("need at least one candidate")

    if args.candidate_name and len(args.candidate_name) != n_cand:
        raise ValueError("--candidate-name count must match candidates")
    cand_names = args.candidate_name if args.candidate_name else [f"cand{i+1}" for i in range(n_cand)]

    train_df = pd.read_csv(args.train_csv, usecols=["track_id", "bird_group", args.time_col, args.group_col])
    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y_all = np.zeros((len(train_df), len(CLASSES)), dtype=np.float32)
    y_all[np.arange(len(train_df)), y_idx] = 1.0

    anchor_ids = np.load(cand_ids_paths[0]).astype(np.int64)
    row_map = dict(zip(train_df["track_id"].astype(np.int64).tolist(), np.arange(len(train_df), dtype=np.int64).tolist()))
    rows = np.asarray([row_map[int(t)] for t in anchor_ids], dtype=np.int64)
    y = y_all[rows]

    sample_weight: np.ndarray | None = None
    if str(args.sample_weights_npy).strip():
        sw = np.asarray(np.load(args.sample_weights_npy), dtype=np.float32).reshape(-1)
        if len(sw) != len(train_df):
            raise ValueError(f"sample_weights length mismatch: {len(sw)} vs train {len(train_df)}")
        sample_weight = np.clip(sw[rows], 1e-8, None).astype(np.float32)

    teacher = _align_probs_from_csv(args.teacher_oof_csv, anchor_ids)
    fold_id = _make_fold_ids(train_df, anchor_ids, args.n_splits, args.time_col, args.group_col)

    cand_oofs: list[np.ndarray] = []
    for i, (oof_path, ids_path) in enumerate(zip(cand_oof_paths, cand_ids_paths)):
        arr = np.asarray(np.load(oof_path), dtype=np.float32)
        ids = np.asarray(np.load(ids_path), dtype=np.int64)
        if arr.ndim != 2 or arr.shape[1] != len(CLASSES):
            raise ValueError(f"{oof_path}: expected [N,{len(CLASSES)}], got {arr.shape}")
        aligned = _align_npy_probs_to_ids(arr, ids, anchor_ids, label=cand_names[i])
        cand_oofs.append(aligned)

    teacher_metrics = _eval_metrics(
        y=y,
        teacher=teacher,
        pred=teacher,
        fold_id=fold_id,
        start_fold=args.start_fold,
        sample_weight=sample_weight,
    )

    seed_delta_by_class: dict[str, float] = {}
    seed_delta_by_candidate: dict[str, dict[str, float]] = {cls: {} for cls in CLASSES}
    candidate_classes: list[str] = []
    for j, cls in enumerate(CLASSES):
        if cls in freeze_classes:
            continue
        ap_t = float(average_precision_score(y[:, j], teacher[:, j], sample_weight=sample_weight))
        best_delta = -1e9
        for name, cand in zip(cand_names, cand_oofs):
            ap_n = float(average_precision_score(y[:, j], cand[:, j], sample_weight=sample_weight))
            d = float(ap_n - ap_t)
            seed_delta_by_candidate[cls][name] = d
            if d > best_delta:
                best_delta = d
        seed_delta_by_class[cls] = float(best_delta)
        if best_delta >= float(args.ap_threshold):
            candidate_classes.append(cls)
    candidate_classes = sorted(candidate_classes, key=lambda c: seed_delta_by_class[c], reverse=True)

    current = teacher.copy()
    current_metrics = teacher_metrics
    chosen: dict[str, dict[str, Any]] = {}
    decisions: list[dict[str, Any]] = []

    for cls in candidate_classes:
        j = CLASSES.index(cls)
        best_local: dict[str, Any] | None = None
        for cand_name, cand_oof in zip(cand_names, cand_oofs):
            for w_new in w_grid:
                trial = current.copy()
                trial[:, j] = _blend_col(teacher[:, j], cand_oof[:, j], w_new=w_new, mode=args.mode)
                m = _eval_metrics(
                    y=y,
                    teacher=teacher,
                    pred=trial,
                    fold_id=fold_id,
                    start_fold=args.start_fold,
                    sample_weight=sample_weight,
                )
                class_gain = float(m["ap_delta"][cls])
                change_frac = float(np.mean(np.abs(trial[:, j] - teacher[:, j]) > float(args.change_threshold)))
                feasible = (
                    class_gain >= float(args.class_min_ap_gain)
                    and m["min_fold_delta"] >= float(args.min_fold_delta)
                    and change_frac <= float(args.max_change_frac)
                    and m["fold_mean_pred"] >= current_metrics["fold_mean_pred"]
                )
                cand = {
                    "class": cls,
                    "candidate": cand_name,
                    "w_teacher": float(1.0 - w_new),
                    "w_new": float(w_new),
                    "macro_pred": float(m["macro_pred"]),
                    "macro_gain_teacher": float(m["macro_gain"]),
                    "fold_mean_pred": float(m["fold_mean_pred"]),
                    "fold_mean_gain_teacher": float(m["fold_mean_gain"]),
                    "min_fold_delta": float(m["min_fold_delta"]),
                    "class_ap_delta": float(class_gain),
                    "change_frac": float(change_frac),
                    "feasible": bool(feasible),
                }
                if feasible and (best_local is None or cand["fold_mean_pred"] > best_local["fold_mean_pred"]):
                    best_local = cand

        if best_local is not None:
            sel_cand = str(best_local["candidate"])
            w_new = float(best_local["w_new"])
            cand_idx = cand_names.index(sel_cand)
            current[:, j] = _blend_col(teacher[:, j], cand_oofs[cand_idx][:, j], w_new=w_new, mode=args.mode)
            current_metrics = _eval_metrics(
                y=y,
                teacher=teacher,
                pred=current,
                fold_id=fold_id,
                start_fold=args.start_fold,
                sample_weight=sample_weight,
            )
            chosen[cls] = {"candidate": sel_cand, "w_new": w_new, "w_teacher": float(1.0 - w_new)}
            best_local["accepted"] = True
            decisions.append(best_local)
        else:
            decisions.append({"class": cls, "accepted": False})

    out_oof = current.astype(np.float32)
    out_csv = Path(args.output_csv).resolve()
    out_json = Path(args.output_json).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if args.output_oof_npy:
        out_oof_path = Path(args.output_oof_npy).resolve()
        out_oof_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_oof_path, out_oof)
    else:
        out_oof_path = None

    if args.teacher_test_csv and args.candidate_test_csv:
        cand_test_paths = [str(Path(p).resolve()) for p in args.candidate_test_csv]
        if len(cand_test_paths) != n_cand:
            raise ValueError("--candidate-test-csv count must match candidates")
        ttest = pd.read_csv(args.teacher_test_csv, usecols=["track_id", *CLASSES])
        test_ids = ttest["track_id"].to_numpy(dtype=np.int64)
        teacher_test = ttest[CLASSES].to_numpy(dtype=np.float32)
        cand_tests = [_align_probs_from_csv(p, test_ids) for p in cand_test_paths]

        sub = pd.DataFrame({"track_id": test_ids})
        for j, cls in enumerate(CLASSES):
            if cls not in chosen:
                sub[cls] = teacher_test[:, j]
                continue
            cfg = chosen[cls]
            cand_idx = cand_names.index(str(cfg["candidate"]))
            w_new = float(cfg["w_new"])
            sub[cls] = _blend_col(teacher_test[:, j], cand_tests[cand_idx][:, j], w_new=w_new, mode=args.mode)
        sub.to_csv(out_csv, index=False)
    else:
        df = pd.DataFrame(out_oof, columns=CLASSES)
        df.insert(0, "track_id", anchor_ids)
        df.to_csv(out_csv, index=False)

    report = {
        "train_csv": str(Path(args.train_csv).resolve()),
        "teacher_oof_csv": str(Path(args.teacher_oof_csv).resolve()),
        "candidate_oof_npy": cand_oof_paths,
        "candidate_track_ids_npy": cand_ids_paths,
        "candidate_names": cand_names,
        "mode": str(args.mode),
        "start_fold": int(args.start_fold),
        "n_splits": int(args.n_splits),
        "constraints": {
            "ap_threshold": float(args.ap_threshold),
            "class_min_ap_gain": float(args.class_min_ap_gain),
            "min_fold_delta": float(args.min_fold_delta),
            "max_change_frac": float(args.max_change_frac),
            "change_threshold": float(args.change_threshold),
            "weights_grid": w_grid,
            "freeze_classes": sorted(freeze_classes),
        },
        "sample_weights_npy": str(Path(args.sample_weights_npy).resolve()) if str(args.sample_weights_npy).strip() else "",
        "sample_weighted_metrics": bool(sample_weight is not None),
        "seed_candidate_classes": candidate_classes,
        "seed_candidate_ap_delta": seed_delta_by_class,
        "seed_candidate_ap_delta_by_model": seed_delta_by_candidate,
        "chosen_weights": chosen,
        "n_selected": int(len(chosen)),
        "teacher_metrics": teacher_metrics,
        "final_metrics": current_metrics,
        "decisions": decisions,
        "output_csv": str(out_csv),
        "output_oof_npy": str(out_oof_path) if out_oof_path is not None else None,
    }
    out_json.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(str(out_csv), flush=True)
    print(f"selected={chosen}", flush=True)
    print(
        f"gain_macro={float(current_metrics['macro_gain']):+.6f} "
        f"gain_fold_mean={float(current_metrics['fold_mean_gain']):+.6f} "
        f"min_fold_delta={float(current_metrics['min_fold_delta']):+.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
