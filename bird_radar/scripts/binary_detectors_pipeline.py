#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.metrics import macro_map_score, per_class_average_precision
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache

EPS = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OOF binary detectors for Waders/Cormorants vs confusing classes.")
    p.add_argument("--train-path", default="train.csv")
    p.add_argument("--teacher-oof-csv", default="bird_radar/artifacts/oof_csv_for_per_class/oof_teacher_fr_complete.csv")
    p.add_argument("--tcn-oof-path", default="bird_radar/artifacts/tcn_trajectory_forward_2stream_distill_full_seed777_cpu/artifacts/tcn_oof.npy")
    p.add_argument("--tcn-train-ids-path", default="bird_radar/artifacts/tcn_trajectory_forward_2stream_distill_full_seed777_cpu/train_track_ids.npy")
    p.add_argument("--tcn-covered-idx-path", default="bird_radar/artifacts/tcn_trajectory_forward_2stream_distill_full_seed777_cpu/oof_covered_idx.npy")
    p.add_argument("--cache-dir", default="bird_radar/artifacts/cache")
    p.add_argument("--output-dir", default="bird_radar/artifacts/binary_detectors_v1")
    p.add_argument("--id-col", default="track_id")
    p.add_argument("--label-col", default="bird_group")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--iterations", type=int, default=600)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--od-wait", type=int, default=50)
    p.add_argument("--thread-count", type=int, default=-1)
    p.add_argument("--grid-step", type=float, default=0.05)
    return p.parse_args()


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict[str, Any]]:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


def _safe_percentile(a: np.ndarray, q: float) -> float:
    if a.size == 0:
        return 0.0
    return float(np.percentile(a, q))


def _extract_features_from_raw(raw: np.ndarray) -> dict[str, float]:
    # raw: [T,9] = x,y,alt,rcs,speed,vspeed,acc,curv,dt
    x = raw[:, 0].astype(np.float32, copy=False)
    y = raw[:, 1].astype(np.float32, copy=False)
    alt = raw[:, 2].astype(np.float32, copy=False)
    speed = raw[:, 4].astype(np.float32, copy=False)
    vspeed = raw[:, 5].astype(np.float32, copy=False)
    acc = np.abs(raw[:, 6].astype(np.float32, copy=False))
    curv = np.abs(raw[:, 7].astype(np.float32, copy=False))

    feats: dict[str, float] = {}

    # altitude features
    feats["alt_mean"] = float(np.mean(alt))
    feats["alt_std"] = float(np.std(alt))
    feats["alt_median"] = float(np.median(alt))
    feats["alt_p10"] = _safe_percentile(alt, 10)
    feats["alt_p90"] = _safe_percentile(alt, 90)
    feats["alt_range"] = float(np.max(alt) - np.min(alt))
    feats["alt_flatness"] = float(1.0 / (np.std(alt) + 1e-6))
    feats["pct_high_alt"] = float(np.mean(alt > _safe_percentile(alt, 75)))
    feats["pct_low_alt"] = float(np.mean(alt < 30.0))

    # speed features
    feats["speed_mean"] = float(np.mean(speed))
    feats["speed_std"] = float(np.std(speed))
    feats["speed_median"] = float(np.median(speed))
    feats["speed_p10"] = _safe_percentile(speed, 10)
    feats["speed_p90"] = _safe_percentile(speed, 90)
    feats["speed_cv"] = float(np.std(speed) / (np.mean(speed) + 1e-6))

    # speed × altitude interaction
    low_alt_thr = _safe_percentile(alt, 33)
    low_alt_mask = alt < low_alt_thr
    feats["speed_at_low_alt"] = float(np.mean(speed[low_alt_mask])) if np.any(low_alt_mask) else 0.0
    if speed.size > 3:
        c = np.corrcoef(speed.astype(np.float64), alt.astype(np.float64))[0, 1]
        feats["speed_alt_corr"] = float(c if np.isfinite(c) else 0.0)
    else:
        feats["speed_alt_corr"] = 0.0
    feats["fast_and_low"] = float(np.mean((speed > _safe_percentile(speed, 66)) & low_alt_mask))

    # curvature
    feats["curv_mean"] = float(np.mean(curv))
    feats["curv_p95"] = _safe_percentile(curv, 95)
    feats["curv_std"] = float(np.std(curv))
    feats["pct_low_curv"] = float(np.mean(curv < 0.05))

    # straightness
    if x.size > 1:
        dx = np.diff(x)
        dy = np.diff(y)
        total_path = float(np.sum(np.sqrt(dx * dx + dy * dy)))
        displacement = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))
        feats["straightness_global"] = float(displacement / (total_path + 1e-6))
    else:
        feats["straightness_global"] = 0.0

    n = int(x.size)
    seg_size = max(n // 5, 2)
    seg_straights: list[float] = []
    for s in range(5):
        a = s * seg_size
        b = (s + 1) * seg_size
        xs = x[a:b]
        ys = y[a:b]
        if xs.size < 2:
            continue
        dx = np.diff(xs)
        dy = np.diff(ys)
        tp = float(np.sum(np.sqrt(dx * dx + dy * dy)))
        d = float(np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2))
        seg_straights.append(float(d / (tp + 1e-6)))
    if seg_straights:
        arr = np.asarray(seg_straights, dtype=np.float32)
        feats["straightness_seg_mean"] = float(np.mean(arr))
        feats["straightness_seg_min"] = float(np.min(arr))
        feats["straightness_seg_std"] = float(np.std(arr))
    else:
        feats["straightness_seg_mean"] = 0.0
        feats["straightness_seg_min"] = 0.0
        feats["straightness_seg_std"] = 0.0

    # vertical speed / glide
    abs_vs = np.abs(vspeed)
    feats["vspeed_std"] = float(np.std(vspeed))
    feats["vspeed_abs_mean"] = float(np.mean(abs_vs))
    feats["pct_level_flight"] = float(np.mean(abs_vs < _safe_percentile(abs_vs, 25)))
    feats["glide_ratio"] = float(np.mean(acc < 0.05))
    feats["accel_p95"] = _safe_percentile(acc, 95)
    feats["accel_mean"] = float(np.mean(acc))
    feats["track_len"] = float(speed.size)
    return feats


def _build_feature_matrix(train_df: pd.DataFrame, cache: dict[int, dict[str, Any]], id_col: str) -> tuple[np.ndarray, list[str]]:
    rows: list[dict[str, float]] = []
    for tid in train_df[id_col].to_numpy(dtype=np.int64).tolist():
        raw = np.asarray(cache[int(tid)]["raw_features"], dtype=np.float32)
        rows.append(_extract_features_from_raw(raw))
    feat_df = pd.DataFrame(rows).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return feat_df.to_numpy(dtype=np.float32), feat_df.columns.tolist()


def _train_binary_detector_oof(
    X: np.ndarray,
    labels: np.ndarray,
    positive_class: str,
    negative_classes: list[str],
    n_splits: int,
    seed: int,
    iterations: int,
    depth: int,
    learning_rate: float,
    od_wait: int,
    thread_count: int,
) -> np.ndarray:
    y_pos = (labels == positive_class).astype(np.int32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(labels),), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_pos), start=1):
        tr_mask = np.isin(labels[tr_idx], [positive_class, *negative_classes])
        tr_use = tr_idx[tr_mask]
        y_tr = (labels[tr_use] == positive_class).astype(np.int32)
        pos = int(np.sum(y_tr == 1))
        neg = int(np.sum(y_tr == 0))
        if pos == 0 or neg == 0:
            prior = float(np.mean(labels[tr_idx] == positive_class))
            oof[va_idx] = prior
            print(f"  fold {fold}: skip (degenerate train), prior={prior:.6f}", flush=True)
            continue

        cw = {0: 1.0, 1: float(neg / (pos + 1e-6))}
        model = CatBoostClassifier(
            iterations=int(iterations),
            learning_rate=float(learning_rate),
            depth=int(depth),
            class_weights=cw,
            eval_metric="AUC",
            random_seed=int(seed + fold),
            od_type="Iter",
            od_wait=int(od_wait),
            verbose=False,
            thread_count=int(thread_count),
            allow_writing_files=False,
        )
        X_tr = X[tr_use]
        X_va = X[va_idx]
        y_va_bin = (labels[va_idx] == positive_class).astype(np.int32)
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va_bin),
            use_best_model=True,
            verbose=False,
        )
        pr = model.predict_proba(X_va)[:, 1].astype(np.float32)
        oof[va_idx] = pr
        ap = average_precision_score(y_va_bin, pr) if int(np.sum(y_va_bin)) > 0 else 0.0
        print(f"  fold {fold}: AP={ap:.6f} pos_val={int(np.sum(y_va_bin))}", flush=True)

    return np.clip(oof, 0.0, 1.0)


def _blend_with_detectors(
    base_probs: np.ndarray,
    wader_scores: np.ndarray,
    cormorant_scores: np.ndarray,
    alpha_w: float,
    alpha_c: float,
) -> np.ndarray:
    out = np.clip(base_probs.copy(), 0.0, 1.0)
    out[:, CLASS_TO_INDEX["Waders"]] = (
        (1.0 - alpha_w) * out[:, CLASS_TO_INDEX["Waders"]] + alpha_w * wader_scores
    )
    out[:, CLASS_TO_INDEX["Cormorants"]] = (
        (1.0 - alpha_c) * out[:, CLASS_TO_INDEX["Cormorants"]] + alpha_c * cormorant_scores
    )
    return np.clip(out, 0.0, 1.0)


def _evaluate_macro_ap_covered(
    probs: np.ndarray,
    y_onehot: np.ndarray,
    covered_idx: np.ndarray,
) -> float:
    return float(macro_map_score(y_onehot[covered_idx], probs[covered_idx]))


def _align_teacher_probs(csv_path: str, ids: np.ndarray, id_col: str) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=[id_col, *CLASSES])
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    mp = {int(r[id_col]): np.asarray([r[c] for c in CLASSES], dtype=np.float32) for _, r in df.iterrows() if pd.notna(r[id_col])}
    miss = [int(t) for t in ids.tolist() if int(t) not in mp]
    if miss:
        raise ValueError(f"teacher_oof_csv missing {len(miss)} ids; first={miss[:10]}")
    return np.stack([mp[int(t)] for t in ids.tolist()], axis=0).astype(np.float32)


def _save_per_class_ap_report(
    y_true: np.ndarray,
    before_probs: np.ndarray,
    after_probs: np.ndarray,
    out_csv: Path,
) -> None:
    ap_before = per_class_average_precision(y_true, before_probs)
    ap_after = per_class_average_precision(y_true, after_probs)
    rows = []
    for c in CLASSES:
        b = float(ap_before.get(c, 0.0))
        a = float(ap_after.get(c, 0.0))
        rows.append(
            {
                "class": c,
                "ap_before": b,
                "ap_after": a,
                "delta_after_minus_before": float(a - b),
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_path, usecols=[args.id_col, args.label_col])
    labels = train_df[args.label_col].astype(str).to_numpy()
    if not np.isin(labels, CLASSES).all():
        bad = sorted(set(labels.tolist()) - set(CLASSES))
        raise ValueError(f"unknown labels in train: {bad[:10]}")
    y_idx = np.array([CLASS_TO_INDEX[x] for x in labels], dtype=np.int64)
    y_onehot = np.zeros((len(labels), len(CLASSES)), dtype=np.float32)
    y_onehot[np.arange(len(labels)), y_idx] = 1.0

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    full_train_df = pd.read_csv(args.train_path)
    train_cache = _load_or_build_cache(full_train_df, cache_dir / "train_track_cache.pkl")

    print("=== Building feature matrix ===", flush=True)
    X, feat_names = _build_feature_matrix(train_df, train_cache, id_col=args.id_col)
    print(f"X_shape={X.shape} n_features={len(feat_names)}", flush=True)

    print("=== OOF detector: Waders vs Gulls+Songbirds ===", flush=True)
    wader_oof = _train_binary_detector_oof(
        X=X,
        labels=labels,
        positive_class="Waders",
        negative_classes=["Gulls", "Songbirds"],
        n_splits=int(args.n_splits),
        seed=int(args.seed),
        iterations=int(args.iterations),
        depth=int(args.depth),
        learning_rate=float(args.learning_rate),
        od_wait=int(args.od_wait),
        thread_count=int(args.thread_count),
    )

    print("=== OOF detector: Cormorants vs Gulls+Songbirds ===", flush=True)
    corm_oof = _train_binary_detector_oof(
        X=X,
        labels=labels,
        positive_class="Cormorants",
        negative_classes=["Gulls", "Songbirds"],
        n_splits=int(args.n_splits),
        seed=int(args.seed + 1000),
        iterations=int(args.iterations),
        depth=int(args.depth),
        learning_rate=float(args.learning_rate),
        od_wait=int(args.od_wait),
        thread_count=int(args.thread_count),
    )

    # align baseline OOF (teacher + tcn)
    base_ids = np.load(args.tcn_train_ids_path).astype(np.int64).reshape(-1)
    train_ids = train_df[args.id_col].to_numpy(dtype=np.int64)
    if not np.array_equal(base_ids, train_ids):
        raise ValueError("train ids mismatch between train csv and tcn train ids path")
    covered_idx = np.load(args.tcn_covered_idx_path).astype(np.int64).reshape(-1)
    tcn_oof = np.asarray(np.load(args.tcn_oof_path), dtype=np.float32)
    if tcn_oof.shape != (len(train_ids), len(CLASSES)):
        raise ValueError(f"tcn_oof shape mismatch: {tcn_oof.shape}, expected {(len(train_ids), len(CLASSES))}")
    teacher_oof = _align_teacher_probs(args.teacher_oof_csv, train_ids, id_col=args.id_col)
    base_probs = np.clip(0.75 * teacher_oof + 0.25 * tcn_oof, 0.0, 1.0)

    baseline_macro = _evaluate_macro_ap_covered(base_probs, y_onehot, covered_idx)
    print(f"baseline_macro_covered={baseline_macro:.6f}", flush=True)

    # grid search alphas
    best_macro = -1.0
    best_aw, best_ac = 0.0, 0.0
    gs = float(args.grid_step)
    grid_vals = np.arange(0.0, 0.50 + gs / 2.0, gs, dtype=np.float32)
    grid_rows: list[dict[str, float]] = []
    for aw in grid_vals.tolist():
        for ac in grid_vals.tolist():
            pr = _blend_with_detectors(base_probs, wader_oof, corm_oof, aw, ac)
            m = _evaluate_macro_ap_covered(pr, y_onehot, covered_idx)
            grid_rows.append({"alpha_w": float(aw), "alpha_c": float(ac), "macro_covered": float(m)})
            if m > best_macro:
                best_macro = float(m)
                best_aw = float(aw)
                best_ac = float(ac)

    final_probs = _blend_with_detectors(base_probs, wader_oof, corm_oof, best_aw, best_ac)
    final_macro = _evaluate_macro_ap_covered(final_probs, y_onehot, covered_idx)

    # per-class report on covered rows
    per_class_csv = out_dir / "per_class_ap_before_after.csv"
    _save_per_class_ap_report(
        y_true=y_onehot[covered_idx],
        before_probs=base_probs[covered_idx],
        after_probs=final_probs[covered_idx],
        out_csv=per_class_csv,
    )

    # detector AP
    w_y = (labels == "Waders").astype(np.int32)
    c_y = (labels == "Cormorants").astype(np.int32)
    w_ap = float(average_precision_score(w_y[covered_idx], wader_oof[covered_idx])) if int(np.sum(w_y[covered_idx])) > 0 else 0.0
    c_ap = float(average_precision_score(c_y[covered_idx], corm_oof[covered_idx])) if int(np.sum(c_y[covered_idx])) > 0 else 0.0

    np.save(out_dir / "wader_oof_scores.npy", wader_oof.astype(np.float32))
    np.save(out_dir / "cormorant_oof_scores.npy", corm_oof.astype(np.float32))
    np.save(out_dir / "base_probs.npy", base_probs.astype(np.float32))
    np.save(out_dir / "final_probs.npy", final_probs.astype(np.float32))
    pd.DataFrame({"feature_name": feat_names}).to_csv(out_dir / "feature_names.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(out_dir / "alpha_grid.csv", index=False)

    summary = {
        "output_dir": str(out_dir),
        "train_path": str(args.train_path),
        "teacher_oof_csv": str(args.teacher_oof_csv),
        "tcn_oof_path": str(args.tcn_oof_path),
        "seed": int(args.seed),
        "n_splits": int(args.n_splits),
        "n_rows": int(len(train_ids)),
        "n_features": int(X.shape[1]),
        "covered_count": int(len(covered_idx)),
        "baseline_macro_covered": float(baseline_macro),
        "best_alpha_w": float(best_aw),
        "best_alpha_c": float(best_ac),
        "best_macro_covered": float(best_macro),
        "delta_macro_covered": float(best_macro - baseline_macro),
        "wader_detector_ap_covered": float(w_ap),
        "cormorant_detector_ap_covered": float(c_ap),
        "artifacts": {
            "per_class_ap_before_after_csv": str(per_class_csv.resolve()),
            "wader_oof_scores_npy": str((out_dir / "wader_oof_scores.npy").resolve()),
            "cormorant_oof_scores_npy": str((out_dir / "cormorant_oof_scores.npy").resolve()),
            "alpha_grid_csv": str((out_dir / "alpha_grid.csv").resolve()),
            "feature_names_csv": str((out_dir / "feature_names.csv").resolve()),
        },
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print("=== BINARY DETECTORS PIPELINE COMPLETE ===", flush=True)
    print(f"baseline_macro_covered={baseline_macro:.6f}", flush=True)
    print(f"best_macro_covered={best_macro:.6f}", flush=True)
    print(f"delta_macro_covered={best_macro - baseline_macro:+.6f}", flush=True)
    print(f"best_alpha_w={best_aw:.3f} best_alpha_c={best_ac:.3f}", flush=True)
    print(f"wader_detector_ap_covered={w_ap:.6f}", flush=True)
    print(f"cormorant_detector_ap_covered={c_ap:.6f}", flush=True)
    print(f"report={out_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()

