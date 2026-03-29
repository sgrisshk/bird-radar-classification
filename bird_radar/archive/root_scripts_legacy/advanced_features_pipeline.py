#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from shapely import wkb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

EPS = 1e-9
R_EARTH = 6371000.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advanced EWKB feature pipeline for Waders/Cormorants detectors")
    p.add_argument("--train_path", default="train.csv")
    p.add_argument("--oof_blend_path", default="bird_radar/artifacts/oof_csv_for_per_class/oof_teacher_fr_complete.csv")
    p.add_argument("--tcn_oof_path", default="bird_radar/artifacts/tcn_trajectory_forward_2stream_distill_full_seed777_cpu/artifacts/tcn_oof.npy")
    p.add_argument("--output_dir", default="bird_radar/artifacts/advanced_features_v1")
    p.add_argument("--id_col", default="track_id")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--n_splits", type=int, default=5)
    return p.parse_args()


def _parse_time_series(val) -> np.ndarray:
    if val is None:
        return np.asarray([], dtype=np.float64)
    if isinstance(val, (list, tuple, np.ndarray)):
        return np.asarray(val, dtype=np.float64)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return np.asarray([], dtype=np.float64)
        try:
            return np.asarray(json.loads(s), dtype=np.float64)
        except Exception:
            try:
                return np.asarray(ast.literal_eval(s), dtype=np.float64)
            except Exception:
                return np.asarray([], dtype=np.float64)
    return np.asarray([], dtype=np.float64)


def parse_ewkb(hex_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    geom = wkb.loads(str(hex_str), hex=True)
    coords = np.asarray(geom.coords, dtype=np.float64)
    if coords.ndim != 2:
        return (np.asarray([], dtype=np.float64),) * 4
    if coords.shape[1] >= 4:
        return coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    if coords.shape[1] == 3:
        z = coords[:, 2]
        return coords[:, 0], coords[:, 1], z, np.zeros_like(z)
    x = coords[:, 0]
    y = coords[:, 1]
    z = np.zeros_like(x)
    return x, y, z, np.zeros_like(x)


def lonlat_to_local_xy_m(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if lon.size == 0:
        return lon, lat
    lon0 = np.deg2rad(lon[0])
    lat0 = np.deg2rad(lat[0])
    lon_r = np.deg2rad(lon)
    lat_r = np.deg2rad(lat)
    x = (lon_r - lon0) * np.cos(lat0) * R_EARTH
    y = (lat_r - lat0) * R_EARTH
    return x, y


def _entropy(values: np.ndarray, bins: int = 18) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    return float(-np.sum(hist * np.log(hist + EPS)) / np.log(float(bins)))


def _safe_percentile(a: np.ndarray, q: float) -> float:
    if a.size == 0:
        return 0.0
    return float(np.percentile(a, q))


def _feature_keys() -> list[str]:
    return [
        "n_points", "total_dist_2d", "total_dist_3d", "displacement_2d", "straightness",
        "dist_3d_vs_2d", "straightness_seg_mean", "straightness_seg_min", "straightness_seg_std",
        "turn_angle_mean", "turn_angle_std", "turn_angle_median", "turn_angle_p95",
        "pct_small_turns", "pct_large_turns", "turn_angle_entropy",
        "alt_mean", "alt_std", "alt_median", "alt_p10", "alt_p90", "alt_range", "alt_flatness",
        "alt_cv", "alt_range_norm", "alt_cv_x_turn_mean", "pct_above_median_alt", "alt_entropy",
        "alt_direction_changes", "alt_autocorr_lag1", "alt_autocorr_lag5", "alt_slope", "alt_slope_abs",
        "speed_raw_mean", "speed_raw_std", "speed_raw_cv", "speed_raw_p10", "speed_raw_p90", "step_cv",
        "fast_straight", "vspeed_raw_mean", "vspeed_raw_std", "pct_level",
        "curv3d_mean", "curv3d_p95", "curv3d_std", "pct_low_curv",
        "dt_mean", "dt_std", "dt_cv", "dt_p90", "dt_regular",
    ]


def extract_raw_features(lon: np.ndarray, lat: np.ndarray, z: np.ndarray, _m: np.ndarray, t_abs: np.ndarray | None) -> dict[str, float]:
    x, y = lonlat_to_local_xy_m(lon, lat)
    feats = {k: 0.0 for k in _feature_keys()}
    n = int(x.size)
    feats["n_points"] = float(n)
    if n < 3:
        return feats

    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    step2 = np.sqrt(dx * dx + dy * dy)
    step3 = np.sqrt(dx * dx + dy * dy + dz * dz)

    total2 = float(np.sum(step2))
    total3 = float(np.sum(step3))
    disp2 = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))
    feats["total_dist_2d"] = total2
    feats["total_dist_3d"] = total3
    feats["displacement_2d"] = disp2
    feats["straightness"] = float(disp2 / (total2 + 1e-6))
    feats["dist_3d_vs_2d"] = float(total3 / (total2 + 1e-6))

    seg_n = max(n // 5, 2)
    seg_st = []
    for s in range(5):
        a, b = s * seg_n, (s + 1) * seg_n
        xs = x[a:b]
        ys = y[a:b]
        if xs.size < 2:
            continue
        ddx = np.diff(xs)
        ddy = np.diff(ys)
        tp = float(np.sum(np.sqrt(ddx * ddx + ddy * ddy)))
        d = float(np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2))
        seg_st.append(d / (tp + 1e-6))
    if seg_st:
        arr = np.asarray(seg_st, dtype=np.float64)
        feats["straightness_seg_mean"] = float(np.mean(arr))
        feats["straightness_seg_min"] = float(np.min(arr))
        feats["straightness_seg_std"] = float(np.std(arr))

    # turn angles
    angles = []
    for i in range(len(dx) - 1):
        v1 = np.array([dx[i], dy[i]], dtype=np.float64)
        v2 = np.array([dx[i + 1], dy[i + 1]], dtype=np.float64)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_a)))
    if angles:
        a = np.asarray(angles, dtype=np.float64)
        feats["turn_angle_mean"] = float(np.mean(a))
        feats["turn_angle_std"] = float(np.std(a))
        feats["turn_angle_median"] = float(np.median(a))
        feats["turn_angle_p95"] = _safe_percentile(a, 95)
        feats["pct_small_turns"] = float(np.mean(a < 10.0))
        feats["pct_large_turns"] = float(np.mean(a > 45.0))
        feats["turn_angle_entropy"] = _entropy(a, bins=18)
        feats["alt_cv_x_turn_mean"] = float((np.std(z) / (np.mean(z) + 1e-6)) * feats["turn_angle_mean"])
    else:
        feats["alt_cv_x_turn_mean"] = 0.0

    # altitude profile
    feats["alt_mean"] = float(np.mean(z))
    feats["alt_std"] = float(np.std(z))
    feats["alt_median"] = float(np.median(z))
    feats["alt_p10"] = _safe_percentile(z, 10)
    feats["alt_p90"] = _safe_percentile(z, 90)
    feats["alt_range"] = float(np.max(z) - np.min(z))
    feats["alt_flatness"] = float(1.0 / (np.std(z) + 1e-6))
    feats["alt_cv"] = float(np.std(z) / (np.mean(z) + 1e-6))
    feats["alt_range_norm"] = float(feats["alt_range"] / (np.mean(z) + 1e-6))
    feats["pct_above_median_alt"] = float(np.mean(z > np.median(z)))
    alt_hist, _ = np.histogram(z, bins=10, density=True)
    alt_hist = alt_hist[alt_hist > 0]
    feats["alt_entropy"] = float(-np.sum(alt_hist * np.log(alt_hist + 1e-9))) if alt_hist.size else 0.0

    dz_sign = np.sign(dz[dz != 0])
    if dz_sign.size > 0:
        runs = np.sum(np.diff(dz_sign) != 0)
        feats["alt_direction_changes"] = float(runs / dz_sign.size)

    if z.size > 5:
        zn = (z - np.mean(z)) / (np.std(z) + 1e-6)
        c1 = np.corrcoef(zn[:-1], zn[1:])[0, 1]
        feats["alt_autocorr_lag1"] = float(c1 if np.isfinite(c1) else 0.0)
        if z.size > 10:
            c5 = np.corrcoef(zn[:-5], zn[5:])[0, 1]
            feats["alt_autocorr_lag5"] = float(c5 if np.isfinite(c5) else 0.0)
    t_idx = np.arange(z.size, dtype=np.float64)
    if z.size > 2:
        slope = np.polyfit(t_idx, z, 1)[0]
        feats["alt_slope"] = float(slope)
        feats["alt_slope_abs"] = float(abs(slope))

    # dt from absolute trajectory time
    dt_seg = None
    if t_abs is not None and t_abs.size >= 2:
        t = t_abs[:n]
        if t.size >= 2:
            dt_seg = np.diff(t)
            dt_seg = np.where(dt_seg < 1e-6, 1e-6, dt_seg)

    if dt_seg is not None and dt_seg.size >= 1:
        sp = step2[:dt_seg.size] / dt_seg
        feats["speed_raw_mean"] = float(np.mean(sp))
        feats["speed_raw_std"] = float(np.std(sp))
        feats["speed_raw_cv"] = float(np.std(sp) / (np.mean(sp) + 1e-6))
        feats["speed_raw_p10"] = _safe_percentile(sp, 10)
        feats["speed_raw_p90"] = _safe_percentile(sp, 90)
        feats["step_cv"] = float(np.std(step2[:dt_seg.size]) / (np.mean(step2[:dt_seg.size]) + 1e-6))
        thr = _safe_percentile(sp, 66)
        feats["fast_straight"] = float(np.mean(sp > thr) * feats["straightness"])

        vz = np.abs(dz[:dt_seg.size] / dt_seg)
        feats["vspeed_raw_mean"] = float(np.mean(vz))
        feats["vspeed_raw_std"] = float(np.std(vz))
        feats["pct_level"] = float(np.mean(vz < _safe_percentile(vz + 1e-9, 25)))

        feats["dt_mean"] = float(np.mean(dt_seg))
        feats["dt_std"] = float(np.std(dt_seg))
        feats["dt_cv"] = float(np.std(dt_seg) / (np.mean(dt_seg) + 1e-6))
        feats["dt_p90"] = _safe_percentile(dt_seg, 90)
        feats["dt_regular"] = float(np.mean(np.abs(dt_seg - np.mean(dt_seg)) < 0.5 * (np.std(dt_seg) + 1e-9)))
    else:
        feats["speed_raw_mean"] = float(np.mean(step2))
        feats["speed_raw_std"] = float(np.std(step2))
        feats["speed_raw_cv"] = float(np.std(step2) / (np.mean(step2) + 1e-6))
        feats["speed_raw_p10"] = _safe_percentile(step2, 10)
        feats["speed_raw_p90"] = _safe_percentile(step2, 90)
        feats["step_cv"] = float(np.std(step2) / (np.mean(step2) + 1e-6))
        feats["fast_straight"] = float(feats["straightness"])
        avz = np.abs(dz)
        feats["vspeed_raw_mean"] = float(np.mean(avz))
        feats["vspeed_raw_std"] = float(np.std(avz))
        feats["pct_level"] = float(np.mean(avz < _safe_percentile(avz + 1e-9, 25)))

    # 3d curvature
    curv3d = []
    for i in range(1, n - 1):
        p0 = np.array([x[i - 1], y[i - 1], z[i - 1]], dtype=np.float64)
        p1 = np.array([x[i], y[i], z[i]], dtype=np.float64)
        p2 = np.array([x[i + 1], y[i + 1], z[i + 1]], dtype=np.float64)
        v1, v2 = p1 - p0, p2 - p1
        cross = np.cross(v1, v2)
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(v1 + v2)
        curv3d.append(np.linalg.norm(cross) / (denom + 1e-9))
    if curv3d:
        c = np.asarray(curv3d, dtype=np.float64)
        feats["curv3d_mean"] = float(np.mean(c))
        feats["curv3d_p95"] = _safe_percentile(c, 95)
        feats["curv3d_std"] = float(np.std(c))
        feats["pct_low_curv"] = float(np.mean(c < _safe_percentile(c + 1e-9, 25)))

    return feats


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    rows = []
    failed = 0
    for _, row in df.iterrows():
        try:
            lon, lat, z, m = parse_ewkb(row["trajectory"])
            t_abs = _parse_time_series(row.get("trajectory_time"))
            feats = extract_raw_features(lon, lat, z, m, t_abs)
        except Exception:
            feats = {k: 0.0 for k in _feature_keys()}
            failed += 1
        rows.append(feats)
    if failed:
        print(f"  parse_fail={failed}", flush=True)
    feat_df = pd.DataFrame(rows).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return feat_df.to_numpy(dtype=np.float32), feat_df.columns.tolist()


def train_binary_detector_oof(
    X: np.ndarray,
    y: np.ndarray,
    positive_class: str,
    negative_classes: list[str],
    n_splits: int,
    seed: int,
) -> np.ndarray:
    mask = np.isin(y, [positive_class, *negative_classes])
    X_bin = X[mask]
    y_bin = (y[mask] == positive_class).astype(np.int32)
    idx_orig = np.where(mask)[0]

    oof = np.zeros((len(y),), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_aps = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_bin, y_bin), start=1):
        y_tr = y_bin[tr_idx]
        pos = int(np.sum(y_tr == 1))
        neg = int(np.sum(y_tr == 0))
        cw = {0: 1.0, 1: float(neg / (pos + 1e-6))}

        model = CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=6,
            class_weights=cw,
            eval_metric="AUC",
            random_seed=seed + fold,
            verbose=False,
            allow_writing_files=False,
        )
        X_tr = X_bin[tr_idx]
        X_va = X_bin[va_idx]
        y_va = y_bin[va_idx]
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=60, use_best_model=True, verbose=False)

        pr = model.predict_proba(X_va)[:, 1].astype(np.float32)
        oof[idx_orig[va_idx]] = pr
        ap = average_precision_score(y_va, pr) if int(np.sum(y_va)) > 0 else 0.0
        fold_aps.append(ap)
        print(f"    fold {fold}: AP={ap:.4f}  pos={int(np.sum(y_va))}", flush=True)

    print(f"  -> mean fold AP: {float(np.mean(fold_aps)):.4f}", flush=True)
    return np.clip(oof, 0.0, 1.0)


def blend_with_detectors(base_probs: np.ndarray, wader_scores: np.ndarray, corm_scores: np.ndarray, class_to_idx: dict[str, int], alpha_w: float, alpha_c: float) -> np.ndarray:
    out = np.clip(base_probs.copy(), 0.0, 1.0)
    out[:, class_to_idx["Waders"]] = (1.0 - alpha_w) * out[:, class_to_idx["Waders"]] + alpha_w * wader_scores
    out[:, class_to_idx["Cormorants"]] = (1.0 - alpha_c) * out[:, class_to_idx["Cormorants"]] + alpha_c * corm_scores
    row_sum = np.sum(out, axis=1, keepdims=True)
    row_sum = np.where(row_sum <= 0, 1.0, row_sum)
    return out / row_sum


def macro_ap(probs: np.ndarray, y_true: np.ndarray, class_order: list[str], mask: np.ndarray) -> float:
    p = probs[mask]
    y = y_true[mask]
    vals = []
    for i, cls in enumerate(class_order):
        yb = (y == cls).astype(np.int32)
        if int(np.sum(yb)) == 0:
            continue
        vals.append(average_precision_score(yb, p[:, i]))
    return float(np.mean(vals)) if vals else 0.0


def tune_alphas(base_probs: np.ndarray, wader_scores: np.ndarray, corm_scores: np.ndarray, y_true: np.ndarray, class_order: list[str], class_to_idx: dict[str, int], mask: np.ndarray) -> tuple[float, float, float]:
    best_ap = -1.0
    best_aw, best_ac = 0.0, 0.0
    for aw in np.arange(0.0, 0.65, 0.05):
        for ac in np.arange(0.0, 0.65, 0.05):
            p = blend_with_detectors(base_probs, wader_scores, corm_scores, class_to_idx, float(aw), float(ac))
            ap = macro_ap(p, y_true, class_order, mask)
            if ap > best_ap:
                best_ap = ap
                best_aw = float(aw)
                best_ac = float(ac)
    return best_aw, best_ac, float(best_ap)


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Загрузка train данных ===", flush=True)
    df = pd.read_csv(args.train_path)
    label_col = "bird_group" if "bird_group" in df.columns else "label"
    labels = df[label_col].astype(str).to_numpy()
    track_ids = pd.to_numeric(df[args.id_col], errors="coerce").astype("Int64")
    if track_ids.isna().any():
        raise ValueError("train contains non-numeric track_id")
    track_ids = track_ids.to_numpy(dtype=np.int64)

    teacher_oof = pd.read_csv(args.oof_blend_path)
    class_order = [c for c in teacher_oof.columns if c != args.id_col]
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    need = {"Waders", "Cormorants"}
    if not need.issubset(class_to_idx):
        raise ValueError(f"Teacher OOF missing required classes: {sorted(list(need - set(class_to_idx)))}")

    print(f"  Всего треков: {len(df)}", flush=True)
    for cls in class_order:
        n = int(np.sum(labels == cls))
        if n > 0:
            print(f"    {cls:<20} {n}", flush=True)

    print("\n=== Извлечение признаков из EWKB ===", flush=True)
    X, feat_names = build_feature_matrix(df)
    print(f"  Feature matrix: {X.shape}", flush=True)

    print("\n=== OOF детектор: Waders vs Gulls+Songbirds ===", flush=True)
    wader_oof = train_binary_detector_oof(X, labels, "Waders", ["Gulls", "Songbirds"], n_splits=args.n_splits, seed=args.seed)

    print("\n=== OOF детектор: Cormorants vs Gulls+Songbirds+Ducks ===", flush=True)
    corm_oof = train_binary_detector_oof(X, labels, "Cormorants", ["Gulls", "Songbirds", "Ducks"], n_splits=args.n_splits, seed=args.seed + 1000)

    print("\n=== Загрузка baseline OOF вероятностей ===", flush=True)
    tcn_probs = np.asarray(np.load(args.tcn_oof_path), dtype=np.float32)
    if tcn_probs.shape != (len(df), len(class_order)):
        raise ValueError(f"tcn_oof shape mismatch: {tcn_probs.shape}, expected {(len(df), len(class_order))}")

    toof = teacher_oof.copy()
    toof[args.id_col] = pd.to_numeric(toof[args.id_col], errors="coerce").astype("Int64")
    toof = toof.dropna(subset=[args.id_col]).copy()
    toof[args.id_col] = toof[args.id_col].astype(np.int64)
    mp = {int(r[args.id_col]): np.asarray([r[c] for c in class_order], dtype=np.float32) for _, r in toof.iterrows()}
    missing = [int(t) for t in track_ids.tolist() if int(t) not in mp]
    if missing:
        raise ValueError(f"teacher_oof missing {len(missing)} track_ids")
    teacher_probs = np.stack([mp[int(t)] for t in track_ids.tolist()], axis=0)

    base_probs = np.clip(0.75 * teacher_probs + 0.25 * tcn_probs, 0.0, 1.0)

    covered_idx_path = Path(args.tcn_oof_path).with_name("oof_covered_idx.npy")
    if covered_idx_path.exists():
        covered_idx = np.asarray(np.load(covered_idx_path), dtype=np.int64).reshape(-1)
        covered_mask = np.zeros((len(df),), dtype=bool)
        covered_mask[covered_idx] = True
    else:
        covered_mask = np.ones((len(df),), dtype=bool)

    print("\n=== Baseline macro AP (covered, w75/25) ===", flush=True)
    baseline = macro_ap(base_probs, labels, class_order, covered_mask)
    print(f"  {baseline:.6f}", flush=True)

    print("\n=== Grid search alpha_w × alpha_c ===", flush=True)
    best_aw, best_ac, best_ap = tune_alphas(base_probs, wader_oof, corm_oof, labels, class_order, class_to_idx, covered_mask)
    print(f"  Best: alpha_w={best_aw:.2f}, alpha_c={best_ac:.2f}  ->  {best_ap:.6f}  (Δ{best_ap - baseline:+.6f})", flush=True)

    final_probs = blend_with_detectors(base_probs, wader_oof, corm_oof, class_to_idx, best_aw, best_ac)

    print("\n=== Per-class AP (covered) ===", flush=True)
    p_cov = base_probs[covered_mask]
    f_cov = final_probs[covered_mask]
    y_cov = labels[covered_mask]

    per_class_rows = []
    for i, cls in enumerate(class_order):
        yb = (y_cov == cls).astype(np.int32)
        support = int(np.sum(yb))
        if support == 0:
            continue
        ap_b = float(average_precision_score(yb, p_cov[:, i]))
        ap_a = float(average_precision_score(yb, f_cov[:, i]))
        delta = ap_a - ap_b
        flag = "🔺" if delta > 0.01 else ("🔻" if delta < -0.01 else "  ")
        print(f"  {flag} {cls:<20} {ap_b:.4f} -> {ap_a:.4f}  ({delta:+.4f})  support={support}", flush=True)
        per_class_rows.append({"class": cls, "ap_before": ap_b, "ap_after": ap_a, "delta": delta, "support": support})

    np.save(out_dir / "wader_oof_scores.npy", wader_oof.astype(np.float32))
    np.save(out_dir / "cormorant_oof_scores.npy", corm_oof.astype(np.float32))
    np.save(out_dir / "final_probs_oof.npy", final_probs.astype(np.float32))
    pd.DataFrame(per_class_rows).to_csv(out_dir / "per_class_ap.csv", index=False)
    pd.DataFrame({"feature": feat_names}).to_csv(out_dir / "feature_names.csv", index=False)

    report = {
        "baseline_macro_ap": float(baseline),
        "best_macro_ap": float(best_ap),
        "delta": float(best_ap - baseline),
        "alpha_w": float(best_aw),
        "alpha_c": float(best_ac),
        "n_features": int(len(feat_names)),
    }
    pd.DataFrame([report]).to_csv(out_dir / "report.csv", index=False)

    print(f"\n✅ Готово. Результаты в {out_dir}/", flush=True)
    print(f"   baseline -> best:  {baseline:.6f} -> {best_ap:.6f}  (Δ{best_ap - baseline:+.6f})", flush=True)


if __name__ == "__main__":
    main(parse_args())
