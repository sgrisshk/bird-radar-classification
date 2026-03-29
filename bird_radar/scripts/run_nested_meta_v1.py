#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from shapely import wkb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leak-free nested OOF stacking for detector + base probabilities.")
    p.add_argument("--train-path", default="train.csv")
    p.add_argument("--teacher-oof-csv", default="bird_radar/artifacts/oof_csv_for_per_class/oof_teacher_fr_complete.csv")
    p.add_argument("--tcn-oof-npy", default="bird_radar/artifacts/tcn_trajectory_forward_2stream_distill_full_seed777_cpu/artifacts/tcn_oof.npy")
    p.add_argument("--covered-idx-npy", default="bird_radar/artifacts/tcn_trajectory_forward_2stream_distill_full_seed777_cpu/oof_covered_idx.npy")
    p.add_argument("--output-dir", default="bird_radar/artifacts/nested_meta_v1")
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--inner-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--detector-iters", type=int, default=600)
    p.add_argument("--meta-iters", type=int, default=300)
    p.add_argument("--thread-count", type=int, default=-1)
    return p.parse_args()


def parse_ewkb(hex_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geom = wkb.loads(str(hex_str), hex=True)
    coords = np.asarray(geom.coords, dtype=np.float64)
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2] if coords.shape[1] >= 3 else np.zeros_like(x)
    return x, y, z


def parse_time_series(val) -> np.ndarray:
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


def _entropy(values: np.ndarray, bins: int = 18) -> float:
    if values.size == 0:
        return 0.0
    h, _ = np.histogram(values, bins=bins, density=True)
    h = h[h > 0]
    if h.size == 0:
        return 0.0
    return float(-np.sum(h * np.log(h + 1e-9)) / np.log(float(bins)))


def _sp(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else 0.0


def extract_features(row: pd.Series) -> np.ndarray:
    try:
        x, y, z = parse_ewkb(row["trajectory"])
        t = parse_time_series(row.get("trajectory_time"))
    except Exception:
        return np.zeros((52,), dtype=np.float32)

    n = len(x)
    if n < 3:
        return np.zeros((52,), dtype=np.float32)

    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
    step2 = np.sqrt(dx * dx + dy * dy)
    step3 = np.sqrt(dx * dx + dy * dy + dz * dz)
    total2 = float(np.sum(step2)) + 1e-9
    total3 = float(np.sum(step3))
    disp = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))

    angles = []
    for i in range(len(dx) - 1):
        v1 = np.array([dx[i], dy[i]], dtype=np.float64)
        v2 = np.array([dx[i + 1], dy[i + 1]], dtype=np.float64)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        angles.append(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))
    angles = np.asarray(angles, dtype=np.float64) if angles else np.asarray([0.0], dtype=np.float64)

    seg_st = []
    seg_len = max(n // 5, 2)
    for s in range(5):
        a, b = s * seg_len, (s + 1) * seg_len
        xs, ys = x[a:b], y[a:b]
        if len(xs) < 2:
            continue
        st = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
        tp = float(np.sum(st)) + 1e-9
        d = float(np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2))
        seg_st.append(d / tp)
    seg_st = np.asarray(seg_st, dtype=np.float64) if seg_st else np.asarray([0.0], dtype=np.float64)

    alt_cv = float(np.std(z) / (np.mean(z) + 1e-6))
    alt_range = float(np.max(z) - np.min(z))
    dz_sign = np.sign(dz[dz != 0])
    alt_dchg = float(np.sum(np.diff(dz_sign) != 0) / len(dz_sign)) if len(dz_sign) > 0 else 0.0
    alt_ac1 = float(np.corrcoef(z[:-1], z[1:])[0, 1]) if len(z) > 5 else 0.0
    if not np.isfinite(alt_ac1):
        alt_ac1 = 0.0

    dt_seg = None
    if t.size >= n:
        dt_seg = np.diff(t[:n])
        dt_seg = np.clip(dt_seg, 1e-6, None)

    if dt_seg is not None and len(dt_seg) > 0:
        spd = step2[: len(dt_seg)] / dt_seg
        dt_mean, dt_std = float(np.mean(dt_seg)), float(np.std(dt_seg))
        dt_cv = float(dt_std / (dt_mean + 1e-6))
        dt_p90 = _sp(dt_seg, 90)
        dt_reg = float(np.mean(np.abs(dt_seg - dt_mean) < 0.5 * (dt_std + 1e-9)))
    else:
        spd = step2
        dt_mean = dt_std = dt_cv = dt_p90 = dt_reg = 0.0

    spd_mean = float(np.mean(spd)) if len(spd) else 0.0
    spd_std = float(np.std(spd)) if len(spd) else 0.0
    spd_cv = float(spd_std / (spd_mean + 1e-6))
    spd_p10 = _sp(spd, 10)
    spd_p90 = _sp(spd, 90)
    fast_straight = float(spd_mean * (disp / total2))

    step_cv = float(np.std(step2) / (np.mean(step2) + 1e-6)) if len(step2) else 0.0
    pct_above_median_alt = float(np.mean(z > np.median(z)))
    alt_hist, _ = np.histogram(z, bins=10, density=True)
    alt_hist = alt_hist[alt_hist > 0]
    alt_entropy = float(-np.sum(alt_hist * np.log(alt_hist + 1e-9))) if len(alt_hist) else 0.0

    # 3D curvature summary
    curv3d = []
    for i in range(1, n - 1):
        p0 = np.array([x[i - 1], y[i - 1], z[i - 1]])
        p1 = np.array([x[i], y[i], z[i]])
        p2 = np.array([x[i + 1], y[i + 1], z[i + 1]])
        v1, v2 = p1 - p0, p2 - p1
        cr = np.cross(v1, v2)
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(v1 + v2)
        curv3d.append(np.linalg.norm(cr) / (denom + 1e-9))
    curv3d = np.asarray(curv3d, dtype=np.float64) if curv3d else np.asarray([0.0], dtype=np.float64)

    feat = np.array([
        float(n), total2, total3, disp, float(disp / total2),  # 0-4
        float(np.mean(seg_st)), float(np.min(seg_st)), float(np.std(seg_st)),  # 5-7
        float(np.mean(angles)), float(np.std(angles)), float(np.median(angles)), _sp(angles, 95),  # 8-11
        float(np.mean(angles < 10.0)), float(np.mean(angles > 45.0)), _entropy(angles, 18),  # 12-14
        float(np.mean(z)), float(np.std(z)), float(np.median(z)), _sp(z, 10), _sp(z, 90), alt_range,  # 15-20
        float(1.0 / (np.std(z) + 1e-6)), alt_cv, float(alt_range / (np.mean(z) + 1e-6)),  # 21-23
        alt_ac1, alt_dchg, float(np.polyfit(np.arange(n), z, 1)[0]),  # 24-26
        spd_mean, spd_std, spd_cv, spd_p10, spd_p90, fast_straight,  # 27-32
        float(np.mean(np.abs(dz))), float(np.std(np.abs(dz))), float(np.mean(np.abs(dz) < _sp(np.abs(dz) + 1e-9, 25))),  # 33-35
        float(alt_cv * np.mean(angles)), step_cv, pct_above_median_alt, alt_entropy,  # 36-39
        dt_mean, dt_std, dt_cv, dt_p90, dt_reg,  # 40-44
        float(np.mean(curv3d)), _sp(curv3d, 95), float(np.std(curv3d)), float(np.mean(curv3d < _sp(curv3d + 1e-9, 25))),  # 45-48
        float(np.mean(step3)), float(np.std(step3)), float(np.mean(step3) / (np.mean(step2) + 1e-6)),  # 49-51
    ], dtype=np.float64)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def macro_ap(probs: np.ndarray, labels: np.ndarray, class_order: list[str], mask: np.ndarray) -> float:
    p = probs[mask]
    y = labels[mask]
    vals = []
    for i, cls in enumerate(class_order):
        yb = (y == cls).astype(np.int32)
        if yb.sum() > 0:
            vals.append(average_precision_score(yb, p[:, i]))
    return float(np.mean(vals))


def train_detector_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    cls_name: str,
    neg_classes: list[str],
    seed: int,
    iterations: int,
    thread_count: int,
) -> np.ndarray:
    mask = np.isin(y_train, [cls_name, *neg_classes])
    Xb = X_train[mask]
    yb = (y_train[mask] == cls_name).astype(np.int32)
    if yb.sum() < 3:
        return np.zeros((len(X_pred),), dtype=np.float32)
    pos = int(yb.sum())
    neg = int((yb == 0).sum())
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.03,
        depth=6,
        class_weights={0: 1.0, 1: float(neg / (pos + 1e-6))},
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=thread_count,
    )
    model.fit(Xb, yb)
    return model.predict_proba(X_pred)[:, 1].astype(np.float32)


def build_meta(bp: np.ndarray, w: np.ndarray, c: np.ndarray, idx_w: int, idx_c: int, idx_g: int, idx_s: int) -> np.ndarray:
    return np.column_stack([
        bp,
        w,
        c,
        w * bp[:, idx_w],
        c * bp[:, idx_c],
        w - bp[:, idx_w],
        c - bp[:, idx_c],
        bp.max(axis=1),
        bp[:, idx_g] + bp[:, idx_s],
    ]).astype(np.float32)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...", flush=True)
    train_df = pd.read_csv(args.train_path)
    label_col = "bird_group" if "bird_group" in train_df.columns else "label"
    labels = train_df[label_col].astype(str).to_numpy()
    track_ids = pd.to_numeric(train_df["track_id"], errors="coerce").astype("Int64")
    if track_ids.isna().any():
        raise ValueError("Non-numeric track_id found.")
    track_ids = track_ids.to_numpy(dtype=np.int64)

    teacher_oof = pd.read_csv(args.teacher_oof_csv)
    class_order = [c for c in teacher_oof.columns if c != "track_id"]
    idx_w = class_order.index("Waders")
    idx_c = class_order.index("Cormorants")
    idx_g = class_order.index("Gulls")
    idx_s = class_order.index("Songbirds")

    teacher_probs = teacher_oof.set_index("track_id").loc[track_ids, class_order].to_numpy(dtype=np.float32)
    tcn_probs = np.asarray(np.load(args.tcn_oof_npy), dtype=np.float32)
    if tcn_probs.shape != teacher_probs.shape:
        raise ValueError(f"tcn_oof shape mismatch: {tcn_probs.shape} vs {teacher_probs.shape}")

    base_probs_raw = np.clip(0.75 * teacher_probs + 0.25 * tcn_probs, 0.0, None)
    base_probs = base_probs_raw / np.clip(base_probs_raw.sum(axis=1, keepdims=True), 1e-9, None)

    if Path(args.covered_idx_npy).exists():
        covered_idx = np.asarray(np.load(args.covered_idx_npy), dtype=np.int64).reshape(-1)
        covered_mask = np.zeros((len(track_ids),), dtype=bool)
        covered_mask[covered_idx] = True
    else:
        covered_mask = np.ones((len(track_ids),), dtype=bool)

    print("Extracting raw features...", flush=True)
    X_raw = np.vstack([extract_features(row) for _, row in train_df.iterrows()])
    print(f"X_raw shape: {X_raw.shape}", flush=True)

    meta_oof = np.zeros_like(base_probs, dtype=np.float32)
    outer = StratifiedKFold(n_splits=args.outer_splits, shuffle=True, random_state=args.seed)

    print(f"Nested OOF: outer={args.outer_splits}, inner={args.inner_splits}", flush=True)
    for ofold, (otr, oval) in enumerate(outer.split(X_raw, labels), start=1):
        print(f"-- Outer fold {ofold}/{args.outer_splits}", flush=True)

        X_tr, X_val = X_raw[otr], X_raw[oval]
        y_tr, y_val = labels[otr], labels[oval]
        bp_tr, bp_val = base_probs[otr], base_probs[oval]

        # Inner OOF detector predictions for outer-train
        w_inner = np.zeros((len(otr),), dtype=np.float32)
        c_inner = np.zeros((len(otr),), dtype=np.float32)
        inner = StratifiedKFold(n_splits=args.inner_splits, shuffle=True, random_state=args.seed + 100 + ofold)

        for cls_name, out_arr, neg_cls, seed_shift in [
            ("Waders", w_inner, ["Gulls", "Songbirds"], 1000),
            ("Cormorants", c_inner, ["Gulls", "Songbirds", "Ducks"], 2000),
        ]:
            mask_bin = np.isin(y_tr, [cls_name, *neg_cls])
            X_bin = X_tr[mask_bin]
            y_bin = (y_tr[mask_bin] == cls_name).astype(np.int32)
            idx_bin = np.where(mask_bin)[0]
            if y_bin.sum() < args.inner_splits:
                continue
            for ifold, (itr, ival) in enumerate(inner.split(X_bin, y_bin), start=1):
                y_itr = y_bin[itr]
                pos = int(y_itr.sum())
                neg = int((y_itr == 0).sum())
                model = CatBoostClassifier(
                    iterations=args.detector_iters,
                    learning_rate=0.03,
                    depth=6,
                    class_weights={0: 1.0, 1: float(neg / (pos + 1e-6))},
                    random_seed=args.seed + seed_shift + ofold * 10 + ifold,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=args.thread_count,
                )
                model.fit(X_bin[itr], y_bin[itr], eval_set=(X_bin[ival], y_bin[ival]), early_stopping_rounds=50, use_best_model=True, verbose=False)
                out_arr[idx_bin[ival]] = model.predict_proba(X_bin[ival])[:, 1].astype(np.float32)

        # Detector predictions for outer-val (trained on full outer-train)
        w_val = train_detector_predict(
            X_train=X_tr,
            y_train=y_tr,
            X_pred=X_val,
            cls_name="Waders",
            neg_classes=["Gulls", "Songbirds"],
            seed=args.seed + 3000 + ofold,
            iterations=args.detector_iters,
            thread_count=args.thread_count,
        )
        c_val = train_detector_predict(
            X_train=X_tr,
            y_train=y_tr,
            X_pred=X_val,
            cls_name="Cormorants",
            neg_classes=["Gulls", "Songbirds", "Ducks"],
            seed=args.seed + 4000 + ofold,
            iterations=args.detector_iters,
            thread_count=args.thread_count,
        )

        meta_tr = build_meta(bp_tr, w_inner, c_inner, idx_w, idx_c, idx_g, idx_s)
        meta_val = build_meta(bp_val, w_val, c_val, idx_w, idx_c, idx_g, idx_s)

        fold_probs = np.zeros((len(oval), len(class_order)), dtype=np.float32)
        for ci, cls in enumerate(class_order):
            y_bin = (y_tr == cls).astype(np.int32)
            pos = int(y_bin.sum())
            neg = int((y_bin == 0).sum())
            if pos < 2:
                fold_probs[:, ci] = bp_val[:, ci]
                continue
            m = CatBoostClassifier(
                iterations=args.meta_iters,
                learning_rate=0.05,
                depth=4,
                class_weights={0: 1.0, 1: float(neg / (pos + 1e-6))},
                random_seed=args.seed + 5000 + ofold + ci,
                verbose=False,
                allow_writing_files=False,
                thread_count=args.thread_count,
            )
            m.fit(meta_tr, y_bin)
            fold_probs[:, ci] = m.predict_proba(meta_val)[:, 1].astype(np.float32)

        fold_probs = np.clip(fold_probs, 0.0, None)
        fold_probs = fold_probs / np.clip(fold_probs.sum(axis=1, keepdims=True), 1e-9, None)
        meta_oof[oval] = fold_probs

        val_cov = covered_mask[oval]
        if val_cov.sum() > 0:
            fold_ap = macro_ap(fold_probs, y_val, class_order, val_cov)
            print(f"   covered macro AP: {fold_ap:.4f}", flush=True)

    print("\nFinal evaluation (covered)...", flush=True)
    p_cov_meta = meta_oof[covered_mask]
    y_cov = labels[covered_mask]
    p_cov_base = base_probs[covered_mask]

    rows = []
    for i, cls in enumerate(class_order):
        yb = (y_cov == cls).astype(np.int32)
        if yb.sum() == 0:
            continue
        ap_b = average_precision_score(yb, p_cov_base[:, i])
        ap_m = average_precision_score(yb, p_cov_meta[:, i])
        rows.append({"class": cls, "ap_base": float(ap_b), "ap_meta": float(ap_m), "delta": float(ap_m - ap_b), "support": int(yb.sum())})

    res_df = pd.DataFrame(rows)
    base_macro = float(res_df["ap_base"].mean())
    meta_macro = float(res_df["ap_meta"].mean())

    summary = pd.DataFrame(
        [
            {
                "baseline_macro_covered": base_macro,
                "meta_macro_covered": meta_macro,
                "delta_macro_covered": meta_macro - base_macro,
                "n_features_raw": int(X_raw.shape[1]),
                "n_features_meta": int(build_meta(base_probs[:1], np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), idx_w, idx_c, idx_g, idx_s).shape[1]),
                "n_rows": int(len(labels)),
                "covered_count": int(covered_mask.sum()),
            }
        ]
    )

    res_df.to_csv(out_dir / "results.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    np.save(out_dir / "meta_oof_probs.npy", meta_oof)
    np.save(out_dir / "base_probs_norm.npy", base_probs)

    print(res_df.to_string(index=False), flush=True)
    print(f"\nmacro AP: {base_macro:.6f} -> {meta_macro:.6f} ({meta_macro - base_macro:+.6f})", flush=True)
    print(f"Saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()

