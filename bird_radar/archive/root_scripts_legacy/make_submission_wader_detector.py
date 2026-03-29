#!/usr/bin/env python3
"""
Final submission: base blend w75/25 + Wader detector (aw=0.65)
"""

from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from shapely import wkb

warnings.filterwarnings("ignore")

# Paths
ROOT = Path("/Users/sgrisshk/Desktop/AI-task")
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
TEACHER_OOF = ROOT / "bird_radar/artifacts/oof_csv_for_per_class/oof_teacher_fr_complete.csv"
TCN_OOF = ROOT / "bird_radar/artifacts/tcn_trajectory_forward_2stream_distill_full_seed777_cpu/artifacts/tcn_oof.npy"
BASE_SUB_CANDIDATES = [
    ROOT / "bird_radar/submissions/sub_teacherfr_tcn_distill_w75_25.csv",
    ROOT / "bird_radar/artifacts/tcn_trajectory_forward_2stream_distill_full_seed777_cpu_blends/sub_teacherfr_tcn_distill_w75_25.csv",
]
OUT_SUB = ROOT / "bird_radar/submissions/sub_wader_detector_aw65.csv"
SAMPLE_SUB = ROOT / "sample_submission.csv"

AW = 0.65


def parse_ewkb(hex_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geom = wkb.loads(str(hex_str), hex=True)
    coords = np.asarray(geom.coords, dtype=np.float64)
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2] if coords.shape[1] >= 3 else np.zeros((len(x),), dtype=np.float64)
    return x, y, z


def parse_dt(val) -> np.ndarray | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (list, tuple, np.ndarray)):
        return np.asarray(val, dtype=np.float64)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            return np.asarray(json.loads(s), dtype=np.float64)
        except Exception:
            try:
                return np.asarray(ast.literal_eval(s), dtype=np.float64)
            except Exception:
                return None
    return None


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
        dt = parse_dt(row.get("trajectory_time"))
    except Exception:
        return np.zeros((47,), dtype=np.float32)

    n = len(x)
    if n < 3:
        return np.zeros((47,), dtype=np.float32)

    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
    step2d = np.sqrt(dx * dx + dy * dy)
    total2d = float(np.sum(step2d)) + 1e-9
    disp = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))

    # turn angles
    angles = []
    for i in range(len(dx) - 1):
        v1 = np.array([dx[i], dy[i]], dtype=np.float64)
        v2 = np.array([dx[i + 1], dy[i + 1]], dtype=np.float64)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        angles.append(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))
    angles = np.asarray(angles, dtype=np.float64) if angles else np.asarray([0.0], dtype=np.float64)

    # segment straightness
    seg_st = []
    seg_len = max(n // 5, 2)
    for s in range(5):
        a, b = s * seg_len, (s + 1) * seg_len
        xs, ys = x[a:b], y[a:b]
        if len(xs) < 2:
            continue
        tp = float(np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2))) + 1e-9
        d = float(np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2))
        seg_st.append(d / tp)

    alt_cv = float(np.std(z) / (np.mean(z) + 1e-6))
    if n > 5:
        alt_ac1 = float(np.corrcoef(z[:-1], z[1:])[0, 1])
        if not np.isfinite(alt_ac1):
            alt_ac1 = 0.0
    else:
        alt_ac1 = 0.0
    dz_sign = np.sign(dz[dz != 0])
    alt_dchg = float(np.sum(np.diff(dz_sign) != 0) / len(dz_sign)) if len(dz_sign) > 0 else 0.0

    if dt is not None and len(dt) >= len(dx):
        dt_seg = np.clip(dt[: len(dx)], 1e-6, None)
        spd = step2d / dt_seg
    else:
        dt_seg = None
        spd = step2d

    # 3d curvature
    curv3d = []
    for i in range(1, n - 1):
        p0 = np.array([x[i - 1], y[i - 1], z[i - 1]], dtype=np.float64)
        p1 = np.array([x[i], y[i], z[i]], dtype=np.float64)
        p2 = np.array([x[i + 1], y[i + 1], z[i + 1]], dtype=np.float64)
        v1, v2 = p1 - p0, p2 - p1
        cr = np.cross(v1, v2)
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(v1 + v2)
        curv3d.append(np.linalg.norm(cr) / (denom + 1e-9))
    curv3d = np.asarray(curv3d, dtype=np.float64) if curv3d else np.asarray([0.0], dtype=np.float64)

    # align z with segment-level vectors
    z_seg = z[: len(spd)] if len(spd) > 0 else np.asarray([], dtype=np.float64)

    feat = np.array(
        [
            float(n),
            total2d,
            disp,
            float(disp / total2d),
            float(np.mean(seg_st)) if seg_st else 0.0,
            float(np.min(seg_st)) if seg_st else 0.0,
            float(np.mean(angles)),
            float(np.std(angles)),
            float(np.median(angles)),
            _sp(angles, 95),
            float(np.mean(angles < 10.0)),
            float(np.mean(angles > 45.0)),
            _entropy(angles, 18),
            float(np.mean(z)),
            float(np.std(z)),
            float(np.median(z)),
            _sp(z, 10),
            _sp(z, 90),
            float(np.max(z) - np.min(z)),
            alt_cv,
            float((np.max(z) - np.min(z)) / (np.mean(z) + 1e-6)),
            alt_ac1,
            alt_dchg,
            float(np.polyfit(np.arange(n), z, 1)[0]),
            float(np.mean(spd)),
            float(np.std(spd)),
            float(np.std(spd) / (np.mean(spd) + 1e-6)),
            _sp(spd, 10),
            _sp(spd, 90),
            float(np.mean(spd[z_seg < _sp(z_seg, 33)])) if len(z_seg) and np.any(z_seg < _sp(z_seg, 33)) else 0.0,
            float(np.corrcoef(spd, z_seg)[0, 1]) if len(spd) > 3 and len(z_seg) == len(spd) else 0.0,
            float(np.mean(spd) * disp / total2d),
            float(np.mean(np.abs(dz))),
            float(np.std(np.abs(dz))),
            float(np.mean(np.abs(dz) < _sp(np.abs(dz) + 1e-9, 25))),
            float(alt_cv * np.mean(angles)),
            float(np.std(step2d) / (np.mean(step2d) + 1e-6)),
            float(np.mean(z > np.median(z))),
            float(np.mean(dt_seg)) if dt_seg is not None else 0.0,
            float(np.std(dt_seg)) if dt_seg is not None else 0.0,
            float(np.std(dt_seg) / (np.mean(dt_seg) + 1e-6)) if dt_seg is not None else 0.0,
            _sp(dt_seg, 90) if dt_seg is not None else 0.0,
            float(np.mean(curv3d)),
            _sp(curv3d, 95),
            float(np.std(curv3d)),
            float(np.mean(curv3d < _sp(curv3d + 1e-9, 25))),
            float(n) / total2d,
        ],
        dtype=np.float64,
    )
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def main() -> None:
    print("=== Loading data ===")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    label_col = "bird_group" if "bird_group" in train_df.columns else "label"
    labels = train_df[label_col].astype(str).to_numpy()
    print(f"train={len(train_df)} test={len(test_df)}")

    # class order from sample submission to match competition schema
    sample_cols = pd.read_csv(SAMPLE_SUB, nrows=1).columns.tolist()
    class_order = [c for c in sample_cols if c != "track_id"]
    wader_idx = class_order.index("Waders")
    if set(class_order) != set(pd.read_csv(TEACHER_OOF, nrows=1).columns.tolist()) - {"track_id"}:
        raise ValueError("Class set mismatch between sample submission and teacher OOF")

    print("\n=== Building features ===")
    X_train = np.vstack([extract_features(row) for _, row in train_df.iterrows()])
    X_test = np.vstack([extract_features(row) for _, row in test_df.iterrows()])
    print(f"X_train={X_train.shape} X_test={X_test.shape}")

    print("\n=== Train Wader detector ===")
    neg_cls = ["Gulls", "Songbirds"]
    mask = np.isin(labels, ["Waders", *neg_cls])
    X_bin = X_train[mask]
    y_bin = (labels[mask] == "Waders").astype(np.int32)
    pos = int(y_bin.sum())
    neg = int((y_bin == 0).sum())
    print(f"Waders={pos} negatives={neg} ratio={neg/(pos+1e-6):.2f}x")

    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.03,
        depth=6,
        class_weights={0: 1.0, 1: float(neg / (pos + 1e-6))},
        random_seed=42,
        verbose=200,
        allow_writing_files=False,
    )
    model.fit(X_bin, y_bin)

    wader_scores_test = model.predict_proba(X_test)[:, 1].astype(np.float32)
    print(
        "Wader score test stats:",
        f"mean={wader_scores_test.mean():.4f}",
        f"p50={np.percentile(wader_scores_test, 50):.4f}",
        f"p95={np.percentile(wader_scores_test, 95):.4f}",
        f"max={wader_scores_test.max():.4f}",
    )

    print("\n=== Load base submission ===")
    base_sub = None
    for p in BASE_SUB_CANDIDATES:
        if p.exists():
            base_sub = pd.read_csv(p)
            print(f"base submission: {p}")
            break
    if base_sub is None:
        raise FileNotFoundError(f"Base submission not found in candidates: {BASE_SUB_CANDIDATES}")

    test_ids = pd.to_numeric(test_df["track_id"], errors="coerce").astype("Int64").to_numpy()
    if pd.isna(test_ids).any():
        raise ValueError("test track_id has non-numeric values")
    test_ids = test_ids.astype(np.int64)

    base_prob = base_sub.set_index("track_id").loc[test_ids, class_order].to_numpy(dtype=np.float32)
    base_prob = base_prob / np.clip(base_prob.sum(axis=1, keepdims=True), 1e-9, None)

    print(f"\n=== Blend aw={AW:.2f} ===")
    final = base_prob.copy()
    final[:, wader_idx] = (1.0 - AW) * final[:, wader_idx] + AW * wader_scores_test
    final = final / np.clip(final.sum(axis=1, keepdims=True), 1e-9, None)
    print(f"Waders mean: base={base_prob[:, wader_idx].mean():.4f} -> final={final[:, wader_idx].mean():.4f}")

    print("\n=== Save submission ===")
    OUT_SUB.parent.mkdir(parents=True, exist_ok=True)
    sub_df = pd.DataFrame(final, columns=class_order)
    sub_df.insert(0, "track_id", test_ids)
    sub_df.to_csv(OUT_SUB, index=False)
    print(f"Saved: {OUT_SUB}")
    print(f"Shape: {sub_df.shape}")


if __name__ == "__main__":
    main()

