#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import build_track_cache, load_track_cache, save_track_cache

EPS = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract physics/dynamics features from trajectory cache.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--train-out-parquet", required=True)
    p.add_argument("--test-out-parquet", required=True)
    p.add_argument("--cache-dir", default="bird_radar/artifacts/cache")
    p.add_argument("--id-col", default="track_id")
    p.add_argument("--label-col", default="bird_group")
    return p.parse_args()


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict[str, Any]]:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


def _safe_diff(a: np.ndarray) -> np.ndarray:
    if a.size <= 1:
        return np.zeros_like(a, dtype=np.float32)
    return np.diff(a, prepend=a[:1]).astype(np.float32, copy=False)


def _wrapped_angle_diff(a: np.ndarray) -> np.ndarray:
    d = _safe_diff(a)
    return np.arctan2(np.sin(d), np.cos(d)).astype(np.float32, copy=False)


def _autocorr_lag(a: np.ndarray, lag: int) -> float:
    if a.size <= lag + 1:
        return 0.0
    x = a[:-lag].astype(np.float64, copy=False)
    y = a[lag:].astype(np.float64, copy=False)
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    den = float(np.sqrt(np.sum(x * x) * np.sum(y * y)) + EPS)
    return float(np.sum(x * y) / den)


def _slope_feature(a: np.ndarray) -> float:
    if a.size <= 1:
        return 0.0
    x = np.arange(a.size, dtype=np.float64)
    x = x - float(np.mean(x))
    y = a.astype(np.float64, copy=False) - float(np.mean(a))
    den = float(np.sum(x * x) + EPS)
    return float(np.sum(x * y) / den)


def _resample_1d(a: np.ndarray, n: int = 128) -> np.ndarray:
    if a.size == 0:
        return np.zeros((n,), dtype=np.float32)
    if a.size == 1:
        return np.full((n,), float(a[0]), dtype=np.float32)
    xp = np.linspace(0.0, 1.0, num=a.size, dtype=np.float32)
    xnew = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    return np.interp(xnew, xp, a).astype(np.float32, copy=False)


def _spectral_features(a: np.ndarray, prefix: str) -> dict[str, float]:
    x = _resample_1d(a, 128).astype(np.float64, copy=False)
    x = x - float(np.mean(x))
    fft = np.fft.rfft(x)
    power = (np.abs(fft) ** 2).astype(np.float64, copy=False)
    total = float(np.sum(power))
    if total <= EPS:
        return {
            f"{prefix}_dom_freq": 0.0,
            f"{prefix}_spec_entropy": 0.0,
            f"{prefix}_spec_centroid": 0.0,
            f"{prefix}_low_band_ratio": 0.0,
            f"{prefix}_high_band_ratio": 0.0,
        }
    p = power / (total + EPS)
    idx = np.arange(len(p), dtype=np.float64)
    dom = int(np.argmax(power[1:]) + 1) if len(power) > 1 else 0
    centroid = float(np.sum(idx * p))
    entropy = float(-np.sum(p * np.log(p + EPS)))
    low_end = max(2, len(p) // 8)
    high_start = max(2, len(p) * 3 // 4)
    return {
        f"{prefix}_dom_freq": float(dom),
        f"{prefix}_spec_entropy": entropy,
        f"{prefix}_spec_centroid": centroid,
        f"{prefix}_low_band_ratio": float(np.sum(p[:low_end])),
        f"{prefix}_high_band_ratio": float(np.sum(p[high_start:])),
    }


def _robust_stats(a: np.ndarray, prefix: str) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    if a.size == 0:
        keys = [
            "mean",
            "std",
            "min",
            "max",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "iqr",
            "skew",
            "kurt",
        ]
        return {f"{prefix}_{k}": 0.0 for k in keys}
    q10, q25, q50, q75, q90 = np.quantile(a, [0.10, 0.25, 0.50, 0.75, 0.90])
    m = float(np.mean(a))
    s = float(np.std(a))
    z = (a - m) / (s + EPS)
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3.0)
    return {
        f"{prefix}_mean": m,
        f"{prefix}_std": s,
        f"{prefix}_min": float(np.min(a)),
        f"{prefix}_max": float(np.max(a)),
        f"{prefix}_p10": float(q10),
        f"{prefix}_p25": float(q25),
        f"{prefix}_p50": float(q50),
        f"{prefix}_p75": float(q75),
        f"{prefix}_p90": float(q90),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_skew": skew,
        f"{prefix}_kurt": kurt,
    }


def _segment_stats(a: np.ndarray, prefix: str) -> dict[str, float]:
    n = int(a.size)
    if n == 0:
        out: dict[str, float] = {}
        for seg in ["all", "beg", "mid", "end"]:
            out.update(_robust_stats(a, f"{prefix}_{seg}"))
        return out
    k1 = n // 3
    k2 = 2 * n // 3
    out: dict[str, float] = {}
    out.update(_robust_stats(a, f"{prefix}_all"))
    out.update(_robust_stats(a[:k1], f"{prefix}_beg"))
    out.update(_robust_stats(a[k1:k2], f"{prefix}_mid"))
    out.update(_robust_stats(a[k2:], f"{prefix}_end"))
    return out


def _longest_run(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    best = 0
    cur = 0
    for v in mask.astype(np.int8).tolist():
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return float(best)


def extract_features_from_raw(raw: np.ndarray) -> dict[str, float]:
    # raw: [T, 9] = x,y,alt,rcs,speed,vz,acc,curv,dt
    if raw.ndim != 2 or raw.shape[1] != 9:
        raise ValueError(f"expected raw shape [T,9], got {raw.shape}")
    x = raw[:, 0].astype(np.float32, copy=False)
    y = raw[:, 1].astype(np.float32, copy=False)
    alt = raw[:, 2].astype(np.float32, copy=False)
    speed = raw[:, 4].astype(np.float32, copy=False)
    vz = raw[:, 5].astype(np.float32, copy=False)
    acc = raw[:, 6].astype(np.float32, copy=False)
    curv = raw[:, 7].astype(np.float32, copy=False)
    dt = np.clip(raw[:, 8].astype(np.float32, copy=False), 1e-3, None)

    dx = _safe_diff(x)
    dy = _safe_diff(y)
    step_dist = np.sqrt(dx * dx + dy * dy).astype(np.float32, copy=False)
    heading = np.arctan2(dy, dx).astype(np.float32, copy=False)
    turn = _wrapped_angle_diff(heading)
    turn_rate = (turn / dt).astype(np.float32, copy=False)
    jerk = (_safe_diff(acc) / dt).astype(np.float32, copy=False)
    dv = _safe_diff(speed)
    da = _safe_diff(acc)
    dcurv = _safe_diff(curv)

    feats: dict[str, float] = {"track_len": float(len(raw))}

    for arr, name in [
        (speed, "speed"),
        (vz, "vz"),
        (acc, "acc"),
        (jerk, "jerk"),
        (dv, "dspeed"),
        (da, "dacc"),
        (dcurv, "dcurv"),
        (np.abs(turn), "turn_abs"),
        (np.abs(turn_rate), "turn_rate_abs"),
        (np.abs(curv), "curv_abs"),
        (alt, "alt"),
        (dt, "dt"),
    ]:
        feats.update(_segment_stats(arr, name))

    path_length = float(np.sum(step_dist))
    if x.size >= 2:
        net_disp = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))
        bbox_w = float(np.max(x) - np.min(x))
        bbox_h = float(np.max(y) - np.min(y))
        radius = np.sqrt((x - float(np.mean(x))) ** 2 + (y - float(np.mean(y))) ** 2).astype(np.float32)
    else:
        net_disp = bbox_w = bbox_h = 0.0
        radius = np.zeros((0,), dtype=np.float32)
    feats["path_length"] = path_length
    feats["net_displacement"] = net_disp
    feats["straightness"] = float(net_disp / (path_length + EPS))
    feats["bbox_w"] = bbox_w
    feats["bbox_h"] = bbox_h
    feats["bbox_area"] = float(bbox_w * bbox_h)
    feats["heading_change_total_abs"] = float(np.sum(np.abs(turn)))
    feats.update(_robust_stats(radius, "radius"))

    feats["alt_range"] = float(np.max(alt) - np.min(alt)) if alt.size else 0.0
    feats["alt_slope"] = _slope_feature(alt)
    feats["speed_slope"] = _slope_feature(speed)
    feats["turn_rate_slope"] = _slope_feature(turn_rate)
    feats["vz_slope"] = _slope_feature(vz)

    climb_mask = vz > 0.0
    desc_mask = vz < 0.0
    flat_mask = np.abs(vz) <= np.quantile(np.abs(vz), 0.2) if vz.size else np.zeros((0,), dtype=bool)
    feats["ratio_climb"] = float(np.mean(climb_mask)) if vz.size else 0.0
    feats["ratio_descent"] = float(np.mean(desc_mask)) if vz.size else 0.0
    feats["ratio_flat_vz"] = float(np.mean(flat_mask)) if flat_mask.size else 0.0
    feats["longest_climb_run"] = _longest_run(climb_mask)
    feats["longest_descent_run"] = _longest_run(desc_mask)
    feats["vz_sign_changes"] = float(np.sum(np.abs(np.diff(np.sign(vz))) > 0)) if vz.size > 1 else 0.0

    if speed.size:
        q25, q75 = np.quantile(speed, [0.25, 0.75])
        feats["ratio_speed_low"] = float(np.mean(speed <= q25))
        feats["ratio_speed_high"] = float(np.mean(speed >= q75))
    else:
        feats["ratio_speed_low"] = 0.0
        feats["ratio_speed_high"] = 0.0
    if turn_rate.size:
        thr75, thr90 = np.quantile(np.abs(turn_rate), [0.75, 0.90])
        feats["ratio_turn_high_p75"] = float(np.mean(np.abs(turn_rate) >= thr75))
        feats["ratio_turn_high_p90"] = float(np.mean(np.abs(turn_rate) >= thr90))
    else:
        feats["ratio_turn_high_p75"] = 0.0
        feats["ratio_turn_high_p90"] = 0.0

    for lag in [1, 3, 5]:
        feats[f"speed_acf_lag{lag}"] = _autocorr_lag(speed, lag)
        feats[f"vz_acf_lag{lag}"] = _autocorr_lag(vz, lag)
        feats[f"turn_rate_acf_lag{lag}"] = _autocorr_lag(turn_rate, lag)

    feats.update(_spectral_features(speed, "speed_spec"))
    feats.update(_spectral_features(vz, "vz_spec"))
    feats.update(_spectral_features(turn_rate, "turn_spec"))
    feats.update(_spectral_features(alt, "alt_spec"))

    return feats


def _build_feature_frame(
    df: pd.DataFrame,
    cache: dict[int, dict[str, Any]],
    id_col: str,
    label_col: str,
    include_label: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ids = df[id_col].to_numpy(dtype=np.int64)
    labels = df[label_col].astype(str).tolist() if include_label and label_col in df.columns else None
    for i, tid in enumerate(ids.tolist()):
        item = cache.get(int(tid))
        if item is None:
            raise ValueError(f"track_id={tid} missing in cache")
        raw = np.asarray(item["raw_features"], dtype=np.float32)
        feat = extract_features_from_raw(raw)
        feat[id_col] = int(tid)
        if labels is not None:
            feat[label_col] = labels[i]
        rows.append(feat)
    out = pd.DataFrame(rows)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def main() -> None:
    args = parse_args()
    id_col = str(args.id_col)
    label_col = str(args.label_col)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    if id_col not in train_df.columns or id_col not in test_df.columns:
        raise ValueError(f"id column '{id_col}' must exist in both train/test")

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")

    feat_train = _build_feature_frame(train_df, train_cache, id_col=id_col, label_col=label_col, include_label=True)
    feat_test = _build_feature_frame(test_df, test_cache, id_col=id_col, label_col=label_col, include_label=False)

    train_out = Path(args.train_out_parquet).resolve()
    test_out = Path(args.test_out_parquet).resolve()
    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)

    feat_train.to_parquet(train_out, index=False)
    feat_test.to_parquet(test_out, index=False)

    print(f"train_shape={feat_train.shape}", flush=True)
    print(f"test_shape={feat_test.shape}", flush=True)
    print(f"train_out={train_out}", flush=True)
    print(f"test_out={test_out}", flush=True)


if __name__ == "__main__":
    main()

