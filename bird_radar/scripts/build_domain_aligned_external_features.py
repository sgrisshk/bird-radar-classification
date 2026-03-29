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

from src.feature_engineering import build_feature_frame, compute_monthly_track_centers
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build domain-aligned external features (Quantile map + CORAL).")
    p.add_argument("--train-csv", type=str, default="train.csv")
    p.add_argument("--test-csv", type=str, default="test.csv")
    p.add_argument("--cache-dir", type=str, default="bird_radar/artifacts/cache")
    p.add_argument("--output-train-parquet", required=True)
    p.add_argument("--output-test-parquet", required=True)
    p.add_argument("--output-json", type=str, default="")
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--prefix", type=str, default="da_qc")
    return p.parse_args()


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict]:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


def _ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    if np.all(a == a[0]) and np.all(b == b[0]):
        return 0.0
    sa = np.sort(a)
    sb = np.sort(b)
    grid = np.unique(np.concatenate([sa, sb]))
    cdf_a = np.searchsorted(sa, grid, side="right") / float(len(sa))
    cdf_b = np.searchsorted(sb, grid, side="right") / float(len(sb))
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _quantile_map_source_to_target(x_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
    src = x_src.astype(np.float64, copy=False)
    tgt = x_tgt.astype(np.float64, copy=False)
    src_sorted = np.sort(src)
    tgt_sorted = np.sort(tgt)
    if len(src_sorted) == 0 or len(tgt_sorted) == 0:
        return x_src.astype(np.float64, copy=False)
    p = np.searchsorted(src_sorted, src, side="right") / float(len(src_sorted))
    q_grid = np.linspace(0.0, 1.0, len(tgt_sorted), dtype=np.float64)
    mapped = np.interp(p, q_grid, tgt_sorted)
    return mapped


def _sqrtm_psd(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, eps, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def _invsqrtm_psd(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, eps, None)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def main() -> None:
    args = parse_args()

    train_csv = Path(args.train_csv).resolve()
    test_csv = Path(args.test_csv).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    out_train = Path(args.output_train_parquet).resolve()
    out_test = Path(args.output_test_parquet).resolve()
    out_json = Path(args.output_json).resolve() if str(args.output_json).strip() else out_train.with_suffix(".json")

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")

    monthly_centers = compute_monthly_track_centers(train_df, track_cache=train_cache)
    feat_train = build_feature_frame(train_df, track_cache=train_cache, monthly_centers=monthly_centers)
    feat_test = build_feature_frame(test_df, track_cache=test_cache, monthly_centers=monthly_centers)

    exclude = {"track_id", "observation_id", "primary_observation_id"}
    cols = [c for c in feat_train.columns if c not in exclude]

    # Rank features by domain shift (KS).
    ks_rows: list[dict[str, float | str]] = []
    for c in cols:
        ks = _ks_stat(feat_train[c].to_numpy(dtype=np.float64), feat_test[c].to_numpy(dtype=np.float64))
        ks_rows.append({"feature": c, "ks": float(ks)})
    ks_df = pd.DataFrame(ks_rows).sort_values("ks", ascending=False)
    top_k = max(1, min(int(args.top_k), len(ks_df)))
    selected = ks_df.head(top_k)["feature"].tolist()

    Xs = feat_train[selected].to_numpy(dtype=np.float64)
    Xt = feat_test[selected].to_numpy(dtype=np.float64)

    # 1) Quantile map source to target marginal per-feature.
    Xs_q = np.zeros_like(Xs)
    for j in range(Xs.shape[1]):
        Xs_q[:, j] = _quantile_map_source_to_target(Xs[:, j], Xt[:, j])

    # 2) CORAL covariance alignment: source -> target.
    mu_s = Xs_q.mean(axis=0, keepdims=True)
    mu_t = Xt.mean(axis=0, keepdims=True)
    Xs0 = Xs_q - mu_s
    Xt0 = Xt - mu_t

    ridge = float(args.ridge)
    Cs = np.cov(Xs0, rowvar=False) + ridge * np.eye(Xs0.shape[1], dtype=np.float64)
    Ct = np.cov(Xt0, rowvar=False) + ridge * np.eye(Xt0.shape[1], dtype=np.float64)

    A = _invsqrtm_psd(Cs) @ _sqrtm_psd(Ct)
    Xs_aligned = Xs0 @ A + mu_t
    Xt_aligned = Xt  # target domain unchanged

    pref = str(args.prefix).strip()
    train_out = pd.DataFrame({"track_id": feat_train["track_id"].to_numpy(dtype=np.int64)})
    test_out = pd.DataFrame({"track_id": feat_test["track_id"].to_numpy(dtype=np.int64)})
    for j, c in enumerate(selected):
        name = f"{pref}_{c}"
        train_out[name] = Xs_aligned[:, j].astype(np.float32)
        test_out[name] = Xt_aligned[:, j].astype(np.float32)

    train_out.to_parquet(out_train, index=False)
    test_out.to_parquet(out_test, index=False)

    report = {
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "output_train_parquet": str(out_train),
        "output_test_parquet": str(out_test),
        "n_train": int(len(train_out)),
        "n_test": int(len(test_out)),
        "top_k": int(top_k),
        "ridge": float(ridge),
        "prefix": pref,
        "selected_features": selected,
        "selected_ks": ks_df.head(top_k).to_dict(orient="records"),
        "max_ks_all": float(ks_df["ks"].max()),
        "mean_ks_all": float(ks_df["ks"].mean()),
    }
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_train": str(out_train), "output_test": str(out_test), "top_k": int(top_k)}))


if __name__ == "__main__":
    main()

