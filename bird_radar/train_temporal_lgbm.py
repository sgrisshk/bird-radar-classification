from __future__ import annotations

import argparse
import json
import random
import fnmatch
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import CLASS_TO_INDEX, CLASSES
from src.cv import (
    make_forward_temporal_group_folds,
    make_temporal_holdout_split,
    make_temporal_holdout_split_with_cutoff,
)
from src.feature_engineering import (
    build_feature_frame,
    compute_monthly_track_centers,
    load_external_features,
)
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/Users/sgrisshk/Desktop/AI-task")
    parser.add_argument("--cache-dir", type=str, default="/Users/sgrisshk/Desktop/AI-task/bird_radar/artifacts/cache")
    parser.add_argument("--output-dir", type=str, default="/Users/sgrisshk/Desktop/AI-task/bird_radar/artifacts/temporal_candidates")
    parser.add_argument("--temporal-quantile", type=float, default=0.8)
    parser.add_argument("--temporal-cutoff-date", type=str, default="")
    parser.add_argument("--holdout-mode", type=str, default="temporal", choices=["temporal", "stratified"])
    parser.add_argument("--stratified-splits", type=int, default=5)
    parser.add_argument("--stratified-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-estimators", type=int, default=8000)
    parser.add_argument("--drop-k-grid", type=str, default="0,10,20")
    parser.add_argument("--blacklist-mode", type=str, default="none", choices=["none", "A", "B"])
    parser.add_argument("--extra-blacklist", type=str, default="")
    parser.add_argument("--force-config", type=str, default="", choices=["", "base", "reg", "aggr", "deep", "wide", "reg_heavy"])
    parser.add_argument("--use-time-weights", action="store_true", default=True)
    parser.add_argument("--no-time-weights", dest="use_time_weights", action="store_false")
    parser.add_argument("--save-meta-artifacts", action="store_true", default=True)
    parser.add_argument("--no-save-meta-artifacts", dest="save_meta_artifacts", action="store_false")
    parser.add_argument("--oof-strategy", type=str, default="none", choices=["none", "forward_cv"])
    parser.add_argument(
        "--cv-direction",
        type=str,
        default="forward",
        choices=["forward", "reverse"],
        help="Temporal direction for CV/holdout splitting; reverse flips timestamp order.",
    )
    parser.add_argument("--oof-forward-splits", type=int, default=5)
    parser.add_argument("--oof-forward-complete", type=str, default="off", choices=["off", "backcast_last", "backcast_all"])
    parser.add_argument("--domain-reweight", type=str, default="none", choices=["none", "odds"])
    parser.add_argument("--domain-reweight-clip-min", type=float, default=0.5)
    parser.add_argument("--domain-reweight-clip-max", type=float, default=3.0)
    parser.add_argument("--domain-reweight-splits", type=int, default=5)
    parser.add_argument("--pseudo-label-csv", type=str, default="")
    parser.add_argument("--pseudo-weight", type=float, default=0.2)
    parser.add_argument("--pseudo-prob-col", type=str, default="pseudo_prob")
    parser.add_argument("--external-train-parquet", type=str, default="")
    parser.add_argument("--external-test-parquet", type=str, default="")
    parser.add_argument("--sample-weight-parquet", type=str, default="")
    parser.add_argument(
        "--feature-allowlist-file",
        type=str,
        default="",
        help="Optional path to newline/csv feature allowlist. If set, only these features are kept.",
    )
    parser.add_argument(
        "--training-objective",
        type=str,
        default="ovr",
        choices=["ovr", "multiclass"],
        help="Modeling objective: legacy OvR binary heads or native multiclass.",
    )
    parser.add_argument("--multiclass-oversample-rare", action="store_true", default=False)
    parser.add_argument("--multiclass-corm-factor", type=float, default=10.0)
    parser.add_argument("--multiclass-wader-factor", type=float, default=5.0)
    parser.add_argument("--multiclass-oversample-noise", type=float, default=0.02)
    parser.add_argument(
        "--tabular-mixup-rare",
        action="store_true",
        default=False,
        help="Enable tabular mixup augmentation within selected rare classes on train folds only.",
    )
    parser.add_argument(
        "--tabular-mixup-alpha",
        type=float,
        default=0.3,
        help="Beta(alpha, alpha) parameter for tabular mixup lambda.",
    )
    parser.add_argument(
        "--tabular-mixup-multiplier",
        type=float,
        default=3.0,
        help="Target multiplier per selected class (e.g. 3.0 means add ~2x synthetic rows).",
    )
    parser.add_argument(
        "--tabular-mixup-classes",
        type=str,
        default="Cormorants,Waders,Ducks",
        help="Comma-separated class names for tabular mixup.",
    )
    parser.add_argument(
        "--tabular-mixup-class-multipliers",
        type=str,
        default="",
        help='Optional per-class multipliers, e.g. "Ducks:4,Birds of Prey:2". Falls back to --tabular-mixup-multiplier.',
    )
    parser.add_argument(
        "--monotone-safe-bio",
        action="store_true",
        default=False,
        help="Enable conservative biology-inspired monotone constraints for OvR heads.",
    )
    return parser.parse_args()


def _load_feature_allowlist(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"feature allowlist file not found: {p}")
    if p.suffix.lower() in {".csv"}:
        df = pd.read_csv(p)
        if "feature" in df.columns:
            vals = df["feature"].astype(str).tolist()
        else:
            vals = df.iloc[:, 0].astype(str).tolist()
    else:
        vals = p.read_text(encoding="utf-8").splitlines()
    out = {v.strip() for v in vals if str(v).strip()}
    if len(out) == 0:
        raise ValueError(f"feature allowlist is empty: {p}")
    return out


def make_stratified_holdout_split(
    y_idx: np.ndarray,
    n_splits: int,
    fold_id: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_splits < 2:
        raise ValueError("stratified n_splits must be >= 2")
    if fold_id < 0 or fold_id >= n_splits:
        raise ValueError(f"stratified fold_id must be in [0,{n_splits - 1}]")
    dummy_x = np.zeros((len(y_idx), 1), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (tr_idx, va_idx) in enumerate(skf.split(dummy_x, y_idx)):
        if i == fold_id:
            return tr_idx, va_idx
    raise RuntimeError("unable to construct stratified holdout split")


def _blacklist_patterns(mode: str) -> list[str]:
    speed_quantile_exact = [
        "speed_p5",
        "speed_p10",
        "speed_p25",
        "speed_p50",
        "speed_p75",
        "speed_p90",
        "speed_p95",
    ]
    base = [
        "mean_speed",
        "interaction_rcs_speed",
        *speed_quantile_exact,
        "speed_seg_*_mean",
    ]
    if mode == "A":
        return base
    if mode == "B":
        return base + [
            "speed_seg_*_std",
            "speed_seg_*_slope",
            "speed_fft_*",
            "speed_dist_*",
        ]
    return []


def _apply_blacklist(feature_cols: list[str], patterns: list[str]) -> tuple[list[str], list[str]]:
    if not patterns:
        return feature_cols, []
    drop: list[str] = []
    keep: list[str] = []
    for col in feature_cols:
        matched = any(fnmatch.fnmatch(col, pat) for pat in patterns)
        if matched:
            drop.append(col)
        else:
            keep.append(col)
    return keep, sorted(drop)


def load_or_build_cache(df: pd.DataFrame, path: Path) -> dict:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


def macro_map(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        vals.append(average_precision_score(yt, yp) if yt.sum() > 0 else 0.0)
    return float(np.mean(vals))


def per_class_ap(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, cls in enumerate(CLASSES):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        out[cls] = float(average_precision_score(yt, yp) if yt.sum() > 0 else 0.0)
    return out


def compute_drift_ks(train_feat: pd.DataFrame, test_feat: pd.DataFrame, feature_cols: list[str]) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for col in feature_cols:
        stat, _ = ks_2samp(
            train_feat[col].to_numpy(dtype=np.float64),
            test_feat[col].to_numpy(dtype=np.float64),
            alternative="two-sided",
            mode="asymp",
        )
        rows.append((col, float(stat)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def adversarial_auc(train_feat: pd.DataFrame, test_feat: pd.DataFrame, feature_cols: list[str], seed: int) -> tuple[float, list[float]]:
    X = pd.concat([train_feat[feature_cols], test_feat[feature_cols]], axis=0).astype(np.float32).reset_index(drop=True)
    y = np.concatenate([
        np.zeros(len(train_feat), dtype=np.int32),
        np.ones(len(test_feat), dtype=np.int32),
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs: list[float] = []

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=63,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            random_state=seed + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            X.iloc[tr],
            y[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        pred = model.predict_proba(X.iloc[va])[:, 1]
        aucs.append(float(roc_auc_score(y[va], pred)))

    return float(np.mean(aucs)), aucs


def domain_reweight_odds(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    n_splits: int,
    clip_min: float,
    clip_max: float,
) -> tuple[np.ndarray, dict[str, object]]:
    X = pd.concat([train_feat[feature_cols], test_feat[feature_cols]], axis=0).astype(np.float32).reset_index(drop=True)
    y = np.concatenate(
        [
            np.zeros(len(train_feat), dtype=np.int32),
            np.ones(len(test_feat), dtype=np.int32),
        ]
    )

    skf = StratifiedKFold(n_splits=max(int(n_splits), 2), shuffle=True, random_state=seed + 17)
    oof_p = np.zeros(len(X), dtype=np.float64)
    aucs: list[float] = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=63,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            random_state=seed + 1000 + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            X.iloc[tr],
            y[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        pred = model.predict_proba(X.iloc[va])[:, 1]
        oof_p[va] = pred
        aucs.append(float(roc_auc_score(y[va], pred)))

    p_train = np.clip(oof_p[: len(train_feat)], 1e-4, 1.0 - 1e-4)
    weights = p_train / (1.0 - p_train)
    weights = np.clip(weights, float(clip_min), float(clip_max))
    weights = weights / (np.mean(weights) + 1e-12)
    info = {
        "mode": "odds",
        "n_splits": int(max(int(n_splits), 2)),
        "auc_mean": float(np.mean(aucs)),
        "auc_folds": [float(x) for x in aucs],
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
        "w_mean": float(np.mean(weights)),
        "w_std": float(np.std(weights)),
        "w_min": float(np.min(weights)),
        "w_max": float(np.max(weights)),
        "w_q10": float(np.quantile(weights, 0.10)),
        "w_q50": float(np.quantile(weights, 0.50)),
        "w_q90": float(np.quantile(weights, 0.90)),
    }
    return weights.astype(np.float32), info


def _combine_sample_weights(parts: list[np.ndarray | None]) -> np.ndarray | None:
    active = [p.astype(np.float32) for p in parts if p is not None]
    if not active:
        return None
    w = np.ones_like(active[0], dtype=np.float32)
    for part in active:
        w *= part
    w /= float(np.mean(w) + 1e-12)
    return w.astype(np.float32)


def build_pseudo_payload(
    pseudo_csv: str,
    feat_test: pd.DataFrame,
    *,
    prob_col: str,
    default_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if not pseudo_csv.strip():
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            {"enabled": False},
        )

    p = Path(pseudo_csv).resolve()
    if not p.exists():
        raise FileNotFoundError(f"pseudo-label csv not found: {p}")

    pseudo = pd.read_csv(p)
    req = {"track_id", "bird_group"}
    missing = [c for c in req if c not in pseudo.columns]
    if missing:
        raise ValueError(f"pseudo-label csv is missing required columns: {missing}")

    pseudo = pseudo[pseudo["bird_group"].isin(CLASSES)].copy()
    if len(pseudo) == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            {"enabled": True, "rows": 0, "note": "no rows after class filter"},
        )

    if prob_col in pseudo.columns:
        pseudo["pseudo_prob_used"] = pd.to_numeric(pseudo[prob_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        pseudo["pseudo_prob_used"] = float(default_weight)

    # Keep one row per track_id (highest confidence).
    pseudo = pseudo.sort_values("pseudo_prob_used", ascending=False).drop_duplicates(subset=["track_id"], keep="first")

    test_idx = feat_test[["track_id"]].reset_index().rename(columns={"index": "_test_idx"})
    merged = pseudo.merge(test_idx, on="track_id", how="inner")
    if len(merged) == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            {"enabled": True, "rows": 0, "note": "no pseudo track_id matched test features"},
        )

    idx = merged["_test_idx"].to_numpy(dtype=np.int64)
    cls_idx = merged["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    conf = merged["pseudo_prob_used"].to_numpy(dtype=np.float32)
    w = (float(default_weight) * np.clip(conf, 0.0, 1.0)).astype(np.float32)
    info = {
        "enabled": True,
        "path": str(p),
        "rows": int(len(idx)),
        "weight_base": float(default_weight),
        "weight_mean": float(np.mean(w)),
        "weight_min": float(np.min(w)),
        "weight_max": float(np.max(w)),
        "class_counts": {
            cls: int(np.sum(cls_idx == ci)) for ci, cls in enumerate(CLASSES)
        },
    }
    return idx, cls_idx, w, info


def build_sample_weight_payload(
    sample_weight_parquet: str,
    feat_train: pd.DataFrame,
) -> tuple[np.ndarray | None, dict[str, object]]:
    if not sample_weight_parquet.strip():
        return None, {"enabled": False}

    p = Path(sample_weight_parquet).resolve()
    if not p.exists():
        raise FileNotFoundError(f"sample-weight parquet not found: {p}")

    sw = pd.read_parquet(p)
    required = {"track_id", "sample_weight"}
    missing = [c for c in required if c not in sw.columns]
    if missing:
        raise ValueError(f"sample-weight parquet is missing required columns: {missing}")

    sw = sw[["track_id", "sample_weight"]].copy()
    sw["sample_weight"] = (
        pd.to_numeric(sw["sample_weight"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=0.0)
    )
    sw = sw.drop_duplicates(subset=["track_id"], keep="first")

    merged = feat_train[["track_id"]].merge(sw, on="track_id", how="left")
    matched = int(merged["sample_weight"].notna().sum())
    w = merged["sample_weight"].fillna(1.0).to_numpy(dtype=np.float32)
    info = {
        "enabled": True,
        "path": str(p),
        "rows": int(len(sw)),
        "matched_rows": matched,
        "coverage": float(matched / max(len(feat_train), 1)),
        "weight_mean": float(np.mean(w)),
        "weight_min": float(np.min(w)),
        "weight_max": float(np.max(w)),
    }
    return w, info


def _oversample_rare(
    X: pd.DataFrame,
    y: np.ndarray,
    min_pos: int = 80,
    noise_std: float = 0.02,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    return _oversample_rare_with_weights(
        X=X,
        y=y,
        min_pos=min_pos,
        noise_std=noise_std,
        seed=seed,
        sample_weight=None,
    )[:2]


def _oversample_rare_with_weights(
    X: pd.DataFrame,
    y: np.ndarray,
    min_pos: int = 80,
    noise_std: float = 0.02,
    seed: int = 0,
    sample_weight: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """
    If positive count is below min_pos, duplicate positive rows with small
    Gaussian jitter on numeric columns.
    """
    pos_idx = np.where(y == 1)[0]
    n_pos = len(pos_idx)
    if n_pos == 0 or n_pos >= min_pos:
        return X, y, sample_weight

    rng = np.random.RandomState(seed)
    n_needed = min_pos - n_pos
    chosen = rng.choice(pos_idx, size=n_needed, replace=True)
    X_pos = X.iloc[chosen].copy().reset_index(drop=True)

    num_cols = X_pos.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        col_std = float(X[col].std())
        if col_std > 0:
            noise = rng.normal(0.0, noise_std * col_std, size=len(X_pos))
            X_pos[col] = X_pos[col] + noise

    y_pos = np.ones(n_needed, dtype=y.dtype)
    X_aug = pd.concat([X.reset_index(drop=True), X_pos], ignore_index=True)
    y_aug = np.concatenate([y, y_pos])
    if sample_weight is not None:
        w_pos = sample_weight[chosen]
        w_aug = np.concatenate([sample_weight, w_pos])
    else:
        w_aug = None
    return X_aug, y_aug, w_aug


def _oversample_multiclass_rare(
    X: pd.DataFrame,
    y_idx: np.ndarray,
    class_multipliers: dict[int, float],
    noise_std: float = 0.02,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Multiclass oversampling with optional jitter on numeric columns.
    For each class k with multiplier m>1, add approximately (m-1)*count(k) rows.
    """
    if len(y_idx) == 0 or not class_multipliers:
        return X, y_idx

    rng = np.random.RandomState(seed)
    add_indices: list[np.ndarray] = []
    for ci, mult in class_multipliers.items():
        if mult <= 1.0:
            continue
        cls_idx = np.where(y_idx == int(ci))[0]
        n_cls = len(cls_idx)
        if n_cls == 0:
            continue
        n_add = int(round((float(mult) - 1.0) * n_cls))
        if n_add <= 0:
            continue
        chosen = rng.choice(cls_idx, size=n_add, replace=True)
        add_indices.append(chosen)

    if not add_indices:
        return X, y_idx

    chosen_all = np.concatenate(add_indices, axis=0)
    X_add = X.iloc[chosen_all].copy().reset_index(drop=True)
    y_add = y_idx[chosen_all].astype(np.int32)

    num_cols = X_add.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        col_std = float(X[col].std())
        if col_std > 0:
            noise = rng.normal(0.0, float(noise_std) * col_std, size=len(X_add))
            X_add[col] = X_add[col] + noise

    X_aug = pd.concat([X.reset_index(drop=True), X_add], ignore_index=True)
    y_aug = np.concatenate([y_idx.astype(np.int32), y_add], axis=0)
    return X_aug, y_aug


def _tabular_mixup_within_classes(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    class_indices: list[int],
    alpha: float = 0.3,
    multiplier: float = 3.0,
    class_multiplier_map: dict[int, float] | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, dict[str, object]]:
    """
    Multiclass tabular mixup restricted to selected classes.
    y is expected to be one-hot of shape (n_samples, n_classes).
    """
    info: dict[str, object] = {
        "enabled": True,
        "alpha": float(alpha),
        "multiplier": float(multiplier),
        "class_indices": [int(i) for i in class_indices],
        "class_multiplier_map": {int(k): float(v) for k, v in (class_multiplier_map or {}).items()},
        "added_total": 0,
        "added_per_class": {},
    }
    if len(X) == 0 or y.size == 0:
        return X, y, sample_weight, info
    if y.ndim != 2 or y.shape[0] != len(X):
        return X, y, sample_weight, {**info, "enabled": False, "note": "invalid_y_shape"}
    if alpha <= 0.0 or multiplier <= 1.0 or not class_indices:
        return X, y, sample_weight, {**info, "enabled": False, "note": "disabled_or_no_classes"}

    rng = np.random.RandomState(seed)
    X_base = X.to_numpy(dtype=np.float32, copy=True)
    y_base = y.astype(np.int32, copy=False)
    n_classes = y_base.shape[1]

    syn_x_blocks: list[np.ndarray] = []
    syn_y_blocks: list[np.ndarray] = []
    syn_w_blocks: list[np.ndarray] = []

    for ci in class_indices:
        if ci < 0 or ci >= n_classes:
            continue
        cls_rows = np.where(y_base[:, ci] == 1)[0]
        n_cls = int(len(cls_rows))
        if n_cls < 2:
            info["added_per_class"][CLASSES[ci]] = 0
            continue
        mult_ci = float(class_multiplier_map.get(int(ci), multiplier)) if class_multiplier_map else float(multiplier)
        n_syn = int(round((mult_ci - 1.0) * n_cls))
        if n_syn <= 0:
            info["added_per_class"][CLASSES[ci]] = 0
            continue

        ii = rng.choice(cls_rows, size=n_syn, replace=True)
        jj = rng.choice(cls_rows, size=n_syn, replace=True)
        lam = rng.beta(alpha, alpha, size=n_syn).astype(np.float32)

        xi = X_base[ii]
        xj = X_base[jj]
        x_syn = lam[:, None] * xi + (1.0 - lam)[:, None] * xj

        y_syn = np.zeros((n_syn, n_classes), dtype=np.int32)
        y_syn[:, ci] = 1

        syn_x_blocks.append(x_syn.astype(np.float32))
        syn_y_blocks.append(y_syn)
        if sample_weight is not None:
            w = sample_weight.astype(np.float32, copy=False)
            w_syn = lam * w[ii] + (1.0 - lam) * w[jj]
            syn_w_blocks.append(w_syn.astype(np.float32))

        info["added_per_class"][CLASSES[ci]] = int(n_syn)
        info["added_total"] = int(info["added_total"]) + int(n_syn)

    if not syn_x_blocks:
        return X, y, sample_weight, info

    X_syn = np.vstack(syn_x_blocks).astype(np.float32)
    y_syn = np.vstack(syn_y_blocks).astype(np.int32)

    X_aug = pd.DataFrame(
        np.vstack([X_base, X_syn]).astype(np.float32),
        columns=X.columns,
    )
    y_aug = np.vstack([y_base, y_syn]).astype(np.int32)

    if sample_weight is None:
        w_aug = None
    else:
        w_syn = np.concatenate(syn_w_blocks, axis=0).astype(np.float32) if syn_w_blocks else np.zeros((0,), dtype=np.float32)
        w_aug = np.concatenate([sample_weight.astype(np.float32), w_syn], axis=0).astype(np.float32)
        w_aug = (w_aug / (np.mean(w_aug) + 1e-12)).astype(np.float32)

    return X_aug, y_aug, w_aug, info


def _build_safe_bio_monotone_constraints(feature_cols: list[str], class_name: str) -> list[int]:
    idx = {c: i for i, c in enumerate(feature_cols)}
    vec = [0] * len(feature_cols)
    class_rules: dict[str, dict[str, int]] = {
        "Geese": {
            "n_points": +1,
            "rcs_mean": +1,
            "radar_bird_size__small_bird": -1,
        },
        "Gulls": {
            "n_points": +1,
        },
        "Clutter": {
            "n_points": -1,
        },
        "Songbirds": {
            "rcs_mean": -1,
        },
        "Birds of Prey": {
            "radar_bird_size__small_bird": +1,
            "rcs_mean": -1,
        },
    }
    for feat, sign in class_rules.get(class_name, {}).items():
        j = idx.get(feat)
        if j is not None:
            vec[j] = int(sign)
    return vec


def train_ovr(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    cfg: dict,
    seed: int,
    n_estimators: int,
    sample_weight: np.ndarray | None,
    monotone_safe_bio: bool = False,
) -> np.ndarray:
    preds = np.zeros((len(X_valid), len(CLASSES)), dtype=np.float32)
    feature_cols = list(X_train.columns)
    for ci in range(len(CLASSES)):
        ytr = y_train[:, ci]
        yva = y_valid[:, ci]
        ytr_sum = int(np.sum(ytr))
        if ytr_sum == 0 or ytr_sum == int(len(ytr)):
            preds[:, ci] = float(ytr_sum / max(len(ytr), 1))
            continue
        pos = max(int(ytr.sum()), 1)
        neg = max(int(len(ytr) - ytr.sum()), 1)
        X_train_aug, ytr_aug, w_aug = _oversample_rare_with_weights(
            X_train,
            ytr,
            min_pos=80,
            seed=seed + ci,
            sample_weight=sample_weight,
        )
        pos_aug = max(int(ytr_aug.sum()), 1)
        neg_aug = max(int(len(ytr_aug) - ytr_aug.sum()), 1)

        model_params = dict(
            objective="binary",
            metric="average_precision",
            n_estimators=n_estimators,
            learning_rate=float(cfg["lr"]),
            num_leaves=int(cfg["leaves"]),
            colsample_bytree=float(cfg["ff"]),
            subsample=float(cfg.get("bf", 0.8)),
            subsample_freq=int(cfg.get("bag_freq", 1)),
            reg_alpha=float(cfg["l1"]),
            reg_lambda=float(cfg["l2"]),
            min_child_samples=int(cfg["mc"]),
            scale_pos_weight=float(neg_aug / pos_aug),
            random_state=seed + ci,
            n_jobs=-1,
            verbosity=-1,
        )
        if monotone_safe_bio:
            mc = _build_safe_bio_monotone_constraints(feature_cols, CLASSES[ci])
            if any(v != 0 for v in mc):
                model_params["monotone_constraints"] = mc
        model = lgb.LGBMClassifier(**model_params)
        model.fit(
            X_train_aug,
            ytr_aug,
            sample_weight=w_aug,
            eval_set=[(X_valid, yva)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)],
        )
        preds[:, ci] = model.predict_proba(X_valid)[:, 1]
    return preds


def train_ovr_full(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    cfg: dict,
    seed: int,
    n_estimators: int,
    sample_weight: np.ndarray | None,
    monotone_safe_bio: bool = False,
) -> np.ndarray:
    preds = np.zeros((len(X_test), len(CLASSES)), dtype=np.float32)
    feature_cols = list(X_train.columns)
    for ci in range(len(CLASSES)):
        ytr = y_train[:, ci]
        ytr_sum = int(np.sum(ytr))
        if ytr_sum == 0 or ytr_sum == int(len(ytr)):
            preds[:, ci] = float(ytr_sum / max(len(ytr), 1))
            continue
        pos = max(int(ytr.sum()), 1)
        neg = max(int(len(ytr) - ytr.sum()), 1)
        X_train_aug, ytr_aug, w_aug = _oversample_rare_with_weights(
            X_train,
            ytr,
            min_pos=80,
            seed=seed + 100 + ci,
            sample_weight=sample_weight,
        )
        pos_aug = max(int(ytr_aug.sum()), 1)
        neg_aug = max(int(len(ytr_aug) - ytr_aug.sum()), 1)

        model_params = dict(
            objective="binary",
            metric="average_precision",
            n_estimators=n_estimators,
            learning_rate=float(cfg["lr"]),
            num_leaves=int(cfg["leaves"]),
            colsample_bytree=float(cfg["ff"]),
            subsample=float(cfg.get("bf", 0.8)),
            subsample_freq=int(cfg.get("bag_freq", 1)),
            reg_alpha=float(cfg["l1"]),
            reg_lambda=float(cfg["l2"]),
            min_child_samples=int(cfg["mc"]),
            scale_pos_weight=float(neg_aug / pos_aug),
            random_state=seed + 100 + ci,
            n_jobs=-1,
            verbosity=-1,
        )
        if monotone_safe_bio:
            mc = _build_safe_bio_monotone_constraints(feature_cols, CLASSES[ci])
            if any(v != 0 for v in mc):
                model_params["monotone_constraints"] = mc
        model = lgb.LGBMClassifier(**model_params)
        model.fit(X_train_aug, ytr_aug, sample_weight=w_aug, callbacks=[lgb.log_evaluation(0)])
        preds[:, ci] = model.predict_proba(X_test)[:, 1]
    return preds


def train_multiclass(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    cfg: dict,
    seed: int,
    n_estimators: int,
    sample_weight: np.ndarray | None,
    multiclass_oversample_rare: bool = False,
    multiclass_corm_factor: float = 10.0,
    multiclass_wader_factor: float = 5.0,
    multiclass_oversample_noise: float = 0.02,
) -> np.ndarray:
    def _balanced_multiclass_weights(y_idx: np.ndarray) -> np.ndarray:
        counts = np.bincount(y_idx, minlength=len(CLASSES)).astype(np.float64)
        total = float(len(y_idx))
        ncls = float(len(CLASSES))
        w_cls = np.ones(len(CLASSES), dtype=np.float64)
        nonzero = counts > 0
        w_cls[nonzero] = total / (ncls * counts[nonzero])
        w = w_cls[y_idx]
        w = w / (np.mean(w) + 1e-12)
        return w.astype(np.float32)

    ytr_idx_raw = np.argmax(y_train, axis=1).astype(np.int32)
    yva_idx = np.argmax(y_valid, axis=1).astype(np.int32)
    X_fit = X_train
    y_fit = ytr_idx_raw
    w_fit = _balanced_multiclass_weights(ytr_idx_raw)
    if sample_weight is not None:
        w_fit = (w_fit * sample_weight.astype(np.float32)).astype(np.float32)

    if multiclass_oversample_rare:
        class_multipliers = {
            int(CLASS_TO_INDEX["Cormorants"]): float(multiclass_corm_factor),
            int(CLASS_TO_INDEX["Waders"]): float(multiclass_wader_factor),
        }
        X_fit, y_fit = _oversample_multiclass_rare(
            X=X_train,
            y_idx=ytr_idx_raw,
            class_multipliers=class_multipliers,
            noise_std=float(multiclass_oversample_noise),
            seed=seed,
        )
        # Recompute balanced weights on augmented labels.
        w_fit = _balanced_multiclass_weights(y_fit)

    params = {
        "objective": "multiclass",
        "num_class": len(CLASSES),
        "metric": "multi_logloss",
        "learning_rate": float(cfg["lr"]),
        "num_leaves": int(cfg["leaves"]),
        "feature_fraction": float(cfg["ff"]),
        "bagging_fraction": float(cfg.get("bf", 0.8)),
        "bagging_freq": int(cfg.get("bag_freq", 1)),
        "lambda_l1": float(cfg["l1"]),
        "lambda_l2": float(cfg["l2"]),
        "min_child_samples": int(cfg["mc"]),
        "seed": int(seed),
        "verbosity": -1,
        "num_threads": -1,
    }
    train_set = lgb.Dataset(X_fit, label=y_fit, weight=w_fit, free_raw_data=False)
    valid_set = lgb.Dataset(X_valid, label=yva_idx, free_raw_data=False)
    model = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=n_estimators,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)],
    )
    best_iter = model.best_iteration if model.best_iteration is not None else n_estimators
    pred = model.predict(X_valid, num_iteration=best_iter)
    return np.asarray(pred, dtype=np.float32)


def train_multiclass_full(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    cfg: dict,
    seed: int,
    n_estimators: int,
    sample_weight: np.ndarray | None,
    multiclass_oversample_rare: bool = False,
    multiclass_corm_factor: float = 10.0,
    multiclass_wader_factor: float = 5.0,
    multiclass_oversample_noise: float = 0.02,
) -> np.ndarray:
    def _balanced_multiclass_weights(y_idx: np.ndarray) -> np.ndarray:
        counts = np.bincount(y_idx, minlength=len(CLASSES)).astype(np.float64)
        total = float(len(y_idx))
        ncls = float(len(CLASSES))
        w_cls = np.ones(len(CLASSES), dtype=np.float64)
        nonzero = counts > 0
        w_cls[nonzero] = total / (ncls * counts[nonzero])
        w = w_cls[y_idx]
        w = w / (np.mean(w) + 1e-12)
        return w.astype(np.float32)

    ytr_idx_raw = np.argmax(y_train, axis=1).astype(np.int32)
    X_fit = X_train
    y_fit = ytr_idx_raw
    w_fit = _balanced_multiclass_weights(ytr_idx_raw)
    if sample_weight is not None:
        w_fit = (w_fit * sample_weight.astype(np.float32)).astype(np.float32)

    if multiclass_oversample_rare:
        class_multipliers = {
            int(CLASS_TO_INDEX["Cormorants"]): float(multiclass_corm_factor),
            int(CLASS_TO_INDEX["Waders"]): float(multiclass_wader_factor),
        }
        X_fit, y_fit = _oversample_multiclass_rare(
            X=X_train,
            y_idx=ytr_idx_raw,
            class_multipliers=class_multipliers,
            noise_std=float(multiclass_oversample_noise),
            seed=seed + 1000,
        )
        w_fit = _balanced_multiclass_weights(y_fit)

    params = {
        "objective": "multiclass",
        "num_class": len(CLASSES),
        "metric": "multi_logloss",
        "learning_rate": float(cfg["lr"]),
        "num_leaves": int(cfg["leaves"]),
        "feature_fraction": float(cfg["ff"]),
        "bagging_fraction": float(cfg.get("bf", 0.8)),
        "bagging_freq": int(cfg.get("bag_freq", 1)),
        "lambda_l1": float(cfg["l1"]),
        "lambda_l2": float(cfg["l2"]),
        "min_child_samples": int(cfg["mc"]),
        "seed": int(seed),
        "verbosity": -1,
        "num_threads": -1,
    }
    train_set = lgb.Dataset(X_fit, label=y_fit, weight=w_fit, free_raw_data=False)
    model = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=n_estimators,
        callbacks=[lgb.log_evaluation(0)],
    )
    pred = model.predict(X_test, num_iteration=n_estimators)
    return np.asarray(pred, dtype=np.float32)


def train_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    cfg: dict,
    seed: int,
    n_estimators: int,
    sample_weight: np.ndarray | None,
    training_objective: str,
    monotone_safe_bio: bool = False,
    multiclass_oversample_rare: bool = False,
    multiclass_corm_factor: float = 10.0,
    multiclass_wader_factor: float = 5.0,
    multiclass_oversample_noise: float = 0.02,
) -> np.ndarray:
    if training_objective == "multiclass":
        return train_multiclass(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            cfg=cfg,
            seed=seed,
            n_estimators=n_estimators,
            sample_weight=sample_weight,
            multiclass_oversample_rare=multiclass_oversample_rare,
            multiclass_corm_factor=multiclass_corm_factor,
            multiclass_wader_factor=multiclass_wader_factor,
            multiclass_oversample_noise=multiclass_oversample_noise,
        )
    return train_ovr(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        cfg=cfg,
        seed=seed,
        n_estimators=n_estimators,
        sample_weight=sample_weight,
        monotone_safe_bio=monotone_safe_bio,
    )


def train_classifier_full(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    cfg: dict,
    seed: int,
    n_estimators: int,
    sample_weight: np.ndarray | None,
    training_objective: str,
    monotone_safe_bio: bool = False,
    multiclass_oversample_rare: bool = False,
    multiclass_corm_factor: float = 10.0,
    multiclass_wader_factor: float = 5.0,
    multiclass_oversample_noise: float = 0.02,
) -> np.ndarray:
    if training_objective == "multiclass":
        return train_multiclass_full(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            cfg=cfg,
            seed=seed,
            n_estimators=n_estimators,
            sample_weight=sample_weight,
            multiclass_oversample_rare=multiclass_oversample_rare,
            multiclass_corm_factor=multiclass_corm_factor,
            multiclass_wader_factor=multiclass_wader_factor,
            multiclass_oversample_noise=multiclass_oversample_noise,
        )
    return train_ovr_full(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        cfg=cfg,
        seed=seed,
        n_estimators=n_estimators,
        sample_weight=sample_weight,
        monotone_safe_bio=monotone_safe_bio,
    )


def _time_weight_from_timestamps(ts: pd.Series, enabled: bool) -> np.ndarray | None:
    if not enabled:
        return None
    ts_ns = ts.astype("int64").to_numpy(dtype=np.float64)
    ts_min = float(np.min(ts_ns))
    ts_max = float(np.max(ts_ns))
    norm = (ts_ns - ts_min) / (ts_max - ts_min + 1e-9)
    return (0.7 + 0.6 * norm).astype(np.float32)


def _run_forward_cv_oof(
    *,
    feat_train: pd.DataFrame,
    feat_test: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    cfg: dict,
    seed: int,
    n_estimators: int,
    use_time_weights: bool,
    domain_weights_all: np.ndarray | None,
    sample_weights_all: np.ndarray | None,
    n_splits: int,
    complete_mode: str,
    training_objective: str,
    monotone_safe_bio: bool,
    multiclass_oversample_rare: bool,
    multiclass_corm_factor: float,
    multiclass_wader_factor: float,
    multiclass_oversample_noise: float,
    pseudo_X: pd.DataFrame | None = None,
    pseudo_y: np.ndarray | None = None,
    pseudo_w: np.ndarray | None = None,
    tabular_mixup_rare: bool = False,
    tabular_mixup_alpha: float = 0.3,
    tabular_mixup_multiplier: float = 3.0,
    tabular_mixup_class_indices: list[int] | None = None,
    tabular_mixup_class_multiplier_map: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    folds = make_forward_temporal_group_folds(
        feat_train[["_cv_ts", "_cv_group"]].copy(),
        timestamp_col="_cv_ts",
        group_col="_cv_group",
        n_splits=max(int(n_splits), 2),
    )
    if len(folds) == 0:
        raise RuntimeError("forward_cv requested but no folds were constructed")

    X_test = feat_test[feature_cols].astype(np.float32)
    oof = np.zeros((len(feat_train), len(CLASSES)), dtype=np.float32)
    covered = np.zeros(len(feat_train), dtype=bool)
    test_folds: list[np.ndarray] = []
    fold_metrics: list[dict[str, object]] = []

    def _augment_with_pseudo_fold(
        Xtr_fold: pd.DataFrame,
        ytr_fold: np.ndarray,
        wtr_fold: np.ndarray | None,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
        if pseudo_X is None or pseudo_y is None or len(pseudo_X) == 0:
            return Xtr_fold, ytr_fold, wtr_fold

        X_aug = pd.concat([Xtr_fold, pseudo_X], axis=0, ignore_index=True)
        y_aug = np.vstack([ytr_fold, pseudo_y]).astype(np.int32)

        if pseudo_w is None or len(pseudo_w) == 0:
            return X_aug, y_aug, wtr_fold

        pw = pseudo_w.astype(np.float32)
        if wtr_fold is None:
            w_aug = np.concatenate(
                [np.ones((len(Xtr_fold),), dtype=np.float32), pw],
                axis=0,
            ).astype(np.float32)
        else:
            w_aug = np.concatenate([wtr_fold.astype(np.float32), pw], axis=0).astype(np.float32)
        w_aug = (w_aug / (np.mean(w_aug) + 1e-12)).astype(np.float32)
        return X_aug, y_aug, w_aug

    def _augment_with_mixup_fold(
        Xtr_fold: pd.DataFrame,
        ytr_fold: np.ndarray,
        wtr_fold: np.ndarray | None,
        mixup_seed: int,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, dict[str, object]]:
        if not tabular_mixup_rare:
            return Xtr_fold, ytr_fold, wtr_fold, {"enabled": False}
        return _tabular_mixup_within_classes(
            X=Xtr_fold,
            y=ytr_fold,
            sample_weight=wtr_fold,
            class_indices=list(tabular_mixup_class_indices or []),
            alpha=float(tabular_mixup_alpha),
            multiplier=float(tabular_mixup_multiplier),
            class_multiplier_map=tabular_mixup_class_multiplier_map,
            seed=int(mixup_seed),
        )

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        Xtr = feat_train.iloc[tr_idx][feature_cols].astype(np.float32)
        Xva = feat_train.iloc[va_idx][feature_cols].astype(np.float32)
        ytr = y[tr_idx]
        yva = y[va_idx]
        wtr_time = _time_weight_from_timestamps(feat_train.iloc[tr_idx]["ts"], enabled=use_time_weights)
        wtr_domain = None if domain_weights_all is None else domain_weights_all[tr_idx].astype(np.float32)
        wtr_custom = None if sample_weights_all is None else sample_weights_all[tr_idx].astype(np.float32)
        wtr = _combine_sample_weights([wtr_time, wtr_domain, wtr_custom])
        Xtr, ytr, wtr = _augment_with_pseudo_fold(Xtr, ytr, wtr)
        Xtr, ytr, wtr, mixup_info = _augment_with_mixup_fold(
            Xtr,
            ytr,
            wtr,
            mixup_seed=seed + fold_id * 101 + 7,
        )

        pred_va = train_classifier(
            X_train=Xtr,
            y_train=ytr,
            X_valid=Xva,
            y_valid=yva,
            cfg=cfg,
            seed=seed + fold_id * 101,
            n_estimators=n_estimators,
            sample_weight=wtr,
            training_objective=training_objective,
            monotone_safe_bio=monotone_safe_bio,
            multiclass_oversample_rare=multiclass_oversample_rare,
            multiclass_corm_factor=multiclass_corm_factor,
            multiclass_wader_factor=multiclass_wader_factor,
            multiclass_oversample_noise=multiclass_oversample_noise,
        )
        pred_te = train_classifier_full(
            X_train=Xtr,
            y_train=ytr,
            X_test=X_test,
            cfg=cfg,
            seed=seed + fold_id * 101,
            n_estimators=n_estimators,
            sample_weight=wtr,
            training_objective=training_objective,
            monotone_safe_bio=monotone_safe_bio,
            multiclass_oversample_rare=multiclass_oversample_rare,
            multiclass_corm_factor=multiclass_corm_factor,
            multiclass_wader_factor=multiclass_wader_factor,
            multiclass_oversample_noise=multiclass_oversample_noise,
        )

        oof[va_idx] = pred_va
        covered[va_idx] = True
        test_folds.append(pred_te)

        fold_metrics.append(
            {
                "fold": int(fold_id),
                "train_size": int(len(tr_idx)),
                "valid_size": int(len(va_idx)),
                "macro_map": float(macro_map(yva, pred_va)),
                "per_class_ap": per_class_ap(yva, pred_va),
                "mixup_info": mixup_info,
            }
        )
        print(
            f"forward_cv fold={fold_id} macro={fold_metrics[-1]['macro_map']:.6f} "
            f"train={len(tr_idx)} valid={len(va_idx)}",
            flush=True,
        )

    test_cv = np.mean(np.stack(test_folds, axis=0), axis=0).astype(np.float32)
    covered_idx = np.where(covered)[0].astype(np.int64)

    # forward complete OOF: fill warmup (fold0 train chunk) with a backcast model.
    oof_complete = oof.copy()
    complete_idx = covered_idx.copy()
    test_complete = test_cv.copy()
    complete_meta: dict[str, object] = {"mode": "off", "added": 0}

    if complete_mode != "off":
        warmup_idx = np.array(folds[0][0], dtype=np.int64)
        warmup_idx = np.unique(warmup_idx)
        missing_warmup = warmup_idx[~covered[warmup_idx]]
        if len(missing_warmup) > 0:
            if complete_mode == "backcast_last":
                # Train backcast on late-history train chunk, excluding warmup.
                backcast_train = np.setdiff1d(np.array(folds[-1][0], dtype=np.int64), warmup_idx, assume_unique=False)
            elif complete_mode == "backcast_all":
                backcast_train = np.setdiff1d(np.arange(len(feat_train), dtype=np.int64), warmup_idx, assume_unique=False)
            else:
                raise ValueError(f"unknown complete_mode: {complete_mode}")

            if len(backcast_train) == 0:
                raise RuntimeError("backcast requested but backcast_train is empty")

            Xtr = feat_train.iloc[backcast_train][feature_cols].astype(np.float32)
            Xva = feat_train.iloc[missing_warmup][feature_cols].astype(np.float32)
            ytr = y[backcast_train]
            yva = y[missing_warmup]
            wtr_time = _time_weight_from_timestamps(feat_train.iloc[backcast_train]["ts"], enabled=use_time_weights)
            wtr_domain = None if domain_weights_all is None else domain_weights_all[backcast_train].astype(np.float32)
            wtr_custom = None if sample_weights_all is None else sample_weights_all[backcast_train].astype(np.float32)
            wtr = _combine_sample_weights([wtr_time, wtr_domain, wtr_custom])
            Xtr, ytr, wtr = _augment_with_pseudo_fold(Xtr, ytr, wtr)
            Xtr, ytr, wtr, _ = _augment_with_mixup_fold(
                Xtr,
                ytr,
                wtr,
                mixup_seed=seed + 99998,
            )

            pred_backcast = train_classifier(
                X_train=Xtr,
                y_train=ytr,
                X_valid=Xva,
                y_valid=yva,
                cfg=cfg,
                seed=seed + 99991,
                n_estimators=n_estimators,
                sample_weight=wtr,
                training_objective=training_objective,
                monotone_safe_bio=monotone_safe_bio,
                multiclass_oversample_rare=multiclass_oversample_rare,
                multiclass_corm_factor=multiclass_corm_factor,
                multiclass_wader_factor=multiclass_wader_factor,
                multiclass_oversample_noise=multiclass_oversample_noise,
            )
            pred_backcast_test = train_classifier_full(
                X_train=Xtr,
                y_train=ytr,
                X_test=feat_test[feature_cols].astype(np.float32),
                cfg=cfg,
                seed=seed + 99991,
                n_estimators=n_estimators,
                sample_weight=wtr,
                training_objective=training_objective,
                monotone_safe_bio=monotone_safe_bio,
                multiclass_oversample_rare=multiclass_oversample_rare,
                multiclass_corm_factor=multiclass_corm_factor,
                multiclass_wader_factor=multiclass_wader_factor,
                multiclass_oversample_noise=multiclass_oversample_noise,
            )

            oof_complete[missing_warmup] = pred_backcast.astype(np.float32)
            covered[missing_warmup] = True
            complete_idx = np.where(covered)[0].astype(np.int64)
            # keep test mean stable by default: do not mix backcast into test_complete
            test_complete = test_cv.copy()
            complete_meta = {
                "mode": complete_mode,
                "added": int(len(missing_warmup)),
                "warmup_size": int(len(warmup_idx)),
                "backcast_train_size": int(len(backcast_train)),
                "backcast_valid_size": int(len(missing_warmup)),
                "backcast_macro_map": float(macro_map(yva, pred_backcast)),
                "backcast_per_class_ap": per_class_ap(yva, pred_backcast),
                "backcast_test_pred_mean": float(np.mean(pred_backcast_test)),
            }

            print(
                f"forward_cv_complete mode={complete_mode} added={len(missing_warmup)} "
                f"backcast_macro={complete_meta['backcast_macro_map']:.6f}",
                flush=True,
            )
        else:
            complete_meta = {
                "mode": complete_mode,
                "added": 0,
                "warmup_size": int(len(warmup_idx)),
                "note": "no missing warmup rows to fill",
            }

    return oof, test_cv, covered_idx, fold_metrics, oof_complete, test_complete, complete_idx, complete_meta


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    train_cache = load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")

    ext_train = None
    ext_test = None
    if args.external_train_parquet.strip() or args.external_test_parquet.strip():
        if not (args.external_train_parquet.strip() and args.external_test_parquet.strip()):
            raise ValueError("both --external-train-parquet and --external-test-parquet must be provided together")
        ext_train = load_external_features(args.external_train_parquet.strip())
        ext_test = load_external_features(args.external_test_parquet.strip())
        print(
            f"external_features enabled: train={args.external_train_parquet} test={args.external_test_parquet}",
            flush=True,
        )

    spatial_monthly_centers = compute_monthly_track_centers(train_df, track_cache=train_cache)
    feat_train = build_feature_frame(
        train_df,
        track_cache=train_cache,
        external_features=ext_train,
        monthly_centers=spatial_monthly_centers,
    )
    feat_test = build_feature_frame(
        test_df,
        track_cache=test_cache,
        external_features=ext_test,
        monthly_centers=spatial_monthly_centers,
    )

    train_df["ts"] = pd.to_datetime(train_df["timestamp_start_radar_utc"], errors="coerce", utc=True)
    feat_train = feat_train.merge(train_df[["track_id", "bird_group", "observation_id", "ts"]], on="track_id", how="left", suffixes=("", "_src"))
    feat_train = feat_train.copy()
    feat_test = feat_test.copy()
    if "observation_id_src" in feat_train.columns:
        feat_train["observation_id"] = feat_train["observation_id_src"]
    feat_train["_cv_group"] = feat_train["observation_id"].astype(np.int64)
    cv_ts = pd.to_datetime(feat_train["ts"], errors="coerce", utc=True)
    if args.cv_direction == "reverse":
        ts_ns = cv_ts.astype("int64").to_numpy(dtype=np.int64)
        ts_min = int(np.min(ts_ns))
        ts_max = int(np.max(ts_ns))
        rev_ns = ts_min + (ts_max - ts_ns)
        cv_ts = pd.to_datetime(rev_ns, utc=True)
    feat_train["_cv_ts"] = cv_ts

    sample_weights_all, sample_weight_info = build_sample_weight_payload(
        args.sample_weight_parquet,
        feat_train,
    )
    if sample_weight_info.get("enabled", False):
        print(
            f"sample_weights enabled: path={args.sample_weight_parquet} "
            f"coverage={sample_weight_info['coverage']:.3f} "
            f"w[min,mean,max]=({sample_weight_info['weight_min']:.3f},"
            f"{sample_weight_info['weight_mean']:.3f},{sample_weight_info['weight_max']:.3f})",
            flush=True,
        )

    feature_cols = [
        c
        for c in feat_train.columns
        if c not in ["track_id", "observation_id", "observation_id_src", "primary_observation_id", "bird_group", "ts", "_cv_group", "_cv_ts"]
    ]

    blacklist_patterns = _blacklist_patterns(args.blacklist_mode)
    if args.extra_blacklist.strip():
        blacklist_patterns.extend([x.strip() for x in args.extra_blacklist.split(",") if x.strip()])
    feature_cols, blacklist_dropped = _apply_blacklist(feature_cols, blacklist_patterns)
    allowlist_info: dict[str, object] = {"enabled": False}
    if args.feature_allowlist_file.strip():
        allow = _load_feature_allowlist(args.feature_allowlist_file.strip())
        before = list(feature_cols)
        feature_cols = [c for c in feature_cols if c in allow]
        allowlist_info = {
            "enabled": True,
            "path": str(Path(args.feature_allowlist_file.strip()).resolve()),
            "requested": int(len(allow)),
            "kept": int(len(feature_cols)),
            "missing_requested": sorted([c for c in allow if c not in set(before)]),
        }
        print(
            f"feature_allowlist enabled: requested={allowlist_info['requested']} "
            f"kept={allowlist_info['kept']}",
            flush=True,
        )
        if len(feature_cols) == 0:
            raise RuntimeError("feature allowlist removed all features")

    pseudo_idx, pseudo_cls_idx, pseudo_w, pseudo_info = build_pseudo_payload(
        args.pseudo_label_csv,
        feat_test,
        prob_col=args.pseudo_prob_col,
        default_weight=args.pseudo_weight,
    )

    y_idx = feat_train["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = np.zeros((len(feat_train), len(CLASSES)), dtype=np.int32)
    y[np.arange(len(feat_train)), y_idx] = 1

    mixup_class_names = [x.strip() for x in str(args.tabular_mixup_classes).split(",") if x.strip()]
    mixup_unknown = [x for x in mixup_class_names if x not in CLASS_TO_INDEX]
    if mixup_unknown:
        raise ValueError(f"unknown --tabular-mixup-classes entries: {mixup_unknown}")
    mixup_class_indices = [int(CLASS_TO_INDEX[x]) for x in mixup_class_names]
    mixup_class_multiplier_map: dict[int, float] = {}
    if str(args.tabular_mixup_class_multipliers).strip():
        for token in str(args.tabular_mixup_class_multipliers).split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(f"invalid token in --tabular-mixup-class-multipliers: {token}")
            cls_name, mult_s = token.split(":", 1)
            cls_name = cls_name.strip()
            if cls_name not in CLASS_TO_INDEX:
                raise ValueError(f"unknown class in --tabular-mixup-class-multipliers: {cls_name}")
            mult = float(mult_s.strip())
            mixup_class_multiplier_map[int(CLASS_TO_INDEX[cls_name])] = float(mult)
    mixup_info = {
        "enabled": bool(args.tabular_mixup_rare),
        "alpha": float(args.tabular_mixup_alpha),
        "multiplier": float(args.tabular_mixup_multiplier),
        "classes": mixup_class_names,
        "class_indices": mixup_class_indices,
        "class_multiplier_map": {CLASSES[k]: float(v) for k, v in mixup_class_multiplier_map.items()},
    }
    if args.tabular_mixup_rare:
        print(
            "tabular_mixup enabled: "
            f"classes={mixup_class_names} alpha={args.tabular_mixup_alpha:.3f} "
            f"multiplier={args.tabular_mixup_multiplier:.3f} "
            f"class_mult={{{', '.join(f'{CLASSES[k]}:{v:.2f}' for k,v in mixup_class_multiplier_map.items())}}}",
            flush=True,
        )

    if args.holdout_mode == "stratified":
        if int(args.stratified_fold) == -1:
            tr_idx = np.arange(len(y_idx), dtype=np.int64)
            va_idx = np.arange(len(y_idx), dtype=np.int64)  # dummy full-train mode
            cutoff = pd.Timestamp("1970-01-01", tz="UTC")
            forward_sanity: list[dict[str, object]] = [
                {
                    "mode": "stratified_full",
                    "train_size": int(len(tr_idx)),
                }
            ]
        else:
            tr_idx, va_idx = make_stratified_holdout_split(
                y_idx=y_idx,
                n_splits=int(args.stratified_splits),
                fold_id=int(args.stratified_fold),
                seed=int(args.seed),
            )
            cutoff = pd.Timestamp("1970-01-01", tz="UTC")
            forward_sanity = [
                {
                    "mode": "stratified",
                    "n_splits": int(args.stratified_splits),
                    "fold_id": int(args.stratified_fold),
                    "train_size": int(len(tr_idx)),
                    "valid_size": int(len(va_idx)),
                }
            ]
    else:
        if args.temporal_cutoff_date.strip():
            tr_idx, va_idx, cutoff = make_temporal_holdout_split_with_cutoff(
                feat_train[["_cv_ts", "_cv_group"]].copy(),
                cutoff=args.temporal_cutoff_date.strip(),
                timestamp_col="_cv_ts",
                group_col="_cv_group",
            )
        else:
            tr_idx, va_idx, cutoff = make_temporal_holdout_split(
                feat_train[["_cv_ts", "_cv_group"]].copy(),
                timestamp_col="_cv_ts",
                group_col="_cv_group",
                holdout_quantile=args.temporal_quantile,
            )

        # Sanity-check for forward temporal ordering.
        forward_folds = make_forward_temporal_group_folds(
            feat_train[["_cv_ts", "_cv_group"]].copy(),
            timestamp_col="_cv_ts",
            group_col="_cv_group",
            n_splits=5,
        )
        forward_sanity = []
        for fold_id, (f_tr, f_va) in enumerate(forward_folds):
            tr_max = feat_train.iloc[f_tr]["_cv_ts"].max()
            va_min = feat_train.iloc[f_va]["_cv_ts"].min()
            forward_sanity.append(
                {
                    "fold": int(fold_id),
                    "train_max_ts": str(tr_max),
                    "val_min_ts": str(va_min),
                    "train_max_lt_val_min": bool(tr_max < va_min),
                }
            )

    drift = compute_drift_ks(feat_train, feat_test, feature_cols)

    adv_auc, adv_folds = adversarial_auc(feat_train, feat_test, feature_cols, seed=args.seed)

    domain_reweight_info: dict[str, object] = {"mode": "none"}
    domain_weights_all: np.ndarray | None = None
    if args.domain_reweight == "odds":
        domain_weights_all, domain_reweight_info = domain_reweight_odds(
            train_feat=feat_train,
            test_feat=feat_test,
            feature_cols=feature_cols,
            seed=args.seed,
            n_splits=args.domain_reweight_splits,
            clip_min=args.domain_reweight_clip_min,
            clip_max=args.domain_reweight_clip_max,
        )
        print(
            "domain_reweight odds "
            f"auc_mean={domain_reweight_info['auc_mean']:.6f} "
            f"w[min,p50,max]=({domain_reweight_info['w_min']:.3f},"
            f"{domain_reweight_info['w_q50']:.3f},{domain_reweight_info['w_max']:.3f})",
            flush=True,
        )

    drop_k_values = [int(x.strip()) for x in args.drop_k_grid.split(",") if x.strip()]
    fulltrain_mode = bool(args.holdout_mode == "stratified" and int(args.stratified_fold) == -1)
    candidate_cfgs = [
        {"name": "base", "lr": 0.03, "leaves": 63, "ff": 0.75, "mc": 20, "l1": 0.05, "l2": 0.2},
        {"name": "reg", "lr": 0.02, "leaves": 47, "ff": 0.70, "mc": 30, "l1": 0.10, "l2": 0.40},
        {"name": "aggr", "lr": 0.03, "leaves": 95, "ff": 0.85, "mc": 10, "l1": 0.00, "l2": 0.05},
        {"name": "deep", "lr": 0.02, "leaves": 127, "ff": 0.70, "mc": 15, "l1": 0.05, "l2": 0.10},
        {"name": "wide", "lr": 0.01, "leaves": 63, "ff": 0.60, "mc": 40, "l1": 0.20, "l2": 0.50},
        {"name": "reg_heavy", "lr": 0.02, "leaves": 15, "ff": 0.50, "mc": 50, "l1": 1.00, "l2": 5.00, "bf": 0.70, "bag_freq": 5},
    ]
    if args.force_config:
        candidate_cfgs = [c for c in candidate_cfgs if c["name"] == args.force_config]
        if not candidate_cfgs:
            raise ValueError(f"unknown force-config: {args.force_config}")

    tr_w_time = _time_weight_from_timestamps(feat_train.iloc[tr_idx]["ts"], enabled=args.use_time_weights)
    tr_w_domain = None if domain_weights_all is None else domain_weights_all[tr_idx].astype(np.float32)
    tr_w_custom = None if sample_weights_all is None else sample_weights_all[tr_idx].astype(np.float32)
    tr_w = _combine_sample_weights([tr_w_time, tr_w_domain, tr_w_custom])

    if fulltrain_mode:
        drop_k = int(drop_k_values[0] if drop_k_values else 0)
        drop_set = set([name for name, _ in drift[:drop_k]])
        best_cols = [c for c in feature_cols if c not in drop_set]
        best_cfg = candidate_cfgs[0]
        best_drop_k = drop_k
        best_macro = 0.0
        best_per_cls = {k: 0.0 for k in CLASSES}
        best_holdout_pred = np.zeros((len(va_idx), len(CLASSES)), dtype=np.float32)
        print(
            f"fulltrain_mode=on cfg={best_cfg['name']} drop_k={best_drop_k} "
            f"blacklist={args.blacklist_mode} n_features={len(best_cols)}",
            flush=True,
        )
    else:
        best: tuple[float, dict, list[str], dict[str, float], int, np.ndarray] | None = None

        for drop_k in drop_k_values:
            drop_set = set([name for name, _ in drift[:drop_k]])
            cols = [c for c in feature_cols if c not in drop_set]

            Xtr = feat_train.iloc[tr_idx][cols].astype(np.float32)
            Xva = feat_train.iloc[va_idx][cols].astype(np.float32)
            ytr = y[tr_idx]
            yva = y[va_idx]

            for cfg_id, cfg in enumerate(candidate_cfgs):
                Xtr_fit = Xtr
                ytr_fit = ytr
                wtr_fit = tr_w
                if args.tabular_mixup_rare:
                    Xtr_fit, ytr_fit, wtr_fit, _ = _tabular_mixup_within_classes(
                        X=Xtr,
                        y=ytr,
                        sample_weight=tr_w,
                        class_indices=mixup_class_indices,
                        alpha=float(args.tabular_mixup_alpha),
                        multiplier=float(args.tabular_mixup_multiplier),
                        class_multiplier_map=mixup_class_multiplier_map,
                        seed=int(args.seed + drop_k * 1009 + cfg_id * 97),
                    )
                pred = train_classifier(
                    X_train=Xtr_fit,
                    y_train=ytr_fit,
                    X_valid=Xva,
                    y_valid=yva,
                    cfg=cfg,
                    seed=args.seed,
                    n_estimators=args.n_estimators,
                    sample_weight=wtr_fit,
                    training_objective=args.training_objective,
                    monotone_safe_bio=args.monotone_safe_bio,
                    multiclass_oversample_rare=args.multiclass_oversample_rare,
                    multiclass_corm_factor=args.multiclass_corm_factor,
                    multiclass_wader_factor=args.multiclass_wader_factor,
                    multiclass_oversample_noise=args.multiclass_oversample_noise,
                )
                m = macro_map(yva, pred)
                per_cls = per_class_ap(yva, pred)
                print(
                    f"cfg={cfg['name']} drop_k={drop_k} blacklist={args.blacklist_mode} "
                    f"temporal_macro={m:.6f} n_features={len(cols)}",
                    flush=True,
                )
                if best is None or m > best[0]:
                    best = (m, cfg, cols, per_cls, drop_k, pred.copy())

        assert best is not None
        best_macro, best_cfg, best_cols, best_per_cls, best_drop_k, best_holdout_pred = best

    all_w_time = _time_weight_from_timestamps(feat_train["ts"], enabled=args.use_time_weights)
    all_w = _combine_sample_weights([all_w_time, domain_weights_all, sample_weights_all])

    X_all_real = feat_train[best_cols].astype(np.float32)
    y_all_real = y
    w_all_real = all_w

    X_test = feat_test[best_cols].astype(np.float32)

    if len(pseudo_idx) > 0:
        X_pseudo = feat_test.iloc[pseudo_idx][best_cols].astype(np.float32)
        y_pseudo = np.zeros((len(pseudo_idx), len(CLASSES)), dtype=np.int32)
        y_pseudo[np.arange(len(pseudo_idx)), pseudo_cls_idx] = 1
        X_all = pd.concat([X_all_real, X_pseudo], axis=0, ignore_index=True)
        y_all = np.vstack([y_all_real, y_pseudo]).astype(np.int32)
        if w_all_real is None:
            w_all = np.concatenate(
                [np.ones((len(X_all_real),), dtype=np.float32), pseudo_w],
                axis=0,
            ).astype(np.float32)
        else:
            w_all = np.concatenate([w_all_real.astype(np.float32), pseudo_w], axis=0).astype(np.float32)
        w_all = (w_all / (np.mean(w_all) + 1e-12)).astype(np.float32)
    else:
        X_all = X_all_real
        y_all = y_all_real
        w_all = w_all_real

    if args.tabular_mixup_rare:
        X_all, y_all, w_all, _ = _tabular_mixup_within_classes(
            X=X_all,
            y=y_all,
            sample_weight=w_all,
            class_indices=mixup_class_indices,
            alpha=float(args.tabular_mixup_alpha),
            multiplier=float(args.tabular_mixup_multiplier),
            class_multiplier_map=mixup_class_multiplier_map,
            seed=int(args.seed + 424242),
        )

    test_pred = train_classifier_full(
        X_train=X_all,
        y_train=y_all,
        X_test=X_test,
        cfg=best_cfg,
        seed=args.seed,
        n_estimators=args.n_estimators,
        sample_weight=w_all,
        training_objective=args.training_objective,
        monotone_safe_bio=args.monotone_safe_bio,
        multiclass_oversample_rare=args.multiclass_oversample_rare,
        multiclass_corm_factor=args.multiclass_corm_factor,
        multiclass_wader_factor=args.multiclass_wader_factor,
        multiclass_oversample_noise=args.multiclass_oversample_noise,
    )

    sub = pd.DataFrame(test_pred, columns=CLASSES)
    sub.insert(0, "track_id", test_df["track_id"].to_numpy())
    sub_path = out_dir / "sub_temporal_best.csv"
    sub.to_csv(sub_path, index=False)

    drift_df = pd.DataFrame(drift, columns=["feature", "ks_stat"])
    drift_df.to_csv(out_dir / "drift_ks.csv", index=False)

    meta_artifacts: dict[str, object] = {}
    if args.save_meta_artifacts:
        artifacts_dir = out_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        train_track_ids = feat_train["track_id"].to_numpy()
        test_track_ids = feat_test["track_id"].to_numpy()
        oof_targets = y.astype(np.float32)

        np.save(artifacts_dir / "train_track_ids.npy", train_track_ids)
        np.save(artifacts_dir / "test_track_ids.npy", test_track_ids)
        np.save(artifacts_dir / "oof_targets.npy", oof_targets)
        np.save(artifacts_dir / "test_full.npy", test_pred.astype(np.float32))

        holdout_oof = np.zeros((len(feat_train), len(CLASSES)), dtype=np.float32)
        holdout_oof[va_idx] = best_holdout_pred.astype(np.float32)
        holdout_idx = np.array(va_idx, dtype=np.int64)
        np.save(artifacts_dir / "oof_holdout.npy", holdout_oof)
        np.save(artifacts_dir / "oof_holdout_idx.npy", holdout_idx)

        meta_artifacts = {
            "artifacts_dir": str(artifacts_dir.resolve()),
            "train_track_ids_path": str((artifacts_dir / "train_track_ids.npy").resolve()),
            "test_track_ids_path": str((artifacts_dir / "test_track_ids.npy").resolve()),
            "oof_targets_path": str((artifacts_dir / "oof_targets.npy").resolve()),
            "test_full_path": str((artifacts_dir / "test_full.npy").resolve()),
            "oof_holdout_path": str((artifacts_dir / "oof_holdout.npy").resolve()),
            "oof_holdout_idx_path": str((artifacts_dir / "oof_holdout_idx.npy").resolve()),
            "oof_holdout_covered": int(len(holdout_idx)),
            "oof_holdout_coverage_ratio": float(len(holdout_idx) / max(len(feat_train), 1)),
        }

        if args.oof_strategy == "forward_cv":
            oof_cv, test_cv, covered_idx, fold_metrics, oof_complete, test_complete, complete_idx, complete_meta = _run_forward_cv_oof(
                feat_train=feat_train,
                feat_test=feat_test,
                y=y,
                feature_cols=best_cols,
                cfg=best_cfg,
                seed=args.seed,
                n_estimators=args.n_estimators,
                use_time_weights=args.use_time_weights,
                domain_weights_all=domain_weights_all,
                sample_weights_all=sample_weights_all,
                n_splits=args.oof_forward_splits,
                complete_mode=args.oof_forward_complete,
                training_objective=args.training_objective,
                monotone_safe_bio=args.monotone_safe_bio,
                multiclass_oversample_rare=args.multiclass_oversample_rare,
                multiclass_corm_factor=args.multiclass_corm_factor,
                multiclass_wader_factor=args.multiclass_wader_factor,
                multiclass_oversample_noise=args.multiclass_oversample_noise,
                pseudo_X=(feat_test.iloc[pseudo_idx][best_cols].astype(np.float32) if len(pseudo_idx) > 0 else None),
                pseudo_y=(
                    np.eye(len(CLASSES), dtype=np.int32)[pseudo_cls_idx]
                    if len(pseudo_idx) > 0
                    else None
                ),
                pseudo_w=(pseudo_w.astype(np.float32) if len(pseudo_idx) > 0 else None),
                tabular_mixup_rare=bool(args.tabular_mixup_rare),
                tabular_mixup_alpha=float(args.tabular_mixup_alpha),
                tabular_mixup_multiplier=float(args.tabular_mixup_multiplier),
                tabular_mixup_class_indices=mixup_class_indices,
                tabular_mixup_class_multiplier_map=mixup_class_multiplier_map,
            )
            np.save(artifacts_dir / "oof_forward_cv.npy", oof_cv.astype(np.float32))
            np.save(artifacts_dir / "oof_forward_cv_idx.npy", covered_idx)
            np.save(artifacts_dir / "test_forward_cv_mean.npy", test_cv.astype(np.float32))
            np.save(artifacts_dir / "oof_forward_cv_complete.npy", oof_complete.astype(np.float32))
            np.save(artifacts_dir / "oof_forward_cv_complete_idx.npy", complete_idx.astype(np.int64))
            np.save(artifacts_dir / "test_forward_cv_complete_mean.npy", test_complete.astype(np.float32))
            meta_artifacts.update(
                {
                    "oof_strategy": "forward_cv",
                    "oof_forward_splits": int(args.oof_forward_splits),
                    "oof_forward_complete": args.oof_forward_complete,
                    "oof_forward_cv_path": str((artifacts_dir / "oof_forward_cv.npy").resolve()),
                    "oof_forward_cv_idx_path": str((artifacts_dir / "oof_forward_cv_idx.npy").resolve()),
                    "test_forward_cv_mean_path": str((artifacts_dir / "test_forward_cv_mean.npy").resolve()),
                    "oof_forward_cv_complete_path": str((artifacts_dir / "oof_forward_cv_complete.npy").resolve()),
                    "oof_forward_cv_complete_idx_path": str((artifacts_dir / "oof_forward_cv_complete_idx.npy").resolve()),
                    "test_forward_cv_complete_mean_path": str((artifacts_dir / "test_forward_cv_complete_mean.npy").resolve()),
                    "oof_forward_cv_covered": int(len(covered_idx)),
                    "oof_forward_cv_coverage_ratio": float(len(covered_idx) / max(len(feat_train), 1)),
                    "oof_forward_cv_complete_covered": int(len(complete_idx)),
                    "oof_forward_cv_complete_coverage_ratio": float(len(complete_idx) / max(len(feat_train), 1)),
                    "oof_forward_cv_fold_metrics": fold_metrics,
                    "oof_forward_cv_complete_meta": complete_meta,
                }
            )

    cutoff_mode = "stratified" if args.holdout_mode == "stratified" else ("fixed_date" if args.temporal_cutoff_date.strip() else "quantile")

    report = {
        "holdout_mode": args.holdout_mode,
        "stratified_splits": int(args.stratified_splits),
        "stratified_fold": int(args.stratified_fold),
        "holdout_metric_available": bool(not fulltrain_mode),
        "cv_direction": args.cv_direction,
        "temporal_cutoff": str(cutoff),
        "temporal_cutoff_mode": cutoff_mode,
        "temporal_cutoff_date_arg": args.temporal_cutoff_date.strip(),
        "train_size": int(len(tr_idx)),
        "valid_size": int(len(va_idx)),
        "temporal_macro_map_best": float(best_macro),
        "temporal_per_class_best": best_per_cls,
        "best_config": best_cfg,
        "best_drop_k": int(best_drop_k),
        "training_objective": args.training_objective,
        "monotone_safe_bio": bool(args.monotone_safe_bio),
        "multiclass_oversample_rare": bool(args.multiclass_oversample_rare),
        "multiclass_corm_factor": float(args.multiclass_corm_factor),
        "multiclass_wader_factor": float(args.multiclass_wader_factor),
        "multiclass_oversample_noise": float(args.multiclass_oversample_noise),
        "tabular_mixup_info": mixup_info,
        "n_features": int(len(best_cols)),
        "drop_features": [name for name, _ in drift[:best_drop_k]],
        "blacklist_mode": args.blacklist_mode,
        "blacklist_patterns": blacklist_patterns,
        "blacklist_dropped_features": blacklist_dropped,
        "feature_allowlist_info": allowlist_info,
        "adversarial_auc_mean": float(adv_auc),
        "adversarial_auc_folds": [float(x) for x in adv_folds],
        "domain_reweight": args.domain_reweight,
        "domain_reweight_info": domain_reweight_info,
        "pseudo_label_info": pseudo_info,
        "top10_drift": [{"feature": k, "ks": float(v)} for k, v in drift[:10]],
        "forward_sanity": forward_sanity,
        "submission_path": str(sub_path),
        "save_meta_artifacts": bool(args.save_meta_artifacts),
        "oof_strategy": args.oof_strategy,
        "external_train_parquet": args.external_train_parquet,
        "external_test_parquet": args.external_test_parquet,
        "sample_weight_parquet": args.sample_weight_parquet,
        "sample_weight_info": sample_weight_info,
        "meta_artifacts": meta_artifacts,
    }

    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)

    print("=== SHIFT REPORT ===", flush=True)
    print(f"Temporal macro mAP: {best_macro:.6f}", flush=True)
    print(f"Adversarial AUC: {adv_auc:.6f}", flush=True)
    print(f"Saved submission: {sub_path}", flush=True)


if __name__ == "__main__":
    main()
