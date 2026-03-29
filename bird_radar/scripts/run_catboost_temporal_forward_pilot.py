from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds
from src.feature_engineering import build_feature_frame, compute_monthly_track_centers
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CatBoost OvR temporal-forward pilot on tabular features.")
    p.add_argument("--data-dir", type=str, default="/Users/sgrisshk/Desktop/AI-task")
    p.add_argument("--cache-dir", type=str, default="/Users/sgrisshk/Desktop/AI-task/bird_radar/artifacts/cache")
    p.add_argument(
        "--feature-allowlist-file",
        type=str,
        default="/Users/sgrisshk/Desktop/AI-task/bird_radar/artifacts/proposal/feature_allowlists/orig146_plus_dtstats.csv",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="/Users/sgrisshk/Desktop/AI-task/bird_radar/artifacts/proposal/catboost_temporal_forward_orig146dt_pilot",
    )
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--l2-leaf-reg", type=float, default=5.0)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=777)
    return p.parse_args()


def _macro_map(y_true_onehot: np.ndarray, y_pred: np.ndarray) -> float:
    vals: list[float] = []
    for i in range(y_true_onehot.shape[1]):
        yt = y_true_onehot[:, i]
        yp = y_pred[:, i]
        vals.append(float(average_precision_score(yt, yp)) if int(yt.sum()) > 0 else 0.0)
    return float(np.mean(vals))


def _per_class_ap(y_true_onehot: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, cls in enumerate(CLASSES):
        yt = y_true_onehot[:, i]
        yp = y_pred[:, i]
        out[cls] = float(average_precision_score(yt, yp)) if int(yt.sum()) > 0 else 0.0
    return out


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


def _load_allowlist(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"allowlist file not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "feature" in df.columns:
            vals = df["feature"].astype(str).tolist()
        else:
            vals = df.iloc[:, 0].astype(str).tolist()
    else:
        vals = path.read_text(encoding="utf-8").splitlines()
    out = {x.strip() for x in vals if str(x).strip()}
    if not out:
        raise ValueError("allowlist is empty")
    return out


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    train_df["ts"] = pd.to_datetime(train_df["timestamp_start_radar_utc"], utc=True)
    test_df["ts"] = pd.to_datetime(test_df["timestamp_start_radar_utc"], utc=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = _load_or_build_cache(train_df, cache_dir / "track_cache_train.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "track_cache_test.pkl")

    monthly_centers = compute_monthly_track_centers(train_df)
    feat_train = build_feature_frame(train_df, track_cache=train_cache, monthly_centers=monthly_centers)
    feat_test = build_feature_frame(test_df, track_cache=test_cache, monthly_centers=monthly_centers)

    feat_train = feat_train.merge(
        train_df[["track_id", "bird_group", "observation_id", "ts"]],
        on="track_id",
        how="left",
        suffixes=("", "_src"),
    )

    allow = _load_allowlist(Path(args.feature_allowlist_file))
    protected = {"track_id", "observation_id", "primary_observation_id", "bird_group", "ts", "observation_id_src"}
    feature_cols = [c for c in feat_train.columns if c not in protected and c in allow]
    if not feature_cols:
        raise RuntimeError("no feature columns after applying allowlist")

    y_idx = feat_train["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y_onehot = np.zeros((len(feat_train), len(CLASSES)), dtype=np.int32)
    y_onehot[np.arange(len(feat_train)), y_idx] = 1

    split_df = feat_train[["timestamp_start_radar_utc"]].copy() if "timestamp_start_radar_utc" in feat_train.columns else None
    if split_df is None:
        split_df = train_df[["timestamp_start_radar_utc", "observation_id"]].copy()
    else:
        split_df["observation_id"] = feat_train["observation_id"]

    folds = make_forward_temporal_group_folds(
        split_df,
        timestamp_col="timestamp_start_radar_utc",
        group_col="observation_id",
        n_splits=int(args.n_splits),
    )

    X_train = feat_train[feature_cols].astype(np.float32)
    X_test = feat_test[feature_cols].astype(np.float32)

    oof = np.zeros((len(feat_train), len(CLASSES)), dtype=np.float32)
    covered = np.zeros(len(feat_train), dtype=bool)
    test_folds: list[np.ndarray] = []
    fold_macro: list[float] = []

    for fi, (tr_idx, va_idx) in enumerate(folds):
        Xtr = X_train.iloc[tr_idx]
        Xva = X_train.iloc[va_idx]
        ytr_idx = y_idx[tr_idx]
        yva_onehot = y_onehot[va_idx]

        pred_va = np.zeros((len(va_idx), len(CLASSES)), dtype=np.float32)
        pred_te = np.zeros((len(X_test), len(CLASSES)), dtype=np.float32)

        for ci, cls in enumerate(CLASSES):
            ytr_bin = (ytr_idx == ci).astype(np.int32)
            pos = int(ytr_bin.sum())
            neg = int(len(ytr_bin) - pos)
            if pos == 0:
                pred_va[:, ci] = 0.0
                pred_te[:, ci] = 0.0
                continue
            if neg == 0:
                pred_va[:, ci] = 1.0
                pred_te[:, ci] = 1.0
                continue
            pos_weight = float(neg / max(pos, 1))
            model = CatBoostClassifier(
                iterations=int(args.iterations),
                learning_rate=float(args.learning_rate),
                depth=int(args.depth),
                l2_leaf_reg=float(args.l2_leaf_reg),
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=int(args.seed + fi * 101 + ci),
                verbose=False,
                allow_writing_files=False,
                class_weights=[1.0, pos_weight],
            )
            model.fit(Xtr, ytr_bin)
            pred_va[:, ci] = model.predict_proba(Xva)[:, 1].astype(np.float32)
            pred_te[:, ci] = model.predict_proba(X_test)[:, 1].astype(np.float32)

        oof[va_idx] = pred_va
        covered[va_idx] = True
        test_folds.append(pred_te)
        fold_macro.append(_macro_map(yva_onehot, pred_va))
        print(f"fold={fi} macro_ap={fold_macro[-1]:.6f} train={len(tr_idx)} valid={len(va_idx)}", flush=True)

    covered_idx = np.where(covered)[0]
    if len(covered_idx) == 0:
        raise RuntimeError("no covered validation rows from temporal folds")
    oof_covered = oof[covered_idx]
    y_covered = y_onehot[covered_idx]
    macro_covered = _macro_map(y_covered, oof_covered)
    per_class = _per_class_ap(y_covered, oof_covered)

    test_mean = np.mean(np.stack(test_folds, axis=0), axis=0).astype(np.float32)

    np.save(artifacts_dir / "oof_forward_cv.npy", oof.astype(np.float32))
    np.save(artifacts_dir / "oof_forward_cv_idx.npy", covered_idx.astype(np.int64))
    np.save(artifacts_dir / "oof_targets.npy", y_onehot.astype(np.float32))
    np.save(artifacts_dir / "test_forward_cv_mean.npy", test_mean.astype(np.float32))
    np.save(artifacts_dir / "train_track_ids.npy", feat_train["track_id"].to_numpy(np.int64))
    np.save(artifacts_dir / "test_track_ids.npy", feat_test["track_id"].to_numpy(np.int64))

    sub = pd.DataFrame(test_mean, columns=CLASSES)
    sub.insert(0, "track_id", test_df["track_id"].to_numpy(np.int64))
    sub_path = out_dir / "submission_catboost_temporal_forward.csv"
    sub.to_csv(sub_path, index=False)

    report = {
        "model": "catboost_ovr_temporal_forward",
        "feature_allowlist": str(Path(args.feature_allowlist_file).resolve()),
        "n_features": int(len(feature_cols)),
        "iterations": int(args.iterations),
        "depth": int(args.depth),
        "learning_rate": float(args.learning_rate),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "n_splits": int(args.n_splits),
        "fold_macro_ap": [float(x) for x in fold_macro],
        "covered_rows": int(len(covered_idx)),
        "coverage_ratio": float(len(covered_idx) / max(len(feat_train), 1)),
        "oof_macro_ap_covered": float(macro_covered),
        "oof_per_class_ap_covered": per_class,
        "submission_csv": str(sub_path.resolve()),
        "artifacts_dir": str(artifacts_dir.resolve()),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"oof_macro_ap_covered={macro_covered:.6f}", flush=True)
    for cls in CLASSES:
        print(f"{cls}: {per_class[cls]:.6f}", flush=True)
    print(f"report={out_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
