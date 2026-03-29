from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from config import CLASSES
from src.metrics import macro_map_score, one_hot_labels, per_class_average_precision


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _train_binary_meta(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    X_te: np.ndarray,
    learner: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    if int(np.sum(y_tr)) == 0 or int(np.sum(y_tr)) == len(y_tr):
        const_prob = float(np.mean(y_tr))
        p_va = np.full(X_va.shape[0], const_prob, dtype=np.float32)
        p_te = np.full(X_te.shape[0], const_prob, dtype=np.float32)
        return p_va, p_te, {"constant": const_prob}

    if learner == "lgbm":
        model = lgb.LGBMClassifier(
            objective="binary",
            metric="average_precision",
            n_estimators=1200,
            learning_rate=0.02,
            num_leaves=15,
            min_child_samples=20,
            colsample_bytree=0.9,
            subsample=0.9,
            subsample_freq=1,
            random_state=int(seed),
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(X_tr, y_tr.astype(np.int32), callbacks=[lgb.log_evaluation(period=0)])
        p_va = model.predict_proba(X_va)[:, 1].astype(np.float32)
        p_te = model.predict_proba(X_te)[:, 1].astype(np.float32)
        return p_va, p_te, model

    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=3000,
        random_state=int(seed),
        class_weight="balanced",
    )
    model.fit(X_tr, y_tr.astype(np.int32))
    p_va = model.predict_proba(X_va)[:, 1].astype(np.float32)
    p_te = model.predict_proba(X_te)[:, 1].astype(np.float32)
    return p_va, p_te, model


def run_stacking_cv(
    oof_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    labels_idx: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 3,
    seed: int = 2026,
    learner: str = "logreg",
) -> dict[str, Any]:
    names = sorted(oof_predictions.keys())
    X_meta = np.hstack([oof_predictions[name].astype(np.float32) for name in names])
    X_meta_test = np.hstack([test_predictions[name].astype(np.float32) for name in names])

    y_all = one_hot_labels(labels_idx, len(CLASSES)).astype(np.float32)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(X_meta, labels_idx, groups=groups))

    oof_meta = np.zeros((X_meta.shape[0], len(CLASSES)), dtype=np.float32)
    test_fold_preds: list[np.ndarray] = []
    fold_scores: list[dict[str, Any]] = []

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        X_tr = X_meta[tr_idx]
        X_va = X_meta[va_idx]
        y_tr = y_all[tr_idx]
        y_va = y_all[va_idx]

        fold_valid = np.zeros((len(va_idx), len(CLASSES)), dtype=np.float32)
        fold_test = np.zeros((X_meta_test.shape[0], len(CLASSES)), dtype=np.float32)

        for class_idx in range(len(CLASSES)):
            p_va, p_te, _ = _train_binary_meta(
                X_tr=X_tr,
                y_tr=y_tr[:, class_idx],
                X_va=X_va,
                X_te=X_meta_test,
                learner=learner,
                seed=seed + fold_id * 31 + class_idx,
            )
            fold_valid[:, class_idx] = p_va
            fold_test[:, class_idx] = p_te

        oof_meta[va_idx] = fold_valid
        test_fold_preds.append(fold_test)

        fold_macro = float(macro_map_score(y_va, fold_valid))
        fold_scores.append(
            {
                "fold": int(fold_id),
                "macro_map": fold_macro,
                "per_class_map": per_class_average_precision(y_va, fold_valid),
                "train_size": int(len(tr_idx)),
                "valid_size": int(len(va_idx)),
            }
        )

    test_meta = np.mean(np.stack(test_fold_preds, axis=0), axis=0).astype(np.float32)
    macro = float(macro_map_score(y_all, oof_meta))

    return {
        "model_names": names,
        "oof": oof_meta,
        "test": test_meta,
        "macro_map": macro,
        "per_class_ap": per_class_average_precision(y_all, oof_meta),
        "fold_scores": fold_scores,
        "fold_std": float(np.std([f["macro_map"] for f in fold_scores])) if fold_scores else 0.0,
        "fold_mean": float(np.mean([f["macro_map"] for f in fold_scores])) if fold_scores else 0.0,
    }


def fit_stacking_full_predict(
    oof_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    labels_idx: np.ndarray,
    learner: str = "logreg",
    seed: int = 2026,
) -> np.ndarray:
    names = sorted(oof_predictions.keys())
    X_meta = np.hstack([oof_predictions[name].astype(np.float32) for name in names])
    X_meta_test = np.hstack([test_predictions[name].astype(np.float32) for name in names])
    y_all = one_hot_labels(labels_idx, len(CLASSES)).astype(np.float32)

    out = np.zeros((X_meta_test.shape[0], len(CLASSES)), dtype=np.float32)

    for class_idx in range(len(CLASSES)):
        y_class = y_all[:, class_idx]
        if int(np.sum(y_class)) == 0 or int(np.sum(y_class)) == len(y_class):
            out[:, class_idx] = float(np.mean(y_class))
            continue

        if learner == "lgbm":
            model = lgb.LGBMClassifier(
                objective="binary",
                metric="average_precision",
                n_estimators=1200,
                learning_rate=0.02,
                num_leaves=15,
                min_child_samples=20,
                colsample_bytree=0.9,
                subsample=0.9,
                subsample_freq=1,
                random_state=int(seed + class_idx),
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(X_meta, y_class.astype(np.int32), callbacks=[lgb.log_evaluation(period=0)])
            out[:, class_idx] = model.predict_proba(X_meta_test)[:, 1].astype(np.float32)
        else:
            model = LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=3000,
                random_state=int(seed + class_idx),
                class_weight="balanced",
            )
            model.fit(X_meta, y_class.astype(np.int32))
            out[:, class_idx] = model.predict_proba(X_meta_test)[:, 1].astype(np.float32)

    return out


def _parse_name_path_pairs(items: list[str]) -> dict[str, Path]:
    pairs: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid input format, expected name=path: {item}")
        name, path = item.split("=", 1)
        key = name.strip()
        if not key:
            raise ValueError(f"Empty model name in: {item}")
        pairs[key] = Path(path.strip())
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True, type=str)
    parser.add_argument("--oof-input", action="append", default=[], help="name=path_to_oof_npy")
    parser.add_argument("--test-input", action="append", default=[], help="name=path_to_test_npy")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--learner", type=str, default="logreg", choices=["logreg", "lgbm"])
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    oof_map_paths = _parse_name_path_pairs(args.oof_input)
    test_map_paths = _parse_name_path_pairs(args.test_input)

    if set(oof_map_paths.keys()) != set(test_map_paths.keys()):
        raise RuntimeError("Model names in --oof-input and --test-input must match")

    oof_preds = {k: np.load(v).astype(np.float32) for k, v in oof_map_paths.items()}
    test_preds = {k: np.load(v).astype(np.float32) for k, v in test_map_paths.items()}

    train_df = pd.read_csv(args.train_csv, usecols=["bird_group", "observation_id"])
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    labels_idx = train_df["bird_group"].map(label_to_idx).to_numpy(dtype=np.int64)
    groups = train_df["observation_id"].to_numpy()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_stacking_cv(
        oof_predictions=oof_preds,
        test_predictions=test_preds,
        labels_idx=labels_idx,
        groups=groups,
        n_splits=3,
        seed=int(args.seed),
        learner=args.learner,
    )

    np.save(out_dir / "oof_stacking.npy", result["oof"])
    np.save(out_dir / "test_stacking.npy", result["test"])
    with (out_dir / "stacking_scores.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_names": result["model_names"],
                "macro_map": result["macro_map"],
                "per_class_ap": result["per_class_ap"],
                "fold_scores": result["fold_scores"],
                "fold_std": result["fold_std"],
                "fold_mean": result["fold_mean"],
                "learner": args.learner,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    print(f"[stacking] macro_mAP={result['macro_map']:.6f} fold_std={result['fold_std']:.6f}", flush=True)


if __name__ == "__main__":
    main()
