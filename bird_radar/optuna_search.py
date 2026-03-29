from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import average_precision_score

from config import CACHE_DIR, CLASSES, DEFAULT_DATA_DIR
from train_lgbm import make_group_folds, prepare_tabular_data


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _safe_ap(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if int(np.sum(y_true)) == 0:
        return 0.0
    try:
        return float(average_precision_score(y_true, y_pred))
    except ValueError:
        return 0.0


def _train_eval_one_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> float:
    pos = max(int(np.sum(y_tr)), 1)
    neg = max(int(len(y_tr) - pos), 1)
    spw = float(neg / pos)

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="average_precision",
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        colsample_bytree=float(params["feature_fraction"]),
        subsample=float(params["bagging_fraction"]),
        subsample_freq=1,
        reg_alpha=float(params["lambda_l1"]),
        reg_lambda=float(params["lambda_l2"]),
        min_child_samples=int(params["min_child_samples"]),
        scale_pos_weight=spw,
        random_state=int(seed),
        n_jobs=int(params.get("n_jobs", -1)),
        verbosity=-1,
    )
    model.fit(
        X_tr,
        y_tr.astype(np.int32),
        eval_set=[(X_va, y_va.astype(np.int32))],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    pred = model.predict_proba(X_va)[:, 1]
    return _safe_ap(y_va, pred)


def run_optuna_lgbm_search(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_trials: int = 50,
    seed: int = 2026,
    n_jobs: int = -1,
    timeout_sec: int | None = None,
) -> dict[str, dict[str, Any]]:
    class_params: dict[str, dict[str, Any]] = {}

    for class_idx, class_name in enumerate(CLASSES):
        y_class = y[:, class_idx].astype(np.int32)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": 8000,
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 127),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.95),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "n_jobs": n_jobs,
            }
            fold_scores: list[float] = []
            for fold_id, (tr_idx, va_idx) in enumerate(folds):
                score = _train_eval_one_fold(
                    X_tr=X[tr_idx],
                    y_tr=y_class[tr_idx],
                    X_va=X[va_idx],
                    y_va=y_class[va_idx],
                    params=params,
                    seed=seed + fold_id,
                )
                fold_scores.append(score)
            return float(np.mean(fold_scores))

        sampler = optuna.samplers.TPESampler(seed=seed + class_idx)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=int(n_trials), timeout=timeout_sec, n_jobs=1, show_progress_bar=False)

        best = dict(study.best_params)
        best["n_estimators"] = 8000
        best["n_jobs"] = n_jobs
        class_params[class_name] = {
            "best_params": best,
            "best_value": float(study.best_value),
            "n_trials": int(len(study.trials)),
        }

    return class_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--cache-dir", type=str, default=str(CACHE_DIR))
    parser.add_argument("--output-path", type=str, default="bird_radar/artifacts/optuna_best_params.json")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cv-seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--timeout-sec", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    prepared = prepare_tabular_data(data_dir=args.data_dir, cache_dir=args.cache_dir)
    labels_idx: np.ndarray = prepared["labels_idx"]
    groups: np.ndarray = prepared["groups"]
    y_all: np.ndarray = prepared["y_all"]
    X_all = prepared["X_all"].to_numpy(dtype=np.float32, copy=False)

    folds = make_group_folds(
        labels_idx=labels_idx,
        groups=groups,
        n_splits=int(args.n_folds),
        random_state=int(args.cv_seed),
    )

    best_params = run_optuna_lgbm_search(
        X=X_all,
        y=y_all,
        folds=folds,
        n_trials=int(args.n_trials),
        seed=int(args.seed),
        n_jobs=int(args.n_jobs),
        timeout_sec=int(args.timeout_sec) if int(args.timeout_sec) > 0 else None,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=True, indent=2)

    for cls in CLASSES:
        info = best_params[cls]
        print(f"[optuna] {cls}: best_ap={info['best_value']:.6f} params={info['best_params']}", flush=True)


if __name__ == "__main__":
    main()
