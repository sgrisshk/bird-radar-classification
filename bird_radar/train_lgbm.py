from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from config import CACHE_DIR, CLASSES, CLASS_TO_INDEX, DEFAULT_DATA_DIR, LGBM_DIR, N_FOLDS, ensure_dirs
from focal_loss import build_class_adaptive_weights, make_focal_surrogate_objective
from src.calibration import apply_temperature_scaling, fit_temperature_scaling
from src.feature_engineering import build_feature_frame
from src.metrics import macro_map_score, one_hot_labels, per_class_average_precision, sigmoid_numpy
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache
from src.training import save_json

EPS = 1e-8


@dataclass
class FoldPrediction:
    fold: int
    macro_map: float
    per_class_map: dict[str, float]
    train_size: int
    valid_size: int


@dataclass
class SeedRunResult:
    seed: int
    oof_raw: np.ndarray
    test_raw: np.ndarray
    oof_selected: np.ndarray
    test_selected: np.ndarray
    calibration: dict[str, Any]
    fold_scores: list[FoldPrediction]
    fold_std: float
    macro_raw: float
    macro_selected: float
    per_class_selected: dict[str, float]


@dataclass
class MultiSeedResult:
    seeds: list[int]
    oof_raw_mean: np.ndarray
    test_raw_mean: np.ndarray
    oof_selected_mean: np.ndarray
    test_selected_mean: np.ndarray
    seed_results: list[SeedRunResult]
    fold_scores: list[FoldPrediction]
    macro_raw: float
    macro_selected: float
    per_class_selected: dict[str, float]
    fold_mean: float
    fold_std: float
    fold_spread: float


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _load_or_build_cache(df: pd.DataFrame, cache_path: Path) -> dict[int, dict[str, Any]]:
    if cache_path.exists():
        return load_track_cache(cache_path)
    cache = build_track_cache(df)
    save_track_cache(cache, cache_path)
    return cache


def parse_seed_list(seed_arg: str | None, fallback_seed: int) -> list[int]:
    if seed_arg is None or str(seed_arg).strip() == "":
        return [int(fallback_seed)]
    items = [s.strip() for s in str(seed_arg).split(",") if s.strip()]
    if not items:
        return [int(fallback_seed)]
    out = []
    for item in items:
        out.append(int(item))
    return out


def prepare_tabular_data(
    data_dir: str | Path,
    cache_dir: str | Path,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    if "bird_group" not in train_df.columns:
        raise RuntimeError("train.csv must contain bird_group")
    if "observation_id" not in train_df.columns:
        raise RuntimeError("train.csv must contain observation_id")

    labels_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y_all = one_hot_labels(labels_idx, len(CLASSES)).astype(np.float32)
    groups = train_df["observation_id"].to_numpy()

    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")

    feat_train = build_feature_frame(train_df, track_cache=train_cache)
    feat_test = build_feature_frame(test_df, track_cache=test_cache)

    drop_cols = ["track_id", "observation_id", "primary_observation_id"]
    feature_cols = [c for c in feat_train.columns if c not in drop_cols]

    X_all = feat_train[feature_cols].astype(np.float32)
    X_test = feat_test[feature_cols].astype(np.float32)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "labels_idx": labels_idx,
        "y_all": y_all,
        "groups": groups,
        "X_all": X_all,
        "X_test": X_test,
        "feature_cols": feature_cols,
    }


def make_group_folds(
    labels_idx: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(splitter.split(np.zeros((len(labels_idx), 1)), labels_idx, groups=groups))
    return [(np.asarray(tr), np.asarray(va)) for tr, va in splits]


def _average_precision_from_scores(y_true: np.ndarray, score: np.ndarray) -> float:
    if int(np.sum(y_true)) == 0:
        return 0.0
    try:
        return float(average_precision_score(y_true, score))
    except ValueError:
        return 0.0


def _train_binary_lgbm(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    X_te: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
    class_idx: int,
    use_focal: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Any]:
    y_tr = y_tr.astype(np.float32)
    y_va = y_va.astype(np.float32)

    adaptive = build_class_adaptive_weights(y_tr.astype(np.int32))
    spw = float(adaptive["scale_pos_weight"])
    gamma = float(adaptive["gamma"])

    n_estimators = int(params.get("n_estimators", 6000))
    learning_rate = float(params.get("learning_rate", 0.03))
    num_leaves = int(params.get("num_leaves", 63))
    feature_fraction = float(params.get("feature_fraction", 0.75))
    bagging_fraction = float(params.get("bagging_fraction", 0.8))
    lambda_l1 = float(params.get("lambda_l1", 0.0))
    lambda_l2 = float(params.get("lambda_l2", 0.0))
    min_child_samples = int(params.get("min_child_samples", 20))

    fallback_meta: dict[str, Any] = {
        "used_focal": False,
        "fallback": False,
        "scale_pos_weight": spw,
        "gamma": gamma,
    }

    if use_focal:
        try:
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)

            booster_params: dict[str, Any] = {
                "objective": "none",
                "metric": "None",
                "learning_rate": learning_rate,
                "num_leaves": num_leaves,
                "feature_fraction": feature_fraction,
                "bagging_fraction": bagging_fraction,
                "bagging_freq": 1,
                "lambda_l1": lambda_l1,
                "lambda_l2": lambda_l2,
                "min_child_samples": min_child_samples,
                "verbosity": -1,
                "seed": int(seed + class_idx),
                "feature_fraction_seed": int(seed + class_idx + 100),
                "bagging_seed": int(seed + class_idx + 200),
                "num_threads": int(params.get("n_jobs", -1)),
            }

            focal_obj = make_focal_surrogate_objective(alpha=0.25, gamma=gamma)

            def _feval_ap(preds: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
                y_local = dataset.get_label()
                p_local = sigmoid_numpy(preds.astype(np.float32))
                ap = _average_precision_from_scores(y_local, p_local)
                return "average_precision", ap, True

            booster = lgb.train(
                params=booster_params,
                train_set=dtrain,
                num_boost_round=n_estimators,
                valid_sets=[dvalid],
                fobj=focal_obj,
                feval=_feval_ap,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=300, first_metric_only=True, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            p_va = sigmoid_numpy(booster.predict(X_va, num_iteration=booster.best_iteration)).astype(np.float32)
            p_te = sigmoid_numpy(booster.predict(X_te, num_iteration=booster.best_iteration)).astype(np.float32)
            fallback_meta["used_focal"] = True
            return p_va, p_te, fallback_meta, booster
        except Exception as exc:
            fallback_meta["fallback"] = True
            fallback_meta["fallback_reason"] = f"{type(exc).__name__}: {exc}"

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="average_precision",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        colsample_bytree=feature_fraction,
        subsample=bagging_fraction,
        subsample_freq=1,
        reg_alpha=lambda_l1,
        reg_lambda=lambda_l2,
        min_child_samples=min_child_samples,
        scale_pos_weight=spw,
        random_state=int(seed + class_idx),
        n_jobs=int(params.get("n_jobs", -1)),
        verbosity=-1,
    )
    model.fit(
        X_tr,
        y_tr.astype(np.int32),
        eval_set=[(X_va, y_va.astype(np.int32))],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(stopping_rounds=300, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    p_va = model.predict_proba(X_va)[:, 1].astype(np.float32)
    p_te = model.predict_proba(X_te)[:, 1].astype(np.float32)
    return p_va, p_te, fallback_meta, model


def run_lgbm_seed_cv(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    X_test: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    params: dict[str, Any],
    seed: int,
    use_focal: bool = True,
    class_params_map: dict[str, dict[str, Any]] | None = None,
    save_fold_models_dir: str | Path | None = None,
) -> dict[str, Any]:
    n_samples = X_all.shape[0]
    n_test = X_test.shape[0]
    n_classes = y_all.shape[1]

    oof_raw = np.zeros((n_samples, n_classes), dtype=np.float32)
    test_fold_preds: list[np.ndarray] = []
    fold_scores: list[FoldPrediction] = []
    class_debug: dict[str, list[dict[str, Any]]] = {c: [] for c in CLASSES}

    if save_fold_models_dir is not None:
        save_fold_models_dir = Path(save_fold_models_dir)
        save_fold_models_dir.mkdir(parents=True, exist_ok=True)

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        X_tr = X_all.iloc[tr_idx]
        X_va = X_all.iloc[va_idx]

        fold_valid = np.zeros((len(va_idx), n_classes), dtype=np.float32)
        fold_test = np.zeros((n_test, n_classes), dtype=np.float32)
        fold_models: list[Any] = []

        for class_idx, class_name in enumerate(CLASSES):
            y_tr = y_all[tr_idx, class_idx]
            y_va = y_all[va_idx, class_idx]
            effective_params = dict(params)
            if class_params_map is not None and class_name in class_params_map:
                effective_params.update(class_params_map[class_name])

            p_va, p_te, meta, fitted_model = _train_binary_lgbm(
                X_tr=X_tr,
                y_tr=y_tr,
                X_va=X_va,
                y_va=y_va,
                X_te=X_test,
                params=effective_params,
                seed=seed + fold_id * 17,
                class_idx=class_idx,
                use_focal=use_focal,
            )
            fold_valid[:, class_idx] = p_va
            fold_test[:, class_idx] = p_te
            fold_models.append(fitted_model)
            class_debug[class_name].append(meta)

        oof_raw[va_idx] = fold_valid
        test_fold_preds.append(fold_test)

        fold_macro = float(macro_map_score(y_all[va_idx], fold_valid))
        fold_per_class = per_class_average_precision(y_all[va_idx], fold_valid)
        fold_scores.append(
            FoldPrediction(
                fold=fold_id,
                macro_map=fold_macro,
                per_class_map=fold_per_class,
                train_size=int(len(tr_idx)),
                valid_size=int(len(va_idx)),
            )
        )

        if save_fold_models_dir is not None:
            payload = {
                "seed": int(seed),
                "fold": int(fold_id),
                "models": fold_models,
                "classes": CLASSES,
            }
            with (save_fold_models_dir / f"lgbm_seed_{seed}_fold_{fold_id}.pkl").open("wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    test_raw = np.mean(np.stack(test_fold_preds, axis=0), axis=0).astype(np.float32)
    macro_raw = float(macro_map_score(y_all, oof_raw))

    temps = fit_temperature_scaling(y_all, oof_raw)
    oof_cal = apply_temperature_scaling(oof_raw, temps)
    test_cal = apply_temperature_scaling(test_raw, temps)

    macro_cal = float(macro_map_score(y_all, oof_cal))
    selected_oof = oof_cal if macro_cal >= macro_raw else oof_raw
    selected_test = test_cal if macro_cal >= macro_raw else test_raw

    fold_macros = [f.macro_map for f in fold_scores]
    fold_std = float(np.std(fold_macros)) if fold_macros else 0.0

    calibration = {
        "selected": "calibrated" if macro_cal >= macro_raw else "raw",
        "raw_macro_map": macro_raw,
        "calibrated_macro_map": macro_cal,
        "selected_macro_map": float(macro_cal if macro_cal >= macro_raw else macro_raw),
        "temperatures": [float(v) for v in temps.tolist()],
    }

    return {
        "seed": int(seed),
        "oof_raw": oof_raw,
        "test_raw": test_raw,
        "oof_selected": selected_oof.astype(np.float32),
        "test_selected": selected_test.astype(np.float32),
        "macro_raw": macro_raw,
        "macro_selected": float(macro_map_score(y_all, selected_oof)),
        "per_class_selected": per_class_average_precision(y_all, selected_oof),
        "fold_scores": [f.__dict__ for f in fold_scores],
        "fold_std": fold_std,
        "calibration": calibration,
        "class_debug": class_debug,
    }


def run_lgbm_multiseed_cv(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    X_test: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seeds: list[int],
    params: dict[str, Any],
    use_focal: bool = True,
    class_params_map: dict[str, dict[str, Any]] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    seed_runs: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = output_path / f"seed_{seed}" if output_path is not None else None
        if seed_dir is not None:
            seed_dir.mkdir(parents=True, exist_ok=True)
        result = run_lgbm_seed_cv(
            X_all=X_all,
            y_all=y_all,
            X_test=X_test,
            folds=folds,
            params=params,
            seed=int(seed),
            use_focal=use_focal,
            class_params_map=class_params_map,
            save_fold_models_dir=seed_dir,
        )
        if seed_dir is not None:
            np.save(seed_dir / "oof_raw.npy", result["oof_raw"])
            np.save(seed_dir / "test_raw.npy", result["test_raw"])
            np.save(seed_dir / "oof_selected.npy", result["oof_selected"])
            np.save(seed_dir / "test_selected.npy", result["test_selected"])
            save_json(seed_dir / "seed_scores.json", {
                "seed": int(seed),
                "macro_raw": float(result["macro_raw"]),
                "macro_selected": float(result["macro_selected"]),
                "fold_std": float(result["fold_std"]),
                "folds": result["fold_scores"],
                "per_class_selected": result["per_class_selected"],
                "calibration": result["calibration"],
                "class_debug": result["class_debug"],
            })
        seed_runs.append(result)

    oof_raw_mean = np.mean(np.stack([r["oof_raw"] for r in seed_runs], axis=0), axis=0).astype(np.float32)
    test_raw_mean = np.mean(np.stack([r["test_raw"] for r in seed_runs], axis=0), axis=0).astype(np.float32)
    oof_sel_mean = np.mean(np.stack([r["oof_selected"] for r in seed_runs], axis=0), axis=0).astype(np.float32)
    test_sel_mean = np.mean(np.stack([r["test_selected"] for r in seed_runs], axis=0), axis=0).astype(np.float32)

    fold_logs: list[FoldPrediction] = []
    for fold_id, (_, va_idx) in enumerate(folds):
        fold_macro = float(macro_map_score(y_all[va_idx], oof_sel_mean[va_idx]))
        fold_per_class = per_class_average_precision(y_all[va_idx], oof_sel_mean[va_idx])
        fold_logs.append(
            FoldPrediction(
                fold=fold_id,
                macro_map=fold_macro,
                per_class_map=fold_per_class,
                train_size=int(len(y_all) - len(va_idx)),
                valid_size=int(len(va_idx)),
            )
        )

    fold_values = [f.macro_map for f in fold_logs]
    fold_mean = float(np.mean(fold_values)) if fold_values else 0.0
    fold_std = float(np.std(fold_values)) if fold_values else 0.0
    fold_spread = float(np.max(fold_values) - np.min(fold_values)) if fold_values else 0.0

    return {
        "seeds": [int(s) for s in seeds],
        "seed_runs": seed_runs,
        "oof_raw_mean": oof_raw_mean,
        "test_raw_mean": test_raw_mean,
        "oof_selected_mean": oof_sel_mean,
        "test_selected_mean": test_sel_mean,
        "macro_raw": float(macro_map_score(y_all, oof_raw_mean)),
        "macro_selected": float(macro_map_score(y_all, oof_sel_mean)),
        "per_class_selected": per_class_average_precision(y_all, oof_sel_mean),
        "fold_scores": [f.__dict__ for f in fold_logs],
        "fold_mean": fold_mean,
        "fold_std": fold_std,
        "fold_spread": fold_spread,
    }


def fit_lgbm_multiseed_full_predict(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    X_test: pd.DataFrame,
    seeds: list[int],
    params: dict[str, Any],
    use_focal: bool,
    class_params_map: dict[str, dict[str, Any]] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    n_classes = y_all.shape[1]

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    seed_test_preds: list[np.ndarray] = []
    seed_models: list[list[Any]] = []

    for seed in seeds:
        class_models: list[Any] = []
        class_test_pred = np.zeros((X_test.shape[0], n_classes), dtype=np.float32)
        for class_idx in range(n_classes):
            class_name = CLASSES[class_idx]
            effective_params = dict(params)
            if class_params_map is not None and class_name in class_params_map:
                effective_params.update(class_params_map[class_name])
            y_tr = y_all[:, class_idx]
            dummy_valid_idx = np.arange(min(64, len(X_all)))
            X_va = X_all.iloc[dummy_valid_idx]
            y_va = y_tr[dummy_valid_idx]
            p_va, p_te, _, model_obj = _train_binary_lgbm(
                X_tr=X_all,
                y_tr=y_tr,
                X_va=X_va,
                y_va=y_va,
                X_te=X_test,
                params=effective_params,
                seed=int(seed),
                class_idx=class_idx,
                use_focal=use_focal,
            )
            _ = p_va
            class_test_pred[:, class_idx] = p_te
            class_models.append(model_obj)
        seed_test_preds.append(class_test_pred)
        seed_models.append(class_models)

    test_mean = np.mean(np.stack(seed_test_preds, axis=0), axis=0).astype(np.float32)

    if output_path is not None:
        np.save(output_path / "test_full_raw_mean.npy", test_mean)
        with (output_path / "full_models.pkl").open("wb") as f:
            pickle.dump({"seeds": seeds, "models": seed_models}, f, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "test_raw_mean": test_mean,
        "seeds": [int(s) for s in seeds],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(LGBM_DIR))
    parser.add_argument("--cache-dir", type=str, default=str(CACHE_DIR))

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--cv-seed", type=int, default=2026)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)

    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--feature-fraction", type=float, default=0.75)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--n-estimators", type=int, default=6000)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument("--use-focal", action="store_true", default=True)
    parser.add_argument("--no-focal", dest="use_focal", action="store_false")
    parser.add_argument("--full-retrain", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_global_seed(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_tabular_data(data_dir=args.data_dir, cache_dir=args.cache_dir)
    train_df: pd.DataFrame = prepared["train_df"]
    test_df: pd.DataFrame = prepared["test_df"]
    labels_idx: np.ndarray = prepared["labels_idx"]
    y_all: np.ndarray = prepared["y_all"]
    groups: np.ndarray = prepared["groups"]
    X_all: pd.DataFrame = prepared["X_all"]
    X_test: pd.DataFrame = prepared["X_test"]
    feature_cols: list[str] = prepared["feature_cols"]

    folds = make_group_folds(
        labels_idx=labels_idx,
        groups=groups,
        n_splits=int(args.n_folds),
        random_state=int(args.cv_seed),
    )

    seeds = parse_seed_list(args.seeds, args.seed)
    params = {
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "num_leaves": int(args.num_leaves),
        "feature_fraction": float(args.feature_fraction),
        "bagging_fraction": float(args.bagging_fraction),
        "lambda_l1": float(args.lambda_l1),
        "lambda_l2": float(args.lambda_l2),
        "min_child_samples": int(args.min_child_samples),
        "n_jobs": int(args.n_jobs),
    }

    result = run_lgbm_multiseed_cv(
        X_all=X_all,
        y_all=y_all,
        X_test=X_test,
        folds=folds,
        seeds=seeds,
        params=params,
        use_focal=bool(args.use_focal),
        output_dir=output_dir,
    )

    if len(seeds) == 1:
        seed_dir = output_dir / f"seed_{seeds[0]}"
        for fold_id in range(len(folds)):
            src = seed_dir / f"lgbm_seed_{seeds[0]}_fold_{fold_id}.pkl"
            dst = output_dir / f"lgbm_fold_{fold_id}.pkl"
            if src.exists():
                dst.write_bytes(src.read_bytes())

    oof_probs = result["oof_raw_mean"].astype(np.float32)
    oof_calibrated = result["oof_selected_mean"].astype(np.float32)
    test_probs = result["test_raw_mean"].astype(np.float32)
    test_calibrated = result["test_selected_mean"].astype(np.float32)
    temps = fit_temperature_scaling(y_all, oof_probs).astype(np.float32)

    np.save(output_dir / "oof_probs.npy", oof_probs)
    np.save(output_dir / "oof_calibrated.npy", oof_calibrated)
    np.save(output_dir / "test_probs.npy", test_probs)
    np.save(output_dir / "test_calibrated.npy", test_calibrated)
    np.save(output_dir / "temperatures.npy", temps)

    if args.full_retrain:
        full_dir = output_dir / "full_retrain"
        full_result = fit_lgbm_multiseed_full_predict(
            X_all=X_all,
            y_all=y_all,
            X_test=X_test,
            seeds=seeds,
            params=params,
            use_focal=bool(args.use_focal),
            output_dir=full_dir,
        )
        np.save(output_dir / "test_full_retrain_raw.npy", full_result["test_raw_mean"])

    fold_scores = [float(f["macro_map"]) for f in result["fold_scores"]]
    fold_mean = float(np.mean(fold_scores)) if fold_scores else 0.0
    fold_std = float(np.std(fold_scores)) if fold_scores else 0.0
    fold_spread = float(np.max(fold_scores) - np.min(fold_scores)) if fold_scores else 0.0

    save_json(
        output_dir / "scores.json",
        {
            "splitter": "StratifiedGroupKFold",
            "group_column": "observation_id",
            "seeds": seeds,
            "hyperparams": params,
            "feature_columns": feature_cols,
            "oof_macro_map_raw": float(macro_map_score(y_all, oof_probs)),
            "oof_macro_map_calibrated": float(macro_map_score(y_all, oof_calibrated)),
            "fold_macro_map_mean": fold_mean,
            "fold_macro_map_std": fold_std,
            "fold_macro_map_spread": fold_spread,
            "fold_macro_map_spread_lt_0_02": bool(fold_spread < 0.02),
            "per_class_map_calibrated": per_class_average_precision(y_all, oof_calibrated),
            "folds": result["fold_scores"],
            "seed_runs": [
                {
                    "seed": r["seed"],
                    "macro_raw": r["macro_raw"],
                    "macro_selected": r["macro_selected"],
                    "fold_std": r["fold_std"],
                    "calibration": r["calibration"],
                }
                for r in result["seed_runs"]
            ],
        },
    )
    save_json(output_dir / "feature_columns.json", {"feature_columns": feature_cols})

    submission = pd.DataFrame(test_calibrated, columns=CLASSES)
    submission.insert(0, "track_id", test_df["track_id"].to_numpy())
    submission.to_csv(output_dir / "submission_lgbm.csv", index=False)

    print(f"[lgbm] seeds={seeds} oof_macro={macro_map_score(y_all, oof_calibrated):.6f}", flush=True)
    print(f"[lgbm] fold_mean={fold_mean:.6f} fold_std={fold_std:.6f} fold_spread={fold_spread:.6f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
