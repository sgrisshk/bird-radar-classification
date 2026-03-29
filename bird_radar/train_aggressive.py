from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from blend_optimizer import apply_blend, optimize_blend_weights
from config import CACHE_DIR, CLASSES, DEFAULT_DATA_DIR
from optuna_search import run_optuna_lgbm_search
from src.calibration import apply_temperature_scaling, fit_temperature_scaling
from src.metrics import macro_map_score, one_hot_labels, per_class_average_precision
from stacking import fit_stacking_full_predict, run_stacking_cv
from train_lgbm import (
    fit_lgbm_multiseed_full_predict,
    make_group_folds,
    prepare_tabular_data,
    run_lgbm_multiseed_cv,
)

EPS = 1e-8


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _parse_seed_list(seed_str: str) -> list[int]:
    return [int(x.strip()) for x in seed_str.split(",") if x.strip()]


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    def _default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, default=_default)
    tmp.replace(path)


def _train_binary_catboost(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    seed: int,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Any]:
    pos = int(np.sum(y_tr))
    if pos == 0 or pos == len(y_tr):
        const = float(np.mean(y_tr))
        return np.full(len(X_va), const, dtype=np.float32), np.full(len(X_te), const, dtype=np.float32), {"constant": const}

    neg = max(len(y_tr) - pos, 1)
    spw = float(neg / max(pos, 1))

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        depth=int(params.get("depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.03)),
        l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
        random_seed=int(seed),
        iterations=int(params.get("iterations", 4000)),
        od_type="Iter",
        od_wait=int(params.get("od_wait", 300)),
        bootstrap_type="Bernoulli",
        subsample=float(params.get("subsample", 0.8)),
        scale_pos_weight=spw,
        task_type="CPU",
        thread_count=int(params.get("thread_count", -1)),
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        X_tr,
        y_tr.astype(np.int32),
        eval_set=(X_va, y_va.astype(np.int32)),
        use_best_model=True,
        verbose=False,
    )
    p_va = model.predict_proba(X_va)[:, 1].astype(np.float32)
    p_te = model.predict_proba(X_te)[:, 1].astype(np.float32)
    return p_va, p_te, model


def _train_binary_xgboost(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    seed: int,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Any]:
    pos = int(np.sum(y_tr))
    if pos == 0 or pos == len(y_tr):
        const = float(np.mean(y_tr))
        return np.full(len(X_va), const, dtype=np.float32), np.full(len(X_te), const, dtype=np.float32), {"constant": const}

    neg = max(len(y_tr) - pos, 1)
    spw = float(neg / max(pos, 1))

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=int(params.get("n_estimators", 4000)),
        learning_rate=float(params.get("learning_rate", 0.03)),
        max_depth=int(params.get("max_depth", 6)),
        min_child_weight=float(params.get("min_child_weight", 1.0)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        random_state=int(seed),
        n_jobs=int(params.get("n_jobs", -1)),
        tree_method="hist",
        early_stopping_rounds=int(params.get("early_stopping_rounds", 300)),
        scale_pos_weight=spw,
        verbosity=0,
        device="cpu",
    )
    model.fit(
        X_tr,
        y_tr.astype(np.int32),
        eval_set=[(X_va, y_va.astype(np.int32))],
        verbose=False,
    )
    p_va = model.predict_proba(X_va)[:, 1].astype(np.float32)
    p_te = model.predict_proba(X_te)[:, 1].astype(np.float32)
    return p_va, p_te, model


def _run_ovr_cv(
    model_type: str,
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    X_test: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    params: dict[str, Any],
    seed: int,
    use_temperature_scaling: bool = False,
) -> dict[str, Any]:
    X_np = X_all.to_numpy(dtype=np.float32, copy=False)
    X_te = X_test.to_numpy(dtype=np.float32, copy=False)
    n_classes = y_all.shape[1]

    oof_raw = np.zeros((X_np.shape[0], n_classes), dtype=np.float32)
    test_fold_preds: list[np.ndarray] = []
    fold_scores: list[dict[str, Any]] = []

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        X_tr = X_np[tr_idx]
        X_va = X_np[va_idx]
        y_tr = y_all[tr_idx]
        y_va = y_all[va_idx]

        fold_valid = np.zeros((len(va_idx), n_classes), dtype=np.float32)
        fold_test = np.zeros((X_te.shape[0], n_classes), dtype=np.float32)

        for class_idx in range(n_classes):
            if model_type == "catboost":
                p_va, p_te, _ = _train_binary_catboost(
                    X_tr=X_tr,
                    y_tr=y_tr[:, class_idx],
                    X_va=X_va,
                    y_va=y_va[:, class_idx],
                    X_te=X_te,
                    seed=seed + fold_id * 31 + class_idx,
                    params=params,
                )
            elif model_type == "xgboost":
                p_va, p_te, _ = _train_binary_xgboost(
                    X_tr=X_tr,
                    y_tr=y_tr[:, class_idx],
                    X_va=X_va,
                    y_va=y_va[:, class_idx],
                    X_te=X_te,
                    seed=seed + fold_id * 31 + class_idx,
                    params=params,
                )
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            fold_valid[:, class_idx] = p_va
            fold_test[:, class_idx] = p_te

        oof_raw[va_idx] = fold_valid
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

    test_raw = np.mean(np.stack(test_fold_preds, axis=0), axis=0).astype(np.float32)

    macro_raw = float(macro_map_score(y_all, oof_raw))
    if use_temperature_scaling:
        temps = fit_temperature_scaling(y_all, oof_raw)
        oof_cal = apply_temperature_scaling(oof_raw, temps)
        test_cal = apply_temperature_scaling(test_raw, temps)
        macro_cal = float(macro_map_score(y_all, oof_cal))
        selected_oof = oof_cal if macro_cal >= macro_raw else oof_raw
        selected_test = test_cal if macro_cal >= macro_raw else test_raw
        calibration = {
            "selected": "calibrated" if macro_cal >= macro_raw else "raw",
            "raw_macro_map": macro_raw,
            "calibrated_macro_map": macro_cal,
            "temperatures": [float(v) for v in temps.tolist()],
        }
    else:
        selected_oof = oof_raw
        selected_test = test_raw
        calibration = {
            "selected": "disabled",
            "raw_macro_map": macro_raw,
            "calibrated_macro_map": None,
            "temperatures": None,
        }

    return {
        "model_type": model_type,
        "oof_raw": oof_raw,
        "test_raw": test_raw,
        "oof_selected": selected_oof.astype(np.float32),
        "test_selected": selected_test.astype(np.float32),
        "macro_raw": macro_raw,
        "macro_selected": float(macro_map_score(y_all, selected_oof)),
        "per_class_selected": per_class_average_precision(y_all, selected_oof),
        "fold_scores": fold_scores,
        "fold_std": float(np.std([f["macro_map"] for f in fold_scores])) if fold_scores else 0.0,
        "calibration": calibration,
    }


def _fit_ovr_full_predict(
    model_type: str,
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    X_test: pd.DataFrame,
    seed: int,
    params: dict[str, Any],
) -> np.ndarray:
    X_np = X_all.to_numpy(dtype=np.float32, copy=False)
    X_te = X_test.to_numpy(dtype=np.float32, copy=False)
    n_classes = y_all.shape[1]

    out = np.zeros((X_te.shape[0], n_classes), dtype=np.float32)
    dummy_va_idx = np.arange(min(64, X_np.shape[0]))

    for class_idx in range(n_classes):
        y_class = y_all[:, class_idx]
        X_va = X_np[dummy_va_idx]
        y_va = y_class[dummy_va_idx]

        if model_type == "catboost":
            _, p_te, _ = _train_binary_catboost(
                X_tr=X_np,
                y_tr=y_class,
                X_va=X_va,
                y_va=y_va,
                X_te=X_te,
                seed=seed + class_idx,
                params=params,
            )
        elif model_type == "xgboost":
            _, p_te, _ = _train_binary_xgboost(
                X_tr=X_np,
                y_tr=y_class,
                X_va=X_va,
                y_va=y_va,
                X_te=X_te,
                seed=seed + class_idx,
                params=params,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        out[:, class_idx] = p_te

    return out


def _evaluate_fold_std(
    oof: np.ndarray,
    y_all: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, list[float]]:
    fold_values: list[float] = []
    for _, va_idx in folds:
        fold_values.append(float(macro_map_score(y_all[va_idx], oof[va_idx])))
    return float(np.std(fold_values)), fold_values


def _blend_with_pruning(
    candidates: dict[str, dict[str, Any]],
    y_all: np.ndarray,
) -> dict[str, Any]:
    active = sorted(candidates.keys())

    def _solve(names: list[str]) -> tuple[dict[str, float], np.ndarray, float]:
        oof_map = {k: candidates[k]["oof_selected"] for k in names}
        weights = optimize_blend_weights(y_true=y_all, oof_predictions=oof_map)
        blend = apply_blend(oof_map, weights)
        score = float(macro_map_score(y_all, blend))
        return weights, blend, score

    weights, blend_oof, best_score = _solve(active)
    improved = True
    while improved and len(active) > 1:
        improved = False
        best_variant = None
        for name in active:
            subset = [x for x in active if x != name]
            w_sub, oof_sub, s_sub = _solve(subset)
            if s_sub > best_score + 1e-7:
                if best_variant is None or s_sub > best_variant[3]:
                    best_variant = (subset, w_sub, oof_sub, s_sub)
        if best_variant is not None:
            active, weights, blend_oof, best_score = best_variant
            improved = True

    test_map = {k: candidates[k]["test_selected"] for k in active}
    blend_test = apply_blend(test_map, weights)

    return {
        "selected_models": active,
        "weights": {k: float(v) for k, v in weights.items() if k in active},
        "oof": blend_oof.astype(np.float32),
        "test": blend_test.astype(np.float32),
        "macro_map": float(macro_map_score(y_all, blend_oof)),
        "per_class_ap": per_class_average_precision(y_all, blend_oof),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--cache-dir", type=str, default=str(CACHE_DIR))
    parser.add_argument("--output-dir", type=str, default="bird_radar/artifacts/aggressive")

    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--lgbm-seeds", type=str, default="42,2026,777,1234,9999,31415,2718,8888")
    parser.add_argument("--skip-catboost", action="store_true")
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--skip-stacking", action="store_true")

    parser.add_argument("--run-optuna", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=50)
    parser.add_argument("--use-temperature-scaling", action="store_true", default=False)

    parser.add_argument("--min-improvement", type=float, default=0.0005)
    parser.add_argument("--full-retrain", action="store_true", default=False)
    parser.add_argument("--no-full-retrain", dest="full_retrain", action="store_false")

    parser.add_argument("--validate-script", type=str, default="validate_submission.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

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

    default_lgbm_params: dict[str, Any] = {
        "n_estimators": 6000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "lambda_l1": 0.05,
        "lambda_l2": 0.2,
        "min_child_samples": 20,
        "n_jobs": -1,
    }

    class_params_map: dict[str, dict[str, Any]] | None = None
    if args.run_optuna:
        class_params_info = run_optuna_lgbm_search(
            X=X_all.to_numpy(dtype=np.float32, copy=False),
            y=y_all,
            folds=folds,
            n_trials=int(args.optuna_trials),
            seed=int(args.seed),
            n_jobs=-1,
            timeout_sec=None,
        )
        class_params_map = {k: dict(v["best_params"]) for k, v in class_params_info.items()}
        _save_json(output_dir / "optuna_best_params.json", class_params_info)

    lgbm_seeds = _parse_seed_list(args.lgbm_seeds)
    lgbm_dir = models_dir / "lgbm_multiseed"
    lgbm_result = run_lgbm_multiseed_cv(
        X_all=X_all,
        y_all=y_all,
        X_test=X_test,
        folds=folds,
        seeds=lgbm_seeds,
        params=default_lgbm_params,
        use_focal=True,
        class_params_map=class_params_map,
        output_dir=lgbm_dir,
    )

    if args.use_temperature_scaling:
        lgbm_oof = lgbm_result["oof_selected_mean"]
        lgbm_test = lgbm_result["test_selected_mean"]
        lgbm_macro = float(lgbm_result["macro_selected"])
        lgbm_per_class = lgbm_result["per_class_selected"]
        lgbm_calibration = {"selected": "seedwise"}
    else:
        lgbm_oof = lgbm_result["oof_raw_mean"]
        lgbm_test = lgbm_result["test_raw_mean"]
        lgbm_macro = float(lgbm_result["macro_raw"])
        lgbm_per_class = per_class_average_precision(y_all, lgbm_oof)
        lgbm_calibration = {"selected": "disabled"}

    artifacts: dict[str, dict[str, Any]] = {}
    artifacts["lgbm_multiseed"] = {
        "model_type": "lgbm",
        "oof_raw": lgbm_result["oof_raw_mean"],
        "test_raw": lgbm_result["test_raw_mean"],
        "oof_selected": lgbm_oof,
        "test_selected": lgbm_test,
        "macro_selected": lgbm_macro,
        "per_class_ap": lgbm_per_class,
        "fold_std": float(lgbm_result["fold_std"]),
        "fold_scores": lgbm_result["fold_scores"],
        "calibration": lgbm_calibration,
        "meta": {"seeds": lgbm_seeds, "params": default_lgbm_params, "class_params": class_params_map},
    }

    if not args.skip_catboost:
        cat_params = {
            "iterations": 5000,
            "learning_rate": 0.025,
            "depth": 6,
            "l2_leaf_reg": 5.0,
            "subsample": 0.85,
            "od_wait": 300,
            "thread_count": -1,
        }
        cat_res = _run_ovr_cv(
            model_type="catboost",
            X_all=X_all,
            y_all=y_all,
            X_test=X_test,
            folds=folds,
            params=cat_params,
            seed=int(args.seed),
            use_temperature_scaling=bool(args.use_temperature_scaling),
        )
        artifacts["catboost_ovr"] = {
            "model_type": "catboost",
            **cat_res,
            "meta": {"params": cat_params},
        }

    if not args.skip_xgboost:
        xgb_params = {
            "n_estimators": 4500,
            "learning_rate": 0.02,
            "max_depth": 6,
            "min_child_weight": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "early_stopping_rounds": 300,
            "n_jobs": -1,
        }
        xgb_res = _run_ovr_cv(
            model_type="xgboost",
            X_all=X_all,
            y_all=y_all,
            X_test=X_test,
            folds=folds,
            params=xgb_params,
            seed=int(args.seed + 11),
            use_temperature_scaling=bool(args.use_temperature_scaling),
        )
        artifacts["xgboost_ovr"] = {
            "model_type": "xgboost",
            **xgb_res,
            "meta": {"params": xgb_params},
        }

    stable_artifacts: dict[str, dict[str, Any]] = {}
    for name, artifact in artifacts.items():
        fold_std = float(artifact.get("fold_std", 0.0))
        if fold_std <= 0.02:
            stable_artifacts[name] = artifact

    if not stable_artifacts:
        best_name = max(artifacts.keys(), key=lambda k: float(artifacts[k]["macro_selected"]))
        stable_artifacts[best_name] = artifacts[best_name]

    stacking_result: dict[str, Any] | None = None
    if not args.skip_stacking and len(stable_artifacts) >= 2:
        oof_map = {k: v["oof_selected"] for k, v in stable_artifacts.items()}
        test_map = {k: v["test_selected"] for k, v in stable_artifacts.items()}
        stacking_result = run_stacking_cv(
            oof_predictions=oof_map,
            test_predictions=test_map,
            labels_idx=labels_idx,
            groups=groups,
            n_splits=3,
            seed=int(args.cv_seed),
            learner="logreg",
        )
        best_base = max(float(v["macro_selected"]) for v in stable_artifacts.values())
        if float(stacking_result["macro_map"]) >= best_base - 1e-6:
            stable_artifacts["stacking_logreg"] = {
                "model_type": "stacking",
                "oof_raw": stacking_result["oof"],
                "test_raw": stacking_result["test"],
                "oof_selected": stacking_result["oof"],
                "test_selected": stacking_result["test"],
                "macro_selected": float(stacking_result["macro_map"]),
                "per_class_ap": stacking_result["per_class_ap"],
                "fold_std": float(stacking_result["fold_std"]),
                "fold_scores": stacking_result["fold_scores"],
                "calibration": {"selected": "raw"},
                "meta": {"base_models": stacking_result["model_names"], "learner": "logreg"},
            }

    blend_result = _blend_with_pruning(stable_artifacts, y_all)

    final_candidates = blend_result["selected_models"]
    final_weights = blend_result["weights"]

    full_test_predictions: dict[str, np.ndarray] = {}
    if args.full_retrain:
        needed_base = set(final_candidates)
        if "stacking_logreg" in final_candidates and stacking_result is not None:
            needed_base.update(stacking_result["model_names"])

        if "lgbm_multiseed" in needed_base:
            full_lgbm = fit_lgbm_multiseed_full_predict(
                X_all=X_all,
                y_all=y_all,
                X_test=X_test,
                seeds=lgbm_seeds,
                params=default_lgbm_params,
                use_focal=True,
                class_params_map=class_params_map,
                output_dir=models_dir / "lgbm_multiseed_full",
            )
            full_lgbm_test = full_lgbm["test_raw_mean"]
            full_test_predictions["lgbm_multiseed"] = full_lgbm_test.astype(np.float32)

        if "catboost_ovr" in needed_base and "catboost_ovr" in artifacts:
            cat_params = artifacts["catboost_ovr"]["meta"]["params"]
            full_cat = _fit_ovr_full_predict(
                model_type="catboost",
                X_all=X_all,
                y_all=y_all,
                X_test=X_test,
                seed=int(args.seed),
                params=cat_params,
            )
            full_test_predictions["catboost_ovr"] = full_cat.astype(np.float32)

        if "xgboost_ovr" in needed_base and "xgboost_ovr" in artifacts:
            xgb_params = artifacts["xgboost_ovr"]["meta"]["params"]
            full_xgb = _fit_ovr_full_predict(
                model_type="xgboost",
                X_all=X_all,
                y_all=y_all,
                X_test=X_test,
                seed=int(args.seed + 11),
                params=xgb_params,
            )
            full_test_predictions["xgboost_ovr"] = full_xgb.astype(np.float32)

        for name in list(full_test_predictions.keys()):
            calib = artifacts.get(name, {}).get("calibration", {})
            if calib.get("selected") == "calibrated":
                temps = np.asarray(calib.get("temperatures", [1.0] * len(CLASSES)), dtype=np.float32)
                full_test_predictions[name] = apply_temperature_scaling(full_test_predictions[name], temps).astype(np.float32)

        if "stacking_logreg" in final_candidates and stacking_result is not None:
            base_for_stacking = {k: stable_artifacts[k]["oof_selected"] for k in stacking_result["model_names"]}
            test_for_stacking = {k: full_test_predictions[k] for k in stacking_result["model_names"]}
            stack_full = fit_stacking_full_predict(
                oof_predictions=base_for_stacking,
                test_predictions=test_for_stacking,
                labels_idx=labels_idx,
                learner="logreg",
                seed=int(args.cv_seed),
            )
            full_test_predictions["stacking_logreg"] = stack_full.astype(np.float32)

        final_test = apply_blend({k: full_test_predictions[k] for k in final_candidates}, final_weights).astype(np.float32)
    else:
        final_test = blend_result["test"].astype(np.float32)

    submission = pd.DataFrame(final_test, columns=CLASSES)
    submission.insert(0, "track_id", test_df["track_id"].to_numpy())
    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    np.save(output_dir / "oof_blend.npy", blend_result["oof"])
    np.save(output_dir / "test_blend.npy", blend_result["test"])

    summary: dict[str, Any] = {
        "feature_count": len(feature_cols),
        "models": {},
        "stable_models": sorted(stable_artifacts.keys()),
        "blend": {
            "selected_models": final_candidates,
            "weights": final_weights,
            "macro_map": float(blend_result["macro_map"]),
            "per_class_ap": blend_result["per_class_ap"],
        },
        "stacking": (
            {
                "model_names": stacking_result.get("model_names", []),
                "macro_map": float(stacking_result.get("macro_map", 0.0)),
                "per_class_ap": stacking_result.get("per_class_ap", {}),
                "fold_scores": stacking_result.get("fold_scores", []),
                "fold_std": float(stacking_result.get("fold_std", 0.0)),
                "fold_mean": float(stacking_result.get("fold_mean", 0.0)),
            }
            if stacking_result is not None
            else None
        ),
        "submission_path": str(submission_path),
    }

    for name, artifact in artifacts.items():
        artifact_per_class = artifact.get("per_class_ap", artifact.get("per_class_selected", {}))
        summary["models"][name] = {
            "model_type": artifact["model_type"],
            "macro_map": float(artifact["macro_selected"]),
            "per_class_ap": artifact_per_class,
            "fold_std": float(artifact.get("fold_std", 0.0)),
            "fold_scores": artifact.get("fold_scores", []),
        }
        model_dir = models_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        np.save(model_dir / "oof_selected.npy", artifact["oof_selected"])
        np.save(model_dir / "test_selected.npy", artifact["test_selected"])
        _save_json(model_dir / "scores.json", summary["models"][name])

    _save_json(output_dir / "summary.json", summary)

    validate_script = Path(args.validate_script)
    if not validate_script.is_absolute():
        validate_script = Path.cwd() / validate_script
    if validate_script.exists():
        report_path = output_dir / "validation_report.txt"
        cmd = [
            sys.executable,
            str(validate_script),
            "--test-csv",
            str(Path(args.data_dir) / "test.csv"),
            "--submission-csv",
            str(submission_path),
            "--train-csv",
            str(Path(args.data_dir) / "train.csv"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        report_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
        summary["validation_report"] = str(report_path)
        summary["validation_returncode"] = int(proc.returncode)
        _save_json(output_dir / "summary.json", summary)

    print("Per-model macro mAP:", flush=True)
    for name, artifact in artifacts.items():
        print(f"{name}: {artifact['macro_selected']:.6f}", flush=True)

    print("Per-model per-class AP:", flush=True)
    for name, artifact in artifacts.items():
        artifact_per_class = artifact.get("per_class_ap", artifact.get("per_class_selected", {}))
        print(name, json.dumps(artifact_per_class, ensure_ascii=True), flush=True)

    print("Per-model fold std:", flush=True)
    for name, artifact in artifacts.items():
        print(f"{name}: {artifact.get('fold_std', 0.0):.6f}", flush=True)

    if stacking_result is not None:
        print(f"Stacked macro mAP: {float(stacking_result['macro_map']):.6f}", flush=True)
    else:
        print("Stacked macro mAP: skipped", flush=True)

    print(f"Ensemble macro mAP: {float(blend_result['macro_map']):.6f}", flush=True)
    print(f"Final blended macro mAP: {float(blend_result['macro_map']):.6f}", flush=True)


if __name__ == "__main__":
    main()
