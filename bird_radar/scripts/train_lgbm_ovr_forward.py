#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
try:
    from imblearn.over_sampling import BorderlineSMOTE
except Exception:  # pragma: no cover
    BorderlineSMOTE = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds
from src.interaction_stability import month_stable_interactions, remap_interaction_constraints
from src.monotone_analysis import find_monotone_candidates
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache
from src.redesign.features import build_tabular_frame, get_feature_columns
from src.redesign.utils import dump_json, macro_map, per_class_ap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LightGBM OVR model on tabular temporal features (forward CV).")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--sample-submission", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default="bird_radar/artifacts/cache")

    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--start-fold", type=int, default=0, help="Skip earliest fold indices before training/averaging.")
    p.add_argument("--max-folds", type=int, default=0, help="0 means use all folds")
    p.add_argument(
        "--train-mask-npy",
        type=str,
        default="",
        help="Optional bool/int mask (len=train) to restrict training rows only; validation/out remain full folds.",
    )
    p.add_argument(
        "--refill-after-mask",
        type=str,
        default="none",
        choices=["none", "class_preserve"],
        help="Optional train-only refill after masking: duplicate clean rows per class to compensate dropped rows.",
    )
    p.add_argument(
        "--refill-ratio",
        type=float,
        default=1.0,
        help="How much class mass to refill after mask (1.0 = refill dropped count per class).",
    )
    p.add_argument(
        "--focus-fold",
        type=int,
        default=-1,
        help="If >=0, train fold-specialist tuned on this fold; OOF for each fold is predicted by models validated on focus fold.",
    )
    p.add_argument("--time-col", default="timestamp_start_radar_utc")
    p.add_argument("--group-col", default="observation_id")
    p.add_argument(
        "--min-class-count",
        type=int,
        default=1,
        help="Per fold/class minimum positives and negatives required for training.",
    )
    p.add_argument(
        "--invalid-class-policy",
        type=str,
        default="fallback_prior",
        choices=["fallback_prior", "fallback_teacher", "skip_class", "skip_fold"],
        help="How to handle fold/classes with insufficient class counts.",
    )
    p.add_argument("--teacher-oof-csv", type=str, default="", help="Required for invalid-class-policy=fallback_teacher.")
    p.add_argument("--teacher-test-csv", type=str, default="", help="Required for invalid-class-policy=fallback_teacher.")
    p.add_argument(
        "--teacher-meta-oof-csv",
        type=str,
        default="",
        help="Optional: append teacher probabilities/meta features to tabular inputs (train OOF).",
    )
    p.add_argument(
        "--teacher-meta-test-csv",
        type=str,
        default="",
        help="Optional: append teacher probabilities/meta features to tabular inputs (test submission).",
    )

    p.add_argument("--num-leaves", type=int, default=127)
    p.add_argument(
        "--boosting-type",
        type=str,
        default="gbdt",
        choices=["gbdt", "dart"],
        help="LightGBM boosting type for OvR heads.",
    )
    p.add_argument(
        "--drop-rate",
        type=float,
        default=0.1,
        help="DART drop_rate (used only when --boosting-type=dart).",
    )
    p.add_argument(
        "--skip-drop",
        type=float,
        default=0.5,
        help="DART skip_drop (used only when --boosting-type=dart).",
    )
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--feature-fraction", type=float, default=0.8)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--bagging-freq", type=int, default=1)
    p.add_argument("--min-data-in-leaf", type=int, default=200)
    p.add_argument("--lambda-l2", type=float, default=1.0)
    p.add_argument("--max-depth", type=int, default=-1)
    p.add_argument(
        "--extra-trees",
        action="store_true",
        help="Enable LightGBM extra_trees mode (random split thresholds) for all OvR heads.",
    )
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--early-stopping-rounds", type=int, default=200)
    p.add_argument("--num-boost-round", type=int, default=5000)
    p.add_argument("--feature-drop-pct", type=float, default=0.0, help="Optional top feature drop by adversarial importances (disabled by default).")
    p.add_argument(
        "--class-feature-mask-json",
        type=str,
        default="",
        help=(
            "Optional JSON with per-class feature masks. "
            "Format: {\"ClassName\": {\"drop\": [patterns...], \"keep\": [patterns...]}}; "
            "patterns support '*' wildcards and are matched against final feature columns."
        ),
    )
    p.add_argument(
        "--hardneg-bank-json",
        type=str,
        default="",
        help="Optional JSON with per-class hard-negative track_id bank (from build_hardneg_bank.py).",
    )
    p.add_argument(
        "--hardneg-repeat",
        type=int,
        default=0,
        help="How many times to duplicate class-specific hard negatives inside each train fold.",
    )
    p.add_argument(
        "--hardpos-bank-json",
        type=str,
        default="",
        help="Optional JSON with per-class hard-positive track_id bank (from build_hardpos_bank.py).",
    )
    p.add_argument(
        "--hardpos-repeat",
        type=int,
        default=0,
        help="How many times to duplicate class-specific hard positives inside each train fold.",
    )
    p.add_argument(
        "--time-weight-mode",
        type=str,
        default="none",
        choices=["none", "fold0_boost", "linear_time"],
        help="Optional temporal sample weighting mode.",
    )
    p.add_argument(
        "--fold0-boost",
        type=float,
        default=0.7,
        help="Additional weight for fold0 validation block samples when time-weight-mode=fold0_boost.",
    )
    p.add_argument(
        "--time-weight-strength",
        type=float,
        default=0.6,
        help="Strength for linear-time weighting when time-weight-mode=linear_time.",
    )
    p.add_argument(
        "--smote-mode",
        type=str,
        default="none",
        choices=["none", "borderline"],
        help="Optional fold-safe oversampling for selected classes on train folds only.",
    )
    p.add_argument(
        "--smote-classes",
        type=str,
        default="Cormorants,Ducks,Waders",
        help="Comma-separated class names to apply SMOTE to when smote-mode!=none.",
    )
    p.add_argument("--smote-k-neighbors", type=int, default=5)
    p.add_argument("--smote-m-neighbors", type=int, default=10)
    p.add_argument(
        "--smote-sampling-strategy",
        type=float,
        default=0.35,
        help="Target minority/majority ratio for BorderlineSMOTE (0,1].",
    )
    p.add_argument(
        "--interaction-top-k",
        type=int,
        default=0,
        help="Enable month-stable interaction constraints and keep top-K stable pairs; 0 disables.",
    )
    p.add_argument(
        "--interaction-min-month-samples",
        type=int,
        default=15,
        help="Minimum rows per month required to score a pair for interaction stability.",
    )
    p.add_argument(
        "--monotone-mode",
        type=str,
        default="none",
        choices=["none", "strict", "loose"],
        help="Auto-build monotone constraints for rare classes using month-stable Spearman signal.",
    )
    p.add_argument(
        "--monotone-classes",
        type=str,
        default="Cormorants,Ducks,Waders",
        help="Comma-separated classes to constrain in monotone-mode.",
    )
    p.add_argument(
        "--monotone-top-k",
        type=int,
        default=3,
        help="Maximum monotone features per constrained class.",
    )
    return p.parse_args()


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict[str, Any]]:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


def _one_hot_targets(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.zeros((len(y_idx), n_classes), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    return y


def _macro_map_subset(y_true: np.ndarray, y_prob: np.ndarray, class_mask: np.ndarray) -> float:
    idx = np.where(class_mask)[0]
    if len(idx) == 0:
        return 0.0
    vals: list[float] = []
    from sklearn.metrics import average_precision_score

    for i in idx:
        yt = y_true[:, i]
        yp = y_prob[:, i]
        vals.append(float(average_precision_score(yt, yp)) if yt.sum() > 0 else 0.0)
    return float(np.mean(vals))


def _align_probs_from_csv(csv_path: str, ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=["track_id", *CLASSES])
    mp = {int(r["track_id"]): np.asarray([r[c] for c in CLASSES], dtype=np.float32) for _, r in df.iterrows()}
    missing = [int(t) for t in ids if int(t) not in mp]
    if missing:
        raise ValueError(f"{csv_path} missing {len(missing)} ids; first={missing[:10]}")
    return np.stack([mp[int(t)] for t in ids], axis=0).astype(np.float32)


def _teacher_meta_matrix(probs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    p = np.clip(probs.astype(np.float32), 1e-6, 1.0 - 1e-6)
    logits = np.log(p / (1.0 - p))
    sort_p = np.sort(p, axis=1)
    top1 = sort_p[:, -1]
    top2 = sort_p[:, -2]
    margin = top1 - top2
    entropy = -np.sum(p * np.log(p), axis=1)
    top1_cls = np.argmax(p, axis=1).astype(np.float32)
    cols = ["teacher_top1_prob", "teacher_top2_prob", "teacher_margin", "teacher_entropy", "teacher_top1_class_id"]
    mats = [top1, top2, margin, entropy, top1_cls]
    for j, cls in enumerate(CLASSES):
        cols.extend([f"teacher_p_{cls}", f"teacher_logit_{cls}"])
        mats.extend([p[:, j], logits[:, j]])
    return np.column_stack(mats).astype(np.float32), cols


def _adversarial_drop_lgbm(
    x_train: np.ndarray,
    x_test: np.ndarray,
    feature_cols: list[str],
    seed: int,
    pct_drop: float,
) -> tuple[list[str], dict[str, Any]]:
    if pct_drop <= 0.0:
        return feature_cols, {"enabled": False, "drop_count": 0, "dropped": [], "top20": []}

    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate(
        [
            np.zeros((len(x_train),), dtype=np.int32),
            np.ones((len(x_test),), dtype=np.int32),
        ]
    )

    ds = lgb.Dataset(x, label=y, free_raw_data=False)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 100,
        "lambda_l2": 1.0,
        "seed": int(seed),
        "verbosity": -1,
    }
    booster = lgb.train(
        params=params,
        train_set=ds,
        num_boost_round=400,
        valid_sets=[ds],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    imp = booster.feature_importance(importance_type="gain")
    order = np.argsort(-imp)
    drop_count = int(max(0, min(len(feature_cols) - 1, round(len(feature_cols) * pct_drop))))
    dropped = [feature_cols[i] for i in order[:drop_count]]
    keep = [c for c in feature_cols if c not in set(dropped)]
    top20 = [{"feature": feature_cols[i], "importance": float(imp[i])} for i in order[:20]]
    return keep, {
        "enabled": True,
        "drop_count": int(drop_count),
        "dropped": dropped,
        "top20": top20,
    }


def _select_feature_idx(
    feature_cols: list[str],
    keep_patterns: list[str] | None,
    drop_patterns: list[str] | None,
) -> tuple[np.ndarray, list[str], list[str]]:
    names = list(feature_cols)
    keep_mask = np.ones((len(names),), dtype=bool)
    used_keep: list[str] = []
    used_drop: list[str] = []

    if keep_patterns:
        keep_mask[:] = False
        for p in keep_patterns:
            matched = False
            for i, n in enumerate(names):
                if fnmatch.fnmatch(n, str(p)):
                    keep_mask[i] = True
                    matched = True
            if matched:
                used_keep.append(str(p))

    if drop_patterns:
        for p in drop_patterns:
            matched = False
            for i, n in enumerate(names):
                if fnmatch.fnmatch(n, str(p)):
                    keep_mask[i] = False
                    matched = True
            if matched:
                used_drop.append(str(p))

    idx = np.where(keep_mask)[0].astype(np.int64)
    if len(idx) <= 0:
        idx = np.arange(len(names), dtype=np.int64)
    return idx, used_keep, used_drop


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = out_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    sample_sub = pd.read_csv(args.sample_submission)

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")

    tab_train = build_tabular_frame(train_df, train_cache)
    tab_test = build_tabular_frame(test_df, test_cache)
    feature_cols = get_feature_columns(tab_train)

    x_train_raw = tab_train[feature_cols].to_numpy(dtype=np.float32)
    x_test_raw = tab_test[feature_cols].to_numpy(dtype=np.float32)
    keep_cols, adv_report = _adversarial_drop_lgbm(
        x_train=x_train_raw,
        x_test=x_test_raw,
        feature_cols=feature_cols,
        seed=int(args.seed),
        pct_drop=float(args.feature_drop_pct),
    )

    train_ids = train_df["track_id"].to_numpy(dtype=np.int64)
    test_ids = test_df["track_id"].to_numpy(dtype=np.int64)
    np.save(out_dir / "train_track_ids.npy", train_ids)
    np.save(out_dir / "test_track_ids.npy", test_ids)

    x_train = tab_train[keep_cols].to_numpy(dtype=np.float32)
    x_test = tab_test[keep_cols].to_numpy(dtype=np.float32)
    model_feature_cols = list(keep_cols)
    interaction_constraints_global: list[list[int]] | None = None
    interaction_info: dict[str, Any] = {"enabled": False}
    monotone_info: dict[str, Any] = {"enabled": False}
    monotone_constraints_map: dict[int, dict[str, int]] = {}

    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = _one_hot_targets(y_idx, n_classes=len(CLASSES))
    global_priors = np.clip(y.mean(axis=0), 1e-5, 1.0 - 1e-5).astype(np.float32)
    np.save(out_dir / "oof_targets.npy", y)
    train_id_to_row = {int(t): int(i) for i, t in enumerate(train_ids)}

    teacher_oof: np.ndarray | None = None
    teacher_test: np.ndarray | None = None
    if str(args.invalid_class_policy) in {"fallback_teacher", "skip_class"}:
        if not str(args.teacher_oof_csv).strip() or not str(args.teacher_test_csv).strip():
            raise ValueError("fallback_teacher/skip_class requires --teacher-oof-csv and --teacher-test-csv")
        teacher_oof = _align_probs_from_csv(str(args.teacher_oof_csv), train_ids)
        teacher_test = _align_probs_from_csv(str(args.teacher_test_csv), test_ids)

    teacher_meta_info: dict[str, Any] = {"enabled": False}
    if str(args.teacher_meta_oof_csv).strip() or str(args.teacher_meta_test_csv).strip():
        if not str(args.teacher_meta_oof_csv).strip() or not str(args.teacher_meta_test_csv).strip():
            raise ValueError("teacher meta requires both --teacher-meta-oof-csv and --teacher-meta-test-csv")
        teacher_meta_oof = _align_probs_from_csv(str(args.teacher_meta_oof_csv), train_ids)
        teacher_meta_test = _align_probs_from_csv(str(args.teacher_meta_test_csv), test_ids)
        x_meta_train, meta_cols = _teacher_meta_matrix(teacher_meta_oof)
        x_meta_test, _ = _teacher_meta_matrix(teacher_meta_test)
        x_train = np.concatenate([x_train, x_meta_train], axis=1).astype(np.float32)
        x_test = np.concatenate([x_test, x_meta_test], axis=1).astype(np.float32)
        model_feature_cols.extend(meta_cols)
        teacher_meta_info = {
            "enabled": True,
            "n_meta_features": int(len(meta_cols)),
            "meta_feature_columns": meta_cols,
            "source_oof": str(Path(args.teacher_meta_oof_csv).resolve()),
            "source_test": str(Path(args.teacher_meta_test_csv).resolve()),
        }

    if int(args.interaction_top_k) > 0:
        ts_month = pd.to_datetime(train_df[str(args.time_col)], errors="coerce", utc=True).dt.month
        month_frame = pd.DataFrame(x_train, columns=model_feature_cols)
        month_frame["month"] = ts_month.fillna(0).astype(int).to_numpy()
        interaction_constraints_global, interaction_info = month_stable_interactions(
            frame=month_frame,
            feature_cols=model_feature_cols,
            month_col="month",
            top_k=int(args.interaction_top_k),
            min_month_samples=int(args.interaction_min_month_samples),
        )
        print(
            "[lgbm] interaction-constraints enabled: "
            f"top_k={int(args.interaction_top_k)} "
            f"pairs_selected={int(interaction_info.get('n_pairs_selected', 0))} "
            f"constraints={int(interaction_info.get('n_constraints', 0))}",
            flush=True,
        )

    if str(args.monotone_mode) != "none":
        ts_month = pd.to_datetime(train_df[str(args.time_col)], errors="coerce", utc=True).dt.month.fillna(0).astype(int).to_numpy()
        mono_classes = [x.strip() for x in str(args.monotone_classes).split(",") if x.strip()]
        if str(args.monotone_mode) == "strict":
            min_spearman = 0.20
            max_flips = 0
        else:
            min_spearman = 0.15
            max_flips = 1
        monotone_constraints_map, monotone_info = find_monotone_candidates(
            labels_idx=y_idx,
            months=ts_month,
            feature_frame=pd.DataFrame(x_train, columns=model_feature_cols),
            class_names=mono_classes,
            min_spearman=float(min_spearman),
            max_cross_month_sign_flip=int(max_flips),
            per_class_top_k=max(1, int(args.monotone_top_k)),
            month_min_samples=10,
        )
        n_total = int(sum(len(v) for v in monotone_constraints_map.values()))
        print(
            f"[lgbm] monotone enabled mode={args.monotone_mode} total_features={n_total} "
            + ", ".join(
                [f"{c}:{len(monotone_constraints_map.get(CLASS_TO_INDEX[c], {}))}" for c in mono_classes if c in CLASS_TO_INDEX]
            ),
            flush=True,
        )

    class_feature_mask_info: dict[str, Any] = {"enabled": False}
    class_feature_indices: dict[int, np.ndarray] = {
        i: np.arange(len(model_feature_cols), dtype=np.int64) for i in range(len(CLASSES))
    }
    if str(args.class_feature_mask_json).strip():
        spec = json.loads(Path(args.class_feature_mask_json).read_text(encoding="utf-8"))
        class_report: dict[str, Any] = {}
        for i, cls in enumerate(CLASSES):
            cfg = spec.get(cls, {}) if isinstance(spec, dict) else {}
            keep_patterns = cfg.get("keep", []) if isinstance(cfg, dict) else []
            drop_patterns = cfg.get("drop", []) if isinstance(cfg, dict) else []
            idx, used_keep, used_drop = _select_feature_idx(
                feature_cols=model_feature_cols,
                keep_patterns=keep_patterns,
                drop_patterns=drop_patterns,
            )
            class_feature_indices[i] = idx
            class_report[cls] = {
                "n_features": int(len(idx)),
                "used_keep_patterns": used_keep,
                "used_drop_patterns": used_drop,
            }
        class_feature_mask_info = {
            "enabled": True,
            "path": str(Path(args.class_feature_mask_json).resolve()),
            "report": class_report,
        }
        print(
            "[lgbm] class-feature-mask enabled: "
            + ", ".join([f"{c}:{class_report[c]['n_features']}" for c in CLASSES]),
            flush=True,
        )

    hardneg_info: dict[str, Any] = {"enabled": False}
    hardneg_idx_by_class: dict[int, np.ndarray] = {}
    if str(args.hardneg_bank_json).strip():
        rep = max(0, int(args.hardneg_repeat))
        bank_obj = json.loads(Path(args.hardneg_bank_json).read_text(encoding="utf-8"))
        bank_map = bank_obj["bank"] if isinstance(bank_obj, dict) and "bank" in bank_obj else bank_obj
        if not isinstance(bank_map, dict):
            raise ValueError("hardneg bank json must be dict or contain {'bank': {...}}")
        cls_counts: dict[str, int] = {}
        for cls in CLASSES:
            ids = bank_map.get(cls, [])
            if ids is None:
                ids = []
            idxs = []
            for t in ids:
                ti = int(t)
                if ti in train_id_to_row:
                    idxs.append(train_id_to_row[ti])
            arr = np.unique(np.asarray(idxs, dtype=np.int64))
            hardneg_idx_by_class[int(CLASS_TO_INDEX[cls])] = arr
            cls_counts[cls] = int(len(arr))
        hardneg_info = {
            "enabled": rep > 0,
            "path": str(Path(args.hardneg_bank_json).resolve()),
            "repeat": int(rep),
            "counts_by_class": cls_counts,
            "applied_events": 0,
            "applied_rows_total": 0,
        }
        print(
            "[lgbm] hardneg enabled: "
            + ", ".join([f"{k}:{v}" for k, v in cls_counts.items()])
            + f" repeat={rep}",
            flush=True,
        )

    hardpos_info: dict[str, Any] = {"enabled": False}
    hardpos_idx_by_class: dict[int, np.ndarray] = {}
    if str(args.hardpos_bank_json).strip():
        rep = max(0, int(args.hardpos_repeat))
        bank_obj = json.loads(Path(args.hardpos_bank_json).read_text(encoding="utf-8"))
        bank_map = bank_obj["bank"] if isinstance(bank_obj, dict) and "bank" in bank_obj else bank_obj
        if not isinstance(bank_map, dict):
            raise ValueError("hardpos bank json must be dict or contain {'bank': {...}}")
        cls_counts: dict[str, int] = {}
        for cls in CLASSES:
            ids = bank_map.get(cls, [])
            if ids is None:
                ids = []
            idxs = []
            for t in ids:
                ti = int(t)
                if ti in train_id_to_row:
                    idxs.append(train_id_to_row[ti])
            arr = np.unique(np.asarray(idxs, dtype=np.int64))
            hardpos_idx_by_class[int(CLASS_TO_INDEX[cls])] = arr
            cls_counts[cls] = int(len(arr))
        hardpos_info = {
            "enabled": rep > 0,
            "path": str(Path(args.hardpos_bank_json).resolve()),
            "repeat": int(rep),
            "counts_by_class": cls_counts,
            "applied_events": 0,
            "applied_rows_total": 0,
        }
        print(
            "[lgbm] hardpos enabled: "
            + ", ".join([f"{k}:{v}" for k, v in cls_counts.items()])
            + f" repeat={rep}",
            flush=True,
        )

    # Same normalization style as catboost branch for comparable behavior.
    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    x_train = (x_train - mu) / sd
    x_test = (x_test - mu) / sd

    cv_df = pd.DataFrame(
        {
            "_cv_ts": train_df[str(args.time_col)],
            "_cv_group": train_df[str(args.group_col)].astype(np.int64),
        }
    )
    folds_all = make_forward_temporal_group_folds(
        cv_df,
        timestamp_col="_cv_ts",
        group_col="_cv_group",
        n_splits=int(args.n_splits),
    )
    folds = list(folds_all)
    if len(folds) == 0:
        raise RuntimeError("no folds were generated")

    focus_fold = int(args.focus_fold)
    jobs: list[dict[str, Any]] = []
    if focus_fold >= 0:
        if focus_fold >= len(folds):
            raise ValueError(f"focus-fold={focus_fold} out of range [0, {len(folds)-1}]")
        focus_val_idx = np.asarray(folds[focus_fold][1], dtype=np.int64)
        fold_val_parts = [np.asarray(va, dtype=np.int64) for _, va in folds]
        if int(args.start_fold) > 0 or int(args.max_folds) > 0:
            print("[lgbm] focus-fold mode ignores --start-fold/--max-fold and uses all folds as prediction targets", flush=True)
        for pred_fold in range(len(folds)):
            out_idx = np.asarray(folds[pred_fold][1], dtype=np.int64)
            if pred_fold == focus_fold:
                tr_parts = [fold_val_parts[i] for i in range(len(folds)) if i != focus_fold]
            else:
                tr_parts = [fold_val_parts[i] for i in range(len(folds)) if i not in {focus_fold, pred_fold}]
            tr_idx = np.concatenate(tr_parts, axis=0) if tr_parts else np.zeros((0,), dtype=np.int64)
            jobs.append(
                {
                    "job_id": int(pred_fold),
                    "pred_fold": int(pred_fold),
                    "focus_fold": int(focus_fold),
                    "tr_idx": tr_idx,
                    "va_idx": focus_val_idx,
                    "out_idx": out_idx,
                    "mode": "focus_fold_specialist",
                }
            )
        print(
            f"[lgbm] focus-fold specialist mode enabled: focus_fold={focus_fold}, jobs={len(jobs)}",
            flush=True,
        )
    else:
        if int(args.start_fold) > 0:
            start = min(int(args.start_fold), len(folds))
            folds = folds[start:]
            print(f"[lgbm] starting from fold index {start} (skipped {start} earliest folds)", flush=True)
        if int(args.max_folds) > 0 and len(folds) > int(args.max_folds):
            folds = folds[: int(args.max_folds)]
            print(f"[lgbm] limiting folds to first {int(args.max_folds)}", flush=True)
        for fold_id, (tr_idx, va_idx) in enumerate(folds):
            tr_idx = np.asarray(tr_idx, dtype=np.int64)
            va_idx = np.asarray(va_idx, dtype=np.int64)
            jobs.append(
                {
                    "job_id": int(fold_id),
                    "pred_fold": int(fold_id),
                    "focus_fold": -1,
                    "tr_idx": tr_idx,
                    "va_idx": va_idx,
                    "out_idx": va_idx,
                    "mode": "standard_cv",
                }
            )

    if len(jobs) == 0:
        raise RuntimeError("no folds selected after start/max fold filters")

    train_mask: np.ndarray | None = None
    train_mask_info: dict[str, Any] = {"enabled": False}
    if str(args.train_mask_npy).strip():
        raw_mask = np.asarray(np.load(args.train_mask_npy))
        if raw_mask.ndim != 1 or len(raw_mask) != len(train_df):
            raise ValueError(f"train-mask length mismatch: {raw_mask.shape} vs train {len(train_df)}")
        if raw_mask.dtype == np.bool_:
            train_mask = raw_mask.astype(bool)
        else:
            # treat non-bool as 0/1 mask
            train_mask = (raw_mask.astype(np.float32) > 0.5)
        if int(train_mask.sum()) <= 0:
            raise ValueError("train-mask selected zero rows")
        train_mask_info = {
            "enabled": True,
            "path": str(Path(args.train_mask_npy).resolve()),
            "selected": int(train_mask.sum()),
            "ratio": float(train_mask.mean()),
        }
        print(
            f"[lgbm] train-mask enabled: selected={int(train_mask.sum())}/{len(train_mask)} "
            f"({float(train_mask.mean()):.4f})",
            flush=True,
        )

    refill_info: dict[str, Any] = {
        "mode": str(args.refill_after_mask),
        "ratio": float(args.refill_ratio),
        "total_added": 0,
        "folds_with_refill": 0,
        "by_class_added": {c: 0 for c in CLASSES},
    }

    train_weights = np.ones((len(train_df),), dtype=np.float32)
    time_weight_info: dict[str, Any] = {"mode": str(args.time_weight_mode)}
    if str(args.time_weight_mode) == "fold0_boost":
        if len(folds_all) == 0:
            raise RuntimeError("no folds available for fold0_boost weighting")
        fold0_val_idx = folds_all[0][1]
        boost = max(0.0, float(args.fold0_boost))
        train_weights[fold0_val_idx] += boost
        time_weight_info.update(
            {
                "fold0_boost": float(boost),
                "fold0_n": int(len(fold0_val_idx)),
                "weights_mean": float(np.mean(train_weights)),
                "weights_max": float(np.max(train_weights)),
            }
        )
        print(
            f"[lgbm] time_weight_mode=fold0_boost fold0_n={len(fold0_val_idx)} boost={boost:.4f} "
            f"w_mean={np.mean(train_weights):.4f} w_max={np.max(train_weights):.4f}",
            flush=True,
        )
    elif str(args.time_weight_mode) == "linear_time":
        ts = pd.to_datetime(train_df[str(args.time_col)], errors="coerce", utc=True)
        if ts.isna().any():
            bad = int(ts.isna().sum())
            raise ValueError(f"timestamp parsing failed for {bad} rows in {args.time_col}")
        ns = ts.astype("int64").to_numpy(dtype=np.int64)
        t_min = int(ns.min())
        t_max = int(ns.max())
        denom = float(max(1, t_max - t_min))
        t_norm = (ns.astype(np.float64) - float(t_min)) / denom
        strength = max(0.0, float(args.time_weight_strength))
        train_weights = (1.0 + strength * (1.0 - t_norm)).astype(np.float32)
        time_weight_info.update(
            {
                "time_weight_strength": float(strength),
                "weights_mean": float(np.mean(train_weights)),
                "weights_max": float(np.max(train_weights)),
                "weights_min": float(np.min(train_weights)),
            }
        )
        print(
            f"[lgbm] time_weight_mode=linear_time strength={strength:.4f} "
            f"w_mean={np.mean(train_weights):.4f} w_min={np.min(train_weights):.4f} w_max={np.max(train_weights):.4f}",
            flush=True,
        )
    else:
        time_weight_info.update(
            {
                "weights_mean": float(np.mean(train_weights)),
                "weights_max": float(np.max(train_weights)),
                "weights_min": float(np.min(train_weights)),
            }
        )

    smote_info: dict[str, Any] = {"enabled": False}
    smote_class_indices: set[int] = set()
    if str(args.smote_mode) != "none":
        if BorderlineSMOTE is None:
            raise RuntimeError("smote-mode requested, but imbalanced-learn is not available")
        raw_classes = [x.strip() for x in str(args.smote_classes).split(",") if x.strip()]
        bad = [c for c in raw_classes if c not in CLASS_TO_INDEX]
        if bad:
            raise ValueError(f"unknown smote classes: {bad}")
        smote_class_indices = {int(CLASS_TO_INDEX[c]) for c in raw_classes}
        smote_info = {
            "enabled": True,
            "mode": str(args.smote_mode),
            "classes": raw_classes,
            "k_neighbors": int(args.smote_k_neighbors),
            "m_neighbors": int(args.smote_m_neighbors),
            "sampling_strategy": float(args.smote_sampling_strategy),
            "applied_events": 0,
            "added_rows_total": 0,
            "skipped_events": 0,
            "errors": [],
        }
        print(
            "[lgbm] smote enabled: "
            f"mode={smote_info['mode']} classes={','.join(raw_classes)} "
            f"sampling={smote_info['sampling_strategy']:.3f} "
            f"k={smote_info['k_neighbors']} m={smote_info['m_neighbors']}",
            flush=True,
        )

    n_train = len(train_df)
    n_test = len(test_df)
    n_classes = len(CLASSES)
    oof = np.zeros((n_train, n_classes), dtype=np.float32)
    test_accum = np.zeros((n_test, n_classes), dtype=np.float32)
    fold_scores: list[float] = []
    fold_scores_valid_only: list[float] = []
    fold_reports: list[dict[str, Any]] = []
    fallback_events: list[dict[str, Any]] = []
    skipped_folds: list[dict[str, Any]] = []
    used_val_indices: list[np.ndarray] = []
    effective_folds = 0

    for job in jobs:
        fold_id = int(job["job_id"])
        pred_fold = int(job["pred_fold"])
        mode = str(job["mode"])
        tr_idx_raw = np.asarray(job["tr_idx"], dtype=np.int64)
        tr_idx = tr_idx_raw
        if train_mask is not None:
            tr_idx = tr_idx[train_mask[tr_idx]]
            if str(args.refill_after_mask) == "class_preserve":
                refill_ratio = max(0.0, float(args.refill_ratio))
                dropped_idx = tr_idx_raw[~train_mask[tr_idx_raw]]
                if refill_ratio > 0.0 and len(dropped_idx) > 0:
                    rng = np.random.RandomState(int(args.seed) + 997 * fold_id + 17 * pred_fold)
                    add_parts: list[np.ndarray] = []
                    y_train_idx = y_idx
                    for c, cls in enumerate(CLASSES):
                        n_drop_cls = int(np.sum(y_train_idx[dropped_idx] == c))
                        if n_drop_cls <= 0:
                            continue
                        n_add = int(round(n_drop_cls * refill_ratio))
                        if n_add <= 0:
                            continue
                        pool = tr_idx[y_train_idx[tr_idx] == c]
                        if len(pool) <= 0:
                            continue
                        add_idx = rng.choice(pool, size=n_add, replace=True).astype(np.int64)
                        add_parts.append(add_idx)
                        refill_info["by_class_added"][cls] = int(refill_info["by_class_added"][cls]) + int(n_add)
                    if add_parts:
                        add_concat = np.concatenate(add_parts, axis=0)
                        tr_idx = np.concatenate([tr_idx, add_concat], axis=0)
                        refill_info["total_added"] = int(refill_info["total_added"]) + int(len(add_concat))
                        refill_info["folds_with_refill"] = int(refill_info["folds_with_refill"]) + 1
        va_idx = np.asarray(job["va_idx"], dtype=np.int64)
        out_idx = np.asarray(job["out_idx"], dtype=np.int64)
        if len(tr_idx) == 0:
            skipped_folds.append(
                {
                    "fold": int(fold_id),
                    "pred_fold": int(pred_fold),
                    "n_train": 0,
                    "n_val_out": int(len(out_idx)),
                    "n_val_es": int(len(va_idx)),
                    "invalid_classes": [],
                    "reason": "empty_train_after_mask",
                    "mode": mode,
                }
            )
            print(
                f"[fold] {fold_id+1}/{len(jobs)} skipped: empty train after mask",
                flush=True,
            )
            continue
        invalid_classes: list[dict[str, Any]] = []
        min_count = max(1, int(args.min_class_count))
        for c, cls in enumerate(CLASSES):
            pos = int(np.sum(y[tr_idx, c]))
            neg = int(len(tr_idx) - pos)
            if pos < min_count or neg < min_count:
                invalid_classes.append(
                    {
                        "class_idx": int(c),
                        "class": str(cls),
                        "n_pos_train": int(pos),
                        "n_neg_train": int(neg),
                        "min_class_count": int(min_count),
                    }
                )

        if invalid_classes and str(args.invalid_class_policy) == "skip_fold":
            skipped_folds.append(
                {
                    "fold": int(fold_id),
                    "pred_fold": int(pred_fold),
                    "n_train": int(len(tr_idx)),
                    "n_val_out": int(len(out_idx)),
                    "n_val_es": int(len(va_idx)),
                    "invalid_classes": invalid_classes,
                    "reason": "invalid_class_counts",
                    "mode": mode,
                }
            )
            print(
                f"[fold] {fold_id+1}/{len(jobs)} skipped due to invalid class counts: "
                f"{[x['class'] for x in invalid_classes]}",
                flush=True,
            )
            continue

        invalid_idx = {int(x["class_idx"]) for x in invalid_classes}
        fold_test = np.zeros((n_test, n_classes), dtype=np.float32)
        for c, cls in enumerate(CLASSES):
            tr_idx_c = tr_idx
            if hardneg_info.get("enabled", False):
                bank_idx = hardneg_idx_by_class.get(c, np.zeros((0,), dtype=np.int64))
                if len(bank_idx) > 0:
                    tr_unique = np.unique(tr_idx)
                    bank_in_fold = np.intersect1d(tr_unique, bank_idx, assume_unique=False)
                    if len(bank_in_fold) > 0:
                        # hard negatives are by construction negatives for class c in source OOF
                        add = np.repeat(bank_in_fold, int(hardneg_info["repeat"]))
                        tr_idx_c = np.concatenate([tr_idx, add.astype(np.int64)], axis=0)
                        hardneg_info["applied_events"] = int(hardneg_info["applied_events"]) + 1
                        hardneg_info["applied_rows_total"] = int(hardneg_info["applied_rows_total"]) + int(len(add))
            if hardpos_info.get("enabled", False):
                bank_idx = hardpos_idx_by_class.get(c, np.zeros((0,), dtype=np.int64))
                if len(bank_idx) > 0:
                    tr_unique = np.unique(tr_idx)
                    bank_in_fold = np.intersect1d(tr_unique, bank_idx, assume_unique=False)
                    if len(bank_in_fold) > 0:
                        add = np.repeat(bank_in_fold, int(hardpos_info["repeat"]))
                        tr_idx_c = np.concatenate([tr_idx_c, add.astype(np.int64)], axis=0)
                        hardpos_info["applied_events"] = int(hardpos_info["applied_events"]) + 1
                        hardpos_info["applied_rows_total"] = int(hardpos_info["applied_rows_total"]) + int(len(add))

            y_tr = y[tr_idx_c, c]
            y_va = y[va_idx, c]
            if c in invalid_idx or np.unique(y_tr).size < 2:
                if str(args.invalid_class_policy) in {"fallback_teacher", "skip_class"}:
                    assert teacher_oof is not None and teacher_test is not None
                    oof[out_idx, c] = teacher_oof[out_idx, c]
                    fold_test[:, c] = teacher_test[:, c]
                    fallback_events.append(
                        {
                            "fold": int(fold_id),
                            "pred_fold": int(pred_fold),
                            "class": str(cls),
                            "reason": "fallback_teacher",
                            "n_pos_train": int(np.sum(y_tr)),
                            "n_neg_train": int(len(y_tr) - np.sum(y_tr)),
                            "mode": mode,
                        }
                    )
                else:
                    prior = float(global_priors[c])
                    oof[out_idx, c] = prior
                    fold_test[:, c] = prior
                    fallback_events.append(
                        {
                            "fold": int(fold_id),
                            "pred_fold": int(pred_fold),
                            "class": str(cls),
                            "reason": "fallback_prior",
                            "prior": prior,
                            "n_pos_train": int(np.sum(y_tr)),
                            "n_neg_train": int(len(y_tr) - np.sum(y_tr)),
                            "mode": mode,
                        }
                    )
                continue

            fidx = class_feature_indices[c]
            xtr_c = x_train[tr_idx_c][:, fidx]
            xva_c = x_train[va_idx][:, fidx]
            xout_c = x_train[out_idx][:, fidx]
            xtest_c = x_test[:, fidx]
            wtr_c = train_weights[tr_idx_c].astype(np.float32)

            # Fold-safe SMOTE: train fold only, selected classes only.
            if smote_info.get("enabled", False) and c in smote_class_indices:
                y_bin = y_tr.astype(np.int32)
                pos_count = int(np.sum(y_bin == 1))
                neg_count = int(np.sum(y_bin == 0))
                min_need = max(2, int(args.smote_k_neighbors) + 1)
                if pos_count >= min_need and neg_count >= min_need:
                    try:
                        sm = BorderlineSMOTE(
                            kind="borderline-1",
                            sampling_strategy=float(args.smote_sampling_strategy),
                            k_neighbors=int(args.smote_k_neighbors),
                            m_neighbors=int(args.smote_m_neighbors),
                            random_state=int(args.seed) + 131 * int(fold_id) + 17 * int(c),
                        )
                        x_res, y_res = sm.fit_resample(xtr_c, y_bin)
                        n_orig = int(len(y_bin))
                        n_added = int(len(y_res) - n_orig)
                        if n_added > 0:
                            pos_w = float(np.mean(wtr_c[y_bin == 1])) if np.any(y_bin == 1) else 1.0
                            neg_w = float(np.mean(wtr_c[y_bin == 0])) if np.any(y_bin == 0) else 1.0
                            y_syn = y_res[n_orig:]
                            w_syn = np.where(y_syn == 1, pos_w, neg_w).astype(np.float32)
                            xtr_c = x_res.astype(np.float32)
                            y_tr = y_res.astype(np.float32)
                            wtr_c = np.concatenate([wtr_c, w_syn], axis=0).astype(np.float32)
                            smote_info["applied_events"] = int(smote_info["applied_events"]) + 1
                            smote_info["added_rows_total"] = int(smote_info["added_rows_total"]) + int(n_added)
                        else:
                            smote_info["skipped_events"] = int(smote_info["skipped_events"]) + 1
                    except Exception as e:
                        smote_info["skipped_events"] = int(smote_info["skipped_events"]) + 1
                        errs = smote_info.get("errors", [])
                        if len(errs) < 20:
                            errs.append({"fold": int(fold_id), "class": str(cls), "error": str(e)})
                        smote_info["errors"] = errs
                else:
                    smote_info["skipped_events"] = int(smote_info["skipped_events"]) + 1

            pos = float(y_tr.sum())
            neg = float(len(y_tr) - y_tr.sum())
            scale_pos_weight = float((neg + 1.0) / (pos + 1.0))

            dtrain = lgb.Dataset(xtr_c, label=y_tr, free_raw_data=False)
            dtrain.set_weight(wtr_c)
            dvalid = lgb.Dataset(xva_c, label=y_va, reference=dtrain, free_raw_data=False)

            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": str(args.boosting_type),
                "learning_rate": float(args.learning_rate),
                "num_leaves": int(args.num_leaves),
                "feature_fraction": float(args.feature_fraction),
                "bagging_fraction": float(args.bagging_fraction),
                "bagging_freq": int(args.bagging_freq),
                "min_data_in_leaf": int(args.min_data_in_leaf),
                "lambda_l2": float(args.lambda_l2),
                "max_depth": int(args.max_depth),
                "scale_pos_weight": scale_pos_weight,
                "seed": int(args.seed) + 97 * fold_id + 13 * c,
                "verbosity": -1,
            }
            if str(args.boosting_type) == "dart":
                params["drop_rate"] = float(args.drop_rate)
                params["skip_drop"] = float(args.skip_drop)
            if bool(args.extra_trees):
                params["extra_trees"] = True
            ic_local = remap_interaction_constraints(interaction_constraints_global, fidx)
            if ic_local is not None:
                params["interaction_constraints"] = ic_local
            if c in monotone_constraints_map and len(monotone_constraints_map[c]) > 0:
                mono_local = [0] * len(fidx)
                cls_map = monotone_constraints_map[c]
                for li, gi in enumerate(fidx.tolist()):
                    fn = model_feature_cols[int(gi)]
                    if fn in cls_map:
                        mono_local[li] = int(cls_map[fn])
                if any(v != 0 for v in mono_local):
                    params["monotone_constraints"] = mono_local
                    params["monotone_constraints_method"] = "advanced"
            if int(args.num_threads) > 0:
                params["num_threads"] = int(args.num_threads)

            booster = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=int(args.num_boost_round),
                valid_sets=[dvalid],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=int(args.early_stopping_rounds), verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            oof[out_idx, c] = booster.predict(xout_c, num_iteration=booster.best_iteration).astype(np.float32)
            fold_test[:, c] = booster.predict(xtest_c, num_iteration=booster.best_iteration).astype(np.float32)

        test_accum += fold_test
        effective_folds += 1
        used_val_indices.append(out_idx)
        fold_score = float(macro_map(y[out_idx], oof[out_idx]))
        valid_class_mask = np.ones((n_classes,), dtype=bool)
        if invalid_idx:
            valid_class_mask[list(invalid_idx)] = False
        fold_score_valid = float(_macro_map_subset(y[out_idx], oof[out_idx], valid_class_mask))
        fold_scores.append(fold_score)
        fold_scores_valid_only.append(fold_score_valid)
        fold_reports.append(
            {
                "fold": int(fold_id),
                "pred_fold": int(pred_fold),
                "mode": mode,
                "n_train": int(len(tr_idx)),
                "n_val_out": int(len(out_idx)),
                "n_val_es": int(len(va_idx)),
                "invalid_classes": [str(CLASSES[i]) for i in sorted(list(invalid_idx))],
                "n_invalid_classes": int(len(invalid_idx)),
                "n_valid_classes": int(np.sum(valid_class_mask)),
                "fold_macro": float(fold_score),
                "fold_macro_valid_only": float(fold_score_valid),
            }
        )
        print(
            f"[fold] {fold_id+1}/{len(jobs)} pred_fold={pred_fold} macro_map={fold_score:.6f} "
            f"macro_valid_only={fold_score_valid:.6f} n_val={len(out_idx)} n_es={len(va_idx)} "
            f"invalid_classes={len(invalid_classes)}",
            flush=True,
        )

    if effective_folds <= 0:
        raise RuntimeError("no folds were used for training after invalid-class policy filters")
    test_accum /= float(effective_folds)

    covered_idx = np.unique(np.concatenate(used_val_indices)).astype(np.int64)
    macro_covered = float(macro_map(y[covered_idx], oof[covered_idx])) if len(covered_idx) > 0 else 0.0
    macro_valid_only = float(np.mean(fold_scores_valid_only)) if fold_scores_valid_only else 0.0
    per_covered = per_class_ap(y[covered_idx], oof[covered_idx]) if len(covered_idx) > 0 else {c: 0.0 for c in CLASSES}

    np.save(art_dir / "lgbm_ovr_oof.npy", oof)
    np.save(art_dir / "lgbm_ovr_test.npy", test_accum)

    sub = sample_sub.copy()
    if "track_id" not in sub.columns:
        raise ValueError("sample submission must contain 'track_id'")
    sub = sub[[c for c in sub.columns if c == "track_id" or c in CLASSES]].copy()
    sub["track_id"] = test_ids
    sub[CLASSES] = test_accum
    sub.to_csv(out_dir / "submission_lgbm_ovr.csv", index=False)

    model_name = f"lgbm_ovr_seed{int(args.seed)}"
    summary = {
        "project_root": str(PROJECT_ROOT.resolve()),
        "output_dir": str(out_dir),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "features_total": int(len(feature_cols)),
        "features_kept": int(len(model_feature_cols)),
        "features_dropped": sorted(list(set(feature_cols) - set(keep_cols))),
        "kept_feature_columns": model_feature_cols,
        "adversarial": adv_report,
        "teacher_meta": teacher_meta_info,
        "interaction_constraints": interaction_info,
        "monotone_constraints": monotone_info,
        "class_feature_mask": class_feature_mask_info,
        "hardneg": hardneg_info,
        "hardpos": hardpos_info,
        "time_weight": time_weight_info,
        "smote": smote_info,
        "train_mask": train_mask_info,
        "refill_after_mask": refill_info,
        "invalid_class_handling": {
            "policy": str(args.invalid_class_policy),
            "min_class_count": int(max(1, int(args.min_class_count))),
            "effective_folds": int(effective_folds),
            "requested_folds": int(len(jobs)),
            "focus_fold": int(focus_fold),
            "skipped_folds": skipped_folds,
        },
        "models": {
            model_name: {
                "type": "tabular",
                "oof_path": str((art_dir / "lgbm_ovr_oof.npy").resolve()),
                "test_path": str((art_dir / "lgbm_ovr_test.npy").resolve()),
                "macro_map": float(macro_covered),
                "macro_map_valid_only": float(macro_valid_only),
                "per_class_ap": per_covered,
                "fold_scores": [float(x) for x in fold_scores],
                "fold_scores_valid_only": [float(x) for x in fold_scores_valid_only],
                "worst_fold": float(np.min(fold_scores)) if fold_scores else 0.0,
                "worst_fold_valid_only": float(np.min(fold_scores_valid_only)) if fold_scores_valid_only else 0.0,
                "fold_reports": fold_reports,
                "fallback_events": fallback_events,
            }
        },
        "track_ids_test_path": str((out_dir / "test_track_ids.npy").resolve()),
        "track_ids_train_path": str((out_dir / "train_track_ids.npy").resolve()),
    }
    dump_json(out_dir / "oof_summary.json", summary)

    print("=== LGBM OVR COMPLETE ===", flush=True)
    print(f"output_dir: {out_dir}", flush=True)
    print(f"oof_summary: {out_dir / 'oof_summary.json'}", flush=True)
    print(f"effective_folds={effective_folds}/{len(jobs)}", flush=True)
    print(
        f"macro_covered={macro_covered:.6f} macro_valid_only={macro_valid_only:.6f} "
        f"fold_mean={float(np.mean(fold_scores)):.6f} fold_worst={float(np.min(fold_scores)):.6f} "
        f"fold_worst_valid_only={float(np.min(fold_scores_valid_only)):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
