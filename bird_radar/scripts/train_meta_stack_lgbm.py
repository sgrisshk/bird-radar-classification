#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache
from src.redesign.features import build_tabular_frame, get_feature_columns


def _parse_name_path_pairs(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected name=path, got: {item}")
        name, path = item.split("=", 1)
        key = name.strip()
        val = Path(path.strip())
        if not key:
            raise ValueError(f"Empty model name in: {item}")
        out[key] = val
    return out


def _align_probs_from_csv(csv_path: str | Path, ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=["track_id", *CLASSES])
    mp = {int(r["track_id"]): np.asarray([r[c] for c in CLASSES], dtype=np.float32) for _, r in df.iterrows()}
    missing = [int(t) for t in ids if int(t) not in mp]
    if missing:
        raise ValueError(f"{csv_path} missing {len(missing)} ids; first={missing[:10]}")
    return np.stack([mp[int(t)] for t in ids], axis=0).astype(np.float32)


def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(p.astype(np.float32), eps, 1.0 - eps)
    return np.log(x / (1.0 - x)).astype(np.float32)


def _macro_map_weighted(y_true: np.ndarray, y_prob: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    vals: list[float] = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_prob[:, j]
        if yt.sum() <= 0:
            vals.append(0.0)
        else:
            vals.append(float(average_precision_score(yt, yp, sample_weight=sample_weight)))
    return float(np.mean(vals))


def _teacher_meta_features(p: np.ndarray) -> tuple[np.ndarray, list[str]]:
    probs = np.clip(p.astype(np.float32), 1e-6, 1.0 - 1e-6)
    logits = _safe_logit(probs)
    order = np.argsort(probs, axis=1)
    top1_idx = order[:, -1]
    top2_idx = order[:, -2]
    top1 = probs[np.arange(len(probs)), top1_idx]
    top2 = probs[np.arange(len(probs)), top2_idx]
    margin = top1 - top2
    entropy = -np.sum(probs * np.log(probs), axis=1)

    mats: list[np.ndarray] = [
        top1.reshape(-1, 1),
        top2.reshape(-1, 1),
        margin.reshape(-1, 1),
        entropy.reshape(-1, 1),
        top1_idx.astype(np.float32).reshape(-1, 1),
        top2_idx.astype(np.float32).reshape(-1, 1),
    ]
    cols = [
        "teacher_top1_prob",
        "teacher_top2_prob",
        "teacher_margin",
        "teacher_entropy",
        "teacher_top1_class_id",
        "teacher_top2_class_id",
    ]
    for j, cls in enumerate(CLASSES):
        mats.append(probs[:, j].reshape(-1, 1))
        mats.append(logits[:, j].reshape(-1, 1))
        cols.extend([f"teacher_p_{cls}", f"teacher_logit_{cls}"])
    return np.concatenate(mats, axis=1).astype(np.float32), cols


def _disagreement_features(
    teacher: np.ndarray, expert_map: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[str]]:
    mats: list[np.ndarray] = []
    cols: list[str] = []
    t_entropy = (-np.sum(np.clip(teacher, 1e-6, 1.0) * np.log(np.clip(teacher, 1e-6, 1.0)), axis=1)).astype(np.float32)
    t_top1 = np.max(teacher, axis=1).astype(np.float32)
    for name, p in expert_map.items():
        diff = np.abs(p - teacher).astype(np.float32)
        mats.extend(
            [
                np.mean(diff, axis=1, keepdims=True),
                np.max(diff, axis=1, keepdims=True),
                (np.max(p, axis=1) - t_top1).astype(np.float32).reshape(-1, 1),
                (
                    (-np.sum(np.clip(p, 1e-6, 1.0) * np.log(np.clip(p, 1e-6, 1.0)), axis=1)).astype(np.float32)
                    - t_entropy
                ).reshape(-1, 1),
            ]
        )
        cols.extend(
            [
                f"diff_meanabs_{name}_vs_teacher",
                f"diff_maxabs_{name}_vs_teacher",
                f"diff_top1prob_{name}_vs_teacher",
                f"diff_entropy_{name}_vs_teacher",
            ]
        )
    return np.concatenate(mats, axis=1).astype(np.float32), cols


def _reconstruct_classwise_oof_from_json(json_path: str | Path, ids_order: np.ndarray) -> np.ndarray:
    j = json.loads(Path(json_path).read_text(encoding="utf-8"))
    new_ids = np.asarray(np.load(j["new_track_ids_npy"]), dtype=np.int64)
    new_oof = np.asarray(np.load(j["new_oof_npy"]), dtype=np.float32)
    teacher_oof = _align_probs_from_csv(j["teacher_oof_csv"], new_ids)
    chosen = {str(k): float(v) for k, v in (j.get("chosen_weights") or {}).items()}
    pred = teacher_oof.copy()
    for cls in CLASSES:
        w_new = float(chosen.get(cls, 0.0))
        c = CLASSES.index(cls)
        pred[:, c] = np.clip((1.0 - w_new) * teacher_oof[:, c] + w_new * new_oof[:, c], 0.0, 1.0)

    pos = {int(t): i for i, t in enumerate(new_ids)}
    missing = [int(t) for t in ids_order if int(t) not in pos]
    if missing:
        raise ValueError(f"classwise json OOF reconstruction missing ids={len(missing)}")
    return np.stack([pred[pos[int(t)]] for t in ids_order], axis=0).astype(np.float32)


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict[str, Any]]:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


def _extract_tabular_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cache_dir: Path
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")
    tab_train = build_tabular_frame(train_df, train_cache)
    tab_test = build_tabular_frame(test_df, test_cache)
    feat_cols = get_feature_columns(tab_train)
    x_train = tab_train[feat_cols].to_numpy(dtype=np.float32)
    x_test = tab_test[feat_cols].to_numpy(dtype=np.float32)
    return x_train, x_test, list(feat_cols)


def _build_feature_matrix(
    *,
    teacher_train: np.ndarray,
    teacher_test: np.ndarray,
    expert_train: dict[str, np.ndarray],
    expert_test: dict[str, np.ndarray],
    adv_train: np.ndarray | None,
    adv_test: np.ndarray | None,
    tab_train: np.ndarray | None,
    tab_test: np.ndarray | None,
    tab_cols: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_tr_parts: list[np.ndarray] = []
    x_te_parts: list[np.ndarray] = []
    cols: list[str] = []

    # Expert probabilities + logits.
    for name in sorted(expert_train.keys()):
        p_tr = expert_train[name].astype(np.float32)
        p_te = expert_test[name].astype(np.float32)
        l_tr = _safe_logit(p_tr)
        l_te = _safe_logit(p_te)
        x_tr_parts.extend([p_tr, l_tr])
        x_te_parts.extend([p_te, l_te])
        cols.extend([f"{name}_p_{c}" for c in CLASSES])
        cols.extend([f"{name}_logit_{c}" for c in CLASSES])

    # Teacher meta features.
    tm_tr, tm_cols = _teacher_meta_features(teacher_train)
    tm_te, _ = _teacher_meta_features(teacher_test)
    x_tr_parts.append(tm_tr)
    x_te_parts.append(tm_te)
    cols.extend(tm_cols)

    # Disagreement features.
    dg_tr, dg_cols = _disagreement_features(teacher_train, expert_train)
    dg_te, _ = _disagreement_features(teacher_test, expert_test)
    x_tr_parts.append(dg_tr)
    x_te_parts.append(dg_te)
    cols.extend(dg_cols)

    # Adversarial test-likeness.
    if adv_train is not None and adv_test is not None:
        x_tr_parts.append(adv_train.reshape(-1, 1).astype(np.float32))
        x_te_parts.append(adv_test.reshape(-1, 1).astype(np.float32))
        cols.append("adversarial_pred_testlike")

    if tab_train is not None and tab_test is not None and tab_cols is not None:
        x_tr_parts.append(tab_train.astype(np.float32))
        x_te_parts.append(tab_test.astype(np.float32))
        cols.extend([f"tab_{c}" for c in tab_cols])

    x_tr = np.concatenate(x_tr_parts, axis=1).astype(np.float32)
    x_te = np.concatenate(x_te_parts, axis=1).astype(np.float32)

    # Median imputation for numeric stability.
    med = np.nanmedian(x_tr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    bad_tr = ~np.isfinite(x_tr)
    bad_te = ~np.isfinite(x_te)
    if np.any(bad_tr):
        x_tr[bad_tr] = med[np.where(bad_tr)[1]]
    if np.any(bad_te):
        x_te[bad_te] = med[np.where(bad_te)[1]]
    return x_tr, x_te, cols


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train forward OOF LGBM meta-stacking model from expert predictions + track meta-features.")
    p.add_argument("--train-csv", default="train.csv")
    p.add_argument("--test-csv", default="test.csv")
    p.add_argument("--sample-submission", default="sample_submission.csv")
    p.add_argument("--output-dir", default="bird_radar/artifacts/meta_stack_lgbm_v1")
    p.add_argument("--cache-dir", default="bird_radar/artifacts/cache")

    p.add_argument("--teacher-oof-csv", default="bird_radar/artifacts/oof_csv_for_per_class/oof_teacher_fr_complete.csv")
    p.add_argument(
        "--teacher-test-csv",
        default="bird_radar/artifacts/temporal_model0_reg_drop10_none_reverse_complete/sub_teacher_fr.csv",
    )

    p.add_argument("--forward-oof-csv", default="bird_radar/artifacts/oof_csv_for_per_class/oof_base_forward_complete.csv")
    p.add_argument("--reverse-oof-csv", default="bird_radar/artifacts/oof_csv_for_per_class/oof_base_reverse_complete.csv")
    p.add_argument(
        "--forward-test-csv",
        default="bird_radar/artifacts/temporal_model0_reg_drop10_none_complete/sub_temporal_best.csv",
    )
    p.add_argument(
        "--reverse-test-csv",
        default="bird_radar/artifacts/temporal_model0_reg_drop10_none_reverse_complete/sub_temporal_best.csv",
    )
    p.add_argument("--fr-forward-weight", type=float, default=0.35)

    p.add_argument(
        "--classwise-json",
        default="bird_radar/artifacts/temporal_model0_reg_drop10_none_reverse_complete/sub_classwise_constrained_search_forward_reverse.json",
    )
    p.add_argument(
        "--classwise-test-csv",
        default="bird_radar/artifacts/temporal_model0_reg_drop10_none_reverse_complete/sub_classwise_constrained_search_forward_reverse.csv",
    )
    p.add_argument(
        "--robust-oof-csv",
        default="bird_radar/artifacts/fold123_softmix_bank3_a1p0_b1p0_c1p0_T0p25_scan_v4_robust3proxy/oof_best.csv",
    )
    p.add_argument(
        "--robust-test-csv",
        default="bird_radar/artifacts/fold123_softmix_bank3_a1p0_b1p0_c1p0_T0p25_scan_v4_robust3proxy/sub_best.csv",
    )

    p.add_argument("--extra-expert-oof-csv", action="append", default=[], help="name=path")
    p.add_argument("--extra-expert-test-csv", action="append", default=[], help="name=path")

    p.add_argument(
        "--adversarial-pred-train-npy",
        default="bird_radar/artifacts/adversarial_weights_tabular_only_v2/artifacts/adversarial_pred_train.npy",
    )
    p.add_argument(
        "--adversarial-pred-test-npy",
        default="bird_radar/artifacts/adversarial_weights_tabular_only_v2/artifacts/adversarial_pred_test.npy",
    )
    p.add_argument("--train-sample-weights-npy", default="")
    p.add_argument("--use-tabular-features", action="store_true", default=True)
    p.add_argument("--no-tabular-features", dest="use_tabular_features", action="store_false")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--start-fold", type=int, default=0)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--feature-fraction", type=float, default=0.8)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--bagging-freq", type=int, default=1)
    p.add_argument("--min-data-in-leaf", type=int, default=120)
    p.add_argument("--lambda-l2", type=float, default=3.0)
    p.add_argument("--num-boost-round", type=int, default=1800)
    p.add_argument("--early-stopping-rounds", type=int, default=120)
    p.add_argument("--num-threads", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = out_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    sample_sub = pd.read_csv(args.sample_submission, usecols=["track_id"])
    train_ids = train_df["track_id"].to_numpy(dtype=np.int64)
    test_ids = test_df["track_id"].to_numpy(dtype=np.int64)
    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = np.zeros((len(train_df), len(CLASSES)), dtype=np.float32)
    y[np.arange(len(train_df)), y_idx] = 1.0

    teacher_train = _align_probs_from_csv(args.teacher_oof_csv, train_ids)
    teacher_test = _align_probs_from_csv(args.teacher_test_csv, test_ids)

    # Build expert bank.
    expert_train: dict[str, np.ndarray] = {}
    expert_test: dict[str, np.ndarray] = {}

    # FR 35/65.
    w_f = float(args.fr_forward_weight)
    f_oof = _align_probs_from_csv(args.forward_oof_csv, train_ids)
    r_oof = _align_probs_from_csv(args.reverse_oof_csv, train_ids)
    f_test = _align_probs_from_csv(args.forward_test_csv, test_ids)
    r_test = _align_probs_from_csv(args.reverse_test_csv, test_ids)
    expert_train["fr35_65"] = np.clip(w_f * f_oof + (1.0 - w_f) * r_oof, 0.0, 1.0).astype(np.float32)
    expert_test["fr35_65"] = np.clip(w_f * f_test + (1.0 - w_f) * r_test, 0.0, 1.0).astype(np.float32)

    # Classwise FR from JSON reconstruction + test CSV.
    if str(args.classwise_json).strip() and Path(args.classwise_json).exists():
        expert_train["classwise_fr"] = _reconstruct_classwise_oof_from_json(args.classwise_json, train_ids)
        expert_test["classwise_fr"] = _align_probs_from_csv(args.classwise_test_csv, test_ids)

    # Robust softmix.
    expert_train["robust_softmix"] = _align_probs_from_csv(args.robust_oof_csv, train_ids)
    expert_test["robust_softmix"] = _align_probs_from_csv(args.robust_test_csv, test_ids)

    # Optional extra experts (csv only).
    extra_oof = _parse_name_path_pairs(args.extra_expert_oof_csv)
    extra_test = _parse_name_path_pairs(args.extra_expert_test_csv)
    if sorted(extra_oof.keys()) != sorted(extra_test.keys()):
        raise ValueError("extra-expert-oof-csv and extra-expert-test-csv must have identical names")
    for name in sorted(extra_oof.keys()):
        expert_train[name] = _align_probs_from_csv(extra_oof[name], train_ids)
        expert_test[name] = _align_probs_from_csv(extra_test[name], test_ids)

    # Adversarial test-likeness.
    adv_train: np.ndarray | None = None
    adv_test: np.ndarray | None = None
    if str(args.adversarial_pred_train_npy).strip() and Path(args.adversarial_pred_train_npy).exists():
        adv_train = np.asarray(np.load(args.adversarial_pred_train_npy), dtype=np.float32).reshape(-1)
        adv_test = np.asarray(np.load(args.adversarial_pred_test_npy), dtype=np.float32).reshape(-1)
        if len(adv_train) != len(train_ids) or len(adv_test) != len(test_ids):
            raise ValueError("adversarial prediction lengths mismatch")

    tab_train: np.ndarray | None = None
    tab_test: np.ndarray | None = None
    tab_cols: list[str] | None = None
    if bool(args.use_tabular_features):
        tab_train, tab_test, tab_cols = _extract_tabular_features(
            train_df=train_df, test_df=test_df, cache_dir=Path(args.cache_dir).resolve()
        )

    x_train, x_test, feature_cols = _build_feature_matrix(
        teacher_train=teacher_train,
        teacher_test=teacher_test,
        expert_train=expert_train,
        expert_test=expert_test,
        adv_train=adv_train,
        adv_test=adv_test,
        tab_train=tab_train,
        tab_test=tab_test,
        tab_cols=tab_cols,
    )

    # Optional per-row training weights.
    sample_weight: np.ndarray | None = None
    if str(args.train_sample_weights_npy).strip() and Path(args.train_sample_weights_npy).exists():
        sw = np.asarray(np.load(args.train_sample_weights_npy), dtype=np.float32).reshape(-1)
        if len(sw) != len(train_ids):
            raise ValueError("train-sample-weights length mismatch")
        sample_weight = np.clip(sw, 1e-8, None)

    # Forward folds.
    cv_df = train_df[["timestamp_start_radar_utc", "observation_id"]].copy()
    folds = make_forward_temporal_group_folds(
        cv_df,
        timestamp_col="timestamp_start_radar_utc",
        group_col="observation_id",
        n_splits=int(args.n_splits),
    )

    oof = np.zeros((len(train_ids), len(CLASSES)), dtype=np.float32)
    test_sum = np.zeros((len(test_ids), len(CLASSES)), dtype=np.float32)
    fold_reports: list[dict[str, Any]] = []
    used_folds = 0

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        if fold_idx < int(args.start_fold):
            continue
        used_folds += 1
        x_tr = x_train[tr_idx]
        x_va = x_train[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        sw_tr = sample_weight[tr_idx] if sample_weight is not None else None
        sw_va = sample_weight[va_idx] if sample_weight is not None else None

        pred_va = np.zeros((len(va_idx), len(CLASSES)), dtype=np.float32)
        pred_te = np.zeros((len(test_ids), len(CLASSES)), dtype=np.float32)
        class_info: list[dict[str, Any]] = []
        for c, cls in enumerate(CLASSES):
            ytr_c = y_tr[:, c]
            yva_c = y_va[:, c]
            pos = int(np.sum(ytr_c))
            neg = int(len(ytr_c) - pos)
            if pos == 0 or neg == 0:
                const_p = float(np.mean(ytr_c))
                pred_va[:, c] = const_p
                pred_te[:, c] = const_p
                class_info.append({"class": cls, "mode": "const", "const_p": const_p, "train_pos": pos, "train_neg": neg})
                continue

            clf = lgb.LGBMClassifier(
                objective="binary",
                metric="average_precision",
                n_estimators=int(args.num_boost_round),
                learning_rate=float(args.learning_rate),
                num_leaves=int(args.num_leaves),
                feature_fraction=float(args.feature_fraction),
                bagging_fraction=float(args.bagging_fraction),
                bagging_freq=int(args.bagging_freq),
                min_data_in_leaf=int(args.min_data_in_leaf),
                lambda_l2=float(args.lambda_l2),
                random_state=int(args.seed + 1009 * fold_idx + 17 * c),
                n_jobs=int(args.num_threads),
                verbosity=-1,
            )
            clf.fit(
                x_tr,
                ytr_c.astype(np.int32),
                sample_weight=sw_tr,
                eval_set=[(x_va, yva_c.astype(np.int32))],
                eval_metric="average_precision",
                callbacks=[
                    lgb.early_stopping(int(args.early_stopping_rounds), verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            pred_va[:, c] = clf.predict_proba(x_va, num_iteration=clf.best_iteration_)[:, 1].astype(np.float32)
            pred_te[:, c] = clf.predict_proba(x_test, num_iteration=clf.best_iteration_)[:, 1].astype(np.float32)
            class_info.append(
                {
                    "class": cls,
                    "mode": "lgbm",
                    "best_iteration": int(clf.best_iteration_ or 0),
                    "train_pos": pos,
                    "train_neg": neg,
                }
            )

        oof[va_idx] = np.clip(pred_va, 0.0, 1.0)
        test_sum += np.clip(pred_te, 0.0, 1.0)

        teacher_fold = _macro_map_weighted(y_va, teacher_train[va_idx], sample_weight=sw_va)
        meta_fold = _macro_map_weighted(y_va, pred_va, sample_weight=sw_va)
        fold_reports.append(
            {
                "fold": int(fold_idx),
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
                "teacher_macro": float(teacher_fold),
                "meta_macro": float(meta_fold),
                "delta": float(meta_fold - teacher_fold),
                "classes": class_info,
            }
        )
        print(
            f"[meta] fold={fold_idx} teacher={teacher_fold:.6f} meta={meta_fold:.6f} delta={meta_fold - teacher_fold:+.6f}",
            flush=True,
        )

    if used_folds == 0:
        raise RuntimeError("No folds used; adjust --start-fold or --n-splits")

    test_pred = np.clip(test_sum / float(used_folds), 0.0, 1.0).astype(np.float32)

    # Global OOF metrics on used rows only.
    used_mask = np.zeros((len(train_ids),), dtype=bool)
    for fr in fold_reports:
        f = int(fr["fold"])
        _, va = folds[f]
        used_mask[np.asarray(va, dtype=np.int64)] = True

    y_used = y[used_mask]
    teacher_used = teacher_train[used_mask]
    meta_used = oof[used_mask]
    sw_used = sample_weight[used_mask] if sample_weight is not None else None
    teacher_mean = _macro_map_weighted(y_used, teacher_used, sample_weight=sw_used)
    meta_mean = _macro_map_weighted(y_used, meta_used, sample_weight=sw_used)
    corr = float(np.corrcoef(teacher_used.reshape(-1), meta_used.reshape(-1))[0, 1])

    fold_delta = [float(r["delta"]) for r in fold_reports]
    min_fold_delta = float(np.min(np.asarray(fold_delta, dtype=np.float64))) if fold_delta else 0.0

    # Save artifacts.
    np.save(art_dir / "meta_oof.npy", oof.astype(np.float32))
    np.save(art_dir / "meta_test.npy", test_pred.astype(np.float32))
    np.save(out_dir / "train_track_ids.npy", train_ids)
    np.save(out_dir / "test_track_ids.npy", test_ids)
    np.save(out_dir / "meta_feature_names.npy", np.asarray(feature_cols, dtype=object))

    # Align test to sample submission order.
    pos_test = {int(t): i for i, t in enumerate(test_ids)}
    test_aligned = np.stack([test_pred[pos_test[int(t)]] for t in sample_sub["track_id"].to_numpy(np.int64)], axis=0)
    sub = sample_sub.copy()
    sub[CLASSES] = test_aligned.astype(np.float32)
    sub_path = out_dir / "submission_meta_stack.csv"
    sub.to_csv(sub_path, index=False)

    report = {
        "train_csv": str(Path(args.train_csv).resolve()),
        "test_csv": str(Path(args.test_csv).resolve()),
        "sample_submission": str(Path(args.sample_submission).resolve()),
        "teacher_oof_csv": str(Path(args.teacher_oof_csv).resolve()),
        "teacher_test_csv": str(Path(args.teacher_test_csv).resolve()),
        "experts": {
            "names": sorted(expert_train.keys()),
            "fr_forward_weight": float(args.fr_forward_weight),
            "classwise_json": str(Path(args.classwise_json).resolve()) if str(args.classwise_json).strip() else "",
            "robust_oof_csv": str(Path(args.robust_oof_csv).resolve()),
        },
        "features": {
            "n_total": int(x_train.shape[1]),
            "n_tabular": int(tab_train.shape[1]) if tab_train is not None else 0,
            "n_expert": int(x_train.shape[1] - (tab_train.shape[1] if tab_train is not None else 0)),
            "uses_adversarial_pred": bool(adv_train is not None and adv_test is not None),
            "uses_tabular_features": bool(tab_train is not None),
        },
        "metrics": {
            "teacher_macro_mean": float(teacher_mean),
            "meta_macro_mean": float(meta_mean),
            "gain_vs_teacher": float(meta_mean - teacher_mean),
            "corr_meta_vs_teacher": float(corr),
            "min_fold_delta": float(min_fold_delta),
            "fold_reports": fold_reports,
        },
        "artifacts": {
            "meta_oof_npy": str((art_dir / "meta_oof.npy").resolve()),
            "meta_test_npy": str((art_dir / "meta_test.npy").resolve()),
            "submission_csv": str(sub_path.resolve()),
        },
        "config": {
            "seed": int(args.seed),
            "n_splits": int(args.n_splits),
            "start_fold": int(args.start_fold),
            "lgbm": {
                "num_leaves": int(args.num_leaves),
                "learning_rate": float(args.learning_rate),
                "feature_fraction": float(args.feature_fraction),
                "bagging_fraction": float(args.bagging_fraction),
                "bagging_freq": int(args.bagging_freq),
                "min_data_in_leaf": int(args.min_data_in_leaf),
                "lambda_l2": float(args.lambda_l2),
                "num_boost_round": int(args.num_boost_round),
                "early_stopping_rounds": int(args.early_stopping_rounds),
            },
        },
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print("=== META STACK COMPLETE ===", flush=True)
    print(f"output_dir: {out_dir}", flush=True)
    print(f"teacher_mean={teacher_mean:.6f} meta_mean={meta_mean:.6f} gain={meta_mean - teacher_mean:+.6f}", flush=True)
    print(f"corr={corr:.6f} min_fold_delta={min_fold_delta:+.6f}", flush=True)
    print(f"submission: {sub_path}", flush=True)
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
