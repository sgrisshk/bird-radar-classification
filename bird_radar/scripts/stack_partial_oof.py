from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.metrics import macro_map_score, one_hot_labels, per_class_average_precision


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


def _project01(x: np.ndarray) -> np.ndarray:
    return np.clip(x.astype(np.float32), 0.0, 1.0)


def _assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{name} contains NaN/Inf")


def _row_available(arr: np.ndarray, eps: float) -> np.ndarray:
    return np.sum(np.abs(arr), axis=1) > float(eps)


def _fit_binary(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    x_te: np.ndarray,
    learner: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_sum = int(np.sum(y_tr))
    if y_sum == 0 or y_sum == len(y_tr):
        const_p = float(np.mean(y_tr))
        return (
            np.full(len(x_va), const_p, dtype=np.float32),
            np.full(len(x_te), const_p, dtype=np.float32),
        )

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
        model.fit(x_tr, y_tr.astype(np.int32), callbacks=[lgb.log_evaluation(period=0)])
        p_va = model.predict_proba(x_va)[:, 1].astype(np.float32)
        p_te = model.predict_proba(x_te)[:, 1].astype(np.float32)
        return p_va, p_te

    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=3000,
        class_weight="balanced",
        random_state=int(seed),
    )
    model.fit(x_tr, y_tr.astype(np.int32))
    p_va = model.predict_proba(x_va)[:, 1].astype(np.float32)
    p_te = model.predict_proba(x_te)[:, 1].astype(np.float32)
    return p_va, p_te


def _build_meta_features(pred_map: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.hstack([pred_map[n].astype(np.float32) for n in names]).astype(np.float32)


def _load_predictions(map_paths: dict[str, Path]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, path in map_paths.items():
        out[name] = np.load(path).astype(np.float32)
    return out


def _mask_nonzero_all(oof_preds: dict[str, np.ndarray]) -> np.ndarray:
    mask = None
    for arr in oof_preds.values():
        row_nonzero = np.sum(np.abs(arr), axis=1) > 0.0
        mask = row_nonzero if mask is None else (mask & row_nonzero)
    if mask is None:
        raise RuntimeError("No OOF predictions supplied")
    return mask


def _mask_nonzero_union(oof_preds: dict[str, np.ndarray], eps: float) -> np.ndarray:
    mask = None
    for arr in oof_preds.values():
        row_nonzero = _row_available(arr, eps=eps)
        mask = row_nonzero if mask is None else (mask | row_nonzero)
    if mask is None:
        raise RuntimeError("No OOF predictions supplied")
    return mask


def _impute_fold_means(x: np.ndarray, avail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # x: [N, C], avail: [N]
    if x.ndim != 2:
        raise RuntimeError(f"Expected 2D x, got {x.shape}")
    if avail.ndim != 1 or avail.shape[0] != x.shape[0]:
        raise RuntimeError(f"Bad avail shape {avail.shape} for x {x.shape}")
    out = x.copy()
    if np.any(avail):
        means = x[avail].mean(axis=0).astype(np.float32)
    else:
        means = np.zeros((x.shape[1],), dtype=np.float32)
    out[~avail] = means
    return out, means


def _build_fold_meta_features(
    *,
    model_names: list[str],
    oof_cov_map: dict[str, np.ndarray],
    test_map: dict[str, np.ndarray],
    avail_cov_map: dict[str, np.ndarray],
    avail_test_map: dict[str, np.ndarray],
    tr_rel: np.ndarray,
    va_rel: np.ndarray,
    add_missing_flags: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_tr_parts: list[np.ndarray] = []
    x_va_parts: list[np.ndarray] = []
    x_te_parts: list[np.ndarray] = []

    for name in model_names:
        m_cov = oof_cov_map[name]
        m_test = test_map[name]
        av_cov = avail_cov_map[name]
        av_test = avail_test_map[name]

        x_tr_m, means = _impute_fold_means(m_cov[tr_rel], av_cov[tr_rel])
        x_va_m = m_cov[va_rel].copy()
        x_va_m[~av_cov[va_rel]] = means
        x_te_m = m_test.copy()
        x_te_m[~av_test] = means

        x_tr_parts.append(x_tr_m.astype(np.float32))
        x_va_parts.append(x_va_m.astype(np.float32))
        x_te_parts.append(x_te_m.astype(np.float32))

        if add_missing_flags:
            x_tr_parts.append(av_cov[tr_rel].astype(np.float32).reshape(-1, 1))
            x_va_parts.append(av_cov[va_rel].astype(np.float32).reshape(-1, 1))
            x_te_parts.append(av_test.astype(np.float32).reshape(-1, 1))

    x_tr = np.hstack(x_tr_parts).astype(np.float32)
    x_va = np.hstack(x_va_parts).astype(np.float32)
    x_te = np.hstack(x_te_parts).astype(np.float32)
    return x_tr, x_va, x_te


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--sample-submission", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--oof-input", action="append", default=[], help="name=path_to_oof.npy")
    parser.add_argument("--test-input", action="append", default=[], help="name=path_to_test.npy")
    parser.add_argument("--learner", type=str, default="logreg", choices=["logreg", "lgbm"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--group-col", type=str, default="observation_id")
    parser.add_argument("--label-col", type=str, default="bird_group")
    parser.add_argument("--mask-mode", type=str, default="nonzero_all", choices=["all", "nonzero_all", "union_nonzero"])
    parser.add_argument("--availability-eps", type=float, default=1e-12)
    parser.add_argument("--add-missing-flags", action="store_true", default=True)
    parser.add_argument("--no-missing-flags", dest="add_missing_flags", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    oof_paths = _parse_name_path_pairs(args.oof_input)
    test_paths = _parse_name_path_pairs(args.test_input)
    if sorted(oof_paths.keys()) != sorted(test_paths.keys()):
        raise RuntimeError("Model names in oof-input and test-input must match exactly")

    model_names = sorted(oof_paths.keys())
    oof_preds = _load_predictions(oof_paths)
    test_preds = _load_predictions(test_paths)

    n_train = next(iter(oof_preds.values())).shape[0]
    n_test = next(iter(test_preds.values())).shape[0]
    for n in model_names:
        if oof_preds[n].shape != (n_train, len(CLASSES)):
            raise RuntimeError(f"Bad OOF shape for {n}: {oof_preds[n].shape}")
        if test_preds[n].shape != (n_test, len(CLASSES)):
            raise RuntimeError(f"Bad test shape for {n}: {test_preds[n].shape}")
        _assert_finite(f"oof[{n}]", oof_preds[n])
        _assert_finite(f"test[{n}]", test_preds[n])

    train_df = pd.read_csv(args.train_csv, usecols=[args.label_col, args.group_col])
    if len(train_df) != n_train:
        raise RuntimeError(f"train-csv rows ({len(train_df)}) != oof rows ({n_train})")
    y_idx = train_df[args.label_col].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y_oh = one_hot_labels(y_idx, len(CLASSES)).astype(np.float32)
    groups = train_df[args.group_col].to_numpy()

    avail_oof = {name: _row_available(oof_preds[name], eps=float(args.availability_eps)) for name in model_names}
    avail_test = {name: _row_available(test_preds[name], eps=float(args.availability_eps)) for name in model_names}

    if args.mask_mode == "nonzero_all":
        covered_mask = np.ones(n_train, dtype=bool)
        for name in model_names:
            covered_mask &= avail_oof[name]
    elif args.mask_mode == "union_nonzero":
        covered_mask = _mask_nonzero_union(oof_preds, eps=float(args.availability_eps))
    else:
        covered_mask = np.ones(n_train, dtype=bool)

    covered_idx = np.where(covered_mask)[0]
    if len(covered_idx) == 0:
        raise RuntimeError("No covered rows after mask")

    oof_cov_map = {name: oof_preds[name][covered_idx] for name in model_names}
    avail_cov_map = {name: avail_oof[name][covered_idx] for name in model_names}

    y_cov = y_oh[covered_idx]
    y_idx_cov = y_idx[covered_idx]
    grp_cov = groups[covered_idx]

    base_scores: dict[str, Any] = {}
    for name in model_names:
        m_cov = covered_mask & avail_oof[name]
        if not np.any(m_cov):
            base_scores[name] = {
                "macro_map": 0.0,
                "per_class_ap": {c: 0.0 for c in CLASSES},
                "n_rows": 0,
                "coverage_ratio": 0.0,
            }
            continue
        p_cov = _project01(oof_preds[name][m_cov])
        y_model = y_oh[m_cov]
        base_scores[name] = {
            "macro_map": float(macro_map_score(y_model, p_cov)),
            "per_class_ap": per_class_average_precision(y_model, p_cov),
            "n_rows": int(np.sum(m_cov)),
            "coverage_ratio": float(np.sum(m_cov) / n_train),
        }

    splitter = StratifiedGroupKFold(
        n_splits=int(args.n_splits),
        shuffle=True,
        random_state=int(args.seed),
    )
    # split inputs are only labels + groups; no features are used to split.
    dummy = np.zeros((len(covered_idx), 1), dtype=np.float32)
    splits = list(splitter.split(dummy, y_idx_cov, groups=grp_cov))

    oof_cov = np.zeros((len(covered_idx), len(CLASSES)), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    fold_scores: list[dict[str, Any]] = []

    for fold_id, (tr_rel, va_rel) in enumerate(splits):
        y_tr = y_cov[tr_rel]
        y_va = y_cov[va_rel]

        grp_tr = grp_cov[tr_rel]
        grp_va = grp_cov[va_rel]
        if len(set(grp_tr).intersection(set(grp_va))) > 0:
            raise RuntimeError(f"Group leakage in fold {fold_id}")

        x_tr, x_va, x_te = _build_fold_meta_features(
            model_names=model_names,
            oof_cov_map=oof_cov_map,
            test_map=test_preds,
            avail_cov_map=avail_cov_map,
            avail_test_map=avail_test,
            tr_rel=tr_rel,
            va_rel=va_rel,
            add_missing_flags=bool(args.add_missing_flags),
        )

        fold_va = np.zeros((len(va_rel), len(CLASSES)), dtype=np.float32)
        fold_te = np.zeros((n_test, len(CLASSES)), dtype=np.float32)
        for class_idx in range(len(CLASSES)):
            p_va, p_te = _fit_binary(
                x_tr=x_tr,
                y_tr=y_tr[:, class_idx],
                x_va=x_va,
                x_te=x_te,
                learner=args.learner,
                seed=int(args.seed + fold_id * 113 + class_idx),
            )
            fold_va[:, class_idx] = p_va
            fold_te[:, class_idx] = p_te

        oof_cov[va_rel] = fold_va
        test_folds.append(fold_te)
        fold_scores.append(
            {
                "fold": int(fold_id),
                "macro_map": float(macro_map_score(y_va, fold_va)),
                "per_class_ap": per_class_average_precision(y_va, fold_va),
                "train_size": int(len(tr_rel)),
                "valid_size": int(len(va_rel)),
            }
        )

    test_meta_cv = np.mean(np.stack(test_folds, axis=0), axis=0).astype(np.float32)
    score_cov = float(macro_map_score(y_cov, oof_cov))

    # Fit full meta for a production test prediction.
    full_rel = np.arange(len(covered_idx), dtype=np.int64)
    x_full, _, x_test_full = _build_fold_meta_features(
        model_names=model_names,
        oof_cov_map=oof_cov_map,
        test_map=test_preds,
        avail_cov_map=avail_cov_map,
        avail_test_map=avail_test,
        tr_rel=full_rel,
        va_rel=full_rel[:1],
        add_missing_flags=bool(args.add_missing_flags),
    )
    test_meta_full = np.zeros((n_test, len(CLASSES)), dtype=np.float32)
    for class_idx in range(len(CLASSES)):
        p_dummy, p_te = _fit_binary(
            x_tr=x_full,
            y_tr=y_cov[:, class_idx],
            x_va=x_full[:1],
            x_te=x_test_full,
            learner=args.learner,
            seed=int(args.seed + 1000 + class_idx),
        )
        _ = p_dummy
        test_meta_full[:, class_idx] = p_te

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_full = np.zeros((n_train, len(CLASSES)), dtype=np.float32)
    oof_full[covered_idx] = oof_cov

    np.save(out_dir / "oof_stacked_covered.npy", oof_cov)
    np.save(out_dir / "oof_stacked_full.npy", oof_full)
    np.save(out_dir / "covered_idx.npy", covered_idx.astype(np.int64))
    np.save(out_dir / "test_stacked_cv.npy", _project01(test_meta_cv))
    np.save(out_dir / "test_stacked_full.npy", _project01(test_meta_full))

    sample_sub = pd.read_csv(args.sample_submission)
    sub = sample_sub.copy()
    sub[CLASSES] = _project01(test_meta_full)
    sub_path = out_dir / "submission_stacked.csv"
    sub.to_csv(sub_path, index=False)

    report = {
        "model_names": model_names,
        "learner": args.learner,
        "seed": int(args.seed),
        "n_splits": int(args.n_splits),
        "mask_mode": args.mask_mode,
        "availability_eps": float(args.availability_eps),
        "add_missing_flags": bool(args.add_missing_flags),
        "n_train_total": int(n_train),
        "n_train_covered": int(len(covered_idx)),
        "coverage_ratio": float(len(covered_idx) / max(n_train, 1)),
        "per_model_coverage": {
            name: {
                "train_rows_available": int(np.sum(avail_oof[name])),
                "train_ratio_available": float(np.mean(avail_oof[name])),
                "test_rows_available": int(np.sum(avail_test[name])),
                "test_ratio_available": float(np.mean(avail_test[name])),
            }
            for name in model_names
        },
        "base_scores_on_covered": base_scores,
        "stack_macro_map_on_covered": float(score_cov),
        "stack_per_class_ap_on_covered": per_class_average_precision(y_cov, oof_cov),
        "fold_scores": fold_scores,
        "fold_mean": float(np.mean([f["macro_map"] for f in fold_scores])) if fold_scores else 0.0,
        "fold_std": float(np.std([f["macro_map"] for f in fold_scores])) if fold_scores else 0.0,
        "submission_path": str(sub_path.resolve()),
        "artifacts": {
            "oof_covered": str((out_dir / "oof_stacked_covered.npy").resolve()),
            "oof_full": str((out_dir / "oof_stacked_full.npy").resolve()),
            "covered_idx": str((out_dir / "covered_idx.npy").resolve()),
            "test_cv": str((out_dir / "test_stacked_cv.npy").resolve()),
            "test_full": str((out_dir / "test_stacked_full.npy").resolve()),
        },
    }
    with (out_dir / "stack_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)

    print("=== STACKING COMPLETE ===", flush=True)
    print(f"covered={len(covered_idx)}/{n_train} ({len(covered_idx)/max(n_train,1):.2%})", flush=True)
    print(f"stack_macro_map_on_covered={score_cov:.6f}", flush=True)
    print(f"submission_path={sub_path}", flush=True)


if __name__ == "__main__":
    main()
