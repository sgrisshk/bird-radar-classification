from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.metrics import macro_map_score, one_hot_labels, per_class_average_precision


def _project01(x: np.ndarray) -> np.ndarray:
    return np.clip(x.astype(np.float32), 0.0, 1.0)


def _assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{name} has NaN/Inf")


def _load_run_artifacts(
    run_dir: Path,
    *,
    oof_name: str,
    idx_name: str,
    test_name: str,
    targets_name: str,
    train_ids_name: str,
    test_ids_name: str,
) -> dict[str, Any]:
    artifacts = run_dir / "artifacts"
    if not artifacts.is_dir():
        raise FileNotFoundError(f"Missing artifacts dir: {artifacts}")

    oof_path = artifacts / oof_name
    idx_path = artifacts / idx_name
    test_path = artifacts / test_name
    targets_path = artifacts / targets_name
    train_ids_path = artifacts / train_ids_name
    test_ids_path = artifacts / test_ids_name

    if not oof_path.exists():
        raise FileNotFoundError(f"Missing {oof_path}")
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing {idx_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}")
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing {targets_path}")
    if not train_ids_path.exists():
        raise FileNotFoundError(f"Missing {train_ids_path}")
    if not test_ids_path.exists():
        raise FileNotFoundError(f"Missing {test_ids_path}")

    oof = np.load(oof_path).astype(np.float32)
    idx = np.load(idx_path).astype(np.int64)
    test = np.load(test_path).astype(np.float32)
    targets = np.load(targets_path).astype(np.float32)
    train_ids = np.load(train_ids_path)
    test_ids = np.load(test_ids_path)

    _assert_finite(f"{run_dir.name}.oof", oof)
    _assert_finite(f"{run_dir.name}.test", test)
    _assert_finite(f"{run_dir.name}.targets", targets)

    if oof.ndim != 2 or oof.shape[1] != len(CLASSES):
        raise RuntimeError(f"{oof_path} bad shape {oof.shape}")
    if test.ndim != 2 or test.shape[1] != len(CLASSES):
        raise RuntimeError(f"{test_path} bad shape {test.shape}")
    if targets.shape != oof.shape:
        raise RuntimeError(f"{targets_path} shape {targets.shape} != {oof.shape}")

    avail = np.zeros(oof.shape[0], dtype=bool)
    avail[idx] = True
    return {
        "run_dir": str(run_dir),
        "name": run_dir.name,
        "oof": oof,
        "idx": idx,
        "test": test,
        "targets": targets,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "avail": avail,
    }


def _impute_by_train_availability(
    x_tr: np.ndarray,
    av_tr: np.ndarray,
    x_va: np.ndarray,
    av_va: np.ndarray,
    x_te: np.ndarray,
    av_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # x_*: [N, C], av_*: [N]
    if np.any(av_tr):
        mean_tr = x_tr[av_tr].mean(axis=0).astype(np.float32)
    else:
        mean_tr = np.zeros((x_tr.shape[1],), dtype=np.float32)

    x_tr_i = x_tr.copy()
    x_va_i = x_va.copy()
    x_te_i = x_te.copy()
    x_tr_i[~av_tr] = mean_tr
    x_va_i[~av_va] = mean_tr
    x_te_i[~av_te] = mean_tr
    return x_tr_i, x_va_i, x_te_i


def _fit_binary(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    x_te: np.ndarray,
    learner: str,
    seed: int,
    ridge_alpha: float,
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
        return model.predict_proba(x_va)[:, 1].astype(np.float32), model.predict_proba(x_te)[:, 1].astype(np.float32)

    if learner == "ridge":
        model = Ridge(alpha=float(ridge_alpha), random_state=int(seed))
        model.fit(x_tr, y_tr.astype(np.float32))
        return model.predict(x_va).astype(np.float32), model.predict(x_te).astype(np.float32)

    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=3000,
        class_weight="balanced",
        random_state=int(seed),
    )
    model.fit(x_tr, y_tr.astype(np.int32))
    return model.predict_proba(x_va)[:, 1].astype(np.float32), model.predict_proba(x_te)[:, 1].astype(np.float32)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True, type=str)
    ap.add_argument("--sample-submission", required=True, type=str)
    ap.add_argument("--run-dirs", nargs="+", required=True, type=str)
    ap.add_argument("--output-dir", required=True, type=str)
    ap.add_argument(
        "--oof-source",
        default="forward_cv",
        choices=["forward_cv", "forward_cv_complete", "holdout", "hybrid", "custom"],
    )
    ap.add_argument("--oof-name", default="oof_forward_cv.npy", type=str)
    ap.add_argument("--oof-idx-name", default="oof_forward_cv_idx.npy", type=str)
    ap.add_argument("--test-name", default="test_forward_cv_mean.npy", type=str)
    ap.add_argument("--targets-name", default="oof_targets.npy", type=str)
    ap.add_argument("--holdout-oof-name", default="oof_holdout.npy", type=str)
    ap.add_argument("--holdout-idx-name", default="oof_holdout_idx.npy", type=str)
    ap.add_argument("--train-ids-name", default="train_track_ids.npy", type=str)
    ap.add_argument("--test-ids-name", default="test_track_ids.npy", type=str)
    ap.add_argument("--coverage-mode", default="intersection", choices=["intersection", "union", "union_backfill"])
    ap.add_argument("--meta-train-scope", default="covered", choices=["covered", "full_missing_aware"])
    ap.add_argument("--learner", default="ridge", choices=["ridge", "logreg", "lgbm"])
    ap.add_argument("--ridge-alpha", default=1.0, type=float)
    ap.add_argument("--seed", default=2026, type=int)
    ap.add_argument("--n-splits", default=3, type=int)
    ap.add_argument("--group-col", default="observation_id", type=str)
    ap.add_argument("--label-col", default="bird_group", type=str)
    ap.add_argument("--id-col", default="track_id", type=str)
    ap.add_argument("--skip-missing-runs", action="store_true", default=False)
    ap.add_argument("--add-missing-flags", action="store_true", default=True)
    ap.add_argument("--no-missing-flags", dest="add_missing_flags", action="store_false")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.oof_source == "forward_cv":
        args.oof_name = "oof_forward_cv.npy"
        args.oof_idx_name = "oof_forward_cv_idx.npy"
        args.test_name = "test_forward_cv_mean.npy"
    elif args.oof_source == "forward_cv_complete":
        args.oof_name = "oof_forward_cv_complete.npy"
        args.oof_idx_name = "oof_forward_cv_complete_idx.npy"
        args.test_name = "test_forward_cv_complete_mean.npy"
    elif args.oof_source == "holdout":
        args.oof_name = "oof_holdout.npy"
        args.oof_idx_name = "oof_holdout_idx.npy"
        args.test_name = "test_full.npy"
    elif args.oof_source == "hybrid":
        args.oof_name = "oof_forward_cv.npy"
        args.oof_idx_name = "oof_forward_cv_idx.npy"
        args.test_name = "test_forward_cv_mean.npy"

    run_dirs = [Path(p).expanduser().resolve() for p in args.run_dirs]
    if len(run_dirs) < 2:
        raise RuntimeError("Need at least 2 run dirs for stacking")

    runs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for d in run_dirs:
        try:
            run_payload = _load_run_artifacts(
                    d,
                    oof_name=args.oof_name,
                    idx_name=args.oof_idx_name,
                    test_name=args.test_name,
                    targets_name=args.targets_name,
                    train_ids_name=args.train_ids_name,
                    test_ids_name=args.test_ids_name,
                )
            if args.oof_source == "hybrid":
                art = d / "artifacts"
                hold_oof_path = art / args.holdout_oof_name
                hold_idx_path = art / args.holdout_idx_name
                if not hold_oof_path.exists() or not hold_idx_path.exists():
                    raise FileNotFoundError(
                        f"Hybrid mode requires holdout files in {art}: "
                        f"{args.holdout_oof_name}, {args.holdout_idx_name}"
                    )
                hold_oof = np.load(hold_oof_path).astype(np.float32)
                hold_idx = np.load(hold_idx_path).astype(np.int64)
                if hold_oof.shape != run_payload["oof"].shape:
                    raise RuntimeError(
                        f"{d.name}: hybrid holdout shape {hold_oof.shape} != forward shape {run_payload['oof'].shape}"
                    )

                avail_forward = run_payload["avail"].copy()
                avail_hold = np.zeros_like(avail_forward, dtype=bool)
                avail_hold[hold_idx] = True
                fallback_mask = avail_hold & (~avail_forward)

                merged_oof = run_payload["oof"].copy()
                merged_oof[fallback_mask] = hold_oof[fallback_mask]
                merged_avail = avail_forward | avail_hold

                run_payload["oof_forward"] = run_payload["oof"].copy()
                run_payload["avail_forward"] = avail_forward
                run_payload["oof_holdout"] = hold_oof
                run_payload["avail_holdout"] = avail_hold
                run_payload["hybrid_added_count"] = int(np.sum(fallback_mask))
                run_payload["oof"] = merged_oof
                run_payload["avail"] = merged_avail

            runs.append(run_payload)
        except FileNotFoundError as e:
            if not args.skip_missing_runs:
                raise
            skipped.append({"run_dir": str(d), "reason": str(e)})
    if len(runs) < 2:
        raise RuntimeError(f"Need at least 2 valid runs after filtering; got {len(runs)}")

    n_train = runs[0]["oof"].shape[0]
    n_test = runs[0]["test"].shape[0]
    for r in runs[1:]:
        if r["oof"].shape[0] != n_train:
            raise RuntimeError(f"n_train mismatch for {r['name']}")
        if r["test"].shape[0] != n_test:
            raise RuntimeError(f"n_test mismatch for {r['name']}")
        if r["targets"].shape != runs[0]["targets"].shape:
            raise RuntimeError(f"targets shape mismatch for {r['name']}")

    train_df = pd.read_csv(args.train_csv, usecols=[args.id_col, args.label_col, args.group_col])
    if len(train_df) != n_train:
        raise RuntimeError(f"train-csv rows ({len(train_df)}) != n_train ({n_train})")
    train_ids_ref = train_df[args.id_col].to_numpy()

    sub_template = pd.read_csv(args.sample_submission)
    if args.id_col not in sub_template.columns:
        raise RuntimeError(f"sample_submission must contain id column '{args.id_col}'")
    for c in CLASSES:
        if c not in sub_template.columns:
            raise RuntimeError(f"sample_submission missing class column: {c}")
    if len(sub_template) != n_test:
        raise RuntimeError(f"sample_submission rows ({len(sub_template)}) != n_test ({n_test})")
    test_ids_ref = sub_template[args.id_col].to_numpy()

    # Align each run to canonical train/test id order.
    def _reindex_rows(arr: np.ndarray, src_ids: np.ndarray, dst_ids: np.ndarray, kind: str, run_name: str) -> np.ndarray:
        if len(src_ids) != arr.shape[0]:
            raise RuntimeError(f"{run_name}: {kind} ids len {len(src_ids)} != rows {arr.shape[0]}")
        if len(np.unique(src_ids)) != len(src_ids):
            raise RuntimeError(f"{run_name}: duplicate {kind} ids")
        idx_map = {int(k): i for i, k in enumerate(src_ids.tolist())}
        missing = [int(k) for k in dst_ids.tolist() if int(k) not in idx_map]
        if missing:
            raise RuntimeError(f"{run_name}: missing {len(missing)} {kind} ids in mapping")
        order = np.array([idx_map[int(k)] for k in dst_ids.tolist()], dtype=np.int64)
        return arr[order]

    for r in runs:
        r["oof"] = _reindex_rows(r["oof"], r["train_ids"], train_ids_ref, "train", r["name"])
        r["targets"] = _reindex_rows(r["targets"], r["train_ids"], train_ids_ref, "train", r["name"])
        r["avail"] = _reindex_rows(r["avail"].reshape(-1, 1).astype(np.float32), r["train_ids"], train_ids_ref, "train", r["name"]).reshape(-1) > 0.5
        r["test"] = _reindex_rows(r["test"], r["test_ids"], test_ids_ref, "test", r["name"])
        r["train_ids"] = train_ids_ref.copy()
        r["test_ids"] = test_ids_ref.copy()

    # All targets must match after canonical id alignment.
    for r in runs[1:]:
        if not np.array_equal(r["targets"], runs[0]["targets"]):
            raise RuntimeError(f"oof_targets mismatch between runs after id alignment: {runs[0]['name']} vs {r['name']}")

    if args.coverage_mode == "intersection":
        covered = np.ones(n_train, dtype=bool)
        for r in runs:
            covered &= r["avail"]
    else:
        covered = np.zeros(n_train, dtype=bool)
        for r in runs:
            covered |= r["avail"]
    covered_idx = np.where(covered)[0].astype(np.int64)
    if len(covered_idx) == 0:
        raise RuntimeError("Covered set is empty")

    if args.meta_train_scope == "full_missing_aware":
        train_scope_idx = np.arange(n_train, dtype=np.int64)
    else:
        train_scope_idx = covered_idx.copy()

    y_idx = train_df[args.label_col].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = one_hot_labels(y_idx, len(CLASSES)).astype(np.float32)
    y_cov = y[covered_idx]
    y_idx_cov = y_idx[covered_idx]
    g_cov = train_df[args.group_col].to_numpy()[covered_idx]
    y_trs = y[train_scope_idx]
    y_idx_trs = y_idx[train_scope_idx]
    g_trs = train_df[args.group_col].to_numpy()[train_scope_idx]

    base_scores: dict[str, Any] = {}
    for r in runs:
        m = covered & r["avail"]
        if np.any(m):
            base_scores[r["name"]] = {
                "macro_map": float(macro_map_score(y[m], _project01(r["oof"][m]))),
                "per_class_ap": per_class_average_precision(y[m], _project01(r["oof"][m])),
                "n_rows": int(np.sum(m)),
                "coverage_ratio": float(np.sum(m) / n_train),
            }
        else:
            base_scores[r["name"]] = {
                "macro_map": 0.0,
                "per_class_ap": {c: 0.0 for c in CLASSES},
                "n_rows": 0,
                "coverage_ratio": 0.0,
            }

    splitter = StratifiedGroupKFold(
        n_splits=int(args.n_splits),
        shuffle=True,
        random_state=int(args.seed),
    )
    splits = list(splitter.split(np.zeros((len(train_scope_idx), 1), dtype=np.float32), y_idx_trs, groups=g_trs))

    oof_scope = np.zeros((len(train_scope_idx), len(CLASSES)), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    fold_scores: list[dict[str, Any]] = []

    for fold_id, (tr_rel, va_rel) in enumerate(splits):
        gtr = set(g_trs[tr_rel].tolist())
        gva = set(g_trs[va_rel].tolist())
        if len(gtr.intersection(gva)) > 0:
            raise RuntimeError(f"Group leakage in fold {fold_id}")

        x_tr_parts: list[np.ndarray] = []
        x_va_parts: list[np.ndarray] = []
        x_te_parts: list[np.ndarray] = []

        for r in runs:
            oof_cov_r = r["oof"][train_scope_idx]
            av_cov_r = r["avail"][train_scope_idx]
            x_tr_i, x_va_i, x_te_i = _impute_by_train_availability(
                x_tr=oof_cov_r[tr_rel],
                av_tr=av_cov_r[tr_rel],
                x_va=oof_cov_r[va_rel],
                av_va=av_cov_r[va_rel],
                x_te=r["test"],
                av_te=np.ones(n_test, dtype=bool),  # test preds are expected to be dense
            )
            x_tr_parts.append(x_tr_i.astype(np.float32))
            x_va_parts.append(x_va_i.astype(np.float32))
            x_te_parts.append(x_te_i.astype(np.float32))

            if args.add_missing_flags:
                x_tr_parts.append(av_cov_r[tr_rel].astype(np.float32).reshape(-1, 1))
                x_va_parts.append(av_cov_r[va_rel].astype(np.float32).reshape(-1, 1))
                x_te_parts.append(np.ones((n_test, 1), dtype=np.float32))

        x_tr = np.hstack(x_tr_parts).astype(np.float32)
        x_va = np.hstack(x_va_parts).astype(np.float32)
        x_te = np.hstack(x_te_parts).astype(np.float32)

        y_tr = y_trs[tr_rel]
        y_va = y_trs[va_rel]
        pred_va = np.zeros((len(va_rel), len(CLASSES)), dtype=np.float32)
        pred_te = np.zeros((n_test, len(CLASSES)), dtype=np.float32)

        for c in range(len(CLASSES)):
            p_va, p_te = _fit_binary(
                x_tr=x_tr,
                y_tr=y_tr[:, c],
                x_va=x_va,
                x_te=x_te,
                learner=args.learner,
                seed=int(args.seed + fold_id * 101 + c),
                ridge_alpha=float(args.ridge_alpha),
            )
            pred_va[:, c] = p_va
            pred_te[:, c] = p_te

        pred_va = _project01(pred_va)
        pred_te = _project01(pred_te)
        oof_scope[va_rel] = pred_va
        test_folds.append(pred_te)

        fold_scores.append(
            {
                "fold": int(fold_id),
                "macro_map": float(macro_map_score(y_va, pred_va)),
                "per_class_ap": per_class_average_precision(y_va, pred_va),
                "train_size": int(len(tr_rel)),
                "valid_size": int(len(va_rel)),
            }
        )

    stack_scope = float(macro_map_score(y_trs, oof_scope))
    if args.meta_train_scope == "covered":
        stack_cov = stack_scope
    else:
        # covered-only diagnostic even when training on full scope
        pos_in_scope = np.full(n_train, -1, dtype=np.int64)
        pos_in_scope[train_scope_idx] = np.arange(len(train_scope_idx), dtype=np.int64)
        cov_rel = pos_in_scope[covered_idx]
        cov_rel = cov_rel[cov_rel >= 0]
        stack_cov = float(macro_map_score(y_cov, oof_scope[cov_rel]))
    test_meta = np.mean(np.stack(test_folds, axis=0), axis=0).astype(np.float32)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_full = np.zeros((n_train, len(CLASSES)), dtype=np.float32)
    oof_full[train_scope_idx] = oof_scope

    # Optional full-train diagnostic metric by deterministic backfill for uncovered rows.
    uncovered_idx = np.where(~covered)[0].astype(np.int64)
    oof_full_backfill = oof_full.copy()
    not_filled = np.where(np.sum(np.abs(oof_full_backfill), axis=1) == 0.0)[0].astype(np.int64)
    if len(not_filled) > 0:
        prior = y_trs.mean(axis=0).astype(np.float32) if len(train_scope_idx) > 0 else y.mean(axis=0).astype(np.float32)
        oof_full_backfill[not_filled] = prior
    stack_full_backfill = float(macro_map_score(y, _project01(oof_full_backfill)))

    uncovered_scores: dict[str, Any] = {}
    if len(uncovered_idx) > 0:
        y_unc = y[uncovered_idx]
        prior_unc = np.tile(y_trs.mean(axis=0, keepdims=True), (len(uncovered_idx), 1)).astype(np.float32)
        uncovered_scores["prior"] = {
            "macro_map": float(macro_map_score(y_unc, _project01(prior_unc))),
            "n_rows": int(len(uncovered_idx)),
        }
        for r in runs:
            avail_unc = r["avail"][uncovered_idx]
            if np.any(avail_unc):
                pred_unc = _project01(r["oof"][uncovered_idx][avail_unc])
                y_run_unc = y_unc[avail_unc]
                uncovered_scores[r["name"]] = {
                    "macro_map": float(macro_map_score(y_run_unc, pred_unc)),
                    "n_rows": int(np.sum(avail_unc)),
                    "coverage_on_uncovered": float(np.mean(avail_unc)),
                }
            else:
                uncovered_scores[r["name"]] = {
                    "macro_map": None,
                    "n_rows": 0,
                    "coverage_on_uncovered": 0.0,
                }

    np.save(out_dir / "oof_stacked_full.npy", oof_full)
    np.save(out_dir / "oof_stacked_covered.npy", oof_full[covered_idx])
    np.save(out_dir / "oof_stacked_scope.npy", oof_scope)
    np.save(out_dir / "train_scope_idx.npy", train_scope_idx)
    np.save(out_dir / "oof_stacked_full_backfill.npy", _project01(oof_full_backfill))
    np.save(out_dir / "covered_idx.npy", covered_idx)
    np.save(out_dir / "uncovered_idx.npy", uncovered_idx)
    np.save(out_dir / "test_stacked.npy", _project01(test_meta))

    sub = sub_template.copy()
    sub[CLASSES] = _project01(test_meta)
    sub_path = out_dir / "submission_stacked.csv"
    sub.to_csv(sub_path, index=False)

    report = {
        "runs": [str(d) for d in run_dirs],
        "runs_used": [r["name"] for r in runs],
        "runs_skipped": skipped,
        "oof_source": args.oof_source,
        "coverage_mode": args.coverage_mode,
        "meta_train_scope": args.meta_train_scope,
        "learner": args.learner,
        "ridge_alpha": float(args.ridge_alpha),
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "n_train_total": int(n_train),
        "n_train_covered": int(len(covered_idx)),
        "n_train_uncovered": int(len(uncovered_idx)),
        "n_train_scope": int(len(train_scope_idx)),
        "coverage_ratio": float(len(covered_idx) / n_train),
        "base_scores": base_scores,
        "hybrid_added_counts": {
            r["name"]: int(r.get("hybrid_added_count", 0))
            for r in runs
        },
        "uncovered_scores": uncovered_scores,
        "stack_macro_map_on_train_scope": float(stack_scope),
        "stack_macro_map_on_covered": float(stack_cov),
        "stack_per_class_ap_on_covered": per_class_average_precision(y_cov, oof_full[covered_idx]),
        "stack_macro_map_full_with_backfill": float(stack_full_backfill),
        "fold_scores": fold_scores,
        "fold_mean": float(np.mean([f["macro_map"] for f in fold_scores])) if fold_scores else 0.0,
        "fold_std": float(np.std([f["macro_map"] for f in fold_scores])) if fold_scores else 0.0,
        "artifacts": {
            "oof_stacked_full": str((out_dir / "oof_stacked_full.npy").resolve()),
            "oof_stacked_covered": str((out_dir / "oof_stacked_covered.npy").resolve()),
            "oof_stacked_scope": str((out_dir / "oof_stacked_scope.npy").resolve()),
            "train_scope_idx": str((out_dir / "train_scope_idx.npy").resolve()),
            "oof_stacked_full_backfill": str((out_dir / "oof_stacked_full_backfill.npy").resolve()),
            "covered_idx": str((out_dir / "covered_idx.npy").resolve()),
            "uncovered_idx": str((out_dir / "uncovered_idx.npy").resolve()),
            "test_stacked": str((out_dir / "test_stacked.npy").resolve()),
            "submission_path": str(sub_path.resolve()),
        },
    }
    with (out_dir / "stack_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)

    print("=== STACK TEMPORAL RUNS COMPLETE ===", flush=True)
    print(f"covered={len(covered_idx)}/{n_train} ({len(covered_idx)/n_train:.2%})", flush=True)
    print(f"stack_macro_map_on_covered={stack_cov:.6f}", flush=True)
    print(f"submission_path={sub_path}", flush=True)


if __name__ == "__main__":
    main()
