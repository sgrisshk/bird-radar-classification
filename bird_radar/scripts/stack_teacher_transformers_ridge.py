from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds
from src.metrics import macro_map_score, one_hot_labels, per_class_average_precision

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float32), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p)).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def entropy_multi(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float32), EPS, 1.0 - EPS)
    ent = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return ent.mean(axis=1, keepdims=True).astype(np.float32)


def _parse_run_items(items: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for raw in items:
        if "=" in raw:
            name, path = raw.split("=", 1)
            out.append((name.strip(), Path(path.strip())))
        else:
            p = Path(raw)
            out.append((p.name, p))
    if not out:
        raise ValueError("no run dirs provided")
    return out


def _resolve_ids_path(run_dir: Path, file_name: str) -> Path:
    p1 = run_dir / file_name
    if p1.exists():
        return p1
    p2 = run_dir / "artifacts" / file_name
    if p2.exists():
        return p2
    raise FileNotFoundError(f"cannot find {file_name} in {run_dir} or {run_dir / 'artifacts'}")


def _detect_pred_path(run_dir: Path, kind: str, explicit_name: str | None) -> Path:
    art = run_dir / "artifacts"
    if not art.exists():
        raise FileNotFoundError(f"artifacts dir missing: {art}")

    if explicit_name:
        p = art / explicit_name
        if p.exists():
            return p
        raise FileNotFoundError(f"missing explicit {kind} file: {p}")

    if kind == "oof":
        cands = sorted(art.glob("deep_transformer_heavy_seed*_oof.npy"))
        if len(cands) == 1:
            return cands[0]
        cands = sorted([p for p in art.glob("*_oof.npy") if "catboost" not in p.name.lower()])
    else:
        cands = sorted(art.glob("deep_transformer_heavy_seed*_test.npy"))
        if len(cands) == 1:
            return cands[0]
        cands = sorted([p for p in art.glob("*_test.npy") if "catboost" not in p.name.lower()])

    if len(cands) != 1:
        raise RuntimeError(f"cannot auto-detect {kind} npy in {art}; candidates={ [p.name for p in cands] }")
    return cands[0]


def _align_by_track_id(src_ids: np.ndarray, src_arr: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    idx = {int(tid): i for i, tid in enumerate(src_ids.tolist())}
    miss = [int(tid) for tid in target_ids.tolist() if int(tid) not in idx]
    if miss:
        raise ValueError(f"missing {len(miss)} track_ids in source array; first={miss[:10]}")
    take = np.array([idx[int(tid)] for tid in target_ids.tolist()], dtype=np.int64)
    return src_arr[take]


def _simplex_weight_grid(n_models: int, step: float, min_teacher: float) -> list[np.ndarray]:
    if n_models < 1:
        raise ValueError("n_models must be >= 1")
    if step <= 0.0 or step > 1.0:
        raise ValueError("step must be in (0, 1]")
    if min_teacher < 0.0 or min_teacher > 1.0:
        raise ValueError("min_teacher must be in [0, 1]")
    units = int(round(1.0 / step))
    if not math.isclose(units * step, 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError(f"step={step} must divide 1 exactly (e.g. 0.1, 0.05, 0.02)")
    min_teacher_units = int(math.ceil(min_teacher / step))
    min_teacher_units = max(0, min(units, min_teacher_units))

    out: list[np.ndarray] = []
    if n_models == 1:
        out.append(np.array([1.0], dtype=np.float32))
        return out

    def rec_fill(parts: list[int], left: int, k: int) -> None:
        if k == 1:
            parts.append(left)
            vec = np.array(parts, dtype=np.float32) * float(step)
            out.append(vec)
            parts.pop()
            return
        for v in range(left + 1):
            parts.append(v)
            rec_fill(parts, left - v, k - 1)
            parts.pop()

    for t_units in range(min_teacher_units, units + 1):
        rem = units - t_units
        rec_fill([t_units], rem, n_models - 1)
    return out


def _select_convex_anchor_weights(
    y_tr: np.ndarray,
    teacher_tr_probs: np.ndarray,
    run_tr_probs: list[np.ndarray],
    *,
    min_teacher: float,
    grid_step: float,
) -> tuple[np.ndarray, float]:
    pred_list = [teacher_tr_probs] + run_tr_probs
    grid = _simplex_weight_grid(len(pred_list), grid_step, min_teacher)
    best_w = grid[0]
    best_score = -1.0
    for w in grid:
        blend = np.zeros_like(teacher_tr_probs, dtype=np.float32)
        for i, p in enumerate(pred_list):
            blend += float(w[i]) * p
        score = float(macro_map_score(y_tr, blend))
        if score > best_score:
            best_score = score
            best_w = w.copy()
    return best_w, best_score


def _fit_predict_binary_probs(
    *,
    learner: str,
    x_tr: np.ndarray,
    y_tr_bin: np.ndarray,
    x_target: np.ndarray,
    seed: int,
    ridge_alpha: float,
) -> np.ndarray:
    y_sum = int(np.sum(y_tr_bin))
    if y_sum == 0 or y_sum == len(y_tr_bin):
        p_const = float(np.mean(y_tr_bin))
        return np.full((len(x_target),), p_const, dtype=np.float32)

    if learner == "logreg":
        m = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=4000,
            class_weight="balanced",
            random_state=int(seed),
        )
        m.fit(x_tr, y_tr_bin.astype(np.int32))
        p = m.predict_proba(x_target)[:, 1].astype(np.float32)
        return np.clip(p, 0.0, 1.0)

    # ridge path is treated as logits in this script
    m = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
    m.fit(x_tr, y_tr_bin.astype(np.float32))
    pred = m.predict(x_target).astype(np.float32)
    return np.clip(sigmoid(pred), 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--sample-submission", required=True)
    p.add_argument("--teacher-oof-csv", required=True)
    p.add_argument("--teacher-test-csv", required=True)
    p.add_argument("--run-dir", action="append", default=[], help="run_dir or name=run_dir; repeat")
    p.add_argument("--oof-npy-name", default=None, help="optional file name under run/artifacts")
    p.add_argument("--test-npy-name", default=None, help="optional file name under run/artifacts")
    p.add_argument(
        "--feature-mode",
        type=str,
        default="full",
        choices=["full", "teacher_plus_mean"],
        help="full=teacher+all-runs; teacher_plus_mean=teacher+mean(run logits)",
    )
    p.add_argument("--run-arrays-are-probs", action="store_true", default=True)
    p.add_argument("--run-arrays-are-logits", dest="run_arrays_are_probs", action="store_false")
    p.add_argument("--learner", type=str, default="ridge", choices=["ridge", "logreg", "convex_anchor", "teacher_passthrough"])
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--min-teacher", type=float, default=0.85, help="for convex_anchor: minimum teacher weight")
    p.add_argument("--grid-step", type=float, default=0.02, help="for convex_anchor: simplex grid step")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--split-mode", type=str, default="forward", choices=["forward", "stratified_group"])
    p.add_argument(
        "--uncovered-fill",
        type=str,
        default="teacher_fallback",
        choices=["teacher_fallback", "meta_backcast", "covered_mean", "zero"],
        help="How to fill forward-uncovered rows for full OOF metric/reporting",
    )
    p.add_argument("--group-col", type=str, default="observation_id")
    p.add_argument("--label-col", type=str, default="bird_group")
    p.add_argument("--timestamp-col", type=str, default="timestamp_start_radar_utc")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--out-name", default="submission_stacked_ridge.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs = _parse_run_items(args.run_dir) if args.run_dir else []

    train_df = pd.read_csv(
        args.train_csv,
        usecols=["track_id", args.group_col, args.label_col, args.timestamp_col],
    )
    y_idx = train_df[args.label_col].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y_oh = one_hot_labels(y_idx, len(CLASSES)).astype(np.float32)
    train_ids = train_df["track_id"].to_numpy(dtype=np.int64)
    groups = train_df[args.group_col].to_numpy()

    sample_sub = pd.read_csv(args.sample_submission, usecols=["track_id", *CLASSES])
    test_ids = sample_sub["track_id"].to_numpy(dtype=np.int64)

    teacher_oof_df = pd.read_csv(args.teacher_oof_csv, usecols=["track_id", *CLASSES])
    teacher_oof_df = pd.DataFrame({"track_id": train_ids}).merge(teacher_oof_df, on="track_id", how="left")
    if teacher_oof_df[CLASSES].isna().any().any():
        raise RuntimeError("teacher-oof-csv has missing track_ids after alignment")
    teacher_oof_probs = teacher_oof_df[CLASSES].to_numpy(dtype=np.float32)
    teacher_oof_logits = logit(teacher_oof_probs)

    teacher_test_df = pd.read_csv(args.teacher_test_csv, usecols=["track_id", *CLASSES])
    teacher_test_df = pd.DataFrame({"track_id": test_ids}).merge(teacher_test_df, on="track_id", how="left")
    if teacher_test_df[CLASSES].isna().any().any():
        raise RuntimeError("teacher-test-csv has missing track_ids after alignment")
    teacher_test_probs = teacher_test_df[CLASSES].to_numpy(dtype=np.float32)
    teacher_test_logits = logit(teacher_test_probs)

    run_oof_logits: list[np.ndarray] = []
    run_test_logits: list[np.ndarray] = []
    run_oof_probs: list[np.ndarray] = []
    run_test_probs: list[np.ndarray] = []
    run_info: list[dict[str, Any]] = []
    for name, run_dir in runs:
        run_dir = run_dir.resolve()
        tr_ids = np.load(_resolve_ids_path(run_dir, "train_track_ids.npy")).astype(np.int64)
        te_ids = np.load(_resolve_ids_path(run_dir, "test_track_ids.npy")).astype(np.int64)

        oof_p = _detect_pred_path(run_dir, kind="oof", explicit_name=args.oof_npy_name)
        test_p = _detect_pred_path(run_dir, kind="test", explicit_name=args.test_npy_name)
        oof_arr = np.load(oof_p).astype(np.float32)
        test_arr = np.load(test_p).astype(np.float32)

        if oof_arr.shape[1] != len(CLASSES) or test_arr.shape[1] != len(CLASSES):
            raise RuntimeError(f"class dim mismatch in {run_dir}: oof={oof_arr.shape}, test={test_arr.shape}")

        oof_arr = _align_by_track_id(tr_ids, oof_arr, train_ids)
        test_arr = _align_by_track_id(te_ids, test_arr, test_ids)

        if args.run_arrays_are_probs:
            oof_probs = np.clip(oof_arr, 0.0, 1.0).astype(np.float32)
            test_probs = np.clip(test_arr, 0.0, 1.0).astype(np.float32)
            oof_logits = logit(oof_probs)
            test_logits = logit(test_probs)
        else:
            oof_logits = oof_arr.astype(np.float32)
            test_logits = test_arr.astype(np.float32)
            oof_probs = sigmoid(oof_logits)
            test_probs = sigmoid(test_logits)

        run_oof_logits.append(oof_logits)
        run_test_logits.append(test_logits)
        run_oof_probs.append(oof_probs)
        run_test_probs.append(test_probs)
        run_info.append(
            {
                "name": name,
                "run_dir": str(run_dir),
                "oof_file": str(oof_p),
                "test_file": str(test_p),
                "oof_shape": list(oof_arr.shape),
                "test_shape": list(test_arr.shape),
            }
        )

    # Add teacher global uncertainty features.
    teacher_oof_conf = teacher_oof_probs.max(axis=1, keepdims=True)
    teacher_oof_ent = entropy_multi(teacher_oof_probs)
    teacher_test_conf = teacher_test_probs.max(axis=1, keepdims=True)
    teacher_test_ent = entropy_multi(teacher_test_probs)

    if args.feature_mode == "teacher_plus_mean":
        if run_oof_logits:
            run_oof_mean = np.mean(np.stack(run_oof_logits, axis=0), axis=0)
            run_test_mean = np.mean(np.stack(run_test_logits, axis=0), axis=0)
            x_train = np.concatenate(
                [teacher_oof_logits, run_oof_mean, teacher_oof_conf, teacher_oof_ent],
                axis=1,
            ).astype(np.float32)
            x_test = np.concatenate(
                [teacher_test_logits, run_test_mean, teacher_test_conf, teacher_test_ent],
                axis=1,
            ).astype(np.float32)
        else:
            x_train = np.concatenate([teacher_oof_logits, teacher_oof_conf, teacher_oof_ent], axis=1).astype(np.float32)
            x_test = np.concatenate([teacher_test_logits, teacher_test_conf, teacher_test_ent], axis=1).astype(np.float32)
    else:
        x_train = np.concatenate(
            [teacher_oof_logits] + run_oof_logits + [teacher_oof_conf, teacher_oof_ent],
            axis=1,
        ).astype(np.float32)
        x_test = np.concatenate(
            [teacher_test_logits] + run_test_logits + [teacher_test_conf, teacher_test_ent],
            axis=1,
        ).astype(np.float32)

    if args.split_mode == "forward":
        folds = make_forward_temporal_group_folds(
            train_df,
            timestamp_col=args.timestamp_col,
            group_col=args.group_col,
            n_splits=int(args.n_splits),
        )
    else:
        splitter = StratifiedGroupKFold(
            n_splits=int(args.n_splits),
            shuffle=True,
            random_state=int(args.seed),
        )
        split_dummy = np.zeros((len(train_ids), 1), dtype=np.float32)
        folds = list(splitter.split(split_dummy, y_idx, groups=groups))

    oof_pred_logits = np.zeros((len(train_ids), len(CLASSES)), dtype=np.float32)
    oof_pred_mask = np.zeros((len(train_ids),), dtype=bool)
    fold_scores: list[dict[str, Any]] = []
    convex_fold_weights: list[list[float]] = []

    for fold, (tr_idx, va_idx) in enumerate(folds):
        y_tr = y_oh[tr_idx]
        y_va = y_oh[va_idx]

        if args.learner == "teacher_passthrough":
            fold_va_prob = teacher_oof_probs[va_idx].astype(np.float32)
            fold_va_logits = logit(fold_va_prob)
        elif args.learner == "convex_anchor":
            teacher_tr = teacher_oof_probs[tr_idx]
            run_tr = [r[tr_idx] for r in run_oof_probs]
            best_w, _best_tr_score = _select_convex_anchor_weights(
                y_tr,
                teacher_tr,
                run_tr,
                min_teacher=float(args.min_teacher),
                grid_step=float(args.grid_step),
            )
            convex_fold_weights.append(best_w.astype(np.float32).tolist())

            fold_va_prob = best_w[0] * teacher_oof_probs[va_idx]
            for i, r in enumerate(run_oof_probs):
                fold_va_prob += best_w[i + 1] * r[va_idx]
            fold_va_prob = np.clip(fold_va_prob.astype(np.float32), 0.0, 1.0)
            fold_va_logits = logit(fold_va_prob)
        else:
            x_tr = x_train[tr_idx]
            x_va = x_train[va_idx]
            fold_va_logits = np.zeros((len(va_idx), len(CLASSES)), dtype=np.float32)
            for c in range(len(CLASSES)):
                y_bin = y_tr[:, c]
                pos = int(np.sum(y_bin))
                neg = int(len(y_bin) - pos)
                if pos == 0 or neg == 0:
                    p_const = float(np.mean(y_bin))
                    fold_va_logits[:, c] = logit(np.full((len(va_idx),), p_const, dtype=np.float32))
                    continue

                if args.learner == "logreg":
                    m = LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=int(args.seed + fold * 97 + c),
                    )
                    m.fit(x_tr, y_bin.astype(np.int32))
                    p_va = m.predict_proba(x_va)[:, 1].astype(np.float32)
                    fold_va_logits[:, c] = logit(p_va)
                else:
                    m = Ridge(alpha=float(args.ridge_alpha), fit_intercept=True)
                    m.fit(x_tr, y_bin)
                    fold_va_logits[:, c] = m.predict(x_va).astype(np.float32)
            fold_va_prob = np.clip(sigmoid(fold_va_logits), 0.0, 1.0)

        oof_pred_logits[va_idx] = fold_va_logits
        oof_pred_mask[va_idx] = True
        fold_scores.append(
            {
                "fold": int(fold),
                "macro_map": float(macro_map_score(y_va, fold_va_prob)),
                "per_class_ap": per_class_average_precision(y_va, fold_va_prob),
                "train_size": int(len(tr_idx)),
                "valid_size": int(len(va_idx)),
            }
        )

    # Fit full models for test.
    test_pred_logits = np.zeros((len(test_ids), len(CLASSES)), dtype=np.float32)
    convex_full_weights: list[float] | None = None
    if args.learner == "teacher_passthrough":
        test_pred_logits = logit(teacher_test_probs)
    elif args.learner == "convex_anchor":
        best_w, _best_tr_score = _select_convex_anchor_weights(
            y_oh,
            teacher_oof_probs,
            run_oof_probs,
            min_teacher=float(args.min_teacher),
            grid_step=float(args.grid_step),
        )
        convex_full_weights = best_w.astype(np.float32).tolist()
        test_pred_prob = best_w[0] * teacher_test_probs
        for i, r in enumerate(run_test_probs):
            test_pred_prob += best_w[i + 1] * r
        test_pred_prob = np.clip(test_pred_prob.astype(np.float32), 0.0, 1.0)
        test_pred_logits = logit(test_pred_prob)
    else:
        for c in range(len(CLASSES)):
            y_bin = y_oh[:, c]
            pos = int(np.sum(y_bin))
            neg = int(len(y_bin) - pos)
            if pos == 0 or neg == 0:
                p_const = float(np.mean(y_bin))
                test_pred_logits[:, c] = logit(np.full((len(test_ids),), p_const, dtype=np.float32))
                continue

            if args.learner == "logreg":
                m = LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=4000,
                    class_weight="balanced",
                    random_state=int(args.seed + 5000 + c),
                )
                m.fit(x_train, y_bin.astype(np.int32))
                p_te = m.predict_proba(x_test)[:, 1].astype(np.float32)
                test_pred_logits[:, c] = logit(p_te)
            else:
                m = Ridge(alpha=float(args.ridge_alpha), fit_intercept=True)
                m.fit(x_train, y_bin)
                test_pred_logits[:, c] = m.predict(x_test).astype(np.float32)

    oof_pred_prob = np.clip(sigmoid(oof_pred_logits), 0.0, 1.0)
    oof_pred_prob_with_teacher_fallback = oof_pred_prob.copy()
    oof_pred_prob_with_teacher_fallback[~oof_pred_mask] = teacher_oof_probs[~oof_pred_mask]
    uncovered_idx = np.where(~oof_pred_mask)[0].astype(np.int64)
    covered_idx = np.where(oof_pred_mask)[0].astype(np.int64)

    oof_pred_prob_full_filled = oof_pred_prob.copy()
    backcast_meta: dict[str, Any] = {}
    if len(uncovered_idx) > 0:
        if args.uncovered_fill == "teacher_fallback":
            oof_pred_prob_full_filled[uncovered_idx] = teacher_oof_probs[uncovered_idx]
            backcast_meta = {"mode": "teacher_fallback"}
        elif args.uncovered_fill == "covered_mean":
            if len(covered_idx) == 0:
                raise RuntimeError("covered_mean requested but covered_idx is empty")
            mean_vec = np.mean(oof_pred_prob[covered_idx], axis=0).astype(np.float32)
            oof_pred_prob_full_filled[uncovered_idx] = mean_vec[None, :]
            backcast_meta = {"mode": "covered_mean"}
        elif args.uncovered_fill == "zero":
            oof_pred_prob_full_filled[uncovered_idx] = 0.0
            backcast_meta = {"mode": "zero"}
        else:
            if args.split_mode != "forward":
                raise RuntimeError("meta_backcast is intended for split_mode=forward")
            if len(folds) == 0:
                raise RuntimeError("meta_backcast requested but folds are empty")

            # Train backcast model on late-history train chunk, excluding uncovered warmup rows.
            backcast_train_idx = np.setdiff1d(np.asarray(folds[-1][0], dtype=np.int64), uncovered_idx, assume_unique=False)
            if len(backcast_train_idx) == 0:
                backcast_train_idx = covered_idx.copy()
            if len(backcast_train_idx) == 0:
                raise RuntimeError("meta_backcast requested but backcast_train_idx is empty")

            if args.learner == "teacher_passthrough":
                oof_pred_prob_full_filled[uncovered_idx] = teacher_oof_probs[uncovered_idx]
                backcast_meta = {
                    "mode": "meta_backcast",
                    "effective_mode": "teacher_passthrough",
                    "backcast_train_size": int(len(backcast_train_idx)),
                    "backcast_valid_size": int(len(uncovered_idx)),
                }
            elif args.learner == "convex_anchor":
                teacher_tr = teacher_oof_probs[backcast_train_idx]
                run_tr = [r[backcast_train_idx] for r in run_oof_probs]
                backcast_w, backcast_tr_score = _select_convex_anchor_weights(
                    y_oh[backcast_train_idx],
                    teacher_tr,
                    run_tr,
                    min_teacher=float(args.min_teacher),
                    grid_step=float(args.grid_step),
                )
                backcast_pred = backcast_w[0] * teacher_oof_probs[uncovered_idx]
                for i, r in enumerate(run_oof_probs):
                    backcast_pred += backcast_w[i + 1] * r[uncovered_idx]
                oof_pred_prob_full_filled[uncovered_idx] = np.clip(backcast_pred.astype(np.float32), 0.0, 1.0)
                backcast_meta = {
                    "mode": "meta_backcast",
                    "effective_mode": "convex_anchor",
                    "backcast_train_size": int(len(backcast_train_idx)),
                    "backcast_valid_size": int(len(uncovered_idx)),
                    "backcast_weights": backcast_w.astype(np.float32).tolist(),
                    "backcast_train_macro_map": float(backcast_tr_score),
                }
            else:
                x_bc_tr = x_train[backcast_train_idx]
                x_bc_va = x_train[uncovered_idx]
                y_bc = y_oh[backcast_train_idx]
                pred_bc = np.zeros((len(uncovered_idx), len(CLASSES)), dtype=np.float32)
                for c in range(len(CLASSES)):
                    pred_bc[:, c] = _fit_predict_binary_probs(
                        learner=args.learner,
                        x_tr=x_bc_tr,
                        y_tr_bin=y_bc[:, c],
                        x_target=x_bc_va,
                        seed=int(args.seed + 7000 + c),
                        ridge_alpha=float(args.ridge_alpha),
                    )
                oof_pred_prob_full_filled[uncovered_idx] = np.clip(pred_bc, 0.0, 1.0)
                backcast_meta = {
                    "mode": "meta_backcast",
                    "effective_mode": str(args.learner),
                    "backcast_train_size": int(len(backcast_train_idx)),
                    "backcast_valid_size": int(len(uncovered_idx)),
                }
    else:
        backcast_meta = {"mode": "none", "reason": "no_uncovered_rows"}

    test_pred_prob = np.clip(sigmoid(test_pred_logits), 0.0, 1.0)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = sample_sub.copy()
    sub[CLASSES] = test_pred_prob
    sub_path = out_dir / args.out_name
    sub.to_csv(sub_path, index=False)

    np.save(out_dir / "oof_stacked_ridge_logits.npy", oof_pred_logits)
    np.save(out_dir / "oof_stacked_ridge_probs.npy", oof_pred_prob)
    np.save(out_dir / "oof_stacked_ridge_probs_full_filled.npy", oof_pred_prob_full_filled)
    np.save(out_dir / "test_stacked_ridge_logits.npy", test_pred_logits)
    np.save(out_dir / "test_stacked_ridge_probs.npy", test_pred_prob)

    teacher_oof_macro = float(macro_map_score(y_oh, teacher_oof_probs))
    stack_oof_macro = float(macro_map_score(y_oh, oof_pred_prob))
    if np.any(oof_pred_mask):
        stack_oof_macro_covered = float(macro_map_score(y_oh[oof_pred_mask], oof_pred_prob[oof_pred_mask]))
        teacher_oof_macro_covered = float(macro_map_score(y_oh[oof_pred_mask], teacher_oof_probs[oof_pred_mask]))
    else:
        stack_oof_macro_covered = 0.0
        teacher_oof_macro_covered = 0.0
    stack_oof_macro_fallback = float(macro_map_score(y_oh, oof_pred_prob_with_teacher_fallback))
    stack_oof_macro_full_filled = float(macro_map_score(y_oh, oof_pred_prob_full_filled))
    if fold_scores:
        teacher_fold_scores = []
        for fold_id, (tr_idx, va_idx) in enumerate(folds):
            teacher_fold_scores.append(
                {
                    "fold": int(fold_id),
                    "macro_map": float(macro_map_score(y_oh[va_idx], teacher_oof_probs[va_idx])),
                }
            )
    else:
        teacher_fold_scores = []
    report = {
        "n_train": int(len(train_ids)),
        "n_test": int(len(test_ids)),
        "n_classes": int(len(CLASSES)),
        "n_features": int(x_train.shape[1]),
        "n_runs": int(len(runs)),
        "feature_mode": str(args.feature_mode),
        "covered_rows": int(np.sum(oof_pred_mask)),
        "uncovered_rows": int(np.sum(~oof_pred_mask)),
        "covered_ratio": float(np.mean(oof_pred_mask)),
        "uncovered_fill": str(args.uncovered_fill),
        "run_arrays_are_probs": bool(args.run_arrays_are_probs),
        "learner": str(args.learner),
        "ridge_alpha": float(args.ridge_alpha),
        "n_splits": int(args.n_splits),
        "split_mode": str(args.split_mode),
        "seed": int(args.seed),
        "teacher_oof_csv": str(Path(args.teacher_oof_csv).resolve()),
        "teacher_test_csv": str(Path(args.teacher_test_csv).resolve()),
        "run_info": run_info,
        "teacher_oof_macro_map": teacher_oof_macro,
        "teacher_oof_macro_map_covered": teacher_oof_macro_covered,
        "stack_oof_macro_map": stack_oof_macro,
        "stack_oof_macro_map_covered": stack_oof_macro_covered,
        "stack_oof_macro_map_teacher_fallback": stack_oof_macro_fallback,
        "stack_oof_macro_map_full_filled": stack_oof_macro_full_filled,
        "stack_gain_vs_teacher": float(stack_oof_macro - teacher_oof_macro),
        "stack_gain_vs_teacher_covered": float(stack_oof_macro_covered - teacher_oof_macro_covered),
        "stack_gain_vs_teacher_fallback": float(stack_oof_macro_fallback - teacher_oof_macro),
        "stack_gain_vs_teacher_full_filled": float(stack_oof_macro_full_filled - teacher_oof_macro),
        "stack_per_class_ap": per_class_average_precision(y_oh, oof_pred_prob),
        "fold_scores": fold_scores,
        "teacher_fold_scores": teacher_fold_scores,
        "fold_mean": float(np.mean([f["macro_map"] for f in fold_scores])) if fold_scores else 0.0,
        "fold_std": float(np.std([f["macro_map"] for f in fold_scores])) if fold_scores else 0.0,
        "teacher_fold_mean": float(np.mean([f["macro_map"] for f in teacher_fold_scores])) if teacher_fold_scores else 0.0,
        "convex_fold_weights": convex_fold_weights,
        "convex_full_weights": convex_full_weights,
        "backcast_meta": backcast_meta,
        "submission_path": str(sub_path),
        "artifacts": {
            "oof_logits": str((out_dir / "oof_stacked_ridge_logits.npy").resolve()),
            "oof_probs": str((out_dir / "oof_stacked_ridge_probs.npy").resolve()),
            "oof_probs_full_filled": str((out_dir / "oof_stacked_ridge_probs_full_filled.npy").resolve()),
            "oof_probs_teacher_fallback": str((out_dir / "oof_stacked_ridge_probs_teacher_fallback.npy").resolve()),
            "test_logits": str((out_dir / "test_stacked_ridge_logits.npy").resolve()),
            "test_probs": str((out_dir / "test_stacked_ridge_probs.npy").resolve()),
        },
    }
    np.save(out_dir / "oof_stacked_ridge_probs_teacher_fallback.npy", oof_pred_prob_with_teacher_fallback)
    report_path = out_dir / "stack_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)

    print("=== STACK RIDGE COMPLETE ===", flush=True)
    print(f"stack_oof_macro_map={stack_oof_macro:.6f}", flush=True)
    print(f"stack_oof_macro_map_covered={stack_oof_macro_covered:.6f}", flush=True)
    print(f"stack_oof_macro_map_teacher_fallback={stack_oof_macro_fallback:.6f}", flush=True)
    print(f"stack_oof_macro_map_full_filled={stack_oof_macro_full_filled:.6f}", flush=True)
    print(f"teacher_oof_macro_map={teacher_oof_macro:.6f}", flush=True)
    print(f"teacher_oof_macro_map_covered={teacher_oof_macro_covered:.6f}", flush=True)
    print(f"stack_gain_vs_teacher={stack_oof_macro - teacher_oof_macro:.6f}", flush=True)
    print(f"stack_gain_vs_teacher_covered={stack_oof_macro_covered - teacher_oof_macro_covered:.6f}", flush=True)
    print(f"stack_gain_vs_teacher_fallback={stack_oof_macro_fallback - teacher_oof_macro:.6f}", flush=True)
    print(f"stack_gain_vs_teacher_full_filled={stack_oof_macro_full_filled - teacher_oof_macro:.6f}", flush=True)
    print(
        f"fold_mean={report['fold_mean']:.6f} teacher_fold_mean={report['teacher_fold_mean']:.6f} covered_ratio={report['covered_ratio']:.4f}",
        flush=True,
    )
    print(f"submission_path={sub_path}", flush=True)
    print(f"report_path={report_path}", flush=True)


if __name__ == "__main__":
    main()
