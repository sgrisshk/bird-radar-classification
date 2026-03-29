#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds

EPS = 1e-6


def _safe_logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    x = np.clip(p.astype(np.float32), eps, 1.0 - eps)
    return np.log(x / (1.0 - x)).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(x.astype(np.float32), -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _macro_map(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    vals: list[float] = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_prob[:, j]
        if yt.sum() <= 0:
            vals.append(0.0)
        else:
            vals.append(float(average_precision_score(yt, yp)))
    return float(np.mean(vals))


def _align_probs_from_csv(csv_path: str | Path, ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=["track_id", *CLASSES])
    mp = {int(r["track_id"]): np.asarray([r[c] for c in CLASSES], dtype=np.float32) for _, r in df.iterrows()}
    missing = [int(t) for t in ids if int(t) not in mp]
    if missing:
        raise ValueError(f"{csv_path} missing {len(missing)} ids; first={missing[:10]}")
    return np.stack([mp[int(t)] for t in ids], axis=0).astype(np.float32)


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
        raise ValueError(f"classwise reconstruction missing ids={len(missing)}")
    return np.stack([pred[pos[int(t)]] for t in ids_order], axis=0).astype(np.float32)


def _fit_one(
    model_type: str,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    x_te: np.ndarray,
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y_sum = int(np.sum(y_tr > 0.5))
    if y_sum == 0 or y_sum == len(y_tr):
        const_p = float(np.mean(y_tr))
        return (
            np.full((len(x_va),), const_p, dtype=np.float32),
            np.full((len(x_te),), const_p, dtype=np.float32),
            {"mode": "const", "const_p": const_p},
        )

    mu = np.mean(x_tr, axis=0, keepdims=True).astype(np.float32)
    sd = np.std(x_tr, axis=0, keepdims=True).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    tr = ((x_tr - mu) / sd).astype(np.float32)
    va = ((x_va - mu) / sd).astype(np.float32)
    te = ((x_te - mu) / sd).astype(np.float32)

    if model_type == "ridge":
        m = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
    elif model_type == "nnls":
        m = LinearRegression(fit_intercept=True, positive=True)
    else:
        raise ValueError(f"unsupported model_type={model_type}")

    m.fit(tr, y_tr.astype(np.float32))
    z_va = m.predict(va).astype(np.float32)
    z_te = m.predict(te).astype(np.float32)
    p_va = _sigmoid(z_va)
    p_te = _sigmoid(z_te)
    info: dict[str, Any] = {"mode": model_type}
    if hasattr(m, "coef_"):
        coef = np.asarray(m.coef_, dtype=np.float32).reshape(-1)
        info["coef"] = [float(x) for x in coef.tolist()]
    if hasattr(m, "intercept_"):
        info["intercept"] = float(np.asarray(m.intercept_).reshape(()))
    return p_va, p_te, info


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forward OOF per-class Ridge/NNLS logit stacking over expert predictions.")
    p.add_argument("--train-csv", default="train.csv")
    p.add_argument("--test-csv", default="test.csv")
    p.add_argument("--sample-submission", default="sample_submission.csv")
    p.add_argument("--output-dir", default="bird_radar/artifacts/meta_stack_ridge_logit_v1")
    p.add_argument("--model-type", choices=["ridge", "nnls"], default="ridge")
    p.add_argument("--ridge-alpha", type=float, default=100.0)
    p.add_argument("--target-smoothing", type=float, default=0.01)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--start-fold", type=int, default=1)
    p.add_argument("--seed", type=int, default=777)  # for reproducibility hooks

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
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = out_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv, usecols=["track_id", "bird_group", "timestamp_start_radar_utc", "observation_id"])
    test_df = pd.read_csv(args.test_csv, usecols=["track_id"])
    sample_sub = pd.read_csv(args.sample_submission, usecols=["track_id"])
    train_ids = train_df["track_id"].to_numpy(dtype=np.int64)
    test_ids = test_df["track_id"].to_numpy(dtype=np.int64)
    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = np.zeros((len(train_ids), len(CLASSES)), dtype=np.float32)
    y[np.arange(len(train_ids)), y_idx] = 1.0

    teacher_oof = _align_probs_from_csv(args.teacher_oof_csv, train_ids)
    teacher_test = _align_probs_from_csv(args.teacher_test_csv, test_ids)

    fwd_oof = _align_probs_from_csv(args.forward_oof_csv, train_ids)
    rev_oof = _align_probs_from_csv(args.reverse_oof_csv, train_ids)
    fwd_test = _align_probs_from_csv(args.forward_test_csv, test_ids)
    rev_test = _align_probs_from_csv(args.reverse_test_csv, test_ids)
    wf = float(args.fr_forward_weight)
    fr_oof = np.clip(wf * fwd_oof + (1.0 - wf) * rev_oof, 0.0, 1.0).astype(np.float32)
    fr_test = np.clip(wf * fwd_test + (1.0 - wf) * rev_test, 0.0, 1.0).astype(np.float32)

    classwise_oof = _reconstruct_classwise_oof_from_json(args.classwise_json, train_ids)
    classwise_test = _align_probs_from_csv(args.classwise_test_csv, test_ids)

    robust_oof = _align_probs_from_csv(args.robust_oof_csv, train_ids)
    robust_test = _align_probs_from_csv(args.robust_test_csv, test_ids)

    experts_oof = {
        "teacher_fr": teacher_oof,
        "fr35_65": fr_oof,
        "classwise_fr": classwise_oof,
        "robust_softmix": robust_oof,
    }
    experts_test = {
        "teacher_fr": teacher_test,
        "fr35_65": fr_test,
        "classwise_fr": classwise_test,
        "robust_softmix": robust_test,
    }

    # Fold map.
    folds = make_forward_temporal_group_folds(
        train_df,
        timestamp_col="timestamp_start_radar_utc",
        group_col="observation_id",
        n_splits=int(args.n_splits),
    )
    fold_id = np.full((len(train_ids),), -1, dtype=np.int64)
    for k, (_, va) in enumerate(folds):
        fold_id[np.asarray(va, dtype=np.int64)] = int(k)

    # Standalone expert diagnostics with exact same fold masks.
    expert_rows: list[dict[str, Any]] = []
    for name, p in experts_oof.items():
        per_fold = []
        for f in sorted(np.unique(fold_id)):
            if int(f) < int(args.start_fold):
                continue
            m = fold_id == f
            per_fold.append(_macro_map(y[m], p[m]))
        if per_fold:
            expert_rows.append(
                {
                    "name": name,
                    "fold_scores": [float(x) for x in per_fold],
                    "fold_mean": float(np.mean(per_fold)),
                    "fold_min": float(np.min(per_fold)),
                }
            )

    # Train OOF stack.
    oof = teacher_oof.copy()  # keep fallback for uncovered rows
    test_sum = np.zeros((len(test_ids), len(CLASSES)), dtype=np.float32)
    fold_reports: list[dict[str, Any]] = []
    n_used_folds = 0
    tgt_s = float(np.clip(args.target_smoothing, 1e-6, 0.49))

    # Pre-build per-class features from experts (only same-class logits).
    expert_names = list(experts_oof.keys())
    x_oof_class = {
        c: np.column_stack([_safe_logit(experts_oof[n][:, c]) for n in expert_names]).astype(np.float32)
        for c in range(len(CLASSES))
    }
    x_test_class = {
        c: np.column_stack([_safe_logit(experts_test[n][:, c]) for n in expert_names]).astype(np.float32)
        for c in range(len(CLASSES))
    }

    for f, (tr_idx, va_idx) in enumerate(folds):
        if int(f) < int(args.start_fold):
            continue
        n_used_folds += 1
        pred_va = np.zeros((len(va_idx), len(CLASSES)), dtype=np.float32)
        pred_te = np.zeros((len(test_ids), len(CLASSES)), dtype=np.float32)
        class_info: list[dict[str, Any]] = []
        for c, cls in enumerate(CLASSES):
            x_tr = x_oof_class[c][tr_idx]
            x_va = x_oof_class[c][va_idx]
            x_te = x_test_class[c]
            y_bin = y[:, c]
            y_tr = y_bin[tr_idx]
            # Logit target with smoothing to enforce logit-space linearity.
            y_tr_s = np.where(y_tr > 0.5, 1.0 - tgt_s, tgt_s).astype(np.float32)
            y_tr_logit = _safe_logit(y_tr_s)
            p_va, p_te, info = _fit_one(
                model_type=str(args.model_type),
                x_tr=x_tr,
                y_tr=y_tr_logit,
                x_va=x_va,
                x_te=x_te,
                ridge_alpha=float(args.ridge_alpha),
            )
            pred_va[:, c] = p_va
            pred_te[:, c] = p_te
            info["class"] = cls
            class_info.append(info)

        oof[va_idx] = np.clip(pred_va, 0.0, 1.0)
        test_sum += np.clip(pred_te, 0.0, 1.0)

        teacher_fold = _macro_map(y[va_idx], teacher_oof[va_idx])
        meta_fold = _macro_map(y[va_idx], pred_va)
        fold_reports.append(
            {
                "fold": int(f),
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
                "teacher_macro": float(teacher_fold),
                "meta_macro": float(meta_fold),
                "delta": float(meta_fold - teacher_fold),
                "classes": class_info,
            }
        )
        print(
            f"[ridge-meta] fold={f} teacher={teacher_fold:.6f} meta={meta_fold:.6f} delta={meta_fold - teacher_fold:+.6f}",
            flush=True,
        )

    if n_used_folds == 0:
        raise RuntimeError("No folds used; adjust --start-fold.")

    test_pred = np.clip(test_sum / float(n_used_folds), 0.0, 1.0).astype(np.float32)

    used_mask = np.zeros((len(train_ids),), dtype=bool)
    for fr in fold_reports:
        ff = int(fr["fold"])
        _, va = folds[ff]
        used_mask[np.asarray(va, dtype=np.int64)] = True

    teacher_mean = _macro_map(y[used_mask], teacher_oof[used_mask])
    meta_mean = _macro_map(y[used_mask], oof[used_mask])
    deltas = np.asarray([float(fr["delta"]) for fr in fold_reports], dtype=np.float64)
    min_fold_delta = float(np.min(deltas)) if len(deltas) else 0.0
    corr = float(np.corrcoef(teacher_oof[used_mask].reshape(-1), oof[used_mask].reshape(-1))[0, 1])

    # Save artifacts.
    np.save(art_dir / "meta_oof.npy", oof.astype(np.float32))
    np.save(art_dir / "meta_test.npy", test_pred.astype(np.float32))
    np.save(out_dir / "train_track_ids.npy", train_ids)
    np.save(out_dir / "test_track_ids.npy", test_ids)

    pos_test = {int(t): i for i, t in enumerate(test_ids)}
    test_aligned = np.stack([test_pred[pos_test[int(t)]] for t in sample_sub["track_id"].to_numpy(np.int64)], axis=0)
    sub = sample_sub.copy()
    sub[CLASSES] = test_aligned
    sub_path = out_dir / "submission_meta_ridge.csv"
    sub.to_csv(sub_path, index=False)

    # Sanity sample rows.
    sanity_rows: list[dict[str, Any]] = []
    for t in train_ids[:5]:
        i = int(np.where(train_ids == t)[0][0])
        top = int(np.argmax(teacher_oof[i]))
        sanity_rows.append(
            {
                "track_id": int(t),
                "y_true_class_id": int(y_idx[i]),
                "teacher_top1_class": CLASSES[top],
                "teacher_top1_prob": float(np.max(teacher_oof[i])),
            }
        )

    report = {
        "status": "ok",
        "model_type": str(args.model_type),
        "ridge_alpha": float(args.ridge_alpha),
        "target_smoothing": float(args.target_smoothing),
        "n_splits": int(args.n_splits),
        "start_fold": int(args.start_fold),
        "n_rows_oof": int(len(train_ids)),
        "n_rows_y": int(len(y)),
        "n_unique_track_id": int(len(np.unique(train_ids))),
        "fold_sizes": {int(f): int(np.sum(fold_id == f)) for f in np.unique(fold_id)},
        "experts": expert_rows,
        "metrics": {
            "teacher_macro_mean": float(teacher_mean),
            "meta_macro_mean": float(meta_mean),
            "gain_vs_teacher": float(meta_mean - teacher_mean),
            "min_fold_delta": float(min_fold_delta),
            "corr_meta_vs_teacher": float(corr),
            "fold_reports": fold_reports,
        },
        "sanity_sample": sanity_rows,
        "artifacts": {
            "meta_oof_npy": str((art_dir / "meta_oof.npy").resolve()),
            "meta_test_npy": str((art_dir / "meta_test.npy").resolve()),
            "submission_csv": str(sub_path.resolve()),
        },
        "inputs": {
            "teacher_oof_csv": str(Path(args.teacher_oof_csv).resolve()),
            "teacher_test_csv": str(Path(args.teacher_test_csv).resolve()),
            "classwise_json": str(Path(args.classwise_json).resolve()),
            "robust_oof_csv": str(Path(args.robust_oof_csv).resolve()),
            "robust_test_csv": str(Path(args.robust_test_csv).resolve()),
        },
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print("=== RIDGE/NNLS META STACK COMPLETE ===", flush=True)
    print(f"output_dir: {out_dir}", flush=True)
    print(f"teacher_mean={teacher_mean:.6f} meta_mean={meta_mean:.6f} gain={meta_mean - teacher_mean:+.6f}", flush=True)
    print(f"corr={corr:.6f} min_fold_delta={min_fold_delta:+.6f}", flush=True)
    print(f"submission: {sub_path}", flush=True)
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
