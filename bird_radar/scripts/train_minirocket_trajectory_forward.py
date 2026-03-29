#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds
from src.metrics import macro_map_score, per_class_average_precision
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MiniROCKET trajectory OVR model with forward CV.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--sample-submission", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-dir", default="bird_radar/artifacts/cache")
    p.add_argument("--out-name", default="submission_minirocket.csv")

    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--start-fold", type=int, default=0)
    p.add_argument("--time-col", default="timestamp_start_radar_utc")
    p.add_argument("--group-col", default="observation_id")
    p.add_argument("--id-col", default="track_id")

    p.add_argument("--seq-len", type=int, default=160)
    p.add_argument("--raw-clip-quantile-low", type=float, default=0.01)
    p.add_argument("--raw-clip-quantile-high", type=float, default=0.99)
    p.add_argument("--n-kernels", type=int, default=10_000)
    p.add_argument("--max-dilations-per-kernel", type=int, default=32)
    p.add_argument("--classifier", choices=["ridge", "logistic"], default="ridge")
    p.add_argument("--ridge-alphas", default="0.1,1.0,10.0,100.0")
    p.add_argument("--logreg-c", type=float, default=2.0)
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    p.add_argument("--sample-weights-npy", default="")

    p.add_argument("--min-class-count", type=int, default=5)
    p.add_argument(
        "--invalid-class-policy",
        choices=["fallback_prior", "skip_class", "skip_fold"],
        default="skip_class",
    )
    p.add_argument("--teacher-oof-csv", default="")
    return p.parse_args()


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    import json

    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict[str, Any]]:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


def _align_teacher_probs(csv_path: str, ids: np.ndarray, id_col: str) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=[id_col, *CLASSES])
    mp = {int(r[id_col]): np.asarray([r[c] for c in CLASSES], dtype=np.float32) for _, r in df.iterrows()}
    missing = [int(t) for t in ids.tolist() if int(t) not in mp]
    if missing:
        raise ValueError(f"teacher_oof_csv missing {len(missing)} ids; first={missing[:10]}")
    return np.stack([mp[int(t)] for t in ids.tolist()], axis=0).astype(np.float32)


def _safe_corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size == 0:
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    return c if np.isfinite(c) else 0.0


def _teacher_diag(y_true: np.ndarray, teacher: np.ndarray, model: np.ndarray) -> dict[str, float]:
    teacher = np.clip(teacher.astype(np.float32), 1e-6, 1.0 - 1e-6)
    model = np.clip(model.astype(np.float32), 1e-6, 1.0 - 1e-6)
    teacher_macro = float(macro_map_score(y_true, teacher))
    model_macro = float(macro_map_score(y_true, model))
    best = teacher_macro
    best_w = 1.0
    for w in [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]:
        blend = np.clip(w * teacher + (1.0 - w) * model, 0.0, 1.0)
        m = float(macro_map_score(y_true, blend))
        if m > best:
            best = m
            best_w = float(w)
    return {
        "teacher_macro": teacher_macro,
        "model_macro": model_macro,
        "corr_with_teacher": float(_safe_corr_flat(teacher, model)),
        "best_blend_macro": float(best),
        "best_blend_w_teacher": float(best_w),
        "best_blend_gain": float(best - teacher_macro),
    }


def _fit_raw_scaler(seqs: list[np.ndarray], q_low: float, q_high: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cat = np.concatenate([np.asarray(s, dtype=np.float32) for s in seqs if len(s) > 0], axis=0)
    if cat.ndim != 2 or cat.shape[1] != 9:
        raise ValueError(f"raw scaler expects [N,9], got {cat.shape}")
    lo = np.quantile(cat, q=float(q_low), axis=0).astype(np.float32)
    hi = np.quantile(cat, q=float(q_high), axis=0).astype(np.float32)
    span = hi - lo
    hi = np.where(span < 1e-6, lo + 1.0, hi)
    clipped = np.clip(cat, lo[None, :], hi[None, :])
    mean = np.mean(clipped, axis=0).astype(np.float32)
    std = np.std(clipped, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return lo, hi, mean, std


def _to_fixed_len(seq: np.ndarray, seq_len: int) -> np.ndarray:
    n, c = int(seq.shape[0]), int(seq.shape[1])
    if n >= seq_len:
        idx = np.linspace(0, n - 1, num=seq_len, dtype=np.float64)
        idx = np.clip(np.round(idx).astype(np.int64), 0, n - 1)
        return seq[idx].astype(np.float32, copy=False)
    out = np.zeros((seq_len, c), dtype=np.float32)
    out[:n] = seq.astype(np.float32, copy=False)
    return out


def _build_2stream_tensor(
    raw_seqs: list[np.ndarray],
    norm_seqs: list[np.ndarray],
    seq_len: int,
    raw_clip_lo: np.ndarray,
    raw_clip_hi: np.ndarray,
    raw_mean: np.ndarray,
    raw_std: np.ndarray,
) -> np.ndarray:
    out = np.zeros((len(raw_seqs), 18, int(seq_len)), dtype=np.float32)
    for i, (r, z) in enumerate(zip(raw_seqs, norm_seqs)):
        rr = np.asarray(r, dtype=np.float32)
        zz = np.asarray(z, dtype=np.float32)
        rr = np.clip(rr, raw_clip_lo[None, :], raw_clip_hi[None, :])
        rr = (rr - raw_mean[None, :]) / raw_std[None, :]
        rfix = _to_fixed_len(rr, int(seq_len))
        zfix = _to_fixed_len(zz, int(seq_len))
        x = np.concatenate([rfix, zfix], axis=1)  # [L,18]
        out[i] = x.T
    return out


def _predict_binary(
    classifier: str,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    x_te: np.ndarray,
    c: float,
    max_iter: int,
    class_weight: str,
    ridge_alphas: list[float],
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cw = None if class_weight == "none" else "balanced"
    if classifier == "logistic":
        clf = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=float(c),
            max_iter=int(max_iter),
            class_weight=cw,
            random_state=int(seed),
            n_jobs=1,
        )
        clf.fit(x_tr, y_tr, sample_weight=sample_weight)
        pv = clf.predict_proba(x_va)[:, 1].astype(np.float32)
        pt = clf.predict_proba(x_te)[:, 1].astype(np.float32)
        return pv, pt

    clf = RidgeClassifierCV(alphas=np.asarray(ridge_alphas, dtype=np.float64), class_weight=cw)
    clf.fit(x_tr, y_tr, sample_weight=sample_weight)
    sv = np.asarray(clf.decision_function(x_va), dtype=np.float32)
    st = np.asarray(clf.decision_function(x_te), dtype=np.float32)
    pv = (1.0 / (1.0 + np.exp(-np.clip(sv, -20.0, 20.0)))).astype(np.float32)
    pt = (1.0 / (1.0 + np.exp(-np.clip(st, -20.0, 20.0)))).astype(np.float32)
    return pv, pt


def main() -> None:
    args = parse_args()
    try:
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
    except Exception as exc:
        raise RuntimeError(
            "MiniROCKET requires sktime+numba in .venv. Run: .venv/bin/pip install sktime numba"
        ) from exc

    out_dir = Path(args.out_dir).resolve()
    art_dir = out_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    sample_sub = pd.read_csv(args.sample_submission, usecols=[str(args.id_col)])

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")

    train_ids = train_df[str(args.id_col)].to_numpy(dtype=np.int64)
    test_ids = test_df[str(args.id_col)].to_numpy(dtype=np.int64)
    np.save(out_dir / "train_track_ids.npy", train_ids)
    np.save(out_dir / "test_track_ids.npy", test_ids)

    raw_train = [np.asarray(train_cache[int(t)]["raw_features"], dtype=np.float32) for t in train_ids.tolist()]
    norm_train = [np.asarray(train_cache[int(t)]["features"], dtype=np.float32) for t in train_ids.tolist()]
    raw_test = [np.asarray(test_cache[int(t)]["raw_features"], dtype=np.float32) for t in test_ids.tolist()]
    norm_test = [np.asarray(test_cache[int(t)]["features"], dtype=np.float32) for t in test_ids.tolist()]

    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = np.zeros((len(train_ids), len(CLASSES)), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    np.save(out_dir / "oof_targets.npy", y.astype(np.float32))

    sample_weights: np.ndarray | None = None
    if str(args.sample_weights_npy).strip():
        sw = np.load(str(args.sample_weights_npy))
        sw = np.asarray(sw, dtype=np.float32).reshape(-1)
        if len(sw) != len(train_ids):
            raise ValueError(f"sample_weights length mismatch: got {len(sw)}, expected {len(train_ids)}")
        sample_weights = sw

    cv_df = pd.DataFrame(
        {
            "_cv_ts": train_df[str(args.time_col)],
            "_cv_group": train_df[str(args.group_col)].astype(np.int64),
        }
    )
    folds_all = make_forward_temporal_group_folds(
        cv_df, timestamp_col="_cv_ts", group_col="_cv_group", n_splits=int(args.n_splits)
    )
    if int(args.start_fold) > 0:
        s = min(int(args.start_fold), len(folds_all))
        folds = folds_all[s:]
        print(f"[minirocket] starting from fold index {s} (skipped {s} earliest folds)", flush=True)
    else:
        folds = folds_all
    if len(folds) == 0:
        raise RuntimeError("no folds selected")

    oof = np.zeros((len(train_ids), len(CLASSES)), dtype=np.float32)
    covered_mask = np.zeros((len(train_ids),), dtype=bool)
    test_accum = np.zeros((len(test_ids), len(CLASSES)), dtype=np.float32)
    fold_scores: list[float] = []
    fold_reports: list[dict[str, Any]] = []
    skipped_folds: list[int] = []

    ridge_alphas = [float(x.strip()) for x in str(args.ridge_alphas).split(",") if x.strip()]
    effective_folds = 0

    for fold_id, (tr_idx_raw, va_idx_raw) in enumerate(folds):
        tr_idx = np.asarray(tr_idx_raw, dtype=np.int64)
        va_idx = np.asarray(va_idx_raw, dtype=np.int64)

        invalid_classes: list[str] = []
        for c, cls in enumerate(CLASSES):
            pos = int(y[tr_idx, c].sum())
            neg = int(len(tr_idx) - pos)
            if min(pos, neg) < int(args.min_class_count):
                invalid_classes.append(cls)

        if invalid_classes and str(args.invalid_class_policy) == "skip_fold":
            skipped_folds.append(int(fold_id))
            fold_reports.append(
                {
                    "fold_id": int(fold_id),
                    "n_val": int(len(va_idx)),
                    "skipped": True,
                    "invalid_classes": invalid_classes,
                }
            )
            print(f"[fold] {fold_id + 1}/{len(folds)} skipped invalid_classes={len(invalid_classes)}", flush=True)
            continue

        raw_clip_lo, raw_clip_hi, raw_mean, raw_std = _fit_raw_scaler(
            seqs=[raw_train[int(i)] for i in tr_idx.tolist()],
            q_low=float(args.raw_clip_quantile_low),
            q_high=float(args.raw_clip_quantile_high),
        )

        xtr_3d = _build_2stream_tensor(
            raw_seqs=[raw_train[int(i)] for i in tr_idx.tolist()],
            norm_seqs=[norm_train[int(i)] for i in tr_idx.tolist()],
            seq_len=int(args.seq_len),
            raw_clip_lo=raw_clip_lo,
            raw_clip_hi=raw_clip_hi,
            raw_mean=raw_mean,
            raw_std=raw_std,
        )
        xva_3d = _build_2stream_tensor(
            raw_seqs=[raw_train[int(i)] for i in va_idx.tolist()],
            norm_seqs=[norm_train[int(i)] for i in va_idx.tolist()],
            seq_len=int(args.seq_len),
            raw_clip_lo=raw_clip_lo,
            raw_clip_hi=raw_clip_hi,
            raw_mean=raw_mean,
            raw_std=raw_std,
        )
        xte_3d = _build_2stream_tensor(
            raw_seqs=raw_test,
            norm_seqs=norm_test,
            seq_len=int(args.seq_len),
            raw_clip_lo=raw_clip_lo,
            raw_clip_hi=raw_clip_hi,
            raw_mean=raw_mean,
            raw_std=raw_std,
        )

        transformer = MiniRocketMultivariate(
            num_kernels=int(args.n_kernels),
            max_dilations_per_kernel=int(args.max_dilations_per_kernel),
            random_state=int(args.seed) + 97 * int(fold_id),
        )
        xtr = transformer.fit_transform(xtr_3d).to_numpy(dtype=np.float32, copy=False)
        xva = transformer.transform(xva_3d).to_numpy(dtype=np.float32, copy=False)
        xte = transformer.transform(xte_3d).to_numpy(dtype=np.float32, copy=False)

        fold_test = np.zeros((len(test_ids), len(CLASSES)), dtype=np.float32)
        for c, cls in enumerate(CLASSES):
            yt = y[tr_idx, c]
            pos = int(yt.sum())
            neg = int(len(tr_idx) - pos)
            if min(pos, neg) < int(args.min_class_count):
                prior = float(np.mean(yt)) if len(yt) > 0 else 0.5
                oof[va_idx, c] = prior
                fold_test[:, c] = prior
                continue

            sw_tr = sample_weights[tr_idx] if sample_weights is not None else None
            pv, pt = _predict_binary(
                classifier=str(args.classifier),
                x_tr=xtr,
                y_tr=yt.astype(np.int64),
                x_va=xva,
                x_te=xte,
                c=float(args.logreg_c),
                max_iter=int(args.max_iter),
                class_weight=str(args.class_weight),
                ridge_alphas=ridge_alphas,
                seed=int(args.seed) + 131 * int(fold_id) + 17 * c,
                sample_weight=sw_tr,
            )
            oof[va_idx, c] = pv
            fold_test[:, c] = pt

        test_accum += fold_test
        covered_mask[va_idx] = True
        effective_folds += 1
        fold_macro = float(macro_map_score(y[va_idx], oof[va_idx]))
        fold_scores.append(fold_macro)
        fold_reports.append(
            {
                "fold_id": int(fold_id),
                "n_val": int(len(va_idx)),
                "skipped": False,
                "invalid_classes": invalid_classes,
                "macro_map": fold_macro,
            }
        )
        print(
            f"[fold] {fold_id + 1}/{len(folds)} macro_map={fold_macro:.6f} n_val={len(va_idx)} invalid_classes={len(invalid_classes)}",
            flush=True,
        )

    if effective_folds <= 0:
        raise RuntimeError("all folds skipped; cannot produce predictions")
    test_accum /= float(effective_folds)

    covered_idx = np.where(covered_mask)[0].astype(np.int64)
    uncovered_idx = np.where(~covered_mask)[0].astype(np.int64)
    macro_raw_full = float(macro_map_score(y, oof))
    macro_covered = float(macro_map_score(y[covered_idx], oof[covered_idx])) if len(covered_idx) else 0.0
    per_covered = per_class_average_precision(y[covered_idx], oof[covered_idx]) if len(covered_idx) else {}

    teacher_diag: dict[str, float] = {}
    oof_full_filled = oof.copy()
    macro_full_filled_teacher = 0.0
    if str(args.teacher_oof_csv).strip():
        teacher = _align_teacher_probs(str(args.teacher_oof_csv), train_ids, id_col=str(args.id_col))
        if len(uncovered_idx) > 0:
            oof_full_filled[uncovered_idx] = teacher[uncovered_idx]
        macro_full_filled_teacher = float(macro_map_score(y, oof_full_filled))
        if len(covered_idx) > 0:
            teacher_diag = _teacher_diag(y_true=y[covered_idx], teacher=teacher[covered_idx], model=oof[covered_idx])
    np.save(out_dir / "oof_forward_cv_complete.npy", oof_full_filled.astype(np.float32))

    np.save(art_dir / "minirocket_oof.npy", oof.astype(np.float32))
    np.save(art_dir / "minirocket_test.npy", test_accum.astype(np.float32))

    sample_ids = sample_sub[str(args.id_col)].to_numpy(dtype=np.int64)
    if not np.array_equal(sample_ids, test_ids):
        pos = {int(tid): i for i, tid in enumerate(test_ids.tolist())}
        miss = [int(tid) for tid in sample_ids.tolist() if int(tid) not in pos]
        if miss:
            raise ValueError(f"sample_submission ids missing in test predictions; first={miss[:10]}")
        take = np.array([pos[int(tid)] for tid in sample_ids.tolist()], dtype=np.int64)
        pred_sub = test_accum[take]
    else:
        pred_sub = test_accum

    sub = pd.DataFrame({str(args.id_col): sample_ids})
    for i, cls in enumerate(CLASSES):
        sub[cls] = np.clip(pred_sub[:, i], 0.0, 1.0).astype(np.float32)
    sub_path = out_dir / str(args.out_name)
    sub.to_csv(sub_path, index=False)

    summary: dict[str, Any] = {
        "output_dir": str(out_dir),
        "seed": int(args.seed),
        "n_splits": int(len(folds)),
        "start_fold": int(args.start_fold),
        "seq_len": int(args.seq_len),
        "n_kernels": int(args.n_kernels),
        "max_dilations_per_kernel": int(args.max_dilations_per_kernel),
        "classifier": str(args.classifier),
        "logreg_c": float(args.logreg_c),
        "ridge_alphas": ridge_alphas,
        "max_iter": int(args.max_iter),
        "class_weight": str(args.class_weight),
        "sample_weights_npy": str(args.sample_weights_npy),
        "n_train": int(len(train_ids)),
        "n_test": int(len(test_ids)),
        "effective_folds": int(effective_folds),
        "skipped_folds": skipped_folds,
        "macro_map_raw_full": float(macro_raw_full),
        "macro_map_covered": float(macro_covered),
        "macro_map_full_filled_teacher": float(macro_full_filled_teacher),
        "covered_ratio": float(len(covered_idx) / len(train_ids)),
        "n_covered": int(len(covered_idx)),
        "n_uncovered": int(len(uncovered_idx)),
        "per_class_ap_covered": {k: float(v) for k, v in per_covered.items()},
        "fold_scores": [float(v) for v in fold_scores],
        "fold_mean": float(np.mean(fold_scores) if fold_scores else 0.0),
        "fold_worst": float(np.min(fold_scores) if fold_scores else 0.0),
        "fold_best": float(np.max(fold_scores) if fold_scores else 0.0),
        "fold_reports": fold_reports,
        "teacher_diag": teacher_diag,
        "models": {
            "minirocket_trajectory": {
                "type": "minirocket_trajectory",
                "oof_path": str((art_dir / "minirocket_oof.npy").resolve()),
                "test_path": str((art_dir / "minirocket_test.npy").resolve()),
                "macro_map_raw_full": float(macro_raw_full),
                "macro_map_covered": float(macro_covered),
                "macro_map_full_filled_teacher": float(macro_full_filled_teacher),
                "fold_scores": [float(v) for v in fold_scores],
            }
        },
        "track_ids_train_path": str((out_dir / "train_track_ids.npy").resolve()),
        "track_ids_test_path": str((out_dir / "test_track_ids.npy").resolve()),
        "submission_path": str(sub_path.resolve()),
    }
    if sample_weights is not None:
        summary["sample_weights_stats"] = {
            "min": float(np.min(sample_weights)),
            "max": float(np.max(sample_weights)),
            "mean": float(np.mean(sample_weights)),
            "p95": float(np.quantile(sample_weights, 0.95)),
        }
    dump_json(out_dir / "oof_summary.json", summary)

    print("=== MINIROCKET TRAJECTORY TRAIN COMPLETE ===", flush=True)
    print(f"output_dir={out_dir}", flush=True)
    print(f"oof_summary={out_dir / 'oof_summary.json'}", flush=True)
    print(
        f"macro_covered={summary['macro_map_covered']:.6f} "
        f"full_filled={summary['macro_map_full_filled_teacher']:.6f} "
        f"fold_mean={summary['fold_mean']:.6f} fold_worst={summary['fold_worst']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

