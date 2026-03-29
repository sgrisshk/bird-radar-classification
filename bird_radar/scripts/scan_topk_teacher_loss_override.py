#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds

EPS = 1e-6


def _parse_grid(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    arr = np.array(vals, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("grid is empty")
    return arr


def _one_hot(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.zeros((len(y_idx), n_classes), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    return y


def _topk_mask(score: np.ndarray, q_frac: float) -> tuple[np.ndarray, float]:
    n = int(score.shape[0])
    k = int(np.round(float(q_frac) * float(n)))
    k = max(1, min(n, k))
    order = np.argsort(score, kind="mergesort")
    sel = order[-k:]
    mask = np.zeros((n,), dtype=bool)
    mask[sel] = True
    thr = float(np.min(score[sel])) if k > 0 else float(np.max(score))
    return mask, thr


def _align_probs_from_csv(csv_path: str, ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=["track_id", *CLASSES]).set_index("track_id")
    return df.loc[ids, CLASSES].to_numpy(dtype=np.float32)


def _align_specialist_oof(npy_path: str, ids_path: str, target_ids: np.ndarray) -> np.ndarray:
    arr = np.load(npy_path).astype(np.float32)
    ids = np.load(ids_path).astype(np.int64)
    pos = {int(t): i for i, t in enumerate(ids.tolist())}
    missing = [int(t) for t in target_ids if int(t) not in pos]
    if missing:
        raise ValueError(f"specialist oof missing {len(missing)} ids; first={missing[:10]}")
    return np.stack([arr[pos[int(t)]] for t in target_ids], axis=0).astype(np.float32)


def _safe_logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, EPS, 1.0 - EPS)
    return np.log(pc / (1.0 - pc)).astype(np.float32)


def _make_bank_probs(
    teacher: np.ndarray,
    spec1: np.ndarray,
    spec2: np.ndarray | None,
    mode: str,
) -> np.ndarray:
    if spec2 is None or str(mode) == "single":
        return spec1.astype(np.float32)
    if str(mode) == "mean":
        return np.clip(0.5 * spec1 + 0.5 * spec2, 0.0, 1.0).astype(np.float32)
    p_t = np.clip(teacher, EPS, 1.0 - EPS)
    entropy = (-np.sum(p_t * np.log(p_t), axis=1, keepdims=True)).astype(np.float32)
    if str(mode) == "best_proxy_track":
        s1 = np.sum(np.abs(spec1 - teacher) * entropy, axis=1)
        s2 = np.sum(np.abs(spec2 - teacher) * entropy, axis=1)
        use1 = s1 >= s2
        out = spec2.copy()
        out[use1] = spec1[use1]
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    if str(mode) == "best_proxy_class":
        s1 = np.abs(spec1 - teacher) * entropy
        s2 = np.abs(spec2 - teacher) * entropy
        use1 = s1 >= s2
        out = np.where(use1, spec1, spec2)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"unknown bank-mode: {mode}")


def _apply_override(
    teacher: np.ndarray,
    spec: np.ndarray,
    use_mask: np.ndarray,
    alpha: float,
    class_idx: list[int],
    top_m_classes: int,
    delta_threshold: float = 0.0,
) -> np.ndarray:
    pred = teacher.copy()
    if not np.any(use_mask):
        return pred
    cls = np.asarray(class_idx, dtype=np.int64)
    lt = _safe_logit(teacher[:, cls])
    ls = _safe_logit(spec[:, cls])
    d = np.abs(ls - lt)
    rows = np.where(use_mask)[0]
    m = int(top_m_classes)
    dt = float(delta_threshold)
    for r in rows.tolist():
        order = np.argsort(d[r])[::-1]
        cnt = 0
        for k in order.tolist():
            if dt > 0.0 and float(d[r, k]) < dt:
                break
            j = int(cls[k])
            pred[r, j] = np.clip((1.0 - alpha) * teacher[r, j] + alpha * spec[r, j], 0.0, 1.0)
            cnt += 1
            if m > 0 and cnt >= m:
                break
    return pred


def _compute_class_strength(
    y: np.ndarray,
    teacher: np.ndarray,
    spec: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    sample_weights: np.ndarray | None,
    class_idx: list[int],
    source: str,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for j in class_idx:
        yt = y[:, j]
        sw = sample_weights
        ap_t = float(average_precision_score(yt, teacher[:, j], sample_weight=sw)) if yt.sum() > 0 else 0.0
        ap_s = float(average_precision_score(yt, spec[:, j], sample_weight=sw)) if yt.sum() > 0 else 0.0
        ap_delta = ap_s - ap_t
        if str(source) == "ap_delta":
            out[j] = max(0.0, float(ap_delta))
            continue
        deltas: list[float] = []
        fw: list[float] = []
        for _, va_idx in folds:
            va = np.asarray(va_idx, dtype=np.int64)
            yv = y[va, j]
            if yv.sum() <= 0:
                d = 0.0
            else:
                swv = sample_weights[va] if sample_weights is not None else None
                at = float(average_precision_score(yv, teacher[va, j], sample_weight=swv))
                as_ = float(average_precision_score(yv, spec[va, j], sample_weight=swv))
                d = as_ - at
            deltas.append(float(d))
            fw.append(float(sample_weights[va].sum()) if sample_weights is not None else float(len(va)))
        if len(deltas) == 0:
            out[j] = 0.0
        else:
            out[j] = max(0.0, float(np.average(np.asarray(deltas, dtype=np.float64), weights=np.asarray(fw, dtype=np.float64))))
    return out


def _budget_from_strength(
    strength: dict[int, float],
    class_idx: list[int],
    total_budget: int,
    tau: float,
    floor_min: int,
    floor_topk: int,
) -> dict[int, int]:
    total_budget = int(max(0, total_budget))
    if total_budget <= 0 or len(class_idx) == 0:
        return {j: 0 for j in class_idx}
    vals = np.asarray([float(strength.get(j, 0.0)) for j in class_idx], dtype=np.float64)
    t = max(float(tau), 1e-6)
    z = vals / t
    z = z - float(np.max(z))
    w = np.exp(z)
    if float(np.sum(w)) <= 0:
        w = np.ones_like(w)
    w = w / float(np.sum(w))
    raw = w * float(total_budget)
    b = np.floor(raw).astype(np.int64)
    rem = int(total_budget - int(np.sum(b)))
    if rem > 0:
        frac_order = np.argsort(-(raw - b))
        for k in frac_order[:rem]:
            b[k] += 1

    if int(floor_min) > 0 and int(floor_topk) > 0:
        positive = np.where(vals > 0.0)[0]
        if len(positive) > 0:
            rank = positive[np.argsort(-vals[positive])]
            k = min(int(floor_topk), len(rank))
            eff_floor = min(int(floor_min), total_budget // max(1, k))
            for ridx in rank[:k]:
                if b[ridx] < eff_floor:
                    b[ridx] = eff_floor

    while int(np.sum(b)) > total_budget:
        order = np.argsort(vals)
        changed = False
        for ridx in order:
            if b[ridx] > 0:
                b[ridx] -= 1
                changed = True
                if int(np.sum(b)) <= total_budget:
                    break
        if not changed:
            break
    while int(np.sum(b)) < total_budget:
        order = np.argsort(-vals)
        b[order[0]] += 1

    return {j: int(bi) for j, bi in zip(class_idx, b.tolist())}


def _apply_override_with_budget(
    teacher: np.ndarray,
    spec: np.ndarray,
    use_mask: np.ndarray,
    alpha: float,
    class_idx: list[int],
    top_m_classes: int,
    class_budget: dict[int, int],
    uncertainty: np.ndarray,
    delta_threshold: float = 0.0,
) -> tuple[np.ndarray, dict[int, int], int, float]:
    pred = teacher.copy()
    if not np.any(use_mask) or len(class_idx) == 0:
        return pred, dict(class_budget), 0, 0.0

    cls = np.asarray(class_idx, dtype=np.int64)
    budget = {int(j): int(class_budget.get(int(j), 0)) for j in cls.tolist()}
    lt = _safe_logit(teacher[:, cls])
    ls = _safe_logit(spec[:, cls])
    d = np.abs(ls - lt)

    idx = np.where(use_mask)[0]
    idx = idx[np.argsort(-uncertainty[idx])]
    applied_ops = 0
    track_applied = np.zeros((teacher.shape[0],), dtype=bool)
    m = int(top_m_classes)
    dt = float(delta_threshold)
    for r in idx.tolist():
        order = np.argsort(-d[r])
        cnt = 0
        for k in order.tolist():
            if dt > 0.0 and float(d[r, k]) < dt:
                break
            j = int(cls[k])
            if budget.get(j, 0) <= 0:
                continue
            pred[r, j] = np.clip((1.0 - alpha) * teacher[r, j] + alpha * spec[r, j], 0.0, 1.0)
            budget[j] -= 1
            applied_ops += 1
            cnt += 1
            track_applied[r] = True
            if m > 0 and cnt >= m:
                break
    changed_track_frac = float(track_applied.mean())
    return pred, budget, int(applied_ops), changed_track_frac


def _macro_map(y_true: np.ndarray, y_prob: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    vals = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        vals.append(0.0 if yt.sum() <= 0 else float(average_precision_score(yt, yp, sample_weight=sample_weight)))
    return float(np.mean(vals))


def _fold_eval(
    y: np.ndarray,
    baseline: np.ndarray,
    pred: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    sample_weights: np.ndarray | None = None,
) -> tuple[float, float, float, dict[int, float], list[dict[str, float | int]]]:
    rows = []
    for fid, (_, va_idx) in enumerate(folds):
        yt = y[va_idx]
        pb = baseline[va_idx]
        pr = pred[va_idx]
        sw = sample_weights[va_idx] if sample_weights is not None else None
        cov = (pr.sum(axis=1) > 0.0)
        if int(cov.sum()) == 0:
            rows.append(
                {
                    "fold": int(fid),
                    "n_cov": 0,
                    "baseline_macro": 0.0,
                    "pred_macro": 0.0,
                    "fold_delta": 0.0,
                    "fold_weight": 0.0,
                }
            )
            continue
        yt = yt[cov]
        pb = pb[cov]
        pr = pr[cov]
        sw_cov = sw[cov] if sw is not None else None
        b = _macro_map(yt, pb, sample_weight=sw_cov)
        p = _macro_map(yt, pr, sample_weight=sw_cov)
        rows.append(
            {
                "fold": int(fid),
                "n_cov": int(cov.sum()),
                "baseline_macro": float(b),
                "pred_macro": float(p),
                "fold_delta": float(p - b),
                "fold_weight": float(sw_cov.sum()) if sw_cov is not None else float(int(cov.sum())),
            }
        )
    valid = [r for r in rows if int(r["n_cov"]) > 0]
    if not valid:
        return 0.0, 0.0, 0.0, {}, rows
    if sample_weights is not None:
        fw = np.asarray([float(r["fold_weight"]) for r in valid], dtype=np.float64)
        b_mean = float(np.average([float(r["baseline_macro"]) for r in valid], weights=fw))
        p_mean = float(np.average([float(r["pred_macro"]) for r in valid], weights=fw))
    else:
        b_mean = float(np.mean([float(r["baseline_macro"]) for r in valid]))
        p_mean = float(np.mean([float(r["pred_macro"]) for r in valid]))
    min_delta = float(np.min([float(r["fold_delta"]) for r in valid]))
    per_fold = {int(r["fold"]): float(r["fold_delta"]) for r in valid}
    return b_mean, p_mean, min_delta, per_fold, rows


def _bce_row(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return (-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))).mean(axis=1).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan top-q teacher-loss selective override with specialist.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--teacher-oof-csv", required=True)
    p.add_argument("--teacher-test-csv", required=True)
    p.add_argument("--spec-oof-npy", required=True)
    p.add_argument("--spec-track-ids-npy", required=True)
    p.add_argument("--spec-test-csv", required=True)
    p.add_argument("--spec2-oof-npy", default="")
    p.add_argument("--spec2-track-ids-npy", default="")
    p.add_argument("--spec2-test-csv", default="")
    p.add_argument("--bank-mode", choices=["single", "mean", "best_proxy_track", "best_proxy_class"], default="single")
    p.add_argument("--top-m-classes", type=int, default=0, help="Override only top-M classes per selected track by |delta_logit|; 0=all.")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--time-col", default="timestamp_start_radar_utc")
    p.add_argument("--group-col", default="observation_id")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--target-fold-index", type=int, default=2)

    p.add_argument("--q-grid", default="0.01,0.02,0.03,0.05,0.08,0.12")
    p.add_argument("--alpha-grid", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    p.add_argument("--delta-threshold-grid", default="0.0", help="Logit |delta| threshold grid for adaptive class override per track.")
    p.add_argument("--oof-selector", choices=["teacher_loss", "teacher_entropy", "policy_score"], default="teacher_loss")
    p.add_argument("--selector-oof-npy", type=str, default="", help="OOF selector score per track (required for oof-selector=policy_score).")
    p.add_argument("--selector-test-npy", type=str, default="", help="Test selector score per track (required for oof-selector=policy_score).")
    p.add_argument("--only-if-better", action="store_true", help="Apply blend only when specialist row BCE < teacher row BCE (OOF only; test uses top-q mask).")
    p.add_argument("--min-fold-delta", type=float, default=-0.002)
    p.add_argument("--max-target-fold-drop", type=float, default=-0.002)
    p.add_argument("--sample-weights-npy", type=str, default="")
    p.add_argument(
        "--selection-objective",
        choices=["gain", "robust"],
        default="gain",
        help="How to pick best feasible row: by weighted gain or minimax robust score.",
    )
    p.add_argument(
        "--time-proxy-bins",
        type=int,
        default=5,
        help="Number of quantile bins for time-equalized proxy weights.",
    )
    p.add_argument("--max-override-rate", type=float, default=1.0)
    p.add_argument("--min-override-rate", type=float, default=0.0)
    p.add_argument("--max-override-rate-test", type=float, default=1.0)
    p.add_argument("--min-override-rate-test", type=float, default=0.0)
    p.add_argument("--override-classes", type=str, default="", help="Comma list of classes to override; empty=all.")
    p.add_argument("--per-class-budget", action="store_true")
    p.add_argument("--budget-strength-source", choices=["ap_delta", "foldmean_delta"], default="ap_delta")
    p.add_argument("--budget-tau", type=float, default=0.02)
    p.add_argument("--budget-floor-min", type=int, default=0)
    p.add_argument("--budget-floor-topk", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv, usecols=["track_id", "bird_group", str(args.time_col), str(args.group_col)])
    test_df = pd.read_csv(args.test_csv, usecols=["track_id"])
    train_ids = train_df["track_id"].to_numpy(dtype=np.int64)
    test_ids = test_df["track_id"].to_numpy(dtype=np.int64)
    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = _one_hot(y_idx, len(CLASSES))
    sample_weights: np.ndarray | None = None
    if str(args.sample_weights_npy).strip():
        sample_weights = np.asarray(np.load(args.sample_weights_npy), dtype=np.float32).reshape(-1)
        if len(sample_weights) != len(train_df):
            raise ValueError(f"sample_weights length mismatch: {len(sample_weights)} vs train {len(train_df)}")
        sample_weights = np.clip(sample_weights, 1e-8, None).astype(np.float32)
    # Time-equalized proxy: inverse-frequency weights by quantile time bins.
    ts = pd.to_datetime(train_df[str(args.time_col)], errors="coerce", utc=True)
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"time parsing failed for {bad} rows in {args.time_col}")
    bins_req = max(2, int(args.time_proxy_bins))
    q = np.linspace(0.0, 1.0, bins_req + 1)
    ts_ns = ts.astype("int64").to_numpy(dtype=np.int64)
    edges = np.quantile(ts_ns, q)
    edges = np.unique(edges)
    if edges.size < 3:
        time_proxy_bins = 1
        time_weights = np.ones((len(train_df),), dtype=np.float32)
    else:
        b = np.digitize(ts_ns, edges[1:-1], right=False)
        cnt = np.bincount(b, minlength=int(edges.size - 1)).astype(np.float64)
        inv = np.zeros_like(cnt)
        nz = cnt > 0
        inv[nz] = 1.0 / cnt[nz]
        time_weights = inv[b].astype(np.float32)
        time_weights /= max(EPS, float(np.mean(time_weights)))
        time_proxy_bins = int(edges.size - 1)

    teacher_oof = _align_probs_from_csv(args.teacher_oof_csv, train_ids)
    teacher_test = _align_probs_from_csv(args.teacher_test_csv, test_ids)
    spec1_oof = _align_specialist_oof(args.spec_oof_npy, args.spec_track_ids_npy, train_ids)
    spec1_test = _align_probs_from_csv(args.spec_test_csv, test_ids)
    if str(args.spec2_oof_npy).strip():
        if not str(args.spec2_track_ids_npy).strip() or not str(args.spec2_test_csv).strip():
            raise ValueError("spec2 requires --spec2-track-ids-npy and --spec2-test-csv")
        spec2_oof = _align_specialist_oof(args.spec2_oof_npy, args.spec2_track_ids_npy, train_ids)
        spec2_test = _align_probs_from_csv(args.spec2_test_csv, test_ids)
    else:
        spec2_oof = None
        spec2_test = None

    bank_oof = _make_bank_probs(teacher_oof, spec1_oof, spec2_oof, mode=str(args.bank_mode))
    bank_test = _make_bank_probs(teacher_test, spec1_test, spec2_test, mode=str(args.bank_mode))

    cv_df = pd.DataFrame(
        {
            "_cv_ts": train_df[str(args.time_col)],
            "_cv_group": train_df[str(args.group_col)].astype(np.int64),
        }
    )
    folds = make_forward_temporal_group_folds(cv_df, timestamp_col="_cv_ts", group_col="_cv_group", n_splits=int(args.n_splits))

    teacher_loss = _bce_row(y, teacher_oof)
    p_oof_clip = np.clip(teacher_oof, EPS, 1.0 - EPS)
    teacher_entropy_oof = -np.sum(p_oof_clip * np.log(p_oof_clip), axis=1)
    selector_score_oof: np.ndarray | None = None
    selector_score_test: np.ndarray | None = None
    if str(args.oof_selector) == "policy_score":
        if not str(args.selector_oof_npy).strip() or not str(args.selector_test_npy).strip():
            raise ValueError("--selector-oof-npy and --selector-test-npy are required for oof-selector=policy_score")
        selector_score_oof = np.asarray(np.load(args.selector_oof_npy), dtype=np.float32).reshape(-1)
        selector_score_test = np.asarray(np.load(args.selector_test_npy), dtype=np.float32).reshape(-1)
        if len(selector_score_oof) != len(train_df):
            raise ValueError(f"selector oof length mismatch: {len(selector_score_oof)} vs train {len(train_df)}")
        if len(selector_score_test) != len(test_df):
            raise ValueError(f"selector test length mismatch: {len(selector_score_test)} vs test {len(test_df)}")
    spec_loss = _bce_row(y, bank_oof)
    if str(args.only_if_better):
        better_mask_oof = spec_loss < teacher_loss
    else:
        better_mask_oof = np.ones((len(train_df),), dtype=bool)

    q_grid = _parse_grid(args.q_grid)
    alpha_grid = _parse_grid(args.alpha_grid)
    delta_grid = _parse_grid(args.delta_threshold_grid)
    target_fold = int(args.target_fold_index)
    override_class_idx = (
        [CLASSES.index(x.strip()) for x in str(args.override_classes).split(",") if x.strip()]
        if str(args.override_classes).strip()
        else list(range(len(CLASSES)))
    )
    override_class_names = [CLASSES[i] for i in override_class_idx]
    class_strength = _compute_class_strength(
        y=y,
        teacher=teacher_oof,
        spec=bank_oof,
        folds=folds,
        sample_weights=sample_weights,
        class_idx=override_class_idx,
        source=str(args.budget_strength_source),
    )

    rows: list[dict[str, float]] = []
    feasible: list[dict[str, float]] = []
    for q in q_grid:
        qf = float(q)
        if not (0.0 < qf < 1.0):
            continue
        if str(args.oof_selector) == "teacher_entropy":
            top_mask_oof, thr = _topk_mask(teacher_entropy_oof, qf)
            p = np.clip(teacher_test, EPS, 1.0 - EPS)
            ent = -np.sum(p * np.log(p), axis=1)
            top_mask_test, _ = _topk_mask(ent, qf)
        elif str(args.oof_selector) == "policy_score":
            assert selector_score_oof is not None
            assert selector_score_test is not None
            top_mask_oof, thr = _topk_mask(selector_score_oof, qf)
            top_mask_test, _ = _topk_mask(selector_score_test, qf)
        else:
            top_mask_oof, thr = _topk_mask(teacher_loss, qf)
            p = np.clip(teacher_test, EPS, 1.0 - EPS)
            ent = -np.sum(p * np.log(p), axis=1)
            top_mask_test, _ = _topk_mask(ent, qf)

        for alpha in alpha_grid:
            a = float(alpha)
            for delta_thr in delta_grid:
                dt = float(delta_thr)
                use_mask_oof = top_mask_oof & better_mask_oof
                if bool(args.per_class_budget):
                    budget_oof = _budget_from_strength(
                        strength=class_strength,
                        class_idx=override_class_idx,
                        total_budget=int(np.sum(use_mask_oof)),
                        tau=float(args.budget_tau),
                        floor_min=int(args.budget_floor_min),
                        floor_topk=int(args.budget_floor_topk),
                    )
                    pred_oof, budget_oof_rem, applied_ops_oof, changed_track_frac_oof = _apply_override_with_budget(
                        teacher=teacher_oof,
                        spec=bank_oof,
                        use_mask=use_mask_oof,
                        alpha=a,
                        class_idx=override_class_idx,
                        top_m_classes=int(args.top_m_classes),
                        class_budget=budget_oof,
                        uncertainty=teacher_entropy_oof,
                        delta_threshold=dt,
                    )
                else:
                    pred_oof = _apply_override(
                        teacher=teacher_oof,
                        spec=bank_oof,
                        use_mask=use_mask_oof,
                        alpha=a,
                        class_idx=override_class_idx,
                        top_m_classes=int(args.top_m_classes),
                        delta_threshold=dt,
                    )
                    budget_oof = {int(j): 0 for j in override_class_idx}
                    budget_oof_rem = budget_oof
                    applied_ops_oof = int(np.sum(np.abs(pred_oof - teacher_oof) > 1e-12))
                    changed_track_frac_oof = float(np.mean(np.any(np.abs(pred_oof - teacher_oof) > 1e-12, axis=1)))

                b_mean_uw, p_mean_uw, min_delta_uw, per_fold_uw, _ = _fold_eval(
                    y,
                    teacher_oof,
                    pred_oof,
                    folds,
                    sample_weights=None,
                )
                if sample_weights is not None:
                    b_mean_adv, p_mean_adv, min_delta_adv, per_fold_adv, _ = _fold_eval(
                        y,
                        teacher_oof,
                        pred_oof,
                        folds,
                        sample_weights=sample_weights,
                    )
                else:
                    b_mean_adv, p_mean_adv, min_delta_adv, per_fold_adv = (
                        b_mean_uw,
                        p_mean_uw,
                        min_delta_uw,
                        per_fold_uw,
                    )
                b_mean_time, p_mean_time, min_delta_time, per_fold_time, _ = _fold_eval(
                    y,
                    teacher_oof,
                    pred_oof,
                    folds,
                    sample_weights=time_weights,
                )
                gain_uw = p_mean_uw - b_mean_uw
                gain_adv = p_mean_adv - b_mean_adv
                gain_time = p_mean_time - b_mean_time
                robust_score = float(min(gain_uw, gain_adv, gain_time))
                target_delta_uw = float(per_fold_uw.get(target_fold, 0.0))
                target_delta_adv = float(per_fold_adv.get(target_fold, 0.0))
                target_delta_time = float(per_fold_time.get(target_fold, 0.0))
                override_frac = float(use_mask_oof.mean())
                row = {
                    "q_frac": qf,
                    "alpha": a,
                    "delta_threshold": dt,
                    "selector_thr": thr,
                    "override_frac_oof": override_frac,
                    "override_frac_test": float(top_mask_test.mean()),
                    "changed_track_frac_oof": float(changed_track_frac_oof),
                    "applied_ops_oof": int(applied_ops_oof),
                    "baseline_mean_uw": b_mean_uw,
                    "pred_mean_uw": p_mean_uw,
                    "gain_uw": gain_uw,
                    "min_fold_delta_uw": min_delta_uw,
                    "target_fold_delta_uw": target_delta_uw,
                    "baseline_mean_adv": b_mean_adv,
                    "pred_mean_adv": p_mean_adv,
                    "gain_adv": gain_adv,
                    "min_fold_delta_adv": min_delta_adv,
                    "target_fold_delta_adv": target_delta_adv,
                    "baseline_mean_time": b_mean_time,
                    "pred_mean_time": p_mean_time,
                    "gain_time": gain_time,
                    "min_fold_delta_time": min_delta_time,
                    "target_fold_delta_time": target_delta_time,
                    "gain": gain_adv,
                    "min_fold_delta": min_delta_adv,
                    "target_fold_delta": target_delta_adv,
                    "robust_score": robust_score,
                }
                rows.append(row)
                if (
                    min_delta_uw >= float(args.min_fold_delta)
                    and min_delta_adv >= float(args.min_fold_delta)
                    and min_delta_time >= float(args.min_fold_delta)
                    and target_delta_uw >= float(args.max_target_fold_drop)
                    and target_delta_adv >= float(args.max_target_fold_drop)
                    and target_delta_time >= float(args.max_target_fold_drop)
                    and override_frac <= float(args.max_override_rate)
                    and override_frac >= float(args.min_override_rate)
                    and float(top_mask_test.mean()) <= float(args.max_override_rate_test)
                    and float(top_mask_test.mean()) >= float(args.min_override_rate_test)
                ):
                    feasible.append(row)

    if str(args.selection_objective) == "robust":
        scan_df = (
            pd.DataFrame(rows)
            .sort_values(["robust_score", "gain_adv", "min_fold_delta_adv"], ascending=[False, False, False])
            .reset_index(drop=True)
        )
    else:
        scan_df = pd.DataFrame(rows).sort_values(["gain_adv", "min_fold_delta_adv"], ascending=[False, False]).reset_index(drop=True)
    scan_df.to_csv(out_dir / "scan.csv", index=False)

    if feasible:
        if str(args.selection_objective) == "robust":
            best = sorted(feasible, key=lambda r: (r["robust_score"], r["gain_adv"], r["min_fold_delta_adv"]), reverse=True)[0]
        else:
            best = sorted(feasible, key=lambda r: (r["gain_adv"], r["min_fold_delta_adv"]), reverse=True)[0]
        status = "GO"
    else:
        if str(args.selection_objective) == "robust":
            best = rows[np.argmax([r["robust_score"] for r in rows])]
        else:
            best = rows[np.argmax([r["gain_adv"] for r in rows])]
        status = "NO_FEASIBLE"

    q = float(best["q_frac"])
    a = float(best["alpha"])
    dt = float(best.get("delta_threshold", 0.0))
    if str(args.oof_selector) == "teacher_entropy":
        top_mask_oof, _ = _topk_mask(teacher_entropy_oof, q)
    elif str(args.oof_selector) == "policy_score":
        assert selector_score_oof is not None
        top_mask_oof, _ = _topk_mask(selector_score_oof, q)
    else:
        top_mask_oof, _ = _topk_mask(teacher_loss, q)
    use_mask_oof = top_mask_oof & better_mask_oof
    if bool(args.per_class_budget):
        budget_oof = _budget_from_strength(
            strength=class_strength,
            class_idx=override_class_idx,
            total_budget=int(np.sum(use_mask_oof)),
            tau=float(args.budget_tau),
            floor_min=int(args.budget_floor_min),
            floor_topk=int(args.budget_floor_topk),
        )
        pred_oof, budget_oof_rem, applied_ops_oof, changed_track_frac_oof = _apply_override_with_budget(
            teacher=teacher_oof,
            spec=bank_oof,
            use_mask=use_mask_oof,
            alpha=a,
            class_idx=override_class_idx,
            top_m_classes=int(args.top_m_classes),
            class_budget=budget_oof,
            uncertainty=teacher_entropy_oof,
            delta_threshold=dt,
        )
    else:
        pred_oof = _apply_override(
            teacher=teacher_oof,
            spec=bank_oof,
            use_mask=use_mask_oof,
            alpha=a,
            class_idx=override_class_idx,
            top_m_classes=int(args.top_m_classes),
            delta_threshold=dt,
        )
        budget_oof = {int(j): 0 for j in override_class_idx}
        budget_oof_rem = budget_oof
        applied_ops_oof = 0
        changed_track_frac_oof = float(use_mask_oof.mean())

    if str(args.oof_selector) == "policy_score":
        assert selector_score_test is not None
        use_mask_test, _ = _topk_mask(selector_score_test, q)
        p = np.clip(teacher_test, EPS, 1.0 - EPS)
        ent = -np.sum(p * np.log(p), axis=1)
    else:
        p = np.clip(teacher_test, EPS, 1.0 - EPS)
        ent = -np.sum(p * np.log(p), axis=1)
        use_mask_test, _ = _topk_mask(ent, q)
    if bool(args.per_class_budget):
        budget_test = _budget_from_strength(
            strength=class_strength,
            class_idx=override_class_idx,
            total_budget=int(np.sum(use_mask_test)),
            tau=float(args.budget_tau),
            floor_min=int(args.budget_floor_min),
            floor_topk=int(args.budget_floor_topk),
        )
        pred_test, budget_test_rem, applied_ops_test, changed_track_frac_test = _apply_override_with_budget(
            teacher=teacher_test,
            spec=bank_test,
            use_mask=use_mask_test,
            alpha=a,
            class_idx=override_class_idx,
            top_m_classes=int(args.top_m_classes),
            class_budget=budget_test,
            uncertainty=ent.astype(np.float32),
            delta_threshold=dt,
        )
    else:
        pred_test = _apply_override(
            teacher=teacher_test,
            spec=bank_test,
            use_mask=use_mask_test,
            alpha=a,
            class_idx=override_class_idx,
            top_m_classes=int(args.top_m_classes),
            delta_threshold=dt,
        )
        budget_test = {int(j): 0 for j in override_class_idx}
        budget_test_rem = budget_test
        applied_ops_test = 0
        changed_track_frac_test = float(use_mask_test.mean())

    oof_df = pd.DataFrame({"track_id": train_ids})
    sub_df = pd.DataFrame({"track_id": test_ids})
    for j, c in enumerate(CLASSES):
        oof_df[c] = pred_oof[:, j]
        sub_df[c] = pred_test[:, j]
    oof_path = out_dir / "oof_best.csv"
    sub_path = out_dir / "sub_best.csv"
    oof_df.to_csv(oof_path, index=False)
    sub_df.to_csv(sub_path, index=False)

    b_mean_uw, p_mean_uw, min_delta_uw, per_fold_uw, fold_rows_uw = _fold_eval(
        y,
        teacher_oof,
        pred_oof,
        folds,
        sample_weights=None,
    )
    if sample_weights is not None:
        b_mean_adv, p_mean_adv, min_delta_adv, per_fold_adv, fold_rows_adv = _fold_eval(
            y,
            teacher_oof,
            pred_oof,
            folds,
            sample_weights=sample_weights,
        )
    else:
        b_mean_adv, p_mean_adv, min_delta_adv, per_fold_adv, fold_rows_adv = (
            b_mean_uw,
            p_mean_uw,
            min_delta_uw,
            per_fold_uw,
            fold_rows_uw,
        )
    b_mean_time, p_mean_time, min_delta_time, per_fold_time, fold_rows_time = _fold_eval(
        y,
        teacher_oof,
        pred_oof,
        folds,
        sample_weights=time_weights,
    )
    robust_score_best = float(min(p_mean_uw - b_mean_uw, p_mean_adv - b_mean_adv, p_mean_time - b_mean_time))
    report = {
        "status": status,
        "selection_objective": str(args.selection_objective),
        "only_if_better": bool(args.only_if_better),
        "oof_selector": str(args.oof_selector),
        "selector_oof_npy": str(Path(args.selector_oof_npy).resolve()) if str(args.selector_oof_npy).strip() else "",
        "selector_test_npy": str(Path(args.selector_test_npy).resolve()) if str(args.selector_test_npy).strip() else "",
        "sample_weights_npy": str(Path(args.sample_weights_npy).resolve()) if str(args.sample_weights_npy).strip() else "",
        "sample_weighted_metrics": bool(sample_weights is not None),
        "time_proxy": {
            "enabled": True,
            "mode": "time_equalized_bins",
            "n_bins_effective": int(time_proxy_bins),
            "weights_mean": float(np.mean(time_weights)),
            "weights_min": float(np.min(time_weights)),
            "weights_max": float(np.max(time_weights)),
        },
        "bank_mode": str(args.bank_mode),
        "top_m_classes": int(args.top_m_classes),
        "delta_threshold_grid": [float(x) for x in delta_grid.tolist()],
        "per_class_budget": bool(args.per_class_budget),
        "budget_strength_source": str(args.budget_strength_source),
        "budget_tau": float(args.budget_tau),
        "budget_floor_min": int(args.budget_floor_min),
        "budget_floor_topk": int(args.budget_floor_topk),
        "override_classes": override_class_names,
        "class_strength": {CLASSES[int(k)]: float(v) for k, v in class_strength.items()},
        "n_scan": int(len(rows)),
        "n_feasible": int(len(feasible)),
        "best": best,
        "best_gain": float(p_mean_adv - b_mean_adv),
        "best_gain_uw": float(p_mean_uw - b_mean_uw),
        "best_gain_adv": float(p_mean_adv - b_mean_adv),
        "best_gain_time": float(p_mean_time - b_mean_time),
        "best_robust_score": float(robust_score_best),
        "best_min_fold_delta": float(min_delta_adv),
        "best_min_fold_delta_uw": float(min_delta_uw),
        "best_min_fold_delta_adv": float(min_delta_adv),
        "best_min_fold_delta_time": float(min_delta_time),
        "best_target_fold_delta": float(per_fold_adv.get(target_fold, 0.0)),
        "best_target_fold_delta_uw": float(per_fold_uw.get(target_fold, 0.0)),
        "best_target_fold_delta_adv": float(per_fold_adv.get(target_fold, 0.0)),
        "best_target_fold_delta_time": float(per_fold_time.get(target_fold, 0.0)),
        "best_override_frac_oof": float(use_mask_oof.mean()),
        "best_override_frac_test": float(use_mask_test.mean()),
        "best_changed_track_frac_oof": float(changed_track_frac_oof),
        "best_changed_track_frac_test": float(changed_track_frac_test),
        "best_applied_ops_oof": int(applied_ops_oof),
        "best_applied_ops_test": int(applied_ops_test),
        "best_budget_oof_initial": {CLASSES[int(k)]: int(v) for k, v in budget_oof.items()},
        "best_budget_oof_remaining": {CLASSES[int(k)]: int(v) for k, v in budget_oof_rem.items()},
        "best_budget_test_initial": {CLASSES[int(k)]: int(v) for k, v in budget_test.items()},
        "best_budget_test_remaining": {CLASSES[int(k)]: int(v) for k, v in budget_test_rem.items()},
        "constraints": {
            "min_fold_delta": float(args.min_fold_delta),
            "target_fold_index": int(target_fold),
            "max_target_fold_drop": float(args.max_target_fold_drop),
            "min_override_rate": float(args.min_override_rate),
            "max_override_rate": float(args.max_override_rate),
            "min_override_rate_test": float(args.min_override_rate_test),
            "max_override_rate_test": float(args.max_override_rate_test),
        },
        "scan_csv": str((out_dir / "scan.csv").resolve()),
        "output_oof": str(oof_path),
        "output_submission": str(sub_path),
        "fold_rows_uw": fold_rows_uw,
        "fold_rows_adv": fold_rows_adv,
        "fold_rows_time": fold_rows_time,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(
        f"status={status} n_feasible={len(feasible)}/{len(rows)} "
        f"gain_adv={float(p_mean_adv - b_mean_adv):+.6f} "
        f"gain_uw={float(p_mean_uw - b_mean_uw):+.6f} "
        f"gain_time={float(p_mean_time - b_mean_time):+.6f} "
        f"robust={float(robust_score_best):+.6f} "
        f"min_fold_delta_adv={float(min_delta_adv):+.6f} "
        f"target_fold_delta_adv={float(per_fold_adv.get(target_fold, 0.0)):+.6f} "
        f"override_oof={float(use_mask_oof.mean()):.4f}",
        flush=True,
    )
    print(str(sub_path), flush=True)
    print(str(out_dir / "report.json"), flush=True)


if __name__ == "__main__":
    main()
