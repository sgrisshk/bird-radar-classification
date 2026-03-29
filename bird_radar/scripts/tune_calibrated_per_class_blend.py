from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASSES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-class temperature scaling + per-class alpha blend tuning.")
    p.add_argument("--base-artifacts-dir", type=str, required=True)
    p.add_argument("--temporal-artifacts-dir", type=str, required=True)
    p.add_argument("--base-test-csv", type=str, required=True)
    p.add_argument("--temporal-test-csv", type=str, required=True)
    p.add_argument("--fit-source", type=str, default="forward_cv_complete", choices=["forward_cv_complete", "holdout"])
    p.add_argument("--eval-source", type=str, default="holdout", choices=["forward_cv_complete", "holdout"])
    p.add_argument("--alpha-grid", type=str, default="0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50")
    p.add_argument("--start-alpha", type=float, default=0.90)
    p.add_argument("--alpha-passes", type=int, default=2)
    p.add_argument("--temp-grid", type=str, default="0.60,0.70,0.80,0.90,1.00,1.10,1.20,1.30,1.40,1.60,1.80,2.00")
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _macro_map(y_true: np.ndarray, y_score: np.ndarray) -> float:
    vals: list[float] = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        ys = y_score[:, i]
        vals.append(float(average_precision_score(yt, ys)) if float(np.sum(yt)) > 0 else 0.0)
    return float(np.mean(vals))


def _per_class_ap(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, cls in enumerate(CLASSES):
        yt = y_true[:, i]
        ys = y_score[:, i]
        out[cls] = float(average_precision_score(yt, ys)) if float(np.sum(yt)) > 0 else 0.0
    return out


def _load_source(art: Path, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # returns: ids, pred, y  for source rows only (indexed by idx)
    ids = np.load(art / "train_track_ids.npy")
    y_all = np.load(art / "oof_targets.npy").astype(np.float32)
    if source == "forward_cv_complete":
        p_all = np.load(art / "oof_forward_cv_complete.npy").astype(np.float32)
        idx = np.load(art / "oof_forward_cv_complete_idx.npy").astype(np.int64)
    else:
        p_all = np.load(art / "oof_holdout.npy").astype(np.float32)
        idx = np.load(art / "oof_holdout_idx.npy").astype(np.int64)
    return ids[idx], p_all[idx], y_all[idx]


def _fit_per_class_temperature(pred_fit: np.ndarray, y_fit: np.ndarray, temp_grid: np.ndarray) -> np.ndarray:
    logits_fit = _logit(pred_fit.astype(np.float64))
    temps = np.ones((pred_fit.shape[1],), dtype=np.float64)
    for ci in range(pred_fit.shape[1]):
        yt = y_fit[:, ci].astype(np.int32)
        if int(np.sum(yt)) == 0 or int(np.sum(yt)) == len(yt):
            temps[ci] = 1.0
            continue
        best_t = 1.0
        best_ll = float("inf")
        for t in temp_grid:
            p = _sigmoid(logits_fit[:, ci] / float(t))
            ll = float(log_loss(yt, np.clip(p, 1e-6, 1.0 - 1e-6), labels=[0, 1]))
            if ll < best_ll:
                best_ll = ll
                best_t = float(t)
        temps[ci] = best_t
    return temps.astype(np.float32)


def _apply_temperature(pred: np.ndarray, temps: np.ndarray) -> np.ndarray:
    logits = _logit(pred.astype(np.float64))
    out = _sigmoid(logits / temps[None, :])
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def _optimize_per_class_alpha(
    y: np.ndarray,
    base_pred: np.ndarray,
    temp_pred: np.ndarray,
    alpha_grid: np.ndarray,
    start_alpha: float,
    passes: int,
) -> tuple[np.ndarray, float]:
    alpha = np.full((base_pred.shape[1],), float(start_alpha), dtype=np.float32)

    def blend(a: np.ndarray) -> np.ndarray:
        return np.clip(base_pred * a[None, :] + temp_pred * (1.0 - a[None, :]), 0.0, 1.0)

    best_pred = blend(alpha)
    best = _macro_map(y, best_pred)
    for _ in range(max(int(passes), 1)):
        changed = False
        for ci in range(base_pred.shape[1]):
            local_best = best
            local_alpha = float(alpha[ci])
            for a in alpha_grid:
                trial = alpha.copy()
                trial[ci] = float(a)
                score = _macro_map(y, blend(trial))
                if score > local_best + 1e-12:
                    local_best = score
                    local_alpha = float(a)
            if local_alpha != float(alpha[ci]):
                alpha[ci] = local_alpha
                best = local_best
                changed = True
        if not changed:
            break
    return alpha, float(best)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_art = Path(args.base_artifacts_dir).resolve()
    temp_art = Path(args.temporal_artifacts_dir).resolve()

    fit_ids_b, fit_b, fit_y_b = _load_source(base_art, args.fit_source)
    fit_ids_t, fit_t, fit_y_t = _load_source(temp_art, args.fit_source)
    eval_ids_b, eval_b, eval_y_b = _load_source(base_art, args.eval_source)
    eval_ids_t, eval_t, eval_y_t = _load_source(temp_art, args.eval_source)

    if not np.array_equal(fit_ids_b, fit_ids_t):
        raise RuntimeError("fit ids mismatch between base and temporal")
    if not np.array_equal(eval_ids_b, eval_ids_t):
        raise RuntimeError("eval ids mismatch between base and temporal")
    if not np.array_equal(fit_y_b, fit_y_t):
        raise RuntimeError("fit targets mismatch between base and temporal")
    if not np.array_equal(eval_y_b, eval_y_t):
        raise RuntimeError("eval targets mismatch between base and temporal")

    fit_y = fit_y_b.astype(np.float32)
    eval_y = eval_y_b.astype(np.float32)
    fit_b = np.clip(fit_b, 0.0, 1.0)
    fit_t = np.clip(fit_t, 0.0, 1.0)
    eval_b = np.clip(eval_b, 0.0, 1.0)
    eval_t = np.clip(eval_t, 0.0, 1.0)

    temp_grid = np.array([float(x.strip()) for x in args.temp_grid.split(",") if x.strip()], dtype=np.float32)
    alpha_grid = np.array([float(x.strip()) for x in args.alpha_grid.split(",") if x.strip()], dtype=np.float32)
    if len(temp_grid) == 0 or len(alpha_grid) == 0:
        raise ValueError("Empty temp-grid or alpha-grid")

    temps_b = _fit_per_class_temperature(fit_b, fit_y, temp_grid)
    temps_t = _fit_per_class_temperature(fit_t, fit_y, temp_grid)

    fit_b_cal = _apply_temperature(fit_b, temps_b)
    fit_t_cal = _apply_temperature(fit_t, temps_t)
    eval_b_cal = _apply_temperature(eval_b, temps_b)
    eval_t_cal = _apply_temperature(eval_t, temps_t)

    alpha, fit_alpha_score = _optimize_per_class_alpha(
        y=fit_y,
        base_pred=fit_b_cal,
        temp_pred=fit_t_cal,
        alpha_grid=alpha_grid,
        start_alpha=args.start_alpha,
        passes=args.alpha_passes,
    )

    eval_blend = np.clip(eval_b_cal * alpha[None, :] + eval_t_cal * (1.0 - alpha[None, :]), 0.0, 1.0)
    eval_base_score = _macro_map(eval_y, eval_b)
    eval_temp_score = _macro_map(eval_y, eval_t)
    eval_blend_score = _macro_map(eval_y, eval_blend)

    base_test = pd.read_csv(args.base_test_csv)
    temp_test = pd.read_csv(args.temporal_test_csv)
    if "track_id" not in base_test.columns or "track_id" not in temp_test.columns:
        raise RuntimeError("base-test-csv and temporal-test-csv must contain track_id")
    if not np.array_equal(base_test["track_id"].to_numpy(), temp_test["track_id"].to_numpy()):
        raise RuntimeError("test track_id mismatch")
    for c in CLASSES:
        if c not in base_test.columns or c not in temp_test.columns:
            raise RuntimeError(f"missing class column in test csv: {c}")

    p_base_test = np.clip(base_test[CLASSES].to_numpy(dtype=np.float32), 0.0, 1.0)
    p_temp_test = np.clip(temp_test[CLASSES].to_numpy(dtype=np.float32), 0.0, 1.0)
    p_base_test_cal = _apply_temperature(p_base_test, temps_b)
    p_temp_test_cal = _apply_temperature(p_temp_test, temps_t)
    p_out = np.clip(p_base_test_cal * alpha[None, :] + p_temp_test_cal * (1.0 - alpha[None, :]), 0.0, 1.0)

    out_sub = base_test[["track_id"]].copy()
    out_sub[CLASSES] = p_out
    sub_path = out_dir / "sub_per_class_alpha_calibrated.csv"
    out_sub.to_csv(sub_path, index=False)

    report = {
        "fit_source": args.fit_source,
        "eval_source": args.eval_source,
        "n_fit": int(len(fit_y)),
        "n_eval": int(len(eval_y)),
        "macro_fit_base": _macro_map(fit_y, fit_b),
        "macro_fit_temp": _macro_map(fit_y, fit_t),
        "macro_fit_alpha_calibrated": float(fit_alpha_score),
        "macro_eval_base_raw": float(eval_base_score),
        "macro_eval_temp_raw": float(eval_temp_score),
        "macro_eval_alpha_calibrated": float(eval_blend_score),
        "eval_delta_vs_base": float(eval_blend_score - eval_base_score),
        "temperature_base_by_class": {cls: float(temps_b[i]) for i, cls in enumerate(CLASSES)},
        "temperature_temporal_by_class": {cls: float(temps_t[i]) for i, cls in enumerate(CLASSES)},
        "alpha_by_class": {cls: float(alpha[i]) for i, cls in enumerate(CLASSES)},
        "per_class_ap_eval_base": _per_class_ap(eval_y, eval_b),
        "per_class_ap_eval_temp": _per_class_ap(eval_y, eval_t),
        "per_class_ap_eval_alpha_calibrated": _per_class_ap(eval_y, eval_blend),
        "output_submission": str(sub_path),
    }
    (out_dir / "calibrated_per_class_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(str(sub_path), flush=True)
    print(str(out_dir / "calibrated_per_class_report.json"), flush=True)


if __name__ == "__main__":
    main()

