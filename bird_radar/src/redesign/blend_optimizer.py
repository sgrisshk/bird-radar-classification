from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score

from config import CLASSES
from src.redesign.utils import (
    apply_temperature_to_probs,
    dump_json,
    macro_map,
    optimize_temperature_per_class,
    per_class_ap,
)


def _load_preds(path: str | Path) -> np.ndarray:
    return np.load(Path(path)).astype(np.float32)


def _project_simplex(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0.0:
        return np.ones_like(w) / len(w)
    return w / s


def _optimize_weights(y_true: np.ndarray, oof_list: list[np.ndarray]) -> np.ndarray:
    n = len(oof_list)
    x0 = np.ones(n, dtype=np.float64) / n

    def obj(w: np.ndarray) -> float:
        ww = _project_simplex(w)
        blend = np.zeros_like(oof_list[0], dtype=np.float64)
        for i in range(n):
            blend += ww[i] * oof_list[i]
        return -macro_map(y_true, blend)

    cons = ({"type": "eq", "fun": lambda w: float(np.sum(np.clip(w, 0.0, None))) - 1.0},)
    bnds = [(0.0, 1.0)] * n
    res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 500})
    if not res.success:
        return x0.astype(np.float32)
    return _project_simplex(res.x).astype(np.float32)


def _optimize_weights_per_class(y_true: np.ndarray, oof_list: list[np.ndarray]) -> np.ndarray:
    n_models = len(oof_list)
    n_classes = y_true.shape[1]
    w_mat = np.zeros((n_classes, n_models), dtype=np.float32)

    for c in range(n_classes):
        yt = y_true[:, c]
        if yt.sum() <= 0:
            w_mat[c] = np.ones(n_models, dtype=np.float32) / n_models
            continue

        x0 = np.ones(n_models, dtype=np.float64) / n_models

        def obj(w: np.ndarray) -> float:
            ww = _project_simplex(w)
            pred = np.zeros_like(yt, dtype=np.float64)
            for i in range(n_models):
                pred += ww[i] * oof_list[i][:, c]
            return -float(average_precision_score(yt, pred))

        cons = ({"type": "eq", "fun": lambda w: float(np.sum(np.clip(w, 0.0, None))) - 1.0},)
        bnds = [(0.0, 1.0)] * n_models
        res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 300})
        if not res.success:
            w_mat[c] = x0.astype(np.float32)
        else:
            w_mat[c] = _project_simplex(res.x).astype(np.float32)

    return w_mat


def _blend(preds: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    out = np.zeros_like(preds[0], dtype=np.float32)
    for p, ww in zip(preds, w):
        out += p.astype(np.float32) * float(ww)
    return np.clip(out, 0.0, 1.0)


def _blend_per_class(preds: list[np.ndarray], w_mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(preds[0], dtype=np.float32)
    for c in range(out.shape[1]):
        for i, p in enumerate(preds):
            out[:, c] += float(w_mat[c, i]) * p[:, c].astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def _threshold_grid(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    # Diagnostic only for collapse monitoring.
    out: dict[str, float] = {}
    grid = np.linspace(0.05, 0.95, 19)
    for i, cls in enumerate(CLASSES):
        yt = y_true[:, i].astype(np.float32)
        yp = probs[:, i].astype(np.float32)
        best_t = 0.5
        best_f1 = -1.0
        for t in grid:
            pr = (yp >= t).astype(np.float32)
            tp = float(np.sum((pr == 1) & (yt == 1)))
            fp = float(np.sum((pr == 1) & (yt == 0)))
            fn = float(np.sum((pr == 0) & (yt == 1)))
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f1 = 2.0 * p * r / (p + r + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        out[cls] = best_t
    return out


def run_blend(cfg: dict[str, Any], oof_summary: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(cfg["paths"]["output_dir"]).resolve()
    blend_dir = out_dir / "blends"
    blend_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.load(out_dir / "oof_targets.npy").astype(np.float32)

    model_names = sorted(oof_summary["models"].keys())
    oof_list = [_load_preds(oof_summary["models"][n]["oof_path"]) for n in model_names]
    test_list = [_load_preds(oof_summary["models"][n]["test_path"]) for n in model_names]

    w_global = _optimize_weights(y_true, oof_list)
    oof_global = _blend(oof_list, w_global)
    test_global = _blend(test_list, w_global)

    w_per_class = _optimize_weights_per_class(y_true, oof_list)
    oof_per_class = _blend_per_class(oof_list, w_per_class)
    test_per_class = _blend_per_class(test_list, w_per_class)

    if macro_map(y_true, oof_per_class) >= macro_map(y_true, oof_global):
        selected_base = "per_class"
        base_weights_global = w_global
        base_weights_per_class = w_per_class
        oof_blend = oof_per_class
        test_blend = test_per_class
    else:
        selected_base = "global"
        base_weights_global = w_global
        base_weights_per_class = w_per_class
        oof_blend = oof_global
        test_blend = test_global

    temps, raw_macro, cal_macro = optimize_temperature_per_class(y_true, oof_blend)
    oof_cal = apply_temperature_to_probs(oof_blend, temps)
    test_cal = apply_temperature_to_probs(test_blend, temps)

    selected = "calibrated" if cal_macro >= raw_macro else "raw"
    final_oof = oof_cal if selected == "calibrated" else oof_blend
    final_test = test_cal if selected == "calibrated" else test_blend

    base_name = "blend_base"
    np.save(blend_dir / f"{base_name}_oof.npy", final_oof)
    np.save(blend_dir / f"{base_name}_test.npy", final_test)

    # Top-3 perturbations around optimal weights.
    rng = np.random.default_rng(int(cfg["seed"]))
    top_blends: list[dict[str, Any]] = []
    cand_weights_global = [base_weights_global]
    cand_weights_per_class = [base_weights_per_class]
    for _ in range(8):
        noise_g = rng.normal(0.0, float(cfg["blend"]["weight_perturb_std"]), size=len(base_weights_global)).astype(np.float32)
        cand_weights_global.append(_project_simplex(base_weights_global + noise_g))

        noise_pc = rng.normal(0.0, float(cfg["blend"]["weight_perturb_std"]), size=base_weights_per_class.shape).astype(np.float32)
        w_pc = np.clip(base_weights_per_class + noise_pc, 0.0, None)
        w_pc = np.where(w_pc.sum(axis=1, keepdims=True) <= 0.0, 1.0, w_pc)
        w_pc = w_pc / w_pc.sum(axis=1, keepdims=True)
        cand_weights_per_class.append(w_pc.astype(np.float32))

    scored: list[tuple[float, str, np.ndarray]] = []
    for cw in cand_weights_global:
        oo = _blend(oof_list, cw)
        oo = apply_temperature_to_probs(oo, temps)
        scored.append((macro_map(y_true, oo), "global", cw))
    for cw in cand_weights_per_class:
        oo = _blend_per_class(oof_list, cw)
        oo = apply_temperature_to_probs(oo, temps)
        scored.append((macro_map(y_true, oo), "per_class", cw))
    scored.sort(key=lambda x: x[0], reverse=True)

    for rank, (score, blend_type, cw) in enumerate(scored[: int(cfg["blend"]["top_n_exports"])], start=1):
        if blend_type == "global":
            oof_r = apply_temperature_to_probs(_blend(oof_list, cw), temps)
            test_r = apply_temperature_to_probs(_blend(test_list, cw), temps)
            weights_payload: dict[str, Any] = {model_names[i]: float(cw[i]) for i in range(len(model_names))}
        else:
            oof_r = apply_temperature_to_probs(_blend_per_class(oof_list, cw), temps)
            test_r = apply_temperature_to_probs(_blend_per_class(test_list, cw), temps)
            weights_payload = {
                CLASSES[c]: {model_names[i]: float(cw[c, i]) for i in range(len(model_names))}
                for c in range(len(CLASSES))
            }

        name = f"blend_top{rank}"
        np.save(blend_dir / f"{name}_oof.npy", oof_r)
        np.save(blend_dir / f"{name}_test.npy", test_r)
        top_blends.append(
            {
                "name": name,
                "blend_type": blend_type,
                "macro_map": float(score),
                "weights": weights_payload,
                "oof_path": str((blend_dir / f"{name}_oof.npy").resolve()),
                "test_path": str((blend_dir / f"{name}_test.npy").resolve()),
            }
        )

    report = {
        "models": model_names,
        "base_selection": selected_base,
        "base_weights_global": {model_names[i]: float(base_weights_global[i]) for i in range(len(model_names))},
        "base_weights_per_class": {
            CLASSES[c]: {model_names[i]: float(base_weights_per_class[c, i]) for i in range(len(model_names))}
            for c in range(len(CLASSES))
        },
        "raw_macro_map": float(raw_macro),
        "calibrated_macro_map": float(cal_macro),
        "selected": selected,
        "selected_macro_map": float(macro_map(y_true, final_oof)),
        "per_class_ap": per_class_ap(y_true, final_oof),
        "temperatures": [float(x) for x in temps.tolist()],
        "thresholds": _threshold_grid(y_true, final_oof),
        "top_blends": top_blends,
    }

    dump_json(blend_dir / "blend_report.json", report)
    return report
