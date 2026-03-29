#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.redesign.utils import dump_json, macro_map, per_class_ap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize blend weights with month-variance and rare-diversity penalties.")
    p.add_argument("--train-csv", type=str, default="train.csv")
    p.add_argument("--test-csv", type=str, default="test.csv")
    p.add_argument("--proposal-root", type=str, default="bird_radar/artifacts/proposal")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--top-n-components", type=int, default=10)
    p.add_argument("--min-component-macro", type=float, default=0.0)
    p.add_argument("--month-var-lambdas", type=str, default="0.0,0.1,0.3,0.5")
    p.add_argument("--diversity-lambdas", type=str, default="0.0,0.02,0.05")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--search-iters", type=int, default=5000)
    return p.parse_args()


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / (e.sum() + 1e-8)


def _macro_by_month(y: np.ndarray, p: np.ndarray, months: np.ndarray) -> dict[int, float]:
    out: dict[int, float] = {}
    for m in sorted(pd.Series(months).dropna().unique().tolist()):
        mask = months == m
        if int(mask.sum()) <= 0:
            continue
        out[int(m)] = float(macro_map(y[mask], p[mask]))
    return out


def _load_prob_csv(path: Path, ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path, usecols=["track_id", *CLASSES])
    mp = {int(r["track_id"]): np.asarray([r[c] for c in CLASSES], dtype=np.float32) for _, r in df.iterrows()}
    miss = [int(t) for t in ids if int(t) not in mp]
    if miss:
        raise ValueError(f"{path} missing {len(miss)} ids (first={miss[:5]})")
    arr = np.stack([mp[int(t)] for t in ids], axis=0).astype(np.float32)
    return arr


def _discover_components(proposal_root: Path, train_ids: np.ndarray, test_ids: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for oof_csv in sorted(proposal_root.glob("**/oof.csv")):
        sub_csv = oof_csv.parent / "submission_lgbm_ovr.csv"
        if not sub_csv.exists():
            continue
        try:
            oof = _load_prob_csv(oof_csv, train_ids)
            test = _load_prob_csv(sub_csv, test_ids)
        except Exception:
            continue
        rows.append(
            {
                "name": str(oof_csv.parent.relative_to(proposal_root)),
                "oof_csv": str(oof_csv.resolve()),
                "sub_csv": str(sub_csv.resolve()),
                "oof": oof,
                "test": test,
            }
        )
    return rows


def _optimize(
    y: np.ndarray,
    months: np.ndarray,
    rare_mask: np.ndarray,
    oof_stack: np.ndarray,
    month_lambda: float,
    div_lambda: float,
    search_iters: int,
    seed: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    n_models = int(oof_stack.shape[0])
    # Correlation only on rare rows for diversity penalty.
    rare_flat = oof_stack[:, rare_mask, :].reshape(n_models, -1)
    if rare_flat.shape[1] > 1:
        rare_corr = np.corrcoef(rare_flat)
    else:
        rare_corr = np.eye(n_models, dtype=np.float64)
    rare_corr = np.nan_to_num(rare_corr, nan=0.0, posinf=0.0, neginf=0.0)

    def objective_from_w(w: np.ndarray) -> tuple[float, dict[str, float]]:
        blend = np.tensordot(w, oof_stack, axes=(0, 0))
        g = float(macro_map(y, blend))
        month_scores = _macro_by_month(y, blend, months)
        month_std = float(np.std(list(month_scores.values()))) if month_scores else 0.0
        corr_pen = float(w @ rare_corr @ w)
        score = g - float(month_lambda) * month_std - float(div_lambda) * corr_pen
        return float(score), {
            "g": g,
            "month_std": month_std,
            "corr_pen": corr_pen,
        }

    rng = np.random.RandomState(int(seed))
    best_w = np.full((n_models,), 1.0 / n_models, dtype=np.float64)
    best_obj, _ = objective_from_w(best_w)
    best_aux: dict[str, float] = {}

    # warm-start: one-hot + uniform + mildly sparse samples
    for i in range(n_models):
        w = np.zeros((n_models,), dtype=np.float64)
        w[i] = 1.0
        obj, aux = objective_from_w(w)
        if obj > best_obj:
            best_obj, best_w, best_aux = obj, w, aux

    for _ in range(max(200, int(search_iters))):
        alpha = rng.uniform(0.3, 2.5, size=(n_models,))
        w = rng.dirichlet(alpha).astype(np.float64)
        obj, aux = objective_from_w(w)
        if obj > best_obj:
            best_obj, best_w, best_aux = obj, w, aux

    # local refinement around current best.
    z_best = np.log(np.clip(best_w, 1e-9, 1.0))
    for _ in range(max(100, int(search_iters // 4))):
        z = z_best + rng.normal(0.0, 0.25, size=z_best.shape)
        w = _softmax(z)
        obj, aux = objective_from_w(w)
        if obj > best_obj:
            best_obj, best_w, best_aux = obj, w, aux
            z_best = np.log(np.clip(best_w, 1e-9, 1.0))

    w = best_w.astype(np.float64)
    blend = np.tensordot(w, oof_stack, axes=(0, 0))
    score = float(macro_map(y, blend))
    month_scores = _macro_by_month(y, blend, months)
    month_std = float(np.std(list(month_scores.values()))) if month_scores else 0.0
    corr_pen = float(w @ rare_corr @ w)
    extras = {
        "month_scores": month_scores,
        "month_std": month_std,
        "rare_corr_penalty": corr_pen,
        "opt_success": True,
        "opt_message": "random_dirichlet_search",
        "objective_value": float(best_obj),
        "objective_parts": best_aux,
    }
    return w.astype(np.float32), score, extras


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    train_ids = train_df["track_id"].to_numpy(dtype=np.int64)
    test_ids = test_df["track_id"].to_numpy(dtype=np.int64)
    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = np.zeros((len(y_idx), len(CLASSES)), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    months = pd.to_datetime(train_df["timestamp_start_radar_utc"], errors="coerce", utc=True).dt.month.fillna(0).astype(int).to_numpy()
    rare_idx = [CLASS_TO_INDEX["Cormorants"], CLASS_TO_INDEX["Waders"], CLASS_TO_INDEX["Ducks"]]
    rare_mask = np.isin(y_idx, rare_idx)

    components = _discover_components(Path(args.proposal_root).resolve(), train_ids, test_ids)
    if len(components) < 2:
        raise RuntimeError("Need at least 2 components with oof.csv + submission_lgbm_ovr.csv")

    for c in components:
        c["macro"] = float(macro_map(y, c["oof"]))
    components = [c for c in components if float(c["macro"]) >= float(args.min_component_macro)]
    components.sort(key=lambda z: float(z["macro"]), reverse=True)
    components = components[: max(2, int(args.top_n_components))]

    names = [c["name"] for c in components]
    oof_stack = np.stack([c["oof"] for c in components], axis=0).astype(np.float32)
    test_stack = np.stack([c["test"] for c in components], axis=0).astype(np.float32)

    month_lams = [float(x) for x in str(args.month_var_lambdas).split(",") if str(x).strip()]
    div_lams = [float(x) for x in str(args.diversity_lambdas).split(",") if str(x).strip()]

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for lm in month_lams:
        for ld in div_lams:
            w, score, extras = _optimize(
                y=y,
                months=months,
                rare_mask=rare_mask,
                oof_stack=oof_stack,
                month_lambda=lm,
                div_lambda=ld,
                search_iters=int(args.search_iters),
                seed=int(args.seed) + int(round(lm * 1000)) + 13 * int(round(ld * 1000)),
            )
            oof_bl = np.tensordot(w, oof_stack, axes=(0, 0)).astype(np.float32)
            test_bl = np.tensordot(w, test_stack, axes=(0, 0)).astype(np.float32)
            name = f"blend_mvar{lm:.2f}_div{ld:.2f}".replace(".", "p")
            sub = pd.DataFrame({"track_id": test_ids})
            sub[CLASSES] = np.clip(test_bl, 0.0, 1.0)
            sub.to_csv(out_dir / f"sub_{name}.csv", index=False)
            oof_csv = pd.DataFrame({"track_id": train_ids})
            oof_csv[CLASSES] = np.clip(oof_bl, 0.0, 1.0)
            oof_csv.to_csv(out_dir / f"oof_{name}.csv", index=False)
            np.save(out_dir / f"oof_{name}.npy", oof_bl)
            np.save(out_dir / f"test_{name}.npy", test_bl)

            pc = per_class_ap(y, oof_bl)
            rare3 = float((pc["Cormorants"] + pc["Waders"] + pc["Ducks"]) / 3.0)
            row = {
                "name": name,
                "month_lambda": float(lm),
                "div_lambda": float(ld),
                "macro_map": float(score),
                "rare3_mean": rare3,
                "ducks_ap": float(pc["Ducks"]),
                "gulls_ap": float(pc["Gulls"]),
                "worst_month_ap": float(min(extras["month_scores"].values())) if extras["month_scores"] else float("nan"),
                "month_std": float(extras["month_std"]),
                "rare_corr_penalty": float(extras["rare_corr_penalty"]),
                "weights": {n: float(v) for n, v in zip(names, w)},
            }
            rows.append(row)
            if best is None or float(row["macro_map"]) > float(best["macro_map"]):
                best = row

    rep = {
        "components": [{"name": c["name"], "macro_map": float(c["macro"]), "oof_csv": c["oof_csv"], "sub_csv": c["sub_csv"]} for c in components],
        "grid": rows,
        "best_by_macro": best,
    }
    dump_json(out_dir / "blend_penalty_report.json", rep)
    pd.DataFrame(rows).sort_values(["macro_map"], ascending=False).to_csv(out_dir / "blend_penalty_grid.csv", index=False)
    print(f"Saved report: {out_dir / 'blend_penalty_report.json'}", flush=True)
    if best is not None:
        print(
            f"BEST {best['name']} macro={best['macro_map']:.6f} "
            f"rare3={best['rare3_mean']:.6f} ducks={best['ducks_ap']:.6f} gulls={best['gulls_ap']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
