from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def month_stable_interactions(
    frame: pd.DataFrame,
    feature_cols: list[str],
    month_col: str = "month",
    top_k: int = 60,
    min_month_samples: int = 15,
) -> tuple[list[list[int]], dict[str, object]]:
    """Build LightGBM interaction_constraints from month-stable feature pairs.

    Strategy:
    - Keep all features as singleton groups (safe baseline behavior).
    - Add top_k pairs with the *lowest* std of Spearman correlation across months.
    """
    if top_k <= 0 or len(feature_cols) <= 1:
        constraints = [[int(i)] for i in range(len(feature_cols))]
        report = {
            "enabled": False,
            "top_k": int(top_k),
            "n_features": int(len(feature_cols)),
            "n_constraints": int(len(constraints)),
            "top_pairs": [],
        }
        return constraints, report

    months = frame[month_col].to_numpy()
    unique_months = [int(m) for m in sorted(pd.Series(months).dropna().unique().tolist())]
    values = frame[feature_cols].to_numpy(dtype=np.float32)
    n = values.shape[1]

    stability: list[tuple[int, int, float, int]] = []
    for i in range(n):
        xi_all = values[:, i]
        for j in range(i + 1, n):
            xj_all = values[:, j]
            cors: list[float] = []
            for m in unique_months:
                mask = months == m
                if int(mask.sum()) < int(min_month_samples):
                    continue
                xi = xi_all[mask]
                xj = xj_all[mask]
                if float(np.std(xi)) < 1e-9 or float(np.std(xj)) < 1e-9:
                    continue
                c, _ = spearmanr(xi, xj)
                if np.isfinite(c):
                    cors.append(float(c))
            if len(cors) >= 2:
                stability.append((i, j, float(np.std(cors)), int(len(cors))))

    stability.sort(key=lambda t: t[2])  # low std first => more stable
    top_pairs = stability[: int(min(top_k, len(stability)))]

    constraints: list[list[int]] = [[int(i)] for i in range(n)]
    constraints.extend([[int(i), int(j)] for i, j, _, _ in top_pairs])

    report_pairs: list[dict[str, object]] = []
    for i, j, s, n_months in top_pairs[:20]:
        report_pairs.append(
            {
                "f1_idx": int(i),
                "f2_idx": int(j),
                "f1": str(feature_cols[i]),
                "f2": str(feature_cols[j]),
                "corr_std": float(s),
                "n_months": int(n_months),
            }
        )

    report = {
        "enabled": True,
        "top_k": int(top_k),
        "n_features": int(n),
        "n_pairs_scored": int(len(stability)),
        "n_pairs_selected": int(len(top_pairs)),
        "n_constraints": int(len(constraints)),
        "unique_months": unique_months,
        "min_month_samples": int(min_month_samples),
        "top_pairs": report_pairs,
    }
    return constraints, report


def remap_interaction_constraints(
    constraints: list[list[int]] | None,
    feature_idx: np.ndarray,
) -> list[list[int]] | None:
    """Remap constraints from global feature indices to class-local indices."""
    if constraints is None:
        return None
    if len(feature_idx) <= 1:
        return None

    local_pos = {int(g): int(i) for i, g in enumerate(feature_idx.tolist())}
    remapped: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for group in constraints:
        mapped = sorted({local_pos[g] for g in group if int(g) in local_pos})
        if len(mapped) <= 0:
            continue
        key = tuple(mapped)
        if key in seen:
            continue
        seen.add(key)
        remapped.append(mapped)

    # Ensure every local feature appears at least as singleton.
    for li in range(len(feature_idx)):
        key = (int(li),)
        if key not in seen:
            remapped.append([int(li)])
            seen.add(key)
    return remapped
