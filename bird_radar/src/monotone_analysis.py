from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import CLASS_TO_INDEX


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 2 or y.size <= 2:
        return float("nan")
    if float(np.std(x)) < 1e-9 or float(np.std(y)) < 1e-9:
        return float("nan")
    c, _ = spearmanr(x, y)
    if not np.isfinite(c):
        return float("nan")
    return float(c)


def find_monotone_candidates(
    labels_idx: np.ndarray,
    months: np.ndarray,
    feature_frame: pd.DataFrame,
    class_names: list[str],
    min_spearman: float = 0.2,
    max_cross_month_sign_flip: int = 0,
    per_class_top_k: int = 3,
    month_min_samples: int = 10,
) -> tuple[dict[int, dict[str, int]], dict[str, object]]:
    """Return per-class monotone feature sign map from month-stable Spearman signal.

    Output map shape:
      {class_idx: {feature_name: +/-1}}
    """
    cols = list(feature_frame.columns)
    x = feature_frame.to_numpy(dtype=np.float32)
    m = months.astype(np.int64)
    unique_months = [int(mm) for mm in sorted(pd.Series(m).dropna().unique().tolist())]

    result: dict[int, dict[str, int]] = {}
    report_cls: dict[str, object] = {}
    for cls in class_names:
        cidx = int(CLASS_TO_INDEX[cls])
        y = (labels_idx == cidx).astype(np.int8)
        picks: list[tuple[str, int, float, int]] = []
        for j, f in enumerate(cols):
            g = _safe_spearman(x[:, j], y)
            if not np.isfinite(g) or abs(float(g)) < float(min_spearman):
                continue
            sign = 1 if float(g) > 0 else -1
            flips = 0
            for mm in unique_months:
                mask = m == mm
                if int(mask.sum()) < int(month_min_samples):
                    continue
                cm = _safe_spearman(x[mask, j], y[mask])
                if not np.isfinite(cm) or abs(float(cm)) <= 0.05:
                    continue
                msign = 1 if float(cm) > 0 else -1
                if msign != sign:
                    flips += 1
            if flips <= int(max_cross_month_sign_flip):
                picks.append((f, int(sign), float(abs(g)), int(flips)))
        picks.sort(key=lambda t: (t[2], -t[3]), reverse=True)
        picks = picks[: max(0, int(per_class_top_k))]
        result[cidx] = {f: int(s) for f, s, _, _ in picks}
        report_cls[cls] = {
            "selected": [
                {
                    "feature": str(f),
                    "sign": int(s),
                    "abs_spearman": float(a),
                    "sign_flips": int(fl),
                }
                for f, s, a, fl in picks
            ],
            "n_selected": int(len(picks)),
        }

    report = {
        "enabled": True,
        "class_names": class_names,
        "min_spearman": float(min_spearman),
        "max_cross_month_sign_flip": int(max_cross_month_sign_flip),
        "per_class_top_k": int(per_class_top_k),
        "month_min_samples": int(month_min_samples),
        "unique_months": unique_months,
        "classes": report_cls,
    }
    return result, report
