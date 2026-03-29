from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.metrics import macro_map_score, one_hot_labels, per_class_average_precision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Greedy per-class alpha optimization for two OOF predictions.")
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--base-oof-csv", type=str, required=True)
    p.add_argument("--temporal-oof-csv", type=str, required=True)
    p.add_argument("--id-col", type=str, default="track_id")
    p.add_argument("--label-col", type=str, default="bird_group")
    p.add_argument("--alpha-grid", type=str, default="0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50")
    p.add_argument("--start-alpha", type=float, default=0.90)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--base-test-csv", type=str, default="")
    p.add_argument("--temporal-test-csv", type=str, default="")
    return p.parse_args()


def _check_submission_like(df: pd.DataFrame, name: str, id_col: str) -> None:
    req = [id_col, *CLASSES]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing columns: {miss}")


def _blend(base: np.ndarray, temp: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    # alpha shape: [C]
    return np.clip(base * alpha[None, :] + temp * (1.0 - alpha[None, :]), 0.0, 1.0)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_csv, usecols=[args.id_col, args.label_col])
    base_oof = pd.read_csv(args.base_oof_csv)
    temp_oof = pd.read_csv(args.temporal_oof_csv)

    _check_submission_like(base_oof, "base-oof-csv", args.id_col)
    _check_submission_like(temp_oof, "temporal-oof-csv", args.id_col)

    m = (
        train.merge(base_oof[[args.id_col, *CLASSES]], on=args.id_col, how="inner", suffixes=("", "_b"))
        .merge(temp_oof[[args.id_col, *CLASSES]], on=args.id_col, how="inner", suffixes=("_base", "_temp"))
        .copy()
    )
    if len(m) == 0:
        raise RuntimeError("No overlapping rows between train and oof predictions")

    y_idx = m[args.label_col].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = one_hot_labels(y_idx, len(CLASSES)).astype(np.float32)
    base = m[[f"{c}_base" for c in CLASSES]].to_numpy(dtype=np.float32)
    temp = m[[f"{c}_temp" for c in CLASSES]].to_numpy(dtype=np.float32)

    grid = np.array([float(x.strip()) for x in args.alpha_grid.split(",") if x.strip()], dtype=np.float32)
    if len(grid) == 0:
        raise ValueError("alpha-grid is empty")

    alpha = np.full((len(CLASSES),), float(args.start_alpha), dtype=np.float32)
    best_pred = _blend(base, temp, alpha)
    best_macro = float(macro_map_score(y, best_pred))

    for _ in range(max(int(args.passes), 1)):
        changed = False
        for ci in range(len(CLASSES)):
            cur_best = alpha[ci]
            cur_macro = best_macro
            for a in grid:
                trial = alpha.copy()
                trial[ci] = float(a)
                pred = _blend(base, temp, trial)
                score = float(macro_map_score(y, pred))
                if score > cur_macro + 1e-12:
                    cur_macro = score
                    cur_best = float(a)
            if cur_best != alpha[ci]:
                alpha[ci] = cur_best
                best_pred = _blend(base, temp, alpha)
                best_macro = float(macro_map_score(y, best_pred))
                changed = True
        if not changed:
            break

    report = {
        "n_rows": int(len(m)),
        "alpha_grid": [float(x) for x in grid.tolist()],
        "start_alpha": float(args.start_alpha),
        "passes": int(args.passes),
        "best_macro_map": best_macro,
        "alpha_by_class": {cls: float(alpha[i]) for i, cls in enumerate(CLASSES)},
        "per_class_ap_base": per_class_average_precision(y, base),
        "per_class_ap_temporal": per_class_average_precision(y, temp),
        "per_class_ap_blend": per_class_average_precision(y, best_pred),
    }
    (out_dir / "per_class_alpha_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.base_test_csv.strip() and args.temporal_test_csv.strip():
        btest = pd.read_csv(args.base_test_csv)
        ttest = pd.read_csv(args.temporal_test_csv)
        _check_submission_like(btest, "base-test-csv", args.id_col)
        _check_submission_like(ttest, "temporal-test-csv", args.id_col)
        if len(btest) != len(ttest) or not np.array_equal(btest[args.id_col].to_numpy(), ttest[args.id_col].to_numpy()):
            raise RuntimeError("test track_id mismatch between base-test-csv and temporal-test-csv")

        p_base = btest[CLASSES].to_numpy(dtype=np.float32)
        p_temp = ttest[CLASSES].to_numpy(dtype=np.float32)
        p_bl = _blend(p_base, p_temp, alpha)
        out_sub = btest[[args.id_col]].copy()
        out_sub[CLASSES] = p_bl
        out_path = out_dir / "sub_blend_per_class_alpha.csv"
        out_sub.to_csv(out_path, index=False)
        print(str(out_path), flush=True)

    print(str(out_dir / "per_class_alpha_report.json"), flush=True)


if __name__ == "__main__":
    main()

