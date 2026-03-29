from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import CLASSES, CLASS_TO_INDEX, DEFAULT_DATA_DIR, ENSEMBLE_DIR, LGBM_DIR, SEQUENCE_DIR, ensure_dirs
from src.inference import make_submission
from src.metrics import macro_map_score, one_hot_labels, per_class_average_precision
from src.training import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--sequence-dir", type=str, default=str(SEQUENCE_DIR))
    parser.add_argument("--lgbm-dir", type=str, default=str(LGBM_DIR))
    parser.add_argument("--output-dir", type=str, default=str(ENSEMBLE_DIR))
    return parser.parse_args()


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / (e.sum() + 1e-8)


def _load_component_pair(base: Path, oof_candidates: list[str], test_candidates: list[str]) -> tuple[np.ndarray, np.ndarray]:
    oof = None
    test = None
    for name in oof_candidates:
        p = base / name
        if p.exists():
            oof = np.load(p)
            break
    for name in test_candidates:
        p = base / name
        if p.exists():
            test = np.load(p)
            break
    if oof is None or test is None:
        raise FileNotFoundError(f"Missing OOF/test predictions under {base}")
    return oof.astype(np.float32), test.astype(np.float32)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    data_dir = Path(args.data_dir)
    sequence_dir = Path(args.sequence_dir)
    lgbm_dir = Path(args.lgbm_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample_sub = pd.read_csv(data_dir / "sample_submission.csv")

    labels_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y_true = one_hot_labels(labels_idx, len(CLASSES))

    component_names: list[str] = []
    oof_components: list[np.ndarray] = []
    test_components: list[np.ndarray] = []

    seed_dirs = sorted([p for p in sequence_dir.glob("seed_*") if p.is_dir()])
    for seed_dir in seed_dirs:
        oof, test = _load_component_pair(
            seed_dir,
            oof_candidates=["oof_calibrated.npy", "oof_probs.npy"],
            test_candidates=["test_calibrated.npy", "test_probs.npy"],
        )
        component_names.append(seed_dir.name)
        oof_components.append(oof)
        test_components.append(test)

    lgbm_oof, lgbm_test = _load_component_pair(
        lgbm_dir,
        oof_candidates=["oof_calibrated.npy", "oof_probs.npy"],
        test_candidates=["test_calibrated.npy", "test_probs.npy"],
    )
    component_names.append("lgbm")
    oof_components.append(lgbm_oof)
    test_components.append(lgbm_test)

    oof_stack = np.stack(oof_components, axis=0)
    test_stack = np.stack(test_components, axis=0)
    n_models = oof_stack.shape[0]
    flat_oof = oof_stack.reshape(n_models, -1)
    corr = np.corrcoef(flat_oof)

    def objective(z: np.ndarray) -> float:
        w = _softmax(z)
        blended = np.tensordot(w, oof_stack, axes=(0, 0))
        return -macro_map_score(y_true, blended)

    z0 = np.zeros(n_models, dtype=np.float64)
    result = minimize(objective, z0, method="L-BFGS-B")
    weights = _softmax(result.x if result.success else z0)

    oof_blend = np.tensordot(weights, oof_stack, axes=(0, 0)).astype(np.float32)
    test_blend = np.tensordot(weights, test_stack, axes=(0, 0)).astype(np.float32)
    component_oof_scores = {name: float(macro_map_score(y_true, oof)) for name, oof in zip(component_names, oof_components)}
    best_single_name = max(component_oof_scores, key=component_oof_scores.get)
    best_single_score = float(component_oof_scores[best_single_name])
    ensemble_score = float(macro_map_score(y_true, oof_blend))
    marginal_gain = ensemble_score - best_single_score

    np.save(output_dir / "oof_ensemble.npy", oof_blend)
    np.save(output_dir / "test_ensemble.npy", test_blend)

    submission = make_submission(test_df["track_id"].to_numpy(), test_blend)
    submission = submission[sample_sub.columns.tolist()]
    submission.to_csv(output_dir / "submission.csv", index=False)

    save_json(
        output_dir / "weights.json",
        {
            "components": component_names,
            "weights": {name: float(w) for name, w in zip(component_names, weights)},
            "opt_success": bool(result.success),
            "opt_message": str(result.message),
            "component_oof_macro_map": component_oof_scores,
            "best_single_component": best_single_name,
            "best_single_oof_macro_map": best_single_score,
            "ensemble_oof_macro_map": ensemble_score,
            "marginal_gain_over_best_single": marginal_gain,
            "oof_per_class_map": per_class_average_precision(y_true, oof_blend),
            "oof_prediction_correlation": {
                component_names[i]: {component_names[j]: float(corr[i, j]) for j in range(n_models)}
                for i in range(n_models)
            },
        },
    )

    print("[ensemble] weights:", {name: round(float(w), 6) for name, w in zip(component_names, weights)}, flush=True)
    print(f"[ensemble] best_single={best_single_name} {best_single_score:.5f}", flush=True)
    print(f"[ensemble] OOF macro mAP: {ensemble_score:.5f} (gain={marginal_gain:+.5f})", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
