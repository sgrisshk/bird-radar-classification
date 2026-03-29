from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading


CLASSES = [
    "Clutter",
    "Cormorants",
    "Pigeons",
    "Ducks",
    "Geese",
    "Gulls",
    "Birds of Prey",
    "Waders",
    "Songbirds",
]

SIZE_CATS = ["Small bird", "Medium bird", "Large bird", "Flock"]


def macro_ap(y_true_idx: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, float]]:
    per_class: dict[str, float] = {}
    for i, c in enumerate(CLASSES):
        y_bin = (y_true_idx == i).astype(int)
        if y_bin.sum() == 0:
            per_class[c] = 0.0
            continue
        per_class[c] = float(average_precision_score(y_bin, probs[:, i]))
    return float(np.mean(list(per_class.values()))), per_class


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label propagation pilot (postprocessing)")
    p.add_argument("--train-csv", type=Path, default=Path("train.csv"))
    p.add_argument("--test-csv", type=Path, default=Path("test.csv"))
    p.add_argument(
        "--train-oof-csv",
        type=Path,
        default=Path("bird_radar/artifacts/oof_csv_for_per_class/oof_temporal_stack9_forward_complete.csv"),
    )
    p.add_argument(
        "--test-base-sub",
        type=Path,
        default=Path("bird_radar/submissions/sub_blend_softmax_w80_sizeprior_b10.csv"),
    )
    p.add_argument("--neighbors", type=int, default=35)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--betas",
        type=str,
        default="0.05,0.10,0.15,0.20,0.30,0.40",
        help="Comma-separated blend weights for LP probs",
    )
    p.add_argument("--out-dir", type=Path, default=Path("bird_radar/artifacts/proposal"))
    p.add_argument("--sub-dir", type=Path, default=Path("bird_radar/submissions"))
    return p.parse_args()


def _count_points_from_time_str(x: object) -> float:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 0.0
    s = str(x).strip()
    if s in {"", "[]", "nan", "None"}:
        return 0.0
    if s[0] == "[" and s[-1] == "]":
        body = s[1:-1].strip()
        if body == "":
            return 0.0
        return float(body.count(",") + 1)
    return 0.0


def build_aux(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["track_id", "radar_bird_size", "airspeed", "min_z", "max_z", "trajectory_time", "timestamp_start_radar_utc", "timestamp_end_radar_utc"]].copy()
    ts0 = pd.to_datetime(out["timestamp_start_radar_utc"], errors="coerce", utc=True)
    ts1 = pd.to_datetime(out["timestamp_end_radar_utc"], errors="coerce", utc=True)
    out["month"] = ts0.dt.month.fillna(0).astype(int)
    out["track_duration_sec"] = (ts1 - ts0).dt.total_seconds().fillna(0.0).astype(float)
    out["n_points"] = out["trajectory_time"].map(_count_points_from_time_str).astype(float)
    out["points_per_second"] = out["n_points"] / np.clip(out["track_duration_sec"], 1.0, None)
    out["altitude_range"] = (out["max_z"].astype(float) - out["min_z"].astype(float)).astype(float)
    for cat in SIZE_CATS:
        out[f"size__{cat.lower().replace(' ', '_')}"] = (out["radar_bird_size"] == cat).astype(float)
    keep = [
        "track_id",
        "month",
        "track_duration_sec",
        "n_points",
        "points_per_second",
        "airspeed",
        "min_z",
        "max_z",
        "altitude_range",
    ] + [f"size__{cat.lower().replace(' ', '_')}" for cat in SIZE_CATS]
    return out[keep]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.sub_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)
    oof = pd.read_csv(args.train_oof_csv)
    sub = pd.read_csv(args.test_base_sub)

    train_aux = build_aux(train)
    test_aux = build_aux(test)

    train_df = train_aux.merge(oof[["track_id"] + CLASSES], on="track_id", how="inner", validate="one_to_one")
    test_df = test_aux.merge(sub[["track_id"] + CLASSES], on="track_id", how="inner", validate="one_to_one")

    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y = train.set_index("track_id").loc[train_df["track_id"], "bird_group"].map(label_to_idx).to_numpy(dtype=int)

    feat_cols = [
        "month",
        "track_duration_sec",
        "n_points",
        "points_per_second",
        "airspeed",
        "min_z",
        "max_z",
        "altitude_range",
    ] + [f"size__{cat.lower().replace(' ', '_')}" for cat in SIZE_CATS] + CLASSES

    X_train = train_df[feat_cols].to_numpy(dtype=float)
    X_test = test_df[feat_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_all = np.vstack([X_train, X_test])
    X_all = scaler.fit_transform(X_all)
    X_train_s = X_all[: len(X_train)]
    X_test_s = X_all[len(X_train) :]

    skf = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.seed)
    oof_lp = np.zeros((len(train_df), len(CLASSES)), dtype=float)
    for _, val_idx in skf.split(X_train_s, y):
        y_masked = y.copy()
        y_masked[val_idx] = -1
        lp = LabelSpreading(
            kernel="knn",
            n_neighbors=args.neighbors,
            alpha=args.alpha,
            max_iter=60,
        )
        lp.fit(X_train_s, y_masked)
        oof_lp[val_idx] = lp.label_distributions_[val_idx]

    oof_base = train_df[CLASSES].to_numpy(dtype=float)
    base_cov, base_pc = macro_ap(y, oof_base)
    lp_cov, lp_pc = macro_ap(y, oof_lp)

    month = train.set_index("track_id").loc[train_df["track_id"], "timestamp_start_radar_utc"]
    month = pd.to_datetime(month, errors="coerce", utc=True).dt.month.to_numpy()
    ms_mask = np.isin(month, [9, 10])
    base_ms, _ = macro_ap(y[ms_mask], oof_base[ms_mask])
    lp_ms, _ = macro_ap(y[ms_mask], oof_lp[ms_mask])

    betas = [float(x) for x in args.betas.split(",") if x.strip()]
    grid_rows: list[dict[str, float]] = []
    best_beta = 0.0
    best_cov = -1.0
    for b in betas:
        blend = (1.0 - b) * oof_base + b * oof_lp
        cov, _ = macro_ap(y, blend)
        ms, _ = macro_ap(y[ms_mask], blend[ms_mask])
        grid_rows.append({"beta": b, "covered_macro_ap": cov, "month_stress_ap": ms})
        if cov > best_cov:
            best_cov = cov
            best_beta = b

    grid_df = pd.DataFrame(grid_rows).sort_values("beta")
    grid_path = args.out_dir / "label_prop_pilot_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    y_all = np.concatenate([y, -np.ones(len(X_test_s), dtype=int)])
    lp_full = LabelSpreading(
        kernel="knn",
        n_neighbors=args.neighbors,
        alpha=args.alpha,
        max_iter=80,
    )
    lp_full.fit(X_all, y_all)
    test_lp = lp_full.label_distributions_[len(X_train_s) :]

    test_base = test_df[CLASSES].to_numpy(dtype=float)
    test_blend = (1.0 - best_beta) * test_base + best_beta * test_lp
    test_blend = np.clip(test_blend, 1e-12, 1.0)
    test_blend = test_blend / np.clip(test_blend.sum(axis=1, keepdims=True), 1e-12, None)

    sub_out = sub.copy()
    sub_out[CLASSES] = test_blend
    if "bird_group" in sub_out.columns:
        sub_out["bird_group"] = np.array(CLASSES)[np.argmax(test_blend, axis=1)]
    out_sub = args.sub_dir / f"sub_w80_labelprop_pilot_b{int(round(best_beta * 100)):02d}.csv"
    sub_out.to_csv(out_sub, index=False)

    report = {
        "train_oof_csv": str(args.train_oof_csv),
        "test_base_sub": str(args.test_base_sub),
        "neighbors": int(args.neighbors),
        "alpha": float(args.alpha),
        "cv_splits": int(args.cv_splits),
        "base_covered_macro_ap": float(base_cov),
        "lp_only_covered_macro_ap": float(lp_cov),
        "base_month_stress_ap": float(base_ms),
        "lp_only_month_stress_ap": float(lp_ms),
        "best_beta": float(best_beta),
        "best_beta_covered_macro_ap": float(best_cov),
        "grid_csv": str(grid_path),
        "submission_csv": str(out_sub),
        "per_class_base": base_pc,
        "per_class_lp_only": lp_pc,
        "per_class_lp_minus_base": {c: float(lp_pc[c] - base_pc[c]) for c in CLASSES},
    }
    report_path = args.out_dir / "label_prop_pilot_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
