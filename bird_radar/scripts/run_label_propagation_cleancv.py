from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label propagation clean-CV pilot (inductive val)")
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
    p.add_argument("--k-list", type=str, default="10,20")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--betas", type=str, default="0.10,0.15,0.20,0.25,0.30")
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
    out = df[
        [
            "track_id",
            "radar_bird_size",
            "airspeed",
            "min_z",
            "max_z",
            "trajectory_time",
            "timestamp_start_radar_utc",
            "timestamp_end_radar_utc",
        ]
    ].copy()
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


def macro_ap(y_true_idx: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, float]]:
    per_class: dict[str, float] = {}
    for i, c in enumerate(CLASSES):
        y_bin = (y_true_idx == i).astype(int)
        if y_bin.sum() == 0:
            per_class[c] = 0.0
            continue
        per_class[c] = float(average_precision_score(y_bin, probs[:, i]))
    return float(np.mean(list(per_class.values()))), per_class


def normalize_rows(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, None)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def inductive_knn_predict(
    x_labeled: np.ndarray,
    y_labeled_soft: np.ndarray,
    x_query: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    k = int(max(1, min(n_neighbors, len(x_labeled))))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(x_labeled)
    distances, indices = nn.kneighbors(x_query)
    w = np.exp(-distances)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    out = (w[:, :, None] * y_labeled_soft[indices]).sum(axis=1)
    return normalize_rows(out)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.sub_dir.mkdir(parents=True, exist_ok=True)

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    betas = [float(x) for x in args.betas.split(",") if x.strip()]

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

    raw_feat_cols = [
        "month",
        "track_duration_sec",
        "n_points",
        "points_per_second",
        "airspeed",
        "min_z",
        "max_z",
        "altitude_range",
    ] + [f"size__{cat.lower().replace(' ', '_')}" for cat in SIZE_CATS]

    x_train_raw = train_df[raw_feat_cols].to_numpy(dtype=float)
    x_test_raw = test_df[raw_feat_cols].to_numpy(dtype=float)
    oof_base = train_df[CLASSES].to_numpy(dtype=float)
    test_base = test_df[CLASSES].to_numpy(dtype=float)

    month = train.set_index("track_id").loc[train_df["track_id"], "timestamp_start_radar_utc"]
    month = pd.to_datetime(month, errors="coerce", utc=True).dt.month.to_numpy()
    ms_mask = np.isin(month, [9, 10])

    base_cov, base_pc = macro_ap(y, oof_base)
    base_ms, _ = macro_ap(y[ms_mask], oof_base[ms_mask])

    rows: list[dict[str, float]] = []
    best_global = {"k": None, "beta": None, "covered": -1.0, "month_stress": None}
    oof_lp_by_k: dict[int, np.ndarray] = {}

    skf = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.seed)
    split_cache = list(skf.split(x_train_raw, y))

    for k in k_list:
        oof_lp = np.zeros((len(train_df), len(CLASSES)), dtype=float)
        for train_idx, val_idx in split_cache:
            # Clean CV: graph uses only train_fold + test (val stays outside graph).
            x_graph_raw = np.vstack([x_train_raw[train_idx], x_test_raw])
            scaler = StandardScaler()
            x_graph = scaler.fit_transform(x_graph_raw)
            x_train_graph = x_graph[: len(train_idx)]
            x_val = scaler.transform(x_train_raw[val_idx])

            y_graph = -np.ones(len(x_graph), dtype=int)
            y_graph[: len(train_idx)] = y[train_idx]

            lp = LabelSpreading(
                kernel="knn",
                n_neighbors=int(max(1, min(k, len(train_idx)))),
                alpha=args.alpha,
                max_iter=80,
            )
            lp.fit(x_graph, y_graph)
            train_soft = normalize_rows(lp.label_distributions_[: len(train_idx)])

            # Inductive prediction for val via kNN on labeled train_fold nodes only.
            val_soft = inductive_knn_predict(
                x_labeled=x_train_graph,
                y_labeled_soft=train_soft,
                x_query=x_val,
                n_neighbors=k,
            )
            oof_lp[val_idx] = val_soft

        oof_lp = normalize_rows(oof_lp)
        oof_lp_by_k[k] = oof_lp
        lp_cov, _ = macro_ap(y, oof_lp)
        lp_ms, _ = macro_ap(y[ms_mask], oof_lp[ms_mask])

        for beta in betas:
            blend = normalize_rows((1.0 - beta) * oof_base + beta * oof_lp)
            cov, _ = macro_ap(y, blend)
            ms, _ = macro_ap(y[ms_mask], blend[ms_mask])
            rows.append(
                {
                    "k": int(k),
                    "beta": float(beta),
                    "base_covered": float(base_cov),
                    "lp_only_covered": float(lp_cov),
                    "covered_macro_ap": float(cov),
                    "base_month_stress": float(base_ms),
                    "lp_only_month_stress": float(lp_ms),
                    "month_stress_ap": float(ms),
                }
            )
            if cov > float(best_global["covered"]):
                best_global = {"k": int(k), "beta": float(beta), "covered": float(cov), "month_stress": float(ms)}

    grid_df = pd.DataFrame(rows).sort_values(["k", "beta"]).reset_index(drop=True)
    grid_path = args.out_dir / "label_prop_cleancv_inductive_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    best_by_k = (
        grid_df.sort_values("covered_macro_ap", ascending=False)
        .groupby("k", as_index=False)
        .first()
        .sort_values("k")
    )
    best_by_k_path = args.out_dir / "label_prop_cleancv_inductive_best_by_k.csv"
    best_by_k.to_csv(best_by_k_path, index=False)

    # Build test submissions for best beta per k and global best.
    submissions: dict[str, str] = {}
    for k in k_list:
        row_k = best_by_k[best_by_k["k"] == k].iloc[0]
        beta_k = float(row_k["beta"])
        x_graph_raw = np.vstack([x_train_raw, x_test_raw])
        scaler = StandardScaler()
        x_graph = scaler.fit_transform(x_graph_raw)
        y_graph = np.concatenate([y, -np.ones(len(x_test_raw), dtype=int)])
        lp_full = LabelSpreading(
            kernel="knn",
            n_neighbors=int(max(1, min(k, len(y)))),
            alpha=args.alpha,
            max_iter=120,
        )
        lp_full.fit(x_graph, y_graph)
        test_lp = normalize_rows(lp_full.label_distributions_[len(y) :])
        test_blend = normalize_rows((1.0 - beta_k) * test_base + beta_k * test_lp)

        sub_out = sub.copy()
        sub_out[CLASSES] = test_blend
        if "bird_group" in sub_out.columns:
            sub_out["bird_group"] = np.array(CLASSES)[np.argmax(test_blend, axis=1)]
        out_name = f"sub_w80_lp_cleancv_inductive_k{k}_b{int(round(beta_k * 100)):02d}.csv"
        out_path = args.sub_dir / out_name
        sub_out.to_csv(out_path, index=False)
        submissions[f"k{k}"] = str(out_path)

    report = {
        "train_oof_csv": str(args.train_oof_csv),
        "test_base_sub": str(args.test_base_sub),
        "k_list": k_list,
        "betas": betas,
        "alpha": float(args.alpha),
        "cv_splits": int(args.cv_splits),
        "base_covered_macro_ap": float(base_cov),
        "base_month_stress_ap": float(base_ms),
        "best_global": best_global,
        "grid_csv": str(grid_path),
        "best_by_k_csv": str(best_by_k_path),
        "submissions": submissions,
        "note": "clean CV: val is never in LP graph; val predicted inductively via kNN from train-fold LP soft labels",
        "per_class_base": base_pc,
    }
    report_path = args.out_dir / "label_prop_cleancv_inductive_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
