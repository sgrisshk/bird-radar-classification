from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
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
    p = argparse.ArgumentParser(description="LP temporal-forward comparable eval")
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
    p.add_argument("--betas", type=str, default="0.10,0.15,0.20,0.25,0.30")
    p.add_argument("--alpha", type=float, default=0.2)
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


def normalize_rows(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, None)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def expand_probs_to_full(probs: np.ndarray, present_classes: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(probs), n_classes), dtype=float)
    idx = np.asarray(present_classes, dtype=int)
    out[:, idx] = probs
    return normalize_rows(out)


def macro_ap(y_true_idx: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, float]]:
    per_class: dict[str, float] = {}
    for i, c in enumerate(CLASSES):
        y_bin = (y_true_idx == i).astype(int)
        if y_bin.sum() == 0:
            per_class[c] = 0.0
            continue
        per_class[c] = float(average_precision_score(y_bin, probs[:, i]))
    return float(np.mean(list(per_class.values()))), per_class


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
    out["year_month"] = ts0.dt.strftime("%Y-%m")
    out["month"] = ts0.dt.month.fillna(0).astype(int)
    out["track_duration_sec"] = (ts1 - ts0).dt.total_seconds().fillna(0.0).astype(float)
    out["n_points"] = out["trajectory_time"].map(_count_points_from_time_str).astype(float)
    out["points_per_second"] = out["n_points"] / np.clip(out["track_duration_sec"], 1.0, None)
    out["altitude_range"] = (out["max_z"].astype(float) - out["min_z"].astype(float)).astype(float)
    for cat in SIZE_CATS:
        out[f"size__{cat.lower().replace(' ', '_')}"] = (out["radar_bird_size"] == cat).astype(float)
    return out


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
    y_all = train.set_index("track_id").loc[train_df["track_id"], "bird_group"].map(label_to_idx).to_numpy(dtype=int)

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
    base_oof = train_df[CLASSES].to_numpy(dtype=float)
    test_base = test_df[CLASSES].to_numpy(dtype=float)

    # Exact comparable temporal-forward month folds.
    split_spec = [
        {"name": "fold1_09_to_10", "train_months": ["2023-09"], "val_months": ["2023-10"]},
        {"name": "fold2_0910_to_01", "train_months": ["2023-09", "2023-10"], "val_months": ["2024-01"]},
        {"name": "fold3_091001_to_04", "train_months": ["2023-09", "2023-10", "2024-01"], "val_months": ["2024-04"]},
    ]

    split_rows: list[dict[str, object]] = []
    indices: list[tuple[np.ndarray, np.ndarray, str]] = []
    ym = train_df["year_month"].to_numpy()
    for sp in split_spec:
        tr_mask = np.isin(ym, np.asarray(sp["train_months"], dtype=object))
        va_mask = np.isin(ym, np.asarray(sp["val_months"], dtype=object))
        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]
        indices.append((tr_idx, va_idx, str(sp["name"])))
        split_rows.append(
            {
                "fold": str(sp["name"]),
                "train_months": ",".join(sp["train_months"]),
                "val_months": ",".join(sp["val_months"]),
                "train_n": int(len(tr_idx)),
                "val_n": int(len(va_idx)),
            }
        )
    split_path = args.out_dir / "label_prop_temporal_forward_splits.csv"
    pd.DataFrame(split_rows).to_csv(split_path, index=False)

    rows: list[dict[str, object]] = []
    best_global = {"k": None, "beta": None, "covered_macro_ap": -1.0, "fold_macro_mean": -1.0}

    for k in k_list:
        lp_oof = np.zeros_like(base_oof)
        fold_lp_only: list[dict[str, object]] = []
        for tr_idx, va_idx, fold_name in indices:
            x_graph_raw = np.vstack([x_train_raw[tr_idx], x_test_raw])  # val excluded from graph
            scaler = StandardScaler()
            x_graph = scaler.fit_transform(x_graph_raw)
            x_tr_graph = x_graph[: len(tr_idx)]
            x_val = scaler.transform(x_train_raw[va_idx])

            y_graph = -np.ones(len(x_graph), dtype=int)
            y_graph[: len(tr_idx)] = y_all[tr_idx]

            lp = LabelSpreading(
                kernel="knn",
                n_neighbors=int(max(1, min(k, len(tr_idx)))),
                alpha=float(args.alpha),
                max_iter=120,
            )
            lp.fit(x_graph, y_graph)
            tr_soft_small = normalize_rows(lp.label_distributions_[: len(tr_idx)])
            tr_soft = expand_probs_to_full(tr_soft_small, lp.classes_, len(CLASSES))
            val_soft = inductive_knn_predict(
                x_labeled=x_tr_graph,
                y_labeled_soft=tr_soft,
                x_query=x_val,
                n_neighbors=k,
            )
            lp_oof[va_idx] = val_soft

            lp_only_fold, _ = macro_ap(y_all[va_idx], val_soft)
            fold_lp_only.append({"k": k, "fold": fold_name, "lp_only_fold_ap": float(lp_only_fold), "val_n": int(len(va_idx))})

        lp_oof = normalize_rows(lp_oof)

        for beta in betas:
            blend = normalize_rows((1.0 - beta) * base_oof + beta * lp_oof)
            fold_vals: list[float] = []
            for tr_idx, va_idx, fold_name in indices:
                fold_ap, _ = macro_ap(y_all[va_idx], blend[va_idx])
                fold_base_ap, _ = macro_ap(y_all[va_idx], base_oof[va_idx])
                rows.append(
                    {
                        "k": int(k),
                        "beta": float(beta),
                        "fold": fold_name,
                        "val_n": int(len(va_idx)),
                        "fold_macro_ap_blend": float(fold_ap),
                        "fold_macro_ap_base": float(fold_base_ap),
                        "fold_delta": float(fold_ap - fold_base_ap),
                    }
                )
                fold_vals.append(float(fold_ap))
            fold_macro_mean = float(np.mean(fold_vals))
            if fold_macro_mean > float(best_global["fold_macro_mean"]):
                best_global = {
                    "k": int(k),
                    "beta": float(beta),
                    "covered_macro_ap": float(macro_ap(y_all, blend)[0]),
                    "fold_macro_mean": fold_macro_mean,
                }

        lp_only_path = args.out_dir / f"label_prop_temporal_forward_lp_only_k{k}.csv"
        pd.DataFrame(fold_lp_only).to_csv(lp_only_path, index=False)

    fold_grid = pd.DataFrame(rows)
    fold_grid_path = args.out_dir / "label_prop_temporal_forward_fold_grid.csv"
    fold_grid.to_csv(fold_grid_path, index=False)

    summary = (
        fold_grid.groupby(["k", "beta"], as_index=False)
        .agg(
            fold_macro_mean=("fold_macro_ap_blend", "mean"),
            fold_base_mean=("fold_macro_ap_base", "mean"),
            fold_delta_mean=("fold_delta", "mean"),
        )
        .sort_values(["k", "beta"])
    )
    summary_path = args.out_dir / "label_prop_temporal_forward_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Build one submission per k using best beta by fold-mean.
    sub_paths: dict[str, str] = {}
    for k in k_list:
        row_k = summary[summary["k"] == k].sort_values("fold_macro_mean", ascending=False).iloc[0]
        beta_k = float(row_k["beta"])
        x_graph_raw = np.vstack([x_train_raw, x_test_raw])
        scaler = StandardScaler()
        x_graph = scaler.fit_transform(x_graph_raw)
        y_graph = np.concatenate([y_all, -np.ones(len(x_test_raw), dtype=int)])
        lp_full = LabelSpreading(
            kernel="knn",
            n_neighbors=int(max(1, min(k, len(y_all)))),
            alpha=float(args.alpha),
            max_iter=140,
        )
        lp_full.fit(x_graph, y_graph)
        test_lp_small = normalize_rows(lp_full.label_distributions_[len(y_all) :])
        test_lp = expand_probs_to_full(test_lp_small, lp_full.classes_, len(CLASSES))
        test_blend = normalize_rows((1.0 - beta_k) * test_base + beta_k * test_lp)

        out = sub.copy()
        out[CLASSES] = test_blend
        if "bird_group" in out.columns:
            out["bird_group"] = np.asarray(CLASSES)[np.argmax(test_blend, axis=1)]
        out_path = args.sub_dir / f"sub_w80_lp_temporal_forward_k{k}_b{int(round(beta_k * 100)):02d}.csv"
        out.to_csv(out_path, index=False)
        sub_paths[f"k{k}"] = str(out_path)

    report = {
        "split_spec": split_spec,
        "split_counts_csv": str(split_path),
        "k_list": k_list,
        "betas": betas,
        "alpha": float(args.alpha),
        "summary_csv": str(summary_path),
        "fold_grid_csv": str(fold_grid_path),
        "best_global": best_global,
        "submissions": sub_paths,
        "note": "val outside graph; fold metric on val only; aggregate by mean across 3 temporal folds",
    }
    report_path = args.out_dir / "label_prop_temporal_forward_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
