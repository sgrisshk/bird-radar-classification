#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds


def _macro_map_weighted(y_true: np.ndarray, y_prob: np.ndarray, sample_weight: np.ndarray | None) -> float:
    vals: list[float] = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_prob[:, j]
        if yt.sum() <= 0:
            vals.append(0.0)
        else:
            vals.append(float(average_precision_score(yt, yp, sample_weight=sample_weight)))
    return float(np.mean(vals))


def _fold_ids(train_df: pd.DataFrame, n_splits: int, time_col: str, group_col: str) -> np.ndarray:
    cv_df = pd.DataFrame({"_cv_ts": train_df[time_col], "_cv_group": train_df[group_col].astype(np.int64)})
    folds = make_forward_temporal_group_folds(
        cv_df,
        timestamp_col="_cv_ts",
        group_col="_cv_group",
        n_splits=int(n_splits),
    )
    out = np.full(len(train_df), -1, dtype=np.int64)
    for k, (_, va_idx) in enumerate(folds):
        out[np.asarray(va_idx, dtype=np.int64)] = int(k)
    return out


def _align_oof_from_csv(path: str, ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path, usecols=["track_id", *CLASSES])
    mp = {int(r["track_id"]): np.asarray([r[c] for c in CLASSES], dtype=np.float32) for _, r in df.iterrows()}
    return np.stack([mp[int(t)] for t in ids], axis=0).astype(np.float32)


def _align_oof_from_npy(path: str, ids_path: str, ids_ref: np.ndarray) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    ids = np.load(ids_path).astype(np.int64)
    pos = {int(t): i for i, t in enumerate(ids)}
    return np.stack([arr[pos[int(t)]] for t in ids_ref], axis=0).astype(np.float32)


def _spearman_flat(a: np.ndarray, b: np.ndarray) -> float:
    ax = a.reshape(-1)
    bx = b.reshape(-1)
    ra = pd.Series(ax).rank(method="average").to_numpy(dtype=np.float64)
    rb = pd.Series(bx).rank(method="average").to_numpy(dtype=np.float64)
    sa = np.std(ra)
    sb = np.std(rb)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _eval_candidate(
    y: np.ndarray,
    teacher: np.ndarray,
    candidate: np.ndarray,
    fold_id: np.ndarray,
    weights: np.ndarray,
    w_grid: np.ndarray,
) -> dict[str, Any]:
    fids = sorted(int(x) for x in np.unique(fold_id))
    teacher_fold: list[float] = []
    cand_fold: list[float] = []
    for f in fids:
        m = fold_id == f
        teacher_fold.append(_macro_map_weighted(y[m], teacher[m], sample_weight=weights[m]))
        cand_fold.append(_macro_map_weighted(y[m], candidate[m], sample_weight=weights[m]))

    teacher_mean = float(np.average(teacher_fold, weights=[float(weights[fold_id == f].sum()) for f in fids]))
    cand_mean = float(np.average(cand_fold, weights=[float(weights[fold_id == f].sum()) for f in fids]))

    best = {"w_teacher": 1.0, "mean": teacher_mean, "min_fold_delta": 0.0}
    for w in w_grid:
        p = np.clip(w * teacher + (1.0 - w) * candidate, 0.0, 1.0)
        pred_fold: list[float] = []
        for f in fids:
            m = fold_id == f
            pred_fold.append(_macro_map_weighted(y[m], p[m], sample_weight=weights[m]))
        mean = float(np.average(pred_fold, weights=[float(weights[fold_id == f].sum()) for f in fids]))
        deltas = np.asarray(pred_fold, dtype=np.float64) - np.asarray(teacher_fold, dtype=np.float64)
        mfd = float(np.min(deltas))
        if mean > float(best["mean"]):
            best = {"w_teacher": float(w), "mean": mean, "min_fold_delta": mfd}

    return {
        "weighted_teacher_mean": teacher_mean,
        "weighted_candidate_mean": cand_mean,
        "weighted_best_blend_mean": float(best["mean"]),
        "weighted_gain": float(best["mean"] - teacher_mean),
        "weighted_min_fold_delta": float(best["min_fold_delta"]),
        "best_w_teacher": float(best["w_teacher"]),
    }


def _candidate_specs(base_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "name": "teacher_forward",
            "oof_csv": "bird_radar/artifacts/oof_csv_for_per_class/oof_base_forward_complete.csv",
            "sub_csv": "bird_radar/artifacts/temporal_model0_reg_drop10_none_complete/sub_temporal_best.csv",
            "notes": "base forward teacher",
        },
        {
            "name": "teacher_reverse",
            "oof_csv": "bird_radar/artifacts/oof_csv_for_per_class/oof_base_reverse_complete.csv",
            "sub_csv": "bird_radar/artifacts/temporal_model0_reg_drop10_none_reverse_complete/sub_temporal_best.csv",
            "notes": "reverse teacher",
        },
        {
            "name": "teacher_fr_oof",
            "oof_csv": "bird_radar/artifacts/oof_csv_for_per_class/oof_teacher_fr_complete.csv",
            "sub_csv": "bird_radar/artifacts/temporal_model0_reg_drop10_none_reverse_complete/sub_teacher_fr.csv",
            "notes": "teacher FR reference anchor",
        },
        {
            "name": "teacher_fr_w35_65",
            "oof_blend": {
                "forward_csv": "bird_radar/artifacts/oof_csv_for_per_class/oof_base_forward_complete.csv",
                "reverse_csv": "bird_radar/artifacts/oof_csv_for_per_class/oof_base_reverse_complete.csv",
                "w_forward": 0.35,
                "w_reverse": 0.65,
            },
            "sub_csv": "bird_radar/artifacts/temporal_model0_forward_reverse_blend_w35_65/sub_blend_mean.csv",
            "notes": "fixed FR blend 35/65",
        },
        {
            "name": "teacher_fr_classwise_constrained",
            "oof_from_json": "bird_radar/artifacts/temporal_model0_reg_drop10_none_reverse_complete/sub_classwise_constrained_search_forward_reverse.json",
            "sub_csv": "bird_radar/artifacts/temporal_model0_reg_drop10_none_reverse_complete/sub_classwise_constrained_search_forward_reverse.csv",
            "notes": "classwise constrained forward+reverse",
        },
        {
            "name": "fold2_specialist_entropy_sub_best",
            "oof_csv": "bird_radar/artifacts/fold2_specialist_topk_override_scan_entropy/oof_best.csv",
            "sub_csv": "bird_radar/artifacts/fold2_specialist_topk_override_scan_entropy/sub_best.csv",
            "report_json": "bird_radar/artifacts/fold2_specialist_topk_override_scan_entropy/report.json",
            "notes": "entropy-selected specialist override",
        },
        {
            "name": "fold2_specialist_entropy_weighted_capped_v1",
            "oof_csv": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_v1/oof_best.csv",
            "sub_csv": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_v1/sub_best.csv",
            "report_json": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_v1/report.json",
            "notes": "entropy weighted capped q<=0.10 alpha<=0.35 classes=Clutter,Gulls",
        },
        {
            "name": "fold2_specialist_entropy_weighted_capped_v2_onlybetter",
            "oof_csv": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_v2_onlybetter/oof_best.csv",
            "sub_csv": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_v2_onlybetter/sub_best.csv",
            "report_json": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_v2_onlybetter/report.json",
            "notes": "entropy weighted capped + only_if_better classes=Clutter,Gulls",
        },
        {
            "name": "fold2_specialist_entropy_weighted_capped_v3_cap10",
            "oof_csv": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_v3_cap10/oof_best.csv",
            "sub_csv": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_v3_cap10/sub_best.csv",
            "report_json": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_v3_cap10/report.json",
            "notes": "entropy weighted cap<=0.10 classes=Clutter,Gulls",
        },
        {
            "name": "fold2_specialist_entropy_weighted_capped_v1b",
            "oof_csv": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_v1b/oof_best.csv",
            "sub_csv": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_v1b/sub_best.csv",
            "report_json": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_v1b/report.json",
            "notes": "entropy weighted cap<=0.08 classes=Clutter,Gulls",
        },
        {
            "name": "fold2_specialist_entropy_weighted_capped_allclasses_v1",
            "oof_csv": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_allclasses_v1/oof_best.csv",
            "sub_csv": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_allclasses_v1/sub_best.csv",
            "report_json": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_allclasses_v1/report.json",
            "notes": "entropy weighted cap<=0.08 all classes",
        },
        {
            "name": "fold2_specialist_entropy_weighted_capped_allclasses_v3_qfine_testcap",
            "oof_csv": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_allclasses_v3_qfine_testcap/oof_best.csv",
            "sub_csv": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_allclasses_v3_qfine_testcap/sub_best.csv",
            "report_json": "bird_radar/artifacts/fold2_specialist_entropy_weighted_capped_allclasses_v3_qfine_testcap/report.json",
            "notes": "entropy weighted cap<=0.08 (train+test), q-fine all classes",
        },
        {
            "name": "fold1_specialist_entropy_weighted_capped_allclasses_v1",
            "oof_csv": "bird_radar/artifacts/fold1_specialist_entropy_weighted_capped_allclasses_v1/oof_best.csv",
            "sub_csv": "bird_radar/artifacts/fold1_specialist_entropy_weighted_capped_allclasses_v1/sub_best.csv",
            "report_json": "bird_radar/artifacts/fold1_specialist_entropy_weighted_capped_allclasses_v1/report.json",
            "notes": "focus-fold1 specialist, cap<=0.08",
        },
        {
            "name": "fold12_specialist_bank_mean_entropy_weighted_capped_allclasses_v1",
            "oof_csv": "bird_radar/artifacts/fold12_specialist_bank_mean_entropy_weighted_capped_allclasses_v1/oof_best.csv",
            "sub_csv": "bird_radar/artifacts/fold12_specialist_bank_mean_entropy_weighted_capped_allclasses_v1/sub_best.csv",
            "report_json": "bird_radar/artifacts/fold12_specialist_bank_mean_entropy_weighted_capped_allclasses_v1/report.json",
            "notes": "bank mean(fold1,fold2) specialist, cap<=0.08",
        },
        {
            "name": "fold12_specialist_bank_bestproxytrack_allclasses_entropy_weighted_capped_v1",
            "oof_csv": "bird_radar/artifacts/fold12_specialist_bank_bestproxytrack_allclasses_entropy_weighted_capped_v1/oof_best.csv",
            "sub_csv": "bird_radar/artifacts/fold12_specialist_bank_bestproxytrack_allclasses_entropy_weighted_capped_v1/sub_best.csv",
            "report_json": "bird_radar/artifacts/fold12_specialist_bank_bestproxytrack_allclasses_entropy_weighted_capped_v1/report.json",
            "notes": "bank best-proxy-track(fold1,fold2), cap<=0.08",
        },
        {
            "name": "fold12_specialist_bank_bestproxyclass_top3_entropy_weighted_capped_v1",
            "oof_csv": "bird_radar/artifacts/fold12_specialist_bank_bestproxyclass_top3_entropy_weighted_capped_v1/oof_best.csv",
            "sub_csv": "bird_radar/artifacts/fold12_specialist_bank_bestproxyclass_top3_entropy_weighted_capped_v1/sub_best.csv",
            "report_json": "bird_radar/artifacts/fold12_specialist_bank_bestproxyclass_top3_entropy_weighted_capped_v1/report.json",
            "notes": "bank best-proxy-class + top3 classes, cap<=0.08",
        },
        {
            "name": "fold12_specialist_bank_bestproxytrack_budget_ap_tau002_floor50_v2",
            "oof_csv": "bird_radar/artifacts/fold12_specialist_bank_bestproxytrack_budget_ap_tau002_floor50_v2/oof_best.csv",
            "sub_csv": "bird_radar/artifacts/fold12_specialist_bank_bestproxytrack_budget_ap_tau002_floor50_v2/sub_best.csv",
            "report_json": "bird_radar/artifacts/fold12_specialist_bank_bestproxytrack_budget_ap_tau002_floor50_v2/report.json",
            "notes": "best-proxy-track + per-class budget AP tau0.02 floor50",
        },
        {
            "name": "fold12_specialist_bank_bestproxytrack_budget_foldmean_tau002_floor50_v1",
            "oof_csv": "bird_radar/artifacts/fold12_specialist_bank_bestproxytrack_budget_foldmean_tau002_floor50_v1/oof_best.csv",
            "sub_csv": "bird_radar/artifacts/fold12_specialist_bank_bestproxytrack_budget_foldmean_tau002_floor50_v1/sub_best.csv",
            "report_json": "bird_radar/artifacts/fold12_specialist_bank_bestproxytrack_budget_foldmean_tau002_floor50_v1/report.json",
            "notes": "best-proxy-track + per-class budget foldmean tau0.02 floor50",
        },
        {
            "name": "fold12_specialist_bank_bestproxytrack_budget_ap_tau002_floor0_v1",
            "oof_csv": "bird_radar/artifacts/fold12_specialist_bank_bestproxytrack_budget_ap_tau002_floor0_v1/oof_best.csv",
            "sub_csv": "bird_radar/artifacts/fold12_specialist_bank_bestproxytrack_budget_ap_tau002_floor0_v1/sub_best.csv",
            "report_json": "bird_radar/artifacts/fold12_specialist_bank_bestproxytrack_budget_ap_tau002_floor0_v1/report.json",
            "notes": "best-proxy-track + per-class budget AP tau0.02 floor0",
        },
        {
            "name": "catboost_rich_temporal_quick",
            "oof_npy": "bird_radar/artifacts/redesign_mps_tabular_catboost_rich_temporal_seed777_quick/artifacts/catboost_ovr_oof.npy",
            "ids_npy": "bird_radar/artifacts/redesign_mps_tabular_catboost_rich_temporal_seed777_quick/train_track_ids.npy",
            "sub_csv": "bird_radar/artifacts/redesign_mps_tabular_catboost_rich_temporal_seed777_quick/submission_catboost_ovr.csv",
            "notes": "catboost temporal quick",
        },
        {
            "name": "lgbm_rich_temporal_quick_start1",
            "oof_npy": "bird_radar/artifacts/redesign_mps_tabular_lgbm_rich_temporal_seed777_quick_start1/artifacts/lgbm_ovr_oof.npy",
            "ids_npy": "bird_radar/artifacts/redesign_mps_tabular_lgbm_rich_temporal_seed777_quick_start1/train_track_ids.npy",
            "sub_csv": "bird_radar/artifacts/redesign_mps_tabular_lgbm_rich_temporal_seed777_quick_start1/submission_lgbm_ovr.csv",
            "notes": "lgbm temporal start1",
        },
        {
            "name": "lgbm_focusfold2_specialist",
            "oof_npy": "bird_radar/artifacts/redesign_mps_tabular_lgbm_rich_temporal_seed777_focusfold2_specialist/artifacts/lgbm_ovr_oof.npy",
            "ids_npy": "bird_radar/artifacts/redesign_mps_tabular_lgbm_rich_temporal_seed777_focusfold2_specialist/train_track_ids.npy",
            "sub_csv": "bird_radar/artifacts/redesign_mps_tabular_lgbm_rich_temporal_seed777_focusfold2_specialist/submission_lgbm_ovr.csv",
            "notes": "fold2 focus specialist raw",
        },
        {
            "name": "lgbm_focusfold2_weighted_constrained_v2",
            "oof_from_json": "bird_radar/artifacts/redesign_mps_tabular_lgbm_rich_temporal_seed777_focusfold2_specialist/sub_classwise_constrained_search_weighted_adv_v2.json",
            "sub_csv": "bird_radar/artifacts/redesign_mps_tabular_lgbm_rich_temporal_seed777_focusfold2_specialist/sub_classwise_constrained_search_weighted_adv_v2.csv",
            "notes": "weighted constrained search (adv v2)",
        },
        {
            "name": "seq_fr_best",
            "oof_npy": "bird_radar/artifacts/redesign_mps_seq_fr_v2_balanced_seed777_from_forward_reverse/artifacts/seq_fr_oof.npy",
            "ids_npy": "bird_radar/artifacts/redesign_mps_seq_fr_v2_balanced_seed777_from_forward_reverse/train_track_ids.npy",
            "sub_csv": "bird_radar/artifacts/redesign_mps_seq_fr_v2_balanced_seed777_from_forward_reverse/submission_seq_fr_best.csv",
            "notes": "seq forward+reverse blend",
        },
    ]


def _load_from_json_blend(json_path: str, ids: np.ndarray) -> tuple[np.ndarray, str]:
    d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    teacher = _align_oof_from_csv(d["teacher_oof_csv"], ids)
    new = _align_oof_from_npy(d["new_oof_npy"], d["new_track_ids_npy"], ids)
    mode = str(d.get("mode", "prob_mean"))
    chosen = d.get("chosen_weights", {})
    pred = teacher.copy()
    classes = []
    for cls, w_new in chosen.items():
        j = CLASSES.index(cls)
        w_new = float(w_new)
        if mode == "rank_mean":
            tb = pd.Series(teacher[:, j]).rank(method="average").to_numpy(dtype=np.float32)
            nb = pd.Series(new[:, j]).rank(method="average").to_numpy(dtype=np.float32)
            if len(tb) > 1:
                tb = (tb - 1.0) / float(len(tb) - 1)
                nb = (nb - 1.0) / float(len(nb) - 1)
            pred[:, j] = np.clip((1.0 - w_new) * tb + w_new * nb, 0.0, 1.0)
        else:
            pred[:, j] = np.clip((1.0 - w_new) * teacher[:, j] + w_new * new[:, j], 0.0, 1.0)
        classes.append(f"{cls}:{w_new:.2f}")
    return pred.astype(np.float32), "chosen=" + ",".join(classes)


def main() -> None:
    p = argparse.ArgumentParser(description="Build weighted shortlist table for offline candidate ranking.")
    p.add_argument("--train-csv", default="train.csv")
    p.add_argument("--teacher-fr-oof-csv", default="bird_radar/artifacts/oof_csv_for_per_class/oof_teacher_fr_complete.csv")
    p.add_argument(
        "--sample-weights-npy",
        default="bird_radar/artifacts/adversarial_weights_tabular_only_v2/artifacts/adversarial_weights_train.npy",
    )
    p.add_argument(
        "--reference-submission-csv",
        default="bird_radar/artifacts/temporal_model0_forward_reverse_blend_w35_65/sub_blend_mean.csv",
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--time-col", default="timestamp_start_radar_utc")
    p.add_argument("--group-col", default="observation_id")
    p.add_argument("--weights-grid", default="1.00,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.40,0.30,0.20,0.10,0.00")
    p.add_argument("--output-csv", default="bird_radar/artifacts/shortlist_weighted_proxy_v1.csv")
    p.add_argument("--output-json", default="bird_radar/artifacts/shortlist_weighted_proxy_v1.json")
    p.add_argument("--go-min-gain", type=float, default=0.006)
    p.add_argument("--go-min-fold-delta", type=float, default=-0.002)
    p.add_argument("--go-max-override-rate", type=float, default=0.10)
    p.add_argument("--go-max-spearman", type=float, default=0.995)
    args = p.parse_args()

    train = pd.read_csv(args.train_csv, usecols=["track_id", "bird_group", args.time_col, args.group_col])
    ids = train["track_id"].to_numpy(dtype=np.int64)
    y_idx = train["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = np.zeros((len(train), len(CLASSES)), dtype=np.float32)
    y[np.arange(len(train)), y_idx] = 1.0
    fold_id = _fold_ids(train, n_splits=int(args.n_splits), time_col=str(args.time_col), group_col=str(args.group_col))
    sample_weights = np.asarray(np.load(args.sample_weights_npy), dtype=np.float32).reshape(-1)
    if len(sample_weights) != len(ids):
        raise ValueError(f"sample_weights length mismatch: {len(sample_weights)} vs train {len(ids)}")
    sample_weights = np.clip(sample_weights, 1e-8, None).astype(np.float32)

    teacher_fr = _align_oof_from_csv(args.teacher_fr_oof_csv, ids)
    w_grid = np.asarray([float(x.strip()) for x in str(args.weights_grid).split(",") if x.strip()], dtype=np.float32)

    ref_sub = pd.read_csv(args.reference_submission_csv, usecols=["track_id", *CLASSES]).set_index("track_id")
    ref_mat = ref_sub.loc[:, CLASSES].to_numpy(dtype=np.float32)

    rows: list[dict[str, Any]] = []
    specs = _candidate_specs(PROJECT_ROOT)
    for spec in specs:
        name = spec["name"]
        sub_path = str(spec["sub_csv"])
        oof_note = ""
        if "oof_csv" in spec:
            candidate_oof = _align_oof_from_csv(str(spec["oof_csv"]), ids)
        elif "oof_npy" in spec:
            candidate_oof = _align_oof_from_npy(str(spec["oof_npy"]), str(spec["ids_npy"]), ids)
        elif "oof_blend" in spec:
            b = spec["oof_blend"]
            fw = _align_oof_from_csv(str(b["forward_csv"]), ids)
            rv = _align_oof_from_csv(str(b["reverse_csv"]), ids)
            candidate_oof = np.clip(float(b["w_forward"]) * fw + float(b["w_reverse"]) * rv, 0.0, 1.0).astype(np.float32)
            oof_note = f"oof_blend={b['w_forward']:.2f}/{b['w_reverse']:.2f}"
        elif "oof_from_json" in spec:
            candidate_oof, oof_note = _load_from_json_blend(str(spec["oof_from_json"]), ids)
        else:
            continue

        eval_m = _eval_candidate(
            y=y,
            teacher=teacher_fr,
            candidate=candidate_oof,
            fold_id=fold_id,
            weights=sample_weights,
            w_grid=w_grid,
        )

        sub_df = pd.read_csv(sub_path, usecols=["track_id", *CLASSES]).set_index("track_id")
        cand_mat = sub_df.loc[ref_sub.index, CLASSES].to_numpy(dtype=np.float32)
        spearman = _spearman_flat(ref_mat, cand_mat)
        mae = float(np.mean(np.abs(ref_mat - cand_mat)))

        override_rate = None
        if "report_json" in spec and Path(spec["report_json"]).exists():
            rep = json.loads(Path(spec["report_json"]).read_text(encoding="utf-8"))
            override_rate = float(rep.get("best_override_frac_test", np.nan))

        notes = str(spec.get("notes", ""))
        if oof_note:
            notes = f"{notes}; {oof_note}" if notes else oof_note

        override_ok = True if override_rate is None or not np.isfinite(override_rate) else bool(override_rate <= float(args.go_max_override_rate))
        gain_ok = bool(eval_m["weighted_gain"] >= float(args.go_min_gain))
        min_fold_ok = bool(eval_m["weighted_min_fold_delta"] >= float(args.go_min_fold_delta))
        spearman_ok = bool(spearman <= float(args.go_max_spearman))
        go_flag = bool(gain_ok and min_fold_ok and override_ok and spearman_ok)

        rows.append(
            {
                "name": name,
                "path": sub_path,
                "weighted_teacher_mean": eval_m["weighted_teacher_mean"],
                "weighted_candidate_mean": eval_m["weighted_candidate_mean"],
                "weighted_best_blend_mean": eval_m["weighted_best_blend_mean"],
                "weighted_gain": eval_m["weighted_gain"],
                "weighted_min_fold_delta": eval_m["weighted_min_fold_delta"],
                "best_w_teacher": eval_m["best_w_teacher"],
                "override_rate": override_rate,
                "spearman_vs_current_best": spearman,
                "mae_vs_current_best": mae,
                "go_flag": go_flag,
                "go_checks": f"gain={gain_ok},min_fold={min_fold_ok},override={override_ok},spearman={spearman_ok}",
                "notes": notes,
            }
        )

    out_df = pd.DataFrame(rows).sort_values(
        by=["weighted_gain", "weighted_min_fold_delta", "spearman_vs_current_best"],
        ascending=[False, False, True],
    )
    out_csv = Path(args.output_csv).resolve()
    out_json = Path(args.output_json).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    summary = {
        "teacher_fr_oof_csv": str(Path(args.teacher_fr_oof_csv).resolve()),
        "sample_weights_npy": str(Path(args.sample_weights_npy).resolve()),
        "reference_submission_csv": str(Path(args.reference_submission_csv).resolve()),
        "n_candidates": int(len(out_df)),
        "n_go": int(out_df["go_flag"].sum()) if len(out_df) else 0,
        "go_thresholds": {
            "min_gain": float(args.go_min_gain),
            "min_fold_delta": float(args.go_min_fold_delta),
            "max_override_rate": float(args.go_max_override_rate),
            "max_spearman": float(args.go_max_spearman),
        },
        "output_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(str(out_csv), flush=True)
    if len(out_df):
        top = out_df.iloc[0]
        print(
            f"top={top['name']} gain={float(top['weighted_gain']):+.6f} "
            f"min_fold_delta={float(top['weighted_min_fold_delta']):+.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
