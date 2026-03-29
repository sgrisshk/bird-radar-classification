from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    c = spearmanr(a, b).correlation
    if c is None or not np.isfinite(c):
        return 0.0
    return float(c)


def _load_submission(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    df = pd.read_csv(path)
    if "track_id" not in df.columns:
        raise ValueError(f"{path} has no track_id")
    cls = [c for c in df.columns if c != "track_id"]
    arr = df[cls].to_numpy(dtype=np.float64)
    ids = df["track_id"].to_numpy()
    return ids, cls, arr


def _rank_shift_vs_baseline(
    baseline_path: Path,
    candidate_paths: list[Path],
) -> pd.DataFrame:
    b_ids, b_cls, b_arr = _load_submission(baseline_path)
    rows: list[dict[str, float | str]] = []
    for p in candidate_paths:
        if not p.exists():
            continue
        try:
            c_ids, c_cls, c_arr = _load_submission(p)
        except Exception:
            continue
        if b_cls != c_cls:
            continue
        if not np.array_equal(b_ids, c_ids):
            tmp_b = pd.DataFrame({"track_id": b_ids})
            tmp_c = pd.DataFrame({"track_id": c_ids})
            inter = tmp_b.merge(tmp_c, on="track_id", how="inner")
            if len(inter) == 0:
                continue
            pos_b = {int(t): i for i, t in enumerate(b_ids.tolist())}
            pos_c = {int(t): i for i, t in enumerate(c_ids.tolist())}
            idx_b = np.array([pos_b[int(t)] for t in inter["track_id"].to_numpy()], dtype=np.int64)
            idx_c = np.array([pos_c[int(t)] for t in inter["track_id"].to_numpy()], dtype=np.int64)
            x = b_arr[idx_b].reshape(-1)
            y = c_arr[idx_c].reshape(-1)
        else:
            x = b_arr.reshape(-1)
            y = c_arr.reshape(-1)
        rows.append(
            {
                "submission_path": str(p),
                "rank_corr_vs_baseline": _safe_spearman(x, y),
                "mae_vs_baseline": float(np.mean(np.abs(x - y))),
            }
        )
    return pd.DataFrame(rows)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in {"public_score"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _best_effort_predictive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, float | str]] = []
    public = pd.to_numeric(df["public_score"], errors="coerce")
    valid = public.notna()
    if valid.sum() < 3:
        return pd.DataFrame(columns=["metric", "pearson", "spearman", "n"])
    p = public[valid].to_numpy(dtype=np.float64)
    if float(np.std(p)) < 1e-12:
        return pd.DataFrame(columns=["metric", "pearson", "spearman", "n"])

    for c in _numeric_columns(df):
        x = pd.to_numeric(df[c], errors="coerce")
        m = valid & x.notna()
        if m.sum() < 3:
            continue
        xv = x[m].to_numpy(dtype=np.float64)
        yv = public[m].to_numpy(dtype=np.float64)
        if float(np.std(xv)) < 1e-12:
            continue
        pear = float(np.corrcoef(xv, yv)[0, 1])
        spear = _safe_spearman(xv, yv)
        if np.isfinite(pear) and np.isfinite(spear):
            out.append({"metric": c, "pearson": pear, "spearman": spear, "n": int(m.sum())})
    res = pd.DataFrame(out)
    if len(res) == 0:
        return res
    res["abs_spearman"] = res["spearman"].abs()
    return res.sort_values("abs_spearman", ascending=False).drop(columns=["abs_spearman"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze offline/public linkage and ranking-shift candidates.")
    ap.add_argument(
        "--merged-csv",
        default="bird_radar/artifacts/lb_proxy_merged_with_scores.csv",
        type=str,
    )
    ap.add_argument(
        "--candidates-csv",
        default="bird_radar/artifacts/lb_proxy_candidates.csv",
        type=str,
    )
    ap.add_argument(
        "--baseline-submission",
        default="bird_radar/artifacts/stack_convex_fast_rescan/mt0p92_gs0p01/submission_stacked_ridge_mt0p92_gs0p01.csv",
        type=str,
    )
    ap.add_argument("--top-k", default=12, type=int)
    ap.add_argument("--out-csv", default="", type=str)
    args = ap.parse_args()

    merged = pd.read_csv(args.merged_csv)
    candidates = pd.read_csv(args.candidates_csv)

    # Normalize duplicates from *_x/*_y merged exports.
    rename_map = {
        "report_path_x": "report_path",
        "submission_path_x": "submission_path",
        "split_mode_x": "split_mode",
        "learner_x": "learner",
        "uncovered_fill_x": "uncovered_fill",
        "stack_gain_vs_teacher_covered_x": "stack_gain_vs_teacher_covered",
        "stack_gain_vs_teacher_fallback_x": "stack_gain_vs_teacher_fallback",
        "stack_oof_macro_map_covered_x": "stack_oof_macro_map_covered",
        "stack_oof_macro_map_teacher_fallback_x": "stack_oof_macro_map_teacher_fallback",
        "stack_oof_macro_map_x": "stack_oof_macro_map",
        "fold_mean_x": "fold_mean",
        "teacher_fold_mean_x": "teacher_fold_mean",
        "covered_ratio_x": "covered_ratio",
    }
    for old, new in rename_map.items():
        if old in merged.columns and new not in merged.columns:
            merged = merged.rename(columns={old: new})

    submitted = merged[["submission_path", "public_score"]].copy()
    submitted["submission_path"] = submitted["submission_path"].astype(str)
    submitted = submitted.dropna(subset=["submission_path"]).drop_duplicates(subset=["submission_path"], keep="last")

    cand = candidates.copy()
    cand["submission_path"] = cand["submission_path"].astype(str)
    cand = cand.merge(submitted, on="submission_path", how="left", suffixes=("_cand", "_submitted"))
    if "public_score_submitted" in cand.columns:
        ps = pd.to_numeric(cand["public_score_submitted"], errors="coerce")
    elif "public_score" in cand.columns:
        ps = pd.to_numeric(cand["public_score"], errors="coerce")
    else:
        ps = pd.Series(np.nan, index=cand.index, dtype=np.float64)
    cand["public_score_submitted"] = ps
    cand["is_submitted"] = cand["public_score_submitted"].notna()

    # Ranking movement vs baseline.
    paths = [Path(p) for p in cand["submission_path"].dropna().unique().tolist()]
    shift = _rank_shift_vs_baseline(Path(args.baseline_submission), paths)
    if len(shift) > 0:
        cand = cand.merge(shift, on="submission_path", how="left")
    else:
        cand["rank_corr_vs_baseline"] = np.nan
        cand["mae_vs_baseline"] = np.nan

    # Predictive metric check on submitted history only.
    merged_predictive = _best_effort_predictive_metrics(merged)

    public = pd.to_numeric(merged.get("public_score"), errors="coerce")
    public_unique = int(public.dropna().nunique())
    print(f"submitted_rows={int(public.notna().sum())} unique_public={public_unique}")
    if public_unique <= 1:
        print("predictive_status=NO_SIGNAL public is constant; cannot estimate offline->public mapping.")
        print("min_ranking_shift_to_move_public=UNDEFINED from current history.")
    else:
        print("predictive_status=WEAK_SIGNAL")
        if len(merged_predictive):
            print("top_predictive_metrics:")
            print(merged_predictive.head(8).to_string(index=False))
        else:
            print("top_predictive_metrics: none")

    # Candidate ranking: prefer good offline covered gain + meaningful ranking shift.
    for c in [
        "stack_gain_vs_teacher_covered",
        "stack_oof_macro_map_covered",
        "rank_corr_vs_baseline",
        "mae_vs_baseline",
    ]:
        if c not in cand.columns:
            cand[c] = np.nan
        cand[c] = pd.to_numeric(cand[c], errors="coerce")

    # Robust z-score style normalization.
    def z(s: pd.Series) -> pd.Series:
        m = float(s.mean()) if s.notna().any() else 0.0
        sd = float(s.std()) if s.notna().any() else 0.0
        if sd < 1e-12:
            return pd.Series(np.zeros(len(s), dtype=np.float64), index=s.index)
        return (s - m) / sd

    cand["movement"] = 1.0 - cand["rank_corr_vs_baseline"].fillna(1.0)
    cand["score_offline_move"] = (
        0.65 * z(cand["stack_gain_vs_teacher_covered"].fillna(0.0))
        + 0.25 * z(cand["stack_oof_macro_map_covered"].fillna(0.0))
        + 0.10 * z(cand["movement"].fillna(0.0))
    )

    shortlist = cand[cand["is_submitted"] == False].copy()
    shortlist = shortlist.sort_values("score_offline_move", ascending=False)
    cols = [
        "submission_path",
        "score_offline_move",
        "stack_gain_vs_teacher_covered",
        "stack_oof_macro_map_covered",
        "rank_corr_vs_baseline",
        "mae_vs_baseline",
        "learner",
        "split_mode",
        "uncovered_fill",
    ]
    cols = [c for c in cols if c in shortlist.columns]
    print(f"\nshortlist_top_{int(args.top_k)}:")
    print(shortlist[cols].head(int(args.top_k)).to_string(index=False))

    if args.out_csv.strip():
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        shortlist.to_csv(out, index=False)
        print(f"\nsaved={out.resolve()}")


if __name__ == "__main__":
    main()
