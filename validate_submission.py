from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

EXPECTED_CLASS_COLUMNS = [
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
EXPECTED_COLUMNS = ["track_id", *EXPECTED_CLASS_COLUMNS]
EPS = 1e-12


@dataclass
class ValidationState:
    hard_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sections: list[tuple[str, list[str]]] = field(default_factory=list)

    def add_section(self, title: str, lines: list[str]) -> None:
        self.sections.append((title, lines))

    def err(self, msg: str) -> None:
        self.hard_errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AI Cup 2026 Performance Track submission CSV.")
    parser.add_argument("--test-csv", required=True, help="Path to test.csv")
    parser.add_argument("--submission-csv", required=True, help="Path to submission.csv")
    parser.add_argument("--oof-predictions-csv", default=None, help="Optional path to OOF predictions CSV")
    parser.add_argument("--train-csv", default=None, help="Optional path to train.csv")
    return parser.parse_args()


def _exists(path: str | os.PathLike[str] | None) -> bool:
    return path is not None and Path(path).is_file()


def _format_bool(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def _read_csv_header(path: Path) -> tuple[bool, str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            line = f.readline()
        return True, line.rstrip("\n\r")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _load_csv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _coerce_probabilities(
    df: pd.DataFrame,
    class_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    converted = pd.DataFrame(index=df.index)
    stats: dict[str, dict[str, Any]] = {}
    for col in class_columns:
        raw = df[col]
        coerced = pd.to_numeric(raw, errors="coerce")
        non_numeric_mask = coerced.isna() & ~raw.isna()
        values = coerced.to_numpy(dtype=np.float64, copy=False)
        finite_mask = np.isfinite(values)
        nan_count = int(np.isnan(values).sum())
        inf_count = int((~finite_mask & ~np.isnan(values)).sum())
        invalid_range_mask = finite_mask & ((values < 0.0) | (values > 1.0))
        stats[col] = {
            "non_numeric_count": int(non_numeric_mask.sum()),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "invalid_range_count": int(invalid_range_mask.sum()),
            "min": float(np.nanmin(values)) if np.isfinite(values).any() else np.nan,
            "max": float(np.nanmax(values)) if np.isfinite(values).any() else np.nan,
            "zero_frac": float(np.mean(np.isclose(np.nan_to_num(values, nan=-1.0), 0.0))),
            "constant": bool(np.isfinite(values).all() and np.nanstd(values) <= 1e-12),
            "mean": float(np.nanmean(values)) if values.size else np.nan,
            "std": float(np.nanstd(values)) if values.size else np.nan,
        }
        converted[col] = coerced.astype(np.float64)
    converted["track_id"] = df["track_id"]
    return converted, stats


def _top5_confident(df_probs: pd.DataFrame, class_columns: list[str]) -> dict[str, list[tuple[Any, float]]]:
    out: dict[str, list[tuple[Any, float]]] = {}
    for col in class_columns:
        top = df_probs[["track_id", col]].nlargest(5, columns=col)
        out[col] = [(row["track_id"], float(row[col])) for _, row in top.iterrows()]
    return out


def _validate_prediction_table(
    name: str,
    df: pd.DataFrame,
    expected_rows: int | None,
    expected_track_ids: pd.Index | None,
    state: ValidationState,
    strict_column_order: bool = True,
    section_prefix: str = "",
) -> dict[str, Any]:
    lines_basic: list[str] = []
    lines_id: list[str] = []
    lines_prob: list[str] = []
    lines_dist: list[str] = []

    lines_basic.append(f"{name} rows: {len(df)}")
    lines_basic.append(f"{name} columns ({len(df.columns)}): {list(df.columns)}")

    exact_col_match = list(df.columns) == EXPECTED_COLUMNS
    has_exact_col_count = len(df.columns) == 10
    lines_basic.append(f"Exactly 10 columns: {_format_bool(has_exact_col_count)}")
    lines_basic.append(f"Header/order matches sample_submission: {_format_bool(exact_col_match)}")

    if not has_exact_col_count:
        state.err(f"{name}: expected exactly 10 columns, got {len(df.columns)}")
    if strict_column_order and not exact_col_match:
        state.err(f"{name}: columns do not match expected order {EXPECTED_COLUMNS}")

    missing_required_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing_required_cols:
        state.err(f"{name}: missing required columns {missing_required_cols}")
        lines_basic.append(f"Missing required columns: {missing_required_cols}")
    else:
        lines_basic.append("Missing required columns: []")
    if extra_cols:
        lines_basic.append(f"Extra columns: {extra_cols}")
    else:
        lines_basic.append("Extra columns: []")

    if "track_id" in df.columns:
        dup_count = int(df["track_id"].duplicated().sum())
        lines_id.append(f"Duplicate track_id count: {dup_count}")
        if dup_count > 0:
            dup_ids = df.loc[df["track_id"].duplicated(), "track_id"].astype(str).head(20).tolist()
            lines_id.append(f"Duplicate track_id examples (up to 20): {dup_ids}")
            state.err(f"{name}: duplicate track_id values found ({dup_count})")
    else:
        lines_id.append("track_id column missing")
        state.err(f"{name}: track_id column missing")

    if expected_rows is not None:
        row_match = len(df) == expected_rows
        lines_basic.append(f"Row count matches expected ({expected_rows}): {_format_bool(row_match)}")
        if not row_match:
            state.err(f"{name}: row count mismatch (expected {expected_rows}, got {len(df)})")

    id_summary: dict[str, Any] = {
        "missing_ids": [],
        "extra_ids": [],
        "intersection_count": 0,
    }
    if expected_track_ids is not None and "track_id" in df.columns:
        sub_ids = pd.Index(df["track_id"])
        missing_ids = expected_track_ids.difference(sub_ids)
        extra_ids = sub_ids.difference(expected_track_ids)
        id_summary = {
            "missing_ids": missing_ids.tolist(),
            "extra_ids": extra_ids.tolist(),
            "intersection_count": int(len(expected_track_ids.intersection(sub_ids))),
        }
        lines_id.append(f"Missing IDs count: {len(missing_ids)}")
        lines_id.append(f"Extra IDs count: {len(extra_ids)}")
        if len(missing_ids):
            lines_id.append(f"Missing IDs (up to 20): {missing_ids[:20].tolist()}")
            state.err(f"{name}: missing track_id values ({len(missing_ids)})")
        if len(extra_ids):
            lines_id.append(f"Extra IDs (up to 20): {extra_ids[:20].tolist()}")
            state.err(f"{name}: extra track_id values ({len(extra_ids)})")
        if len(missing_ids) == 0 and len(extra_ids) == 0:
            lines_id.append("ID set matches expected: OK")

    usable_class_cols = [c for c in EXPECTED_CLASS_COLUMNS if c in df.columns]
    prob_df = None
    prob_stats: dict[str, dict[str, Any]] = {}
    if len(usable_class_cols) == len(EXPECTED_CLASS_COLUMNS) and "track_id" in df.columns:
        prob_df, prob_stats = _coerce_probabilities(df[["track_id", *EXPECTED_CLASS_COLUMNS]].copy(), EXPECTED_CLASS_COLUMNS)
        for col in EXPECTED_CLASS_COLUMNS:
            s = prob_stats[col]
            lines_prob.append(
                f"{col}: min={s['min']:.6f} max={s['max']:.6f} "
                f"non_numeric={s['non_numeric_count']} nan={s['nan_count']} inf={s['inf_count']} "
                f"out_of_range={s['invalid_range_count']}"
            )
            if s["non_numeric_count"] > 0:
                state.err(f"{name}: non-numeric values in {col}")
            if s["nan_count"] > 0:
                state.err(f"{name}: NaN values in {col}")
            if s["inf_count"] > 0:
                state.err(f"{name}: Inf values in {col}")
            if s["invalid_range_count"] > 0:
                state.err(f"{name}: out-of-range probabilities in {col}")

        row_matrix = prob_df[EXPECTED_CLASS_COLUMNS].to_numpy(dtype=np.float64, copy=False)
        finite_all = np.isfinite(row_matrix).all()
        if finite_all:
            constant_cols = [c for c in EXPECTED_CLASS_COLUMNS if prob_stats[c]["constant"]]
            if constant_cols:
                state.warn(f"{name}: constant prediction columns detected {constant_cols}")
                lines_dist.append(f"Constant columns: {constant_cols}")
            else:
                lines_dist.append("Constant columns: []")

            row_unique_count = int(np.unique(np.round(row_matrix, 12), axis=0).shape[0])
            all_rows_identical = row_unique_count == 1
            lines_dist.append(f"All rows identical: {'YES' if all_rows_identical else 'NO'} (unique_rows={row_unique_count})")
            if all_rows_identical:
                state.warn(f"{name}: all prediction rows are identical")

            too_many_zeros = []
            for col in EXPECTED_CLASS_COLUMNS:
                zf = prob_stats[col]["zero_frac"]
                if zf > 0.95:
                    too_many_zeros.append((col, zf))
                    state.warn(f"{name}: >95% zeros in {col} ({zf:.2%})")
            if too_many_zeros:
                lines_dist.append(
                    "Columns with >95% zeros: "
                    + str([(c, round(frac, 4)) for c, frac in too_many_zeros])
                )
            else:
                lines_dist.append("Columns with >95% zeros: []")

            extreme_means = []
            for col in EXPECTED_CLASS_COLUMNS:
                m = prob_stats[col]["mean"]
                if np.isfinite(m) and (m < 1e-4 or m > 0.999):
                    extreme_means.append((col, m))
                    state.warn(f"{name}: extreme mean probability in {col} ({m:.6f})")
            if extreme_means:
                lines_dist.append(
                    "Extreme mean probs: " + str([(c, round(v, 6)) for c, v in extreme_means])
                )
            else:
                lines_dist.append("Extreme mean probs: []")

            lines_dist.append("Mean per class:")
            for col in EXPECTED_CLASS_COLUMNS:
                lines_dist.append(f"  {col}: {prob_stats[col]['mean']:.6f}")
            lines_dist.append("Std per class:")
            for col in EXPECTED_CLASS_COLUMNS:
                lines_dist.append(f"  {col}: {prob_stats[col]['std']:.6f}")

            top5 = _top5_confident(prob_df, EXPECTED_CLASS_COLUMNS)
            lines_dist.append("Top 5 highest confident predictions per class:")
            for col in EXPECTED_CLASS_COLUMNS:
                pretty = ", ".join([f"(track_id={tid}, p={p:.6f})" for tid, p in top5[col]])
                lines_dist.append(f"  {col}: {pretty}")
        else:
            lines_dist.append("Distribution diagnostics skipped due to non-finite probabilities.")
    else:
        lines_prob.append("Probability checks skipped due to missing required columns.")
        lines_dist.append("Distribution diagnostics skipped due to missing required columns.")

    state.add_section(f"{section_prefix}BASIC CHECKS".strip(), lines_basic)
    state.add_section(f"{section_prefix}ID CHECKS".strip(), lines_id)
    state.add_section(f"{section_prefix}PROBABILITY CHECKS".strip(), lines_prob)
    state.add_section(f"{section_prefix}DISTRIBUTION".strip(), lines_dist)

    return {
        "prob_df": prob_df,
        "prob_stats": prob_stats,
        "id_summary": id_summary,
    }


def _build_train_targets(train_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if "bird_group" not in train_df.columns:
        raise ValueError("train.csv must contain bird_group column for OOF evaluation/calibration")
    if "track_id" not in train_df.columns:
        raise ValueError("train.csv must contain track_id column")

    y = np.zeros((len(train_df), len(EXPECTED_CLASS_COLUMNS)), dtype=np.uint8)
    label_to_idx = {c: i for i, c in enumerate(EXPECTED_CLASS_COLUMNS)}
    unknown = sorted(set(train_df["bird_group"].dropna().astype(str)) - set(label_to_idx))
    if unknown:
        raise ValueError(f"train.csv contains unknown labels not in expected class list: {unknown}")
    for i, label in enumerate(train_df["bird_group"].astype(str).tolist()):
        y[i, label_to_idx[label]] = 1
    return train_df[["track_id", "bird_group"]].copy(), y


def _compute_ap_report(y_true: np.ndarray, probs: np.ndarray) -> tuple[dict[str, float], float]:
    per_class: dict[str, float] = {}
    for i, cls in enumerate(EXPECTED_CLASS_COLUMNS):
        yt = y_true[:, i]
        yp = probs[:, i]
        if yt.sum() == 0:
            per_class[cls] = 0.0
            continue
        try:
            per_class[cls] = float(average_precision_score(yt, yp))
        except ValueError:
            per_class[cls] = 0.0
    macro = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return per_class, macro


def _print_report(state: ValidationState) -> None:
    for title, lines in state.sections:
        print(f"=== {title} ===")
        if lines:
            for line in lines:
                print(line)
        else:
            print("(none)")
    print("=== FINAL VERDICT ===")
    if state.hard_errors:
        print("FAIL")
    else:
        print("PASS")
    print(f"Hard errors: {len(state.hard_errors)}")
    if state.hard_errors:
        for msg in state.hard_errors:
            print(f"ERROR: {msg}")
    print(f"Warnings: {len(state.warnings)}")
    if state.warnings:
        for msg in state.warnings:
            print(f"WARN: {msg}")


def main() -> int:
    args = parse_args()
    state = ValidationState()

    test_path = Path(args.test_csv)
    sub_path = Path(args.submission_csv)
    oof_path = Path(args.oof_predictions_csv) if args.oof_predictions_csv else None
    train_path = Path(args.train_csv) if args.train_csv else None

    basic_lines: list[str] = []
    for label, path in [
        ("test.csv", test_path),
        ("submission.csv", sub_path),
        ("oof_predictions.csv", oof_path),
        ("train.csv", train_path),
    ]:
        if path is None:
            basic_lines.append(f"{label}: not provided")
            continue
        exists = path.is_file()
        basic_lines.append(f"{label}: {path} exists={exists}")
        if not exists and label in {"test.csv", "submission.csv"}:
            state.err(f"{label} not found: {path}")
        if exists:
            ok, first_line = _read_csv_header(path)
            basic_lines.append(f"{label} header readable: {_format_bool(ok)}")
            if ok:
                first_cell = first_line.split(",")[0].strip().lstrip("\ufeff")
                header_present = first_cell == "track_id" or label == "train.csv"
                basic_lines.append(f"{label} header present: {_format_bool(header_present)}")
                if label != "train.csv" and not header_present:
                    state.err(f"{label}: header missing or first column is not track_id")
                basic_lines.append(f"{label} first line: {first_line[:200]}")
            else:
                state.err(f"{label}: unreadable header ({first_line})")
    state.add_section("FILE CHECKS", basic_lines)

    if state.hard_errors and (not test_path.is_file() or not sub_path.is_file()):
        _print_report(state)
        return 1

    test_df, test_err = _load_csv(test_path)
    if test_err is not None or test_df is None:
        state.err(f"test.csv unreadable: {test_err}")
        _print_report(state)
        return 1
    sub_df, sub_err = _load_csv(sub_path)
    if sub_err is not None or sub_df is None:
        state.err(f"submission.csv unreadable: {sub_err}")
        _print_report(state)
        return 1

    test_meta_lines: list[str] = [f"test.csv shape: {test_df.shape}", f"test.csv columns: {list(test_df.columns)}"]
    if "track_id" not in test_df.columns:
        state.err("test.csv missing track_id column")
        state.add_section("TEST CHECKS", test_meta_lines)
        _print_report(state)
        return 1
    if int(test_df["track_id"].duplicated().sum()) > 0:
        dup_test = int(test_df["track_id"].duplicated().sum())
        state.err(f"test.csv duplicate track_id values found ({dup_test})")
        test_meta_lines.append(f"Duplicate test track_id count: {dup_test}")
    else:
        test_meta_lines.append("Duplicate test track_id count: 0")
    state.add_section("TEST CHECKS", test_meta_lines)

    expected_test_ids = pd.Index(test_df["track_id"])
    sub_result = _validate_prediction_table(
        name="SUBMISSION",
        df=sub_df,
        expected_rows=len(test_df),
        expected_track_ids=expected_test_ids,
        state=state,
        strict_column_order=True,
        section_prefix="",
    )

    train_df: pd.DataFrame | None = None
    train_targets: np.ndarray | None = None
    train_label_frame: pd.DataFrame | None = None
    if train_path is not None:
        if not train_path.is_file():
            state.err(f"train.csv not found: {train_path}")
        else:
            train_df, train_err = _load_csv(train_path)
            if train_err is not None or train_df is None:
                state.err(f"train.csv unreadable: {train_err}")
            else:
                train_lines = [f"train.csv shape: {train_df.shape}", f"train.csv columns: {list(train_df.columns)}"]
                if "track_id" not in train_df.columns:
                    state.err("train.csv missing track_id")
                if "bird_group" not in train_df.columns:
                    state.err("train.csv missing bird_group")
                if "track_id" in train_df.columns:
                    dup_train = int(train_df["track_id"].duplicated().sum())
                    train_lines.append(f"Duplicate train track_id count: {dup_train}")
                    if dup_train > 0:
                        state.err(f"train.csv duplicate track_id values found ({dup_train})")
                try:
                    if train_df is not None:
                        train_label_frame, train_targets = _build_train_targets(train_df)
                        class_freq = train_targets.mean(axis=0)
                        train_lines.append("True class frequency (train):")
                        for cls, freq in zip(EXPECTED_CLASS_COLUMNS, class_freq):
                            train_lines.append(f"  {cls}: {float(freq):.6f}")
                except Exception as e:
                    state.err(f"Failed to build train targets: {type(e).__name__}: {e}")
                state.add_section("TRAIN CHECKS", train_lines)

    leakage_lines: list[str] = []
    if train_df is not None and sub_df is not None and "track_id" in sub_df.columns and "track_id" in train_df.columns:
        overlap = pd.Index(sub_df["track_id"]).intersection(pd.Index(train_df["track_id"]))
        leakage_lines.append(f"Train/submission track_id overlap count: {len(overlap)}")
        if len(overlap):
            leakage_lines.append(f"Overlapping IDs (up to 20): {overlap[:20].tolist()}")
            state.err(f"Leakage guard failed: {len(overlap)} train track_id values found in submission")
    else:
        leakage_lines.append("Leakage guard skipped (train.csv not provided or invalid).")
    state.add_section("LEAKAGE GUARD", leakage_lines)

    calibration_lines: list[str] = []
    sub_prob_df = sub_result.get("prob_df")
    if sub_prob_df is not None:
        sub_means = sub_prob_df[EXPECTED_CLASS_COLUMNS].mean(axis=0).to_dict()
        calibration_lines.append("Average predicted probability per class (submission):")
        for cls in EXPECTED_CLASS_COLUMNS:
            calibration_lines.append(f"  {cls}: {float(sub_means[cls]):.6f}")
        if train_targets is not None:
            true_freq = train_targets.mean(axis=0)
            calibration_lines.append("Calibration gap vs train class frequency (submission_mean - train_freq):")
            for i, cls in enumerate(EXPECTED_CLASS_COLUMNS):
                gap = float(sub_means[cls] - true_freq[i])
                calibration_lines.append(f"  {cls}: {gap:+.6f}")
    else:
        calibration_lines.append("Submission calibration diagnostic skipped due to invalid/missing probability columns.")

    oof_metrics_lines: list[str] = []
    if oof_path is not None:
        if not oof_path.is_file():
            state.err(f"oof_predictions.csv not found: {oof_path}")
            oof_metrics_lines.append("OOF file missing.")
        else:
            oof_df, oof_err = _load_csv(oof_path)
            if oof_err is not None or oof_df is None:
                state.err(f"oof_predictions.csv unreadable: {oof_err}")
                oof_metrics_lines.append(f"Failed to read OOF CSV: {oof_err}")
            else:
                oof_result = _validate_prediction_table(
                    name="OOF",
                    df=oof_df,
                    expected_rows=len(train_df) if train_df is not None else None,
                    expected_track_ids=pd.Index(train_df["track_id"]) if train_df is not None and "track_id" in train_df.columns else None,
                    state=state,
                    strict_column_order=True,
                    section_prefix="OOF ",
                )
                oof_prob_df = oof_result.get("prob_df")
                if train_df is None or train_targets is None:
                    oof_metrics_lines.append("OOF metrics skipped because train.csv is not provided or invalid.")
                elif oof_prob_df is None:
                    oof_metrics_lines.append("OOF metrics skipped due to invalid OOF probability columns.")
                else:
                    if "track_id" not in oof_prob_df.columns:
                        state.err("OOF predictions missing track_id")
                    else:
                        train_ids = pd.Index(train_df["track_id"])
                        oof_ids = pd.Index(oof_prob_df["track_id"])
                        missing = train_ids.difference(oof_ids)
                        extra = oof_ids.difference(train_ids)
                        if len(missing) > 0:
                            state.err(f"OOF missing train track_id values ({len(missing)})")
                        if len(extra) > 0:
                            state.err(f"OOF contains extra track_id values ({len(extra)})")
                        if len(missing) == 0 and len(extra) == 0:
                            train_indexed = train_df[["track_id", "bird_group"]].copy().set_index("track_id")
                            oof_indexed = oof_prob_df.set_index("track_id")[EXPECTED_CLASS_COLUMNS].copy()
                            oof_aligned = oof_indexed.loc[train_indexed.index].to_numpy(dtype=np.float64)
                            y_true = np.zeros((len(train_indexed), len(EXPECTED_CLASS_COLUMNS)), dtype=np.uint8)
                            label_to_idx = {c: i for i, c in enumerate(EXPECTED_CLASS_COLUMNS)}
                            for i, label in enumerate(train_indexed["bird_group"].astype(str).tolist()):
                                y_true[i, label_to_idx[label]] = 1

                            per_class_ap, macro_map = _compute_ap_report(y_true, oof_aligned)
                            oof_metrics_lines.append("Per-class AP:")
                            for cls in EXPECTED_CLASS_COLUMNS:
                                oof_metrics_lines.append(f"  {cls}: {per_class_ap[cls]:.6f}")
                            oof_metrics_lines.append(f"Overall macro mAP: {macro_map:.6f}")

                            calibration_lines.append("Average predicted probability per class (OOF):")
                            oof_means = oof_indexed.mean(axis=0).to_dict()
                            true_freq = y_true.mean(axis=0)
                            for cls in EXPECTED_CLASS_COLUMNS:
                                calibration_lines.append(f"  {cls}: {float(oof_means[cls]):.6f}")
                            calibration_lines.append("Calibration gap vs true train frequency (OOF_mean - true_freq):")
                            for i, cls in enumerate(EXPECTED_CLASS_COLUMNS):
                                gap = float(oof_means[cls] - true_freq[i])
                                calibration_lines.append(f"  {cls}: {gap:+.6f}")
    else:
        oof_metrics_lines.append("OOF metrics skipped (oof_predictions.csv not provided).")
    state.add_section("OOF METRICS", oof_metrics_lines)
    state.add_section("CALIBRATION DIAGNOSTIC", calibration_lines)

    _print_report(state)
    return 1 if state.hard_errors else 0


if __name__ == "__main__":
    sys.exit(main())
