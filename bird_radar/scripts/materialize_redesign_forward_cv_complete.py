from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASSES
from src.cv import make_forward_temporal_group_folds


def _resolve_ids_path(run_dir: Path, file_name: str) -> Path:
    p1 = run_dir / file_name
    if p1.exists():
        return p1
    p2 = run_dir / "artifacts" / file_name
    if p2.exists():
        return p2
    raise FileNotFoundError(f"cannot find {file_name} in {run_dir} or {run_dir / 'artifacts'}")


def _detect_pred_path(run_dir: Path, kind: str, explicit_name: str | None) -> Path:
    art = run_dir / "artifacts"
    if not art.exists():
        raise FileNotFoundError(f"artifacts dir missing: {art}")

    if explicit_name:
        p = art / explicit_name
        if p.exists():
            return p
        raise FileNotFoundError(f"missing explicit {kind} file: {p}")

    if kind == "oof":
        cands = sorted(art.glob("deep_transformer_heavy_seed*_oof.npy"))
        if len(cands) == 1:
            return cands[0]
        cands = sorted([p for p in art.glob("*_oof.npy") if "catboost" not in p.name.lower()])
    else:
        cands = sorted(art.glob("deep_transformer_heavy_seed*_test.npy"))
        if len(cands) == 1:
            return cands[0]
        cands = sorted([p for p in art.glob("*_test.npy") if "catboost" not in p.name.lower()])

    if len(cands) != 1:
        raise RuntimeError(f"cannot auto-detect {kind} npy in {art}; candidates={ [p.name for p in cands] }")
    return cands[0]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True, type=str)
    ap.add_argument("--teacher-oof-csv", required=True, type=str)
    ap.add_argument("--run-dir", action="append", required=True, help="repeat for each redesign run dir")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--timestamp-col", type=str, default="timestamp_start_radar_utc")
    ap.add_argument("--group-col", type=str, default="observation_id")
    ap.add_argument("--id-col", type=str, default="track_id")
    ap.add_argument("--fill-mode", choices=["teacher_anchor", "covered_mean", "zero"], default="teacher_anchor")
    ap.add_argument("--oof-npy-name", type=str, default=None)
    ap.add_argument("--test-npy-name", type=str, default=None)
    ap.add_argument("--force", action="store_true", default=False)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(
        args.train_csv,
        usecols=[args.id_col, args.group_col, args.timestamp_col],
    )
    canonical_ids = train_df[args.id_col].to_numpy(dtype=np.int64)
    n_train = len(canonical_ids)

    folds = make_forward_temporal_group_folds(
        train_df,
        timestamp_col=args.timestamp_col,
        group_col=args.group_col,
        n_splits=int(args.n_splits),
    )
    covered_idx_canon = np.unique(np.concatenate([va for _, va in folds])).astype(np.int64)
    covered_mask_canon = np.zeros(n_train, dtype=bool)
    covered_mask_canon[covered_idx_canon] = True
    covered_by_track_id = {int(tid): bool(covered_mask_canon[i]) for i, tid in enumerate(canonical_ids.tolist())}

    teacher_df = pd.read_csv(args.teacher_oof_csv, usecols=[args.id_col, *CLASSES])
    teacher_map = {
        int(row[args.id_col]): np.clip(np.asarray([row[c] for c in CLASSES], dtype=np.float32), 0.0, 1.0)
        for _, row in teacher_df.iterrows()
    }
    missing_teacher = sorted(list(set(int(x) for x in canonical_ids.tolist()).difference(teacher_map.keys())))
    if missing_teacher:
        raise RuntimeError(f"teacher_oof_csv missing {len(missing_teacher)} ids; first={missing_teacher[:10]}")

    out_rows: list[dict[str, Any]] = []

    for run_dir_raw in args.run_dir:
        run_dir = Path(run_dir_raw).resolve()
        art = run_dir / "artifacts"
        art.mkdir(parents=True, exist_ok=True)

        out_oof_complete = art / "oof_forward_cv_complete.npy"
        out_oof_complete_idx = art / "oof_forward_cv_complete_idx.npy"
        out_oof_forward = art / "oof_forward_cv.npy"
        out_oof_forward_idx = art / "oof_forward_cv_idx.npy"
        out_test_complete = art / "test_forward_cv_complete_mean.npy"
        out_test_forward = art / "test_forward_cv_mean.npy"

        if (not args.force) and out_oof_complete.exists() and out_oof_complete_idx.exists() and out_test_complete.exists():
            print(f"[skip] complete artifacts already exist: {run_dir}", flush=True)
            out_rows.append(
                {
                    "run_dir": str(run_dir),
                    "skipped": True,
                    "reason": "already_exists",
                }
            )
            continue

        train_ids_path = _resolve_ids_path(run_dir, "train_track_ids.npy")
        test_ids_path = _resolve_ids_path(run_dir, "test_track_ids.npy")
        oof_path = _detect_pred_path(run_dir, kind="oof", explicit_name=args.oof_npy_name)
        test_path = _detect_pred_path(run_dir, kind="test", explicit_name=args.test_npy_name)

        run_train_ids = np.load(train_ids_path).astype(np.int64)
        run_test_ids = np.load(test_ids_path).astype(np.int64)
        oof = np.load(oof_path).astype(np.float32)
        test = np.load(test_path).astype(np.float32)

        if oof.shape != (len(run_train_ids), len(CLASSES)):
            raise RuntimeError(f"{run_dir.name}: oof shape {oof.shape} mismatch with train ids len {len(run_train_ids)}")
        if test.shape != (len(run_test_ids), len(CLASSES)):
            raise RuntimeError(f"{run_dir.name}: test shape {test.shape} mismatch with test ids len {len(run_test_ids)}")

        run_covered_mask = np.array([covered_by_track_id.get(int(tid), False) for tid in run_train_ids.tolist()], dtype=bool)
        run_covered_idx = np.where(run_covered_mask)[0].astype(np.int64)
        run_uncovered_idx = np.where(~run_covered_mask)[0].astype(np.int64)

        oof_forward = np.zeros_like(oof, dtype=np.float32)
        oof_forward[run_covered_idx] = np.clip(oof[run_covered_idx], 0.0, 1.0)

        oof_complete = oof_forward.copy()
        if len(run_uncovered_idx) > 0:
            if args.fill_mode == "teacher_anchor":
                fill = np.stack([teacher_map[int(tid)] for tid in run_train_ids[run_uncovered_idx].tolist()], axis=0)
                oof_complete[run_uncovered_idx] = fill.astype(np.float32)
            elif args.fill_mode == "covered_mean":
                if len(run_covered_idx) == 0:
                    raise RuntimeError(f"{run_dir.name}: covered_idx is empty")
                mean_vec = np.mean(oof_forward[run_covered_idx], axis=0).astype(np.float32)
                oof_complete[run_uncovered_idx] = mean_vec[None, :]
            else:  # zero
                oof_complete[run_uncovered_idx] = 0.0

        np.save(out_oof_forward, oof_forward.astype(np.float32))
        np.save(out_oof_forward_idx, run_covered_idx.astype(np.int64))
        np.save(out_oof_complete, oof_complete.astype(np.float32))
        np.save(out_oof_complete_idx, np.arange(len(run_train_ids), dtype=np.int64))
        np.save(out_test_complete, np.clip(test, 0.0, 1.0).astype(np.float32))
        np.save(out_test_forward, np.clip(test, 0.0, 1.0).astype(np.float32))

        out_rows.append(
            {
                "run_dir": str(run_dir),
                "skipped": False,
                "oof_source": str(oof_path),
                "test_source": str(test_path),
                "fill_mode": args.fill_mode,
                "n_train": int(len(run_train_ids)),
                "n_test": int(len(run_test_ids)),
                "covered": int(len(run_covered_idx)),
                "uncovered": int(len(run_uncovered_idx)),
                "coverage_ratio": float(len(run_covered_idx) / max(1, len(run_train_ids))),
                "oof_forward_cv_complete_path": str(out_oof_complete.resolve()),
                "oof_forward_cv_complete_idx_path": str(out_oof_complete_idx.resolve()),
                "test_forward_cv_complete_mean_path": str(out_test_complete.resolve()),
            }
        )
        print(
            f"[ok] {run_dir.name} covered={len(run_covered_idx)} uncovered={len(run_uncovered_idx)} "
            f"fill_mode={args.fill_mode}",
            flush=True,
        )

    report = {
        "train_csv": str(Path(args.train_csv).resolve()),
        "teacher_oof_csv": str(Path(args.teacher_oof_csv).resolve()),
        "n_splits": int(args.n_splits),
        "fill_mode": str(args.fill_mode),
        "timestamp_col": str(args.timestamp_col),
        "group_col": str(args.group_col),
        "id_col": str(args.id_col),
        "runs": out_rows,
    }

    # Store a single report under project artifacts for reproducibility.
    report_dir = PROJECT_ROOT / "artifacts" / "redesign_forward_complete_materialize"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "materialize_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)
    print(f"[report] {report_path}", flush=True)


if __name__ == "__main__":
    main()
