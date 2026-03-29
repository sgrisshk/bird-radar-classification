#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds
from src.preprocessing import parse_ewkb_linestring_zm_hex, parse_time_array
from src.redesign.utils import dump_json, macro_map, per_class_ap


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TF-IDF trajectory OVR model with forward CV.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--sample-submission", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--start-fold", type=int, default=1)
    p.add_argument("--time-col", default="timestamp_start_radar_utc")
    p.add_argument("--group-col", default="observation_id")
    p.add_argument("--n-bins", type=int, default=24)
    p.add_argument("--sample-points-per-feature", type=int, default=300_000)
    p.add_argument("--ngram-min", type=int, default=3)
    p.add_argument("--ngram-max", type=int, default=5)
    p.add_argument("--min-df", type=int, default=3)
    p.add_argument("--max-features", type=int, default=200_000)
    p.add_argument("--max-iter", type=int, default=3000)
    p.add_argument("--c", type=float, default=2.0)
    p.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    p.add_argument("--min-class-count", type=int, default=5)
    p.add_argument(
        "--invalid-class-policy",
        choices=["fallback_prior", "skip_class", "skip_fold"],
        default="skip_class",
    )
    return p.parse_args()


def _safe_step_time(t: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[:1]).astype(np.float32)
    if len(dt) > 1:
        pos = dt[1:][dt[1:] > 0]
        fill = float(np.median(pos)) if pos.size else 1.0
        dt[0] = fill
    return np.clip(dt, 1e-3, 10.0).astype(np.float32)


def _parse_track_features(row: Any) -> dict[str, np.ndarray] | None:
    track_id = int(getattr(row, "track_id"))
    try:
        lon, lat, alt, rcs = parse_ewkb_linestring_zm_hex(getattr(row, "trajectory"), track_id=track_id)
        t = parse_time_array(getattr(row, "trajectory_time"), track_id=track_id)
    except Exception:
        return None

    n = min(len(lon), len(t))
    if n < 3:
        return None
    lon = lon[:n].astype(np.float32, copy=False)
    lat = lat[:n].astype(np.float32, copy=False)
    alt = alt[:n].astype(np.float32, copy=False)
    rcs = rcs[:n].astype(np.float32, copy=False)
    t = t[:n].astype(np.float32, copy=False)

    m_per_deg_lat = 110_540.0
    m_per_deg_lon = 111_320.0 * float(np.cos(np.deg2rad(float(lat[0]))))
    x = (lon - lon[0]) * m_per_deg_lon
    y = (lat - lat[0]) * m_per_deg_lat

    dx = np.diff(x, prepend=x[:1]).astype(np.float32)
    dy = np.diff(y, prepend=y[:1]).astype(np.float32)
    dz = np.diff(alt, prepend=alt[:1]).astype(np.float32)
    dt = _safe_step_time(t)

    speed = (np.sqrt(dx * dx + dy * dy) / dt).astype(np.float32)
    vertical_speed = (dz / dt).astype(np.float32)
    heading = np.unwrap(np.arctan2(dy.astype(np.float64), dx.astype(np.float64))).astype(np.float32)
    turn = np.diff(heading, prepend=heading[:1]).astype(np.float32)

    return {
        "dx": dx,
        "dy": dy,
        "speed": speed,
        "vs": vertical_speed,
        "turn": turn,
        "rcs": rcs,
        "alt": alt,
    }


def _sample_for_bins(values: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if len(values) <= cap:
        return values
    idx = rng.choice(len(values), size=cap, replace=False)
    return values[idx]


def _build_bin_edges(
    track_feats: list[dict[str, np.ndarray] | None],
    n_bins: int,
    cap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    edges: dict[str, np.ndarray] = {}
    feat_names = ["dx", "dy", "speed", "vs", "turn", "rcs", "alt"]
    for name in feat_names:
        chunks: list[np.ndarray] = []
        for feats in track_feats:
            if feats is None:
                continue
            arr = feats[name]
            if arr.size == 0:
                continue
            chunks.append(arr.astype(np.float32, copy=False))
        if not chunks:
            edges[name] = np.asarray([0.0], dtype=np.float32)
            continue
        vals = np.concatenate(chunks, axis=0)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            edges[name] = np.asarray([0.0], dtype=np.float32)
            continue
        vals = _sample_for_bins(vals, cap=cap, rng=rng)
        q = np.quantile(vals, np.linspace(0.0, 1.0, int(max(2, n_bins + 1))), method="linear")
        inner = np.unique(q[1:-1].astype(np.float32))
        if inner.size == 0:
            inner = np.asarray([float(np.median(vals))], dtype=np.float32)
        edges[name] = inner
    return edges


def _q(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, v, side="right").astype(np.int16)


def _bucket_scalar(v: float, lo: float, hi: float, n_bins: int = 16) -> int:
    vv = float(np.clip(v, lo, hi))
    return int(np.floor((vv - lo) / (hi - lo + 1e-8) * n_bins))


def _build_doc(row: Any, feats: dict[str, np.ndarray] | None, edges: dict[str, np.ndarray]) -> str:
    if feats is None:
        return "invalid_track"

    qdx = _q(feats["dx"], edges["dx"])
    qdy = _q(feats["dy"], edges["dy"])
    qsp = _q(feats["speed"], edges["speed"])
    qvs = _q(feats["vs"], edges["vs"])
    qtr = _q(feats["turn"], edges["turn"])
    qrc = _q(feats["rcs"], edges["rcs"])
    qal = _q(feats["alt"], edges["alt"])

    n = min(len(qdx), 256)
    tokens: list[str] = []
    for i in range(n):
        tokens.append(
            f"s{int(qsp[i])}_t{int(qtr[i])}_v{int(qvs[i])}_x{int(qdx[i])}_y{int(qdy[i])}_r{int(qrc[i])}_a{int(qal[i])}"
        )

    airspeed = float(getattr(row, "airspeed", 0.0) or 0.0)
    min_z = float(getattr(row, "min_z", 0.0) or 0.0)
    max_z = float(getattr(row, "max_z", 0.0) or 0.0)
    size = str(getattr(row, "radar_bird_size", "Unknown") or "Unknown").replace(" ", "_")
    tokens.append(f"meta_len_{min(n // 8, 64)}")
    tokens.append(f"meta_air_{_bucket_scalar(airspeed, 0.0, 80.0)}")
    tokens.append(f"meta_minz_{_bucket_scalar(min_z, -200.0, 3000.0)}")
    tokens.append(f"meta_maxz_{_bucket_scalar(max_z, -200.0, 3000.0)}")
    tokens.append(f"meta_size_{size}")
    return " ".join(tokens)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = out_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    sample_sub = pd.read_csv(args.sample_submission)

    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = np.zeros((len(train_df), len(CLASSES)), dtype=np.float32)
    y[np.arange(len(train_df)), y_idx] = 1.0

    train_track_feats = [_parse_track_features(r) for r in train_df.itertuples(index=False)]
    test_track_feats = [_parse_track_features(r) for r in test_df.itertuples(index=False)]
    edges = _build_bin_edges(
        track_feats=train_track_feats,
        n_bins=int(args.n_bins),
        cap=int(args.sample_points_per_feature),
        seed=int(args.seed),
    )
    train_docs = [_build_doc(r, f, edges) for r, f in zip(train_df.itertuples(index=False), train_track_feats)]
    test_docs = [_build_doc(r, f, edges) for r, f in zip(test_df.itertuples(index=False), test_track_feats)]

    cv_df = pd.DataFrame(
        {
            "_cv_ts": train_df[str(args.time_col)],
            "_cv_group": train_df[str(args.group_col)].astype(np.int64),
        }
    )
    folds_full = make_forward_temporal_group_folds(
        cv_df, timestamp_col="_cv_ts", group_col="_cv_group", n_splits=int(args.n_splits)
    )
    folds = list(folds_full)
    if int(args.start_fold) > 0:
        s = min(int(args.start_fold), len(folds))
        folds = folds[s:]
        print(f"[tfidf] starting from fold index {s} (skipped {s} earliest folds)", flush=True)
    if len(folds) == 0:
        raise RuntimeError("no folds selected")

    oof = np.zeros((len(train_df), len(CLASSES)), dtype=np.float32)
    test_accum = np.zeros((len(test_df), len(CLASSES)), dtype=np.float32)
    fold_scores: list[float] = []
    fold_reports: list[dict[str, Any]] = []
    skipped_folds: list[int] = []
    effective_folds = 0
    class_weight = None if str(args.class_weight) == "none" else "balanced"

    for fold_id, (tr_idx_raw, va_idx) in enumerate(folds):
        tr_idx = np.asarray(tr_idx_raw, dtype=np.int64)
        va_idx = np.asarray(va_idx, dtype=np.int64)

        invalid_classes: list[str] = []
        for c, cls in enumerate(CLASSES):
            pos = int(y[tr_idx, c].sum())
            neg = int(len(tr_idx) - pos)
            if min(pos, neg) < int(args.min_class_count):
                invalid_classes.append(cls)

        if invalid_classes and str(args.invalid_class_policy) == "skip_fold":
            skipped_folds.append(fold_id)
            fold_reports.append(
                {
                    "fold_id": int(fold_id),
                    "n_val": int(len(va_idx)),
                    "skipped": True,
                    "invalid_classes": invalid_classes,
                }
            )
            print(f"[fold] {fold_id+1}/{len(folds)} skipped invalid_classes={len(invalid_classes)}", flush=True)
            continue

        docs_tr = [train_docs[int(i)] for i in tr_idx.tolist()]
        docs_va = [train_docs[int(i)] for i in va_idx.tolist()]
        vec = TfidfVectorizer(
            lowercase=False,
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(int(args.ngram_min), int(args.ngram_max)),
            min_df=int(args.min_df),
            max_features=int(args.max_features),
            sublinear_tf=True,
        )
        xtr = vec.fit_transform(docs_tr)
        xva = vec.transform(docs_va)
        xte = vec.transform(test_docs)

        fold_test = np.zeros((len(test_df), len(CLASSES)), dtype=np.float32)
        for c, cls in enumerate(CLASSES):
            yt = y[tr_idx, c]
            pos = int(yt.sum())
            neg = int(len(tr_idx) - pos)
            if min(pos, neg) < int(args.min_class_count):
                prior = float(np.mean(yt)) if len(yt) > 0 else 0.5
                oof[va_idx, c] = prior
                fold_test[:, c] = prior
                continue

            model = LogisticRegression(
                solver="saga",
                C=float(args.c),
                class_weight=class_weight,
                max_iter=int(args.max_iter),
                random_state=int(args.seed) + 97 * fold_id + 13 * c,
            )
            model.fit(xtr, yt)
            pva = model.predict_proba(xva)[:, 1].astype(np.float32)
            pte = model.predict_proba(xte)[:, 1].astype(np.float32)
            oof[va_idx, c] = np.clip(pva, 1e-6, 1.0 - 1e-6)
            fold_test[:, c] = np.clip(pte, 1e-6, 1.0 - 1e-6)

        effective_folds += 1
        test_accum += fold_test
        fold_macro = float(macro_map(y[va_idx], oof[va_idx]))
        fold_scores.append(fold_macro)
        fold_reports.append(
            {
                "fold_id": int(fold_id),
                "n_val": int(len(va_idx)),
                "skipped": False,
                "invalid_classes": invalid_classes,
                "macro_map": fold_macro,
            }
        )
        print(
            f"[fold] {fold_id+1}/{len(folds)} macro_map={fold_macro:.6f} n_val={len(va_idx)} invalid_classes={len(invalid_classes)}",
            flush=True,
        )

    if effective_folds <= 0:
        raise RuntimeError("all folds skipped")
    test_accum /= float(effective_folds)

    train_ids = train_df["track_id"].to_numpy(dtype=np.int64)
    test_ids = test_df["track_id"].to_numpy(dtype=np.int64)
    np.save(out_dir / "train_track_ids.npy", train_ids)
    np.save(out_dir / "test_track_ids.npy", test_ids)
    np.save(out_dir / "oof_targets.npy", y)
    np.save(art_dir / "tfidf_ovr_oof.npy", oof)
    np.save(art_dir / "tfidf_ovr_test.npy", test_accum)

    sub = sample_sub.copy()
    if "track_id" not in sub.columns:
        raise ValueError("sample submission must contain track_id")
    sub = sub[[c for c in sub.columns if c == "track_id" or c in CLASSES]].copy()
    sub["track_id"] = test_ids
    sub[CLASSES] = test_accum
    sub.to_csv(out_dir / "submission_tfidf_ovr.csv", index=False)

    macro_all = float(macro_map(y, oof))
    summary = {
        "project_root": str(PROJECT_ROOT.resolve()),
        "output_dir": str(out_dir),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "start_fold": int(args.start_fold),
        "model": {
            "type": "trajectory_tfidf_logreg_ovr",
            "n_bins": int(args.n_bins),
            "ngram_range": [int(args.ngram_min), int(args.ngram_max)],
            "min_df": int(args.min_df),
            "max_features": int(args.max_features),
            "c": float(args.c),
            "max_iter": int(args.max_iter),
            "class_weight": str(args.class_weight),
        },
        "invalid_class_handling": {
            "policy": str(args.invalid_class_policy),
            "min_class_count": int(args.min_class_count),
            "requested_folds": int(len(folds)),
            "effective_folds": int(effective_folds),
            "skipped_folds": skipped_folds,
        },
        "models": {
            f"tfidf_ovr_seed{int(args.seed)}": {
                "type": "trajectory_text",
                "oof_path": str((art_dir / "tfidf_ovr_oof.npy").resolve()),
                "test_path": str((art_dir / "tfidf_ovr_test.npy").resolve()),
                "macro_map": macro_all,
                "per_class_ap": per_class_ap(y, oof),
                "fold_scores": [float(x) for x in fold_scores],
                "worst_fold": float(np.min(fold_scores)) if fold_scores else 0.0,
                "fold_reports": fold_reports,
            }
        },
    }
    dump_json(out_dir / "oof_summary.json", summary)

    print("=== TFIDF TRAJECTORY OVR COMPLETE ===", flush=True)
    print(f"output_dir: {out_dir}", flush=True)
    print(f"oof_summary: {out_dir / 'oof_summary.json'}", flush=True)
    print(
        f"macro_oof={macro_all:.6f} fold_mean={float(np.mean(fold_scores)):.6f} fold_worst={float(np.min(fold_scores)):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
