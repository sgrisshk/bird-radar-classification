#!/usr/bin/env python3
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

from config import CLASSES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build consensus pseudo labels from two submission CSVs.")
    p.add_argument("--anchor-sub-csv", required=True)
    p.add_argument("--aux-sub-csv", required=True, help="Second model submission (e.g., CatBoost).")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-json", default="")
    p.add_argument("--id-col", default="track_id")

    p.add_argument("--default-anchor-thr", type=float, default=0.985)
    p.add_argument("--default-aux-thr", type=float, default=0.90)
    p.add_argument("--geese-anchor-thr", type=float, default=0.995)
    p.add_argument("--geese-aux-thr", type=float, default=0.95)
    p.add_argument("--exclude-classes", type=str, default="Cormorants")

    p.add_argument("--max-per-class", type=int, default=150)
    p.add_argument("--min-per-class", type=int, default=0)
    p.add_argument("--per-class-cap", type=str, default="", help="Comma list: Class:cap")
    p.add_argument("--soft-anchor-weight", type=float, default=0.7)
    p.add_argument("--soft-aux-weight", type=float, default=0.3)
    return p.parse_args()


def _parse_class_set(text: str, arg_name: str) -> set[str]:
    s = str(text).strip()
    if not s:
        return set()
    out: set[str] = set()
    for name in [x.strip() for x in s.split(",") if x.strip()]:
        if name not in CLASSES:
            raise ValueError(f"{arg_name}: unknown class '{name}'")
        out.add(name)
    return out


def _parse_class_int_map(text: str, arg_name: str) -> dict[str, int]:
    s = str(text).strip()
    if not s:
        return {}
    out: dict[str, int] = {}
    for part in [x.strip() for x in s.split(",") if x.strip()]:
        if ":" not in part:
            raise ValueError(f"{arg_name}: bad item '{part}', expected Class:value")
        name, val = [x.strip() for x in part.split(":", 1)]
        if name not in CLASSES:
            raise ValueError(f"{arg_name}: unknown class '{name}'")
        out[name] = int(val)
    return out


def _validate_submission(df: pd.DataFrame, id_col: str, name: str) -> None:
    req = {id_col, *CLASSES}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing columns: {miss}")


def main() -> None:
    args = parse_args()

    id_col = str(args.id_col)
    excl = _parse_class_set(args.exclude_classes, "--exclude-classes")
    per_class_cap = _parse_class_int_map(args.per_class_cap, "--per-class-cap")
    max_per_class = int(max(1, args.max_per_class))
    min_per_class = int(max(0, args.min_per_class))

    wa = float(max(0.0, args.soft_anchor_weight))
    wb = float(max(0.0, args.soft_aux_weight))
    wsum = wa + wb
    if wsum <= 0.0:
        raise ValueError("soft weights sum must be > 0")
    wa /= wsum
    wb /= wsum

    a = pd.read_csv(args.anchor_sub_csv)
    b = pd.read_csv(args.aux_sub_csv)
    _validate_submission(a, id_col=id_col, name="anchor")
    _validate_submission(b, id_col=id_col, name="aux")

    if not a[id_col].equals(b[id_col]):
        b = b.set_index(id_col).reindex(a[id_col]).reset_index()
        if b[id_col].isna().any():
            raise ValueError("aux submission could not be aligned to anchor track ids")

    pa = np.clip(a[CLASSES].to_numpy(dtype=np.float64), 1e-9, 1.0)
    pb = np.clip(b[CLASSES].to_numpy(dtype=np.float64), 1e-9, 1.0)

    topa_idx = pa.argmax(axis=1)
    topb_idx = pb.argmax(axis=1)
    topa_prob = pa[np.arange(len(pa)), topa_idx]
    topb_prob = pb[np.arange(len(pb)), topb_idx]

    same_top = topa_idx == topb_idx
    top_cls = np.array([CLASSES[int(i)] for i in topa_idx], dtype=object)

    keep = np.zeros((len(pa),), dtype=bool)
    for i in range(len(pa)):
        if not same_top[i]:
            continue
        c = str(top_cls[i])
        if c in excl:
            continue
        if c == "Geese":
            thr_a = float(args.geese_anchor_thr)
            thr_b = float(args.geese_aux_thr)
        else:
            thr_a = float(args.default_anchor_thr)
            thr_b = float(args.default_aux_thr)
        keep[i] = bool(topa_prob[i] >= thr_a and topb_prob[i] >= thr_b)

    p_soft = np.clip(wa * pa + wb * pb, 1e-6, 1.0 - 1e-6)
    pseudo = pd.DataFrame(
        {
            id_col: a.loc[keep, id_col].to_numpy(dtype=np.int64),
            "bird_group": top_cls[keep].tolist(),
            "pseudo_prob": np.minimum(topa_prob[keep], topb_prob[keep]).astype(np.float32),
            "pseudo_prob_anchor": topa_prob[keep].astype(np.float32),
            "pseudo_prob_aux": topb_prob[keep].astype(np.float32),
        }
    )
    for ci, c in enumerate(CLASSES):
        pseudo[c] = p_soft[keep, ci].astype(np.float32)

    parts: list[pd.DataFrame] = []
    for c in CLASSES:
        if c in excl:
            continue
        part = pseudo[pseudo["bird_group"] == c].sort_values("pseudo_prob", ascending=False)
        if len(part) == 0:
            continue
        cls_cap = int(per_class_cap.get(c, max_per_class))
        if cls_cap <= 0:
            continue
        take_n = min(cls_cap, len(part))
        if min_per_class > 0:
            take_n = max(min_per_class, take_n)
            take_n = min(take_n, len(part))
        parts.append(part.head(take_n))
    out = pd.concat(parts, axis=0, ignore_index=True) if parts else pseudo.iloc[:0].copy()

    out_csv = Path(args.out_csv).resolve()
    out_json = Path(args.out_json).resolve() if str(args.out_json).strip() else out_csv.with_suffix(".json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    report = {
        "anchor_sub_csv": str(Path(args.anchor_sub_csv).resolve()),
        "aux_sub_csv": str(Path(args.aux_sub_csv).resolve()),
        "out_csv": str(out_csv),
        "id_col": id_col,
        "n_input": int(len(a)),
        "n_same_top1": int(same_top.sum()),
        "n_after_thresholds": int(keep.sum()),
        "n_output": int(len(out)),
        "default_anchor_thr": float(args.default_anchor_thr),
        "default_aux_thr": float(args.default_aux_thr),
        "geese_anchor_thr": float(args.geese_anchor_thr),
        "geese_aux_thr": float(args.geese_aux_thr),
        "exclude_classes": sorted(list(excl)),
        "max_per_class": int(max_per_class),
        "per_class_cap": {k: int(v) for k, v in per_class_cap.items()},
        "min_per_class": int(min_per_class),
        "soft_anchor_weight": float(wa),
        "soft_aux_weight": float(wb),
        "class_counts": {str(k): int(v) for k, v in out["bird_group"].value_counts().to_dict().items()},
        "pseudo_prob_stats": {
            "min": float(out["pseudo_prob"].min()) if len(out) else 0.0,
            "max": float(out["pseudo_prob"].max()) if len(out) else 0.0,
            "mean": float(out["pseudo_prob"].mean()) if len(out) else 0.0,
            "p95": float(out["pseudo_prob"].quantile(0.95)) if len(out) else 0.0,
        },
    }
    out_json.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(str(out_csv), flush=True)
    print(f"n_output={len(out)}", flush=True)
    if len(out):
        print(out["bird_group"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
