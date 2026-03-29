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
    p = argparse.ArgumentParser(description="Build pseudo labels from submission (top1 with confidence threshold).")
    p.add_argument("--sub-csv", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--prob-thr", type=float, default=0.985)
    p.add_argument("--max-per-class", type=int, default=150)
    p.add_argument("--per-class-thr", type=str, default="")
    p.add_argument("--per-class-cap", type=str, default="")
    p.add_argument("--exclude-classes", type=str, default="")
    p.add_argument("--id-col", type=str, default="track_id")
    p.add_argument("--out-json", type=str, default="")
    return p.parse_args()


def _parse_class_float_map(text: str, arg_name: str) -> dict[str, float]:
    out: dict[str, float] = {}
    s = str(text).strip()
    if not s:
        return out
    for part in [x.strip() for x in s.split(",") if x.strip()]:
        if ":" not in part:
            raise ValueError(f"{arg_name}: bad item '{part}', expected Class:value")
        name, val = [x.strip() for x in part.split(":", 1)]
        if name not in CLASSES:
            raise ValueError(f"{arg_name}: unknown class '{name}'")
        out[name] = float(val)
    return out


def _parse_class_int_map(text: str, arg_name: str) -> dict[str, int]:
    out: dict[str, int] = {}
    s = str(text).strip()
    if not s:
        return out
    for part in [x.strip() for x in s.split(",") if x.strip()]:
        if ":" not in part:
            raise ValueError(f"{arg_name}: bad item '{part}', expected Class:value")
        name, val = [x.strip() for x in part.split(":", 1)]
        if name not in CLASSES:
            raise ValueError(f"{arg_name}: unknown class '{name}'")
        out[name] = int(val)
    return out


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


def main() -> None:
    args = parse_args()
    thr_map = _parse_class_float_map(args.per_class_thr, "--per-class-thr")
    cap_map = _parse_class_int_map(args.per_class_cap, "--per-class-cap")
    exclude_classes = _parse_class_set(args.exclude_classes, "--exclude-classes")
    df = pd.read_csv(args.sub_csv)
    req = {str(args.id_col), *CLASSES}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"{args.sub_csv} missing columns: {miss}")

    probs = np.clip(df[CLASSES].to_numpy(dtype=np.float64), 0.0, 1.0)
    top_prob = probs.max(axis=1)
    top_idx = probs.argmax(axis=1)
    pred_class = np.array([CLASSES[int(i)] for i in top_idx], dtype=object)
    keep = np.ones((len(df),), dtype=bool)
    for i, cls_name in enumerate(pred_class.tolist()):
        if cls_name in exclude_classes:
            keep[i] = False
            continue
        thr = float(thr_map.get(cls_name, float(args.prob_thr)))
        keep[i] = bool(top_prob[i] >= thr)

    pseudo = pd.DataFrame(
        {
            str(args.id_col): df.loc[keep, str(args.id_col)].to_numpy(dtype=np.int64),
            "bird_group": pred_class[keep].tolist(),
            "pseudo_prob": top_prob[keep].astype(np.float32),
        }
    )
    for c in CLASSES:
        pseudo[c] = probs[keep, CLASSES.index(c)].astype(np.float32)

    out_parts: list[pd.DataFrame] = []
    cap_default = int(max(0, args.max_per_class))
    for c in CLASSES:
        if c in exclude_classes:
            continue
        cap = int(cap_map.get(c, cap_default))
        if cap <= 0:
            continue
        part = pseudo[pseudo["bird_group"] == c].sort_values("pseudo_prob", ascending=False)
        out_parts.append(part.head(cap))
    pseudo2 = pd.concat(out_parts, axis=0, ignore_index=True) if out_parts else pseudo.iloc[:0].copy()

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pseudo2.to_csv(out_csv, index=False)

    out_json_path = Path(args.out_json).resolve() if str(args.out_json).strip() else out_csv.with_suffix(".json")
    report = {
        "sub_csv": str(Path(args.sub_csv).resolve()),
        "out_csv": str(out_csv),
        "prob_thr": float(args.prob_thr),
        "max_per_class": int(cap_default),
        "per_class_thr": {k: float(v) for k, v in thr_map.items()},
        "per_class_cap": {k: int(v) for k, v in cap_map.items()},
        "exclude_classes": sorted(list(exclude_classes)),
        "n_input": int(len(df)),
        "n_keep_thr": int(keep.sum()),
        "n_output": int(len(pseudo2)),
        "class_counts": {str(k): int(v) for k, v in pseudo2["bird_group"].value_counts().to_dict().items()},
        "pseudo_prob_stats": {
            "min": float(pseudo2["pseudo_prob"].min()) if len(pseudo2) else 0.0,
            "max": float(pseudo2["pseudo_prob"].max()) if len(pseudo2) else 0.0,
            "mean": float(pseudo2["pseudo_prob"].mean()) if len(pseudo2) else 0.0,
            "p95": float(pseudo2["pseudo_prob"].quantile(0.95)) if len(pseudo2) else 0.0,
        },
    }
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(str(out_csv), flush=True)
    print(
        f"pseudo_total={len(pseudo2)} "
        f"(default_thr={args.prob_thr}, default_cap={cap_default}, exclude={sorted(list(exclude_classes))})",
        flush=True,
    )
    print(pseudo2["bird_group"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
