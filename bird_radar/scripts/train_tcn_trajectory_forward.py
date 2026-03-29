#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds
from src.metrics import macro_map_score, per_class_average_precision
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 1D TCN on raw trajectory channels with forward CV.")
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--test-csv", type=str, required=True)
    p.add_argument("--sample-submission", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--cache-dir", type=str, default="bird_radar/artifacts/cache")
    p.add_argument(
        "--trajectory-normalization",
        type=str,
        default="none",
        choices=["none", "heading_rel", "rcs_only"],
        help="Pre-model trajectory normalization: none, heading-relative, or rcs-only (keep only rcs/drcs/dt channels).",
    )

    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--lr-scheduler", type=str, default="none", choices=["none", "cosine_warmup"])
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--sample-weights-npy", type=str, default="")
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--use-pos-weight", type=int, default=1)

    p.add_argument("--seq-len", type=int, default=160)
    p.add_argument("--model-type", type=str, default="tcn", choices=["tcn", "transformer"])
    p.add_argument("--stem-channels", type=int, default=64)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--dilations", type=str, default="1,2,4,8,16,32")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--ffn-dim", type=int, default=512)
    p.add_argument("--transformer-multires", type=int, default=0)
    p.add_argument("--multires-pool-factor", type=int, default=2)
    p.add_argument("--physics-token-mode", type=str, default="none", choices=["none", "v1"])
    p.add_argument("--multitask-group-scheme", type=str, default="none", choices=["none", "water_land_clutter", "water_focus_v1"])
    p.add_argument("--multitask-group-weight", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--logit-reg-lambda", type=float, default=0.0)
    p.add_argument("--logit-reg-threshold", type=float, default=8.0)
    p.add_argument("--pos-weight-mode", type=str, default="neg_pos", choices=["neg_pos", "inv_sqrt_freq"])

    p.add_argument("--augment-time-crop-p", type=float, default=0.5)
    p.add_argument("--augment-noise-std-xyz", type=float, default=0.01)
    p.add_argument("--augment-rcs-drop-p", type=float, default=0.05)
    p.add_argument("--augment-speed-drop-p", type=float, default=0.0)
    p.add_argument("--raw-clip-quantile-low", type=float, default=0.01)
    p.add_argument("--raw-clip-quantile-high", type=float, default=0.99)

    p.add_argument("--teacher-oof-csv", type=str, default="")
    p.add_argument("--teacher-ensemble-fr-oof-csv", type=str, default="")
    p.add_argument("--teacher-ensemble-tcn-oof-npy", type=str, default="")
    p.add_argument("--teacher-ensemble-tcn-track-ids-npy", type=str, default="")
    p.add_argument("--teacher-ensemble-trf-oof-npy", type=str, default="")
    p.add_argument("--teacher-ensemble-trf-track-ids-npy", type=str, default="")
    p.add_argument("--teacher-ensemble-weights", type=str, default="0.55,0.25,0.20")
    p.add_argument("--teacher-temp", type=float, default=1.6)
    p.add_argument("--teacher-logit-clip", type=float, default=8.0)

    p.add_argument("--distill-teacher-weight", type=float, default=0.0)
    p.add_argument("--distill-label-weight", type=float, default=1.0)
    p.add_argument("--distill-warmup-epochs", type=int, default=0)
    p.add_argument("--distill-temperature", type=float, default=1.0)
    p.add_argument("--distill-lambda-hard", type=float, default=-1.0)
    p.add_argument("--distill-lambda-soft", type=float, default=-1.0)
    p.add_argument("--distill-class-downweight", type=str, default="")
    p.add_argument("--consistency-lambda", type=float, default=0.0)
    p.add_argument("--aux-mask-ratio", type=float, default=0.0)
    p.add_argument("--aux-lambda", type=float, default=0.0)
    p.add_argument("--aux-ramp-ratio", type=float, default=0.2)
    p.add_argument("--id-col", type=str, default="track_id")
    p.add_argument("--time-col", type=str, default="timestamp_start_radar_utc")
    p.add_argument("--group-col", type=str, default="observation_id")
    p.add_argument("--cv-direction", type=str, default="forward", choices=["forward", "reverse"])
    p.add_argument("--out-name", type=str, default="submission_tcn.csv")
    p.add_argument("--pretrain-ssl", type=int, default=0)
    p.add_argument("--finetune-from-ssl", type=str, default="")
    p.add_argument("--pseudo-csv", type=str, default="")
    p.add_argument("--pseudo-label-csv", type=str, default="")
    p.add_argument("--pseudo-weight", type=float, default=0.4)
    p.add_argument("--pseudo-min-prob", type=float, default=0.97)
    p.add_argument("--pseudo-prob-power", type=float, default=1.0)
    p.add_argument("--pseudo-ramp-ratio", type=float, default=0.0)
    p.add_argument("--hard-loss-class-weight", type=str, default="")
    args = p.parse_args()
    if str(args.pseudo_csv).strip() and not str(args.pseudo_label_csv).strip():
        args.pseudo_label_csv = str(args.pseudo_csv)
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _resolve_device(name: str) -> torch.device:
    n = str(name).strip().lower()
    if n == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if n == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict[str, Any]]:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


def _align_teacher_probs(csv_path: str, ids: np.ndarray, id_col: str) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=[id_col, *CLASSES])
    mp = {int(r[id_col]): np.asarray([r[c] for c in CLASSES], dtype=np.float32) for _, r in df.iterrows()}
    missing = [int(t) for t in ids.tolist() if int(t) not in mp]
    if missing:
        raise ValueError(f"teacher_oof_csv missing {len(missing)} ids; first={missing[:10]}")
    return np.stack([mp[int(t)] for t in ids.tolist()], axis=0).astype(np.float32)


def _align_probs_from_npy(npy_path: str, ids_path: str, ids_order: np.ndarray) -> np.ndarray:
    arr = np.asarray(np.load(npy_path), dtype=np.float32)
    ids = np.asarray(np.load(ids_path), dtype=np.int64).reshape(-1)
    if len(arr) != len(ids):
        raise ValueError(f"{npy_path}: len(pred) != len(ids): {len(arr)} vs {len(ids)}")
    if np.array_equal(ids, ids_order):
        return arr.astype(np.float32)
    pos = {int(t): i for i, t in enumerate(ids.tolist())}
    missing = [int(t) for t in ids_order.tolist() if int(t) not in pos]
    if missing:
        raise ValueError(f"{npy_path}: ids missing {len(missing)} rows; first={missing[:10]}")
    return np.stack([arr[pos[int(t)]] for t in ids_order.tolist()], axis=0).astype(np.float32)


def _parse_teacher_ensemble_weights(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError("--teacher-ensemble-weights must contain exactly 3 numbers: fr,tcn,trf")
    w = np.asarray(vals, dtype=np.float32)
    sw = float(np.sum(np.abs(w)))
    if sw <= 1e-8:
        raise ValueError("--teacher-ensemble-weights sum is zero")
    return (w / sw).astype(np.float32)


def _build_teacher_ensemble_probs(
    p_fr: np.ndarray,
    p_tcn: np.ndarray,
    p_trf: np.ndarray,
    weights: np.ndarray,
    teacher_temp: float,
    logit_clip: float,
) -> np.ndarray:
    eps = 1e-6
    tc = max(1e-6, float(teacher_temp))
    lc = max(1e-6, float(logit_clip))
    def _logit(p: np.ndarray) -> np.ndarray:
        q = np.clip(p.astype(np.float32), eps, 1.0 - eps)
        return np.log(q / (1.0 - q)).astype(np.float32)

    l_fr = np.clip(_logit(p_fr), -lc, lc)
    l_tcn = np.clip(_logit(p_tcn), -lc, lc)
    l_trf = np.clip(_logit(p_trf), -lc, lc)
    l_ens = weights[0] * l_fr + weights[1] * l_tcn + weights[2] * l_trf
    p_ens = 1.0 / (1.0 + np.exp(-np.clip(l_ens / tc, -40.0, 40.0)))
    return np.clip(p_ens.astype(np.float32), eps, 1.0 - eps)


def _parse_class_downweight(text: str) -> np.ndarray:
    w = np.ones((len(CLASSES),), dtype=np.float32)
    s = str(text).strip()
    if not s:
        return w
    parts = [x.strip() for x in s.split(",") if x.strip()]
    for part in parts:
        if ":" not in part:
            raise ValueError(f"bad distill-class-downweight item '{part}', expected Class:weight")
        name, val = [x.strip() for x in part.split(":", 1)]
        if name not in CLASSES:
            raise ValueError(f"unknown class in --distill-class-downweight: {name}")
        ww = float(val)
        if ww < 0.0:
            raise ValueError(f"class downweight must be >=0, got {ww} for {name}")
        w[CLASSES.index(name)] = float(ww)
    return w


def _parse_hard_loss_class_weight(text: str) -> np.ndarray:
    w = np.ones((len(CLASSES),), dtype=np.float32)
    s = str(text).strip()
    if not s:
        return w
    parts = [x.strip() for x in s.split(",") if x.strip()]
    for part in parts:
        if ":" not in part:
            raise ValueError(f"bad hard-loss-class-weight item '{part}', expected Class:weight")
        name, val = [x.strip() for x in part.split(":", 1)]
        if name not in CLASSES:
            raise ValueError(f"unknown class in --hard-loss-class-weight: {name}")
        ww = float(val)
        if ww < 0.0:
            raise ValueError(f"hard loss class weight must be >=0, got {ww} for {name}")
        w[CLASSES.index(name)] = float(ww)
    return w


def _safe_corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size == 0:
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return c


def _load_pseudo_labels(
    pseudo_csv: str,
    test_ids: np.ndarray,
    seqs_test: list[np.ndarray],
    min_prob: float,
    base_weight: float,
    prob_power: float,
    id_col: str,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(pseudo_csv)
    req = {str(id_col), "bird_group", "pseudo_prob"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"pseudo csv missing columns: {miss}")

    df = df.copy()
    df["pseudo_prob"] = pd.to_numeric(df["pseudo_prob"], errors="coerce").fillna(0.0)
    df = df[df["pseudo_prob"] >= float(min_prob)].reset_index(drop=True)
    if len(df) == 0:
        return [], np.zeros((0, len(CLASSES)), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0, len(CLASSES)), dtype=np.float32)

    bad_cls = [c for c in df["bird_group"].astype(str).unique().tolist() if c not in CLASS_TO_INDEX]
    if bad_cls:
        raise ValueError(f"pseudo csv has unknown classes: {bad_cls[:10]}")

    id_to_pos = {int(t): i for i, t in enumerate(test_ids.astype(np.int64).tolist())}
    keep_rows: list[int] = []
    seqs: list[np.ndarray] = []
    for i, tid in enumerate(df[str(id_col)].astype(np.int64).tolist()):
        pos = id_to_pos.get(int(tid))
        if pos is None:
            continue
        keep_rows.append(i)
        seqs.append(seqs_test[int(pos)])
    if not keep_rows:
        return [], np.zeros((0, len(CLASSES)), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0, len(CLASSES)), dtype=np.float32)

    df = df.iloc[keep_rows].reset_index(drop=True)
    y_idx = df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0

    p = np.clip(df["pseudo_prob"].to_numpy(dtype=np.float32), 1e-6, 1.0)
    pw = float(max(0.0, base_weight)) * np.power(p, float(max(0.0, prob_power)))
    sample_w = np.clip(pw.astype(np.float32), 1e-6, None)

    if all(c in df.columns for c in CLASSES):
        soft = np.clip(df[CLASSES].to_numpy(dtype=np.float32), 1e-6, 1.0 - 1e-6)
    else:
        soft = y.copy()
    return seqs, y, sample_w, soft


def _build_group_matrix(scheme: str) -> tuple[np.ndarray, list[str]]:
    sch = str(scheme).strip().lower()
    if sch in {"", "none"}:
        return np.zeros((len(CLASSES), 0), dtype=np.float32), []

    groups: list[tuple[str, list[str]]] = []
    if sch == "water_land_clutter":
        groups = [
            ("Clutter", ["Clutter"]),
            ("Waterbirds", ["Cormorants", "Ducks", "Geese", "Gulls", "Waders"]),
            ("Landbirds", ["Pigeons", "Birds of Prey", "Songbirds"]),
        ]
    elif sch == "water_focus_v1":
        groups = [
            ("Clutter", ["Clutter"]),
            ("WaterFocus", ["Cormorants", "Ducks", "Geese"]),
            ("OtherWater", ["Gulls", "Waders"]),
            ("Landbirds", ["Pigeons", "Birds of Prey", "Songbirds"]),
        ]
    else:
        raise ValueError(f"unknown --multitask-group-scheme: {scheme}")

    mat = np.zeros((len(CLASSES), len(groups)), dtype=np.float32)
    names: list[str] = []
    for gi, (name, cls_list) in enumerate(groups):
        names.append(name)
        for c in cls_list:
            if c not in CLASS_TO_INDEX:
                raise ValueError(f"group scheme '{scheme}' has unknown class '{c}'")
            mat[int(CLASS_TO_INDEX[c]), gi] = 1.0
    return mat, names


def _teacher_diag(y_true: np.ndarray, teacher: np.ndarray, model: np.ndarray) -> dict[str, float]:
    teacher = np.clip(teacher.astype(np.float32), 1e-6, 1.0 - 1e-6)
    model = np.clip(model.astype(np.float32), 1e-6, 1.0 - 1e-6)
    teacher_macro = float(macro_map_score(y_true, teacher))
    model_macro = float(macro_map_score(y_true, model))
    best = teacher_macro
    best_w = 1.0
    for w in [0.95, 0.90, 0.85, 0.80, 0.70]:
        blend = np.clip(w * teacher + (1.0 - w) * model, 0.0, 1.0)
        m = float(macro_map_score(y_true, blend))
        if m > best:
            best = m
            best_w = float(w)
    return {
        "teacher_macro": teacher_macro,
        "model_macro": model_macro,
        "corr_with_teacher": float(_safe_corr_flat(teacher, model)),
        "best_blend_macro": best,
        "best_blend_w_teacher": best_w,
        "best_blend_gain": float(best - teacher_macro),
    }


def _save_per_class_delta_vs_teacher(
    y_true: np.ndarray,
    teacher_prob: np.ndarray,
    model_prob: np.ndarray,
    out_csv: Path,
) -> None:
    ap_teacher = per_class_average_precision(y_true, teacher_prob)
    ap_model = per_class_average_precision(y_true, model_prob)
    rows: list[dict[str, Any]] = []
    for c in CLASSES:
        at = float(ap_teacher.get(c, 0.0))
        am = float(ap_model.get(c, 0.0))
        rows.append(
            {
                "class": c,
                "ap_teacher": at,
                "ap_model": am,
                "delta_model_vs_teacher": float(am - at),
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def _normalize_per_track(arr: np.ndarray, valid_len: int) -> np.ndarray:
    out = arr.copy()
    if valid_len <= 1:
        return out
    m = np.mean(out[:valid_len], axis=0, keepdims=True)
    s = np.std(out[:valid_len], axis=0, keepdims=True)
    s = np.where(s < 1e-5, 1.0, s)
    out[:valid_len] = (out[:valid_len] - m) / s
    return out


def _normalize_raw_sequence(arr: np.ndarray, mode: str) -> np.ndarray:
    m = str(mode).strip().lower()
    if m in {"", "none"}:
        return arr
    if m not in {"heading_rel", "rcs_only"}:
        raise ValueError(f"unknown trajectory normalization mode: {mode}")

    if arr.ndim != 2 or arr.shape[1] < 9:
        return arr
    out = arr.copy().astype(np.float32, copy=False)
    if len(out) <= 1:
        return out

    x = out[:, 0]
    y = out[:, 1]
    alt = out[:, 2]
    rcs = out[:, 3]

    net_dx = float(x[-1] - x[0])
    net_dy = float(y[-1] - y[0])
    if abs(net_dx) + abs(net_dy) > 1e-10:
        angle = float(np.arctan2(net_dx, net_dy))
        cos_a = float(np.cos(-angle))
        sin_a = float(np.sin(-angle))
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
    else:
        x_rot = x
        y_rot = y

    if m == "heading_rel":
        out[:, 0] = x_rot.astype(np.float32)
        out[:, 1] = y_rot.astype(np.float32)
        out[:, 2] = (alt - alt[0]).astype(np.float32)
        out[:, 3] = np.diff(rcs, prepend=rcs[:1]).astype(np.float32)
        return out

    # rcs_only: retain only rcs, drcs, dt channels to isolate sequence wingbeat signal.
    # Channel layout in raw_features: [x, y, z, rcs, speed, vertical_speed, accel, curvature, dt]
    drcs = np.diff(rcs, prepend=rcs[:1]).astype(np.float32)
    dtc = out[:, 8].copy().astype(np.float32)
    out[:, :] = 0.0
    out[:, 3] = rcs.astype(np.float32)
    out[:, 4] = drcs
    out[:, 8] = dtc
    return out


def _fit_raw_scaler(
    seqs: list[np.ndarray],
    q_low: float,
    q_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(seqs) == 0:
        raise ValueError("empty seqs for raw scaler")
    cat = np.concatenate(seqs, axis=0).astype(np.float64, copy=False)
    if cat.ndim != 2 or cat.shape[1] != 9:
        raise ValueError(f"expected concatenated shape [N,9], got {cat.shape}")
    lo = np.quantile(cat, q_low, axis=0).astype(np.float32)
    hi = np.quantile(cat, q_high, axis=0).astype(np.float32)
    span = hi - lo
    hi = np.where(span < 1e-6, lo + 1.0, hi)
    clipped = np.clip(cat, lo[None, :], hi[None, :])
    mean = np.mean(clipped, axis=0).astype(np.float32)
    std = np.std(clipped, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return lo, hi, mean, std


class TrajectoryDataset(
    Dataset[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]
):
    def __init__(
        self,
        seqs: list[np.ndarray],
        y: np.ndarray,
        seq_len: int,
        raw_clip_lo: np.ndarray,
        raw_clip_hi: np.ndarray,
        raw_mean: np.ndarray,
        raw_std: np.ndarray,
        augment: bool,
        time_crop_p: float,
        noise_std_xyz: float,
        rcs_drop_p: float,
        speed_drop_p: float,
        sample_weights: np.ndarray | None = None,
        soft_targets: np.ndarray | None = None,
        pseudo_mask: np.ndarray | None = None,
    ) -> None:
        self.seqs = seqs
        self.y = y.astype(np.float32, copy=False)
        self.seq_len = int(seq_len)
        self.raw_clip_lo = raw_clip_lo.astype(np.float32, copy=False)
        self.raw_clip_hi = raw_clip_hi.astype(np.float32, copy=False)
        self.raw_mean = raw_mean.astype(np.float32, copy=False)
        self.raw_std = raw_std.astype(np.float32, copy=False)
        self.augment = bool(augment)
        self.time_crop_p = float(time_crop_p)
        self.noise_std_xyz = float(noise_std_xyz)
        self.rcs_drop_p = float(rcs_drop_p)
        self.speed_drop_p = float(speed_drop_p)
        if sample_weights is None:
            self.sample_weights = np.ones((len(self.seqs),), dtype=np.float32)
        else:
            sw = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
            if sw.ndim != 1 or len(sw) != len(self.seqs):
                raise ValueError(
                    f"sample_weights must be shape [N], got {sw.shape} for N={len(self.seqs)}"
                )
            self.sample_weights = sw
        if soft_targets is None:
            self.soft_targets = self.y.copy()
        else:
            st = np.asarray(soft_targets, dtype=np.float32)
            if st.shape != self.y.shape:
                raise ValueError(
                    f"soft_targets must have shape {self.y.shape}, got {st.shape}"
                )
            self.soft_targets = st
        if pseudo_mask is None:
            self.pseudo_mask = np.zeros((len(self.seqs),), dtype=np.float32)
        else:
            pm = np.asarray(pseudo_mask, dtype=np.float32).reshape(-1)
            if pm.ndim != 1 or len(pm) != len(self.seqs):
                raise ValueError(
                    f"pseudo_mask must be shape [N], got {pm.shape} for N={len(self.seqs)}"
                )
            self.pseudo_mask = np.clip(pm, 0.0, 1.0).astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.seqs)

    def _time_transform(self, seq: np.ndarray) -> np.ndarray:
        n = int(seq.shape[0])
        if not self.augment or n <= self.seq_len:
            return seq
        if random.random() >= self.time_crop_p:
            return seq
        crop_n = random.randint(self.seq_len, n)
        start = random.randint(0, n - crop_n)
        return seq[start : start + crop_n]

    def _to_fixed_len(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        n, c = int(seq.shape[0]), int(seq.shape[1])
        if n >= self.seq_len:
            idx = np.linspace(0, n - 1, num=self.seq_len, dtype=np.float64)
            idx = np.clip(np.round(idx).astype(np.int64), 0, n - 1)
            out = seq[idx]
            mask = np.ones((self.seq_len,), dtype=np.float32)
            valid_len = self.seq_len
            return out.astype(np.float32), mask, valid_len

        out = np.zeros((self.seq_len, c), dtype=np.float32)
        out[:n] = seq
        mask = np.zeros((self.seq_len,), dtype=np.float32)
        mask[:n] = 1.0
        return out, mask, n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.seqs[idx]
        seq = self._time_transform(seq)
        arr, mask, valid_len = self._to_fixed_len(seq)

        if self.augment and valid_len > 0:
            if self.noise_std_xyz > 0:
                arr[:valid_len, 0:3] += np.random.normal(
                    loc=0.0,
                    scale=self.noise_std_xyz,
                    size=(valid_len, 3),
                ).astype(np.float32)
            if self.rcs_drop_p > 0 and random.random() < self.rcs_drop_p:
                arr[:valid_len, 3] = 0.0
            if self.speed_drop_p > 0 and random.random() < self.speed_drop_p:
                arr[:valid_len, 4] = 0.0

        # Stream-1 (raw): robust clip + global standardization.
        raw = arr.copy()
        if valid_len > 0:
            raw[:valid_len] = np.clip(
                raw[:valid_len],
                self.raw_clip_lo[None, :],
                self.raw_clip_hi[None, :],
            )
            raw[:valid_len] = (raw[:valid_len] - self.raw_mean[None, :]) / self.raw_std[None, :]

        # Stream-2 (znorm): per-track z-normalization on valid timesteps.
        zn = _normalize_per_track(arr.copy(), valid_len=valid_len)

        x18 = np.concatenate([raw, zn], axis=1).astype(np.float32, copy=False)  # [L, 18]
        x = torch.from_numpy(x18.T.copy())  # [18, L]
        m = torch.from_numpy(mask)
        y = torch.from_numpy(self.y[idx])
        w = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
        ys = torch.from_numpy(self.soft_targets[idx])
        is_pseudo = torch.tensor(self.pseudo_mask[idx], dtype=torch.float32)
        return x, m, y, w, ys, is_pseudo


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.act(z)
        z = self.drop(z)
        z = self.conv2(z)
        z = self.bn2(z)
        z = self.drop(z)
        return self.act(x + z)


class MaskedAttentionPooling(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L], mask: [B, L] with 1.0 for valid timesteps.
        s = self.score(x).squeeze(1)  # [B, L]
        s = s.masked_fill(mask <= 0.0, -1e4)
        a = torch.softmax(s, dim=1)
        a = a * mask
        a = a / (a.sum(dim=1, keepdim=True) + 1e-8)
        pooled = torch.sum(x * a.unsqueeze(1), dim=2)
        return pooled


class TwoStreamTCNClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int,
        stem_channels: int,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.stem_raw = nn.Sequential(
            nn.Conv1d(9, stem_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.SiLU(inplace=True),
        )
        self.stem_zn = nn.Sequential(
            nn.Conv1d(9, stem_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(stem_channels * 2, stem_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualTCNBlock(
                    channels=stem_channels,
                    kernel_size=kernel_size,
                    dilation=int(d),
                    dropout=dropout,
                )
                for d in dilations
            ]
        )
        self.pool = MaskedAttentionPooling(stem_channels)
        self.head = nn.Linear(stem_channels, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        xr = x[:, :9, :]
        xz = x[:, 9:, :]
        z = self.fuse(torch.cat([self.stem_raw(xr), self.stem_zn(xz)], dim=1))
        z = self.blocks(z)
        p = self.pool(z, mask)
        return self.head(p)


class SequenceAttentionPooling(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D], mask: [B, L]
        s = self.score(x).squeeze(-1)  # [B, L]
        s = s.masked_fill(mask <= 0.0, -1e4)
        a = torch.softmax(s, dim=1)
        a = a * mask
        a = a / (a.sum(dim=1, keepdim=True) + 1e-8)
        return torch.sum(x * a.unsqueeze(-1), dim=1)


class TwoStreamTransformerClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float,
        physics_token_mode: str = "none",
        n_group_classes: int = 0,
        use_multires: bool = False,
        multires_pool_factor: int = 2,
        aux_out_dim: int = 0,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.n_group_classes = int(max(0, n_group_classes))
        self.use_multires = bool(use_multires)
        self.multires_pool_factor = int(max(2, multires_pool_factor))
        self.aux_out_dim = int(max(0, aux_out_dim))
        self.physics_token_mode = str(physics_token_mode).strip().lower()
        self.n_physics_tokens = 6 if self.physics_token_mode == "v1" else 0
        self.stem_raw = nn.Sequential(
            nn.Linear(9, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.stem_zn = nn.Sequential(
            nn.Linear(9, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + self.n_physics_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        low_len = int(np.ceil(float(self.seq_len) / float(self.multires_pool_factor)))
        self.pos_embed_low = nn.Parameter(torch.zeros(1, low_len + self.n_physics_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed_low, std=0.02)
        self.physics_proj = nn.Sequential(
            nn.Linear(18, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.encoder_low = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool = SequenceAttentionPooling(d_model)
        self.head = nn.Linear(d_model * (2 if self.use_multires else 1), n_classes)
        self.group_head = nn.Linear(d_model * (2 if self.use_multires else 1), self.n_group_classes) if self.n_group_classes > 0 else None
        self.aux_head = nn.Linear(d_model, self.aux_out_dim) if self.aux_out_dim > 0 else None

    @staticmethod
    def _masked_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C], w: [B, L, 1]
        den = w.sum(dim=1).clamp_min(1e-6)
        return (x * w).sum(dim=1) / den

    @staticmethod
    def _masked_std(x: torch.Tensor, w: torch.Tensor, mean: torch.Tensor | None = None) -> torch.Tensor:
        if mean is None:
            mean = TwoStreamTransformerClassifier._masked_mean(x, w)
        xc = x - mean.unsqueeze(1)
        den = w.sum(dim=1).clamp_min(1e-6)
        var = ((xc * xc) * w).sum(dim=1) / den
        return torch.sqrt(var.clamp_min(1e-8))

    def _build_physics_tokens(self, xt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # xt: [B, L, 18], mask: [B, L]
        bsz, length, _ = xt.shape
        if self.n_physics_tokens <= 0:
            return xt.new_zeros((bsz, 0, xt.size(-1)))

        raw = xt[:, :, :9]
        m = mask.unsqueeze(-1).clamp(0.0, 1.0)

        rel = torch.linspace(0.0, 1.0, steps=length, device=xt.device, dtype=xt.dtype).view(1, length, 1)
        w_early = m * (rel < (1.0 / 3.0)).to(xt.dtype)
        w_mid = m * ((rel >= (1.0 / 3.0)) & (rel < (2.0 / 3.0))).to(xt.dtype)
        w_late = m * (rel >= (2.0 / 3.0)).to(xt.dtype)

        def _fallback_weights(w_part: torch.Tensor) -> torch.Tensor:
            part_mass = w_part.sum(dim=1, keepdim=True)
            return torch.where(part_mass > 0.0, w_part, m)

        w_early = _fallback_weights(w_early)
        w_mid = _fallback_weights(w_mid)
        w_late = _fallback_weights(w_late)

        t_early = self._masked_mean(xt, w_early)
        t_mid = self._masked_mean(xt, w_mid)
        t_late = self._masked_mean(xt, w_late)
        t_global = self._masked_mean(xt, m)

        pair_mask = (mask[:, 1:] * mask[:, :-1]).unsqueeze(-1).to(xt.dtype)
        d_raw = raw[:, 1:, :] - raw[:, :-1, :]
        dxyz = d_raw[:, :, :3]
        step3d = torch.sqrt((dxyz * dxyz).sum(dim=-1, keepdim=True).clamp_min(1e-8))
        path_len = self._masked_mean(step3d, pair_mask).squeeze(-1)
        speed_jump = torch.abs(d_raw[:, :, 4:5])
        spike_ratio = self._masked_mean((speed_jump > 1.5).to(xt.dtype), pair_mask).squeeze(-1)
        turn_var = self._masked_std(raw[:, :, 7:8], m).squeeze(-1)
        v_abs = self._masked_mean(torch.abs(raw[:, :, 5:6]), m).squeeze(-1)
        dt_mean = self._masked_mean(raw[:, :, 8:9], m).squeeze(-1)

        t_events = xt.new_zeros((bsz, 18))
        t_events[:, 0] = spike_ratio
        t_events[:, 1] = path_len
        t_events[:, 2] = turn_var
        t_events[:, 3] = v_abs
        t_events[:, 4] = dt_mean
        t_events[:, 5:] = t_global[:, 5:]

        min_raw = raw.masked_fill(m <= 0.0, torch.inf).amin(dim=1)
        max_raw = raw.masked_fill(m <= 0.0, -torch.inf).amax(dim=1)
        min_raw = torch.where(torch.isfinite(min_raw), min_raw, torch.zeros_like(min_raw))
        max_raw = torch.where(torch.isfinite(max_raw), max_raw, torch.zeros_like(max_raw))
        raw_range = max_raw - min_raw
        t_shape = xt.new_zeros((bsz, 18))
        t_shape[:, 0] = raw_range[:, 0]
        t_shape[:, 1] = raw_range[:, 1]
        t_shape[:, 2] = raw_range[:, 2]
        t_shape[:, 3] = self._masked_mean(raw[:, :, 4:5], m).squeeze(-1)
        t_shape[:, 4] = self._masked_mean(torch.abs(raw[:, :, 6:7]), m).squeeze(-1)
        t_shape[:, 5] = self._masked_mean(torch.abs(raw[:, :, 7:8]), m).squeeze(-1)
        t_shape[:, 6:] = t_global[:, 6:]

        return torch.stack([t_early, t_mid, t_late, t_global, t_events, t_shape], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # x: [B, 18, L] -> [B, L, 18]
        xt = x.transpose(1, 2)
        xr = xt[:, :, :9]
        xz = xt[:, :, 9:]
        z_base = self.fuse(torch.cat([self.stem_raw(xr), self.stem_zn(xz)], dim=-1))
        full_mask = mask
        z = z_base
        if self.n_physics_tokens > 0:
            phys_tokens = self.physics_proj(self._build_physics_tokens(xt, mask))
            z = torch.cat([phys_tokens, z], dim=1)
            token_mask = mask.new_ones((mask.size(0), self.n_physics_tokens))
            full_mask = torch.cat([token_mask, mask], dim=1)
        z = z + self.pos_embed[:, : z.size(1), :]
        pad_mask = full_mask <= 0.0
        z = self.encoder(z, src_key_padding_mask=pad_mask)
        z_time = z[:, self.n_physics_tokens :, :]
        p_main = self.pool(z, full_mask)

        if self.use_multires:
            z_low = F.avg_pool1d(
                z_base.transpose(1, 2),
                kernel_size=self.multires_pool_factor,
                stride=self.multires_pool_factor,
                ceil_mode=True,
            ).transpose(1, 2)
            m_low = F.avg_pool1d(
                mask.unsqueeze(1),
                kernel_size=self.multires_pool_factor,
                stride=self.multires_pool_factor,
                ceil_mode=True,
            ).squeeze(1)
            m_low = (m_low > 0.0).to(mask.dtype)
            full_mask_low = m_low
            if self.n_physics_tokens > 0:
                phys_tokens_low = self.physics_proj(self._build_physics_tokens(xt, mask))
                z_low = torch.cat([phys_tokens_low, z_low], dim=1)
                tok_low = mask.new_ones((mask.size(0), self.n_physics_tokens))
                full_mask_low = torch.cat([tok_low, m_low], dim=1)
            z_low = z_low + self.pos_embed_low[:, : z_low.size(1), :]
            pad_mask_low = full_mask_low <= 0.0
            z_low = self.encoder_low(z_low, src_key_padding_mask=pad_mask_low)
            p_low = self.pool(z_low, full_mask_low)
            p = torch.cat([p_main, p_low], dim=1)
        else:
            p = p_main

        logits = self.head(p)
        out: list[torch.Tensor] = [logits]
        if self.group_head is not None:
            out.append(self.group_head(p))
        if self.aux_head is not None:
            out.append(self.aux_head(z_time))
        if len(out) == 1:
            return out[0]
        return tuple(out)


def _unpack_model_output(out: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if isinstance(out, tuple):
        if len(out) == 3:
            return out[0], out[1], out[2]
        if len(out) == 2:
            # Ambiguous tuple: could be (logits, group_logits) or (logits, aux_pred)
            second = out[1]
            if second.dim() == 3:
                return out[0], None, second
            return out[0], second, None
        if len(out) == 1:
            return out[0], None, None
    return out, None, None


def _load_pretrained_encoder_weights(model: nn.Module, ckpt_path: str) -> None:
    path = Path(str(ckpt_path).strip())
    if not str(path):
        return
    if not path.exists():
        raise FileNotFoundError(f"ssl checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    if not isinstance(state, dict):
        raise ValueError(f"bad ssl checkpoint format: expected state_dict, got {type(state)}")

    filtered: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if not isinstance(k, str):
            continue
        if (
            k.startswith("head.")
            or k.startswith("group_head.")
            or k.startswith("aux_head.")
        ):
            continue
        filtered[k] = v

    incompat = model.load_state_dict(filtered, strict=False)
    miss = list(getattr(incompat, "missing_keys", []))
    unexp = list(getattr(incompat, "unexpected_keys", []))
    print(
        f"[ssl] loaded encoder checkpoint='{path}' "
        f"loaded_keys={len(filtered)} missing={len(miss)} unexpected={len(unexp)}",
        flush=True,
    )


def _pretrain_ssl(
    seqs_train: list[np.ndarray],
    seqs_test: list[np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> None:
    if str(args.model_type).lower() != "transformer":
        raise ValueError("--pretrain-ssl currently supports only --model-type transformer")

    seqs_all = list(seqs_train) + list(seqs_test)
    if len(seqs_all) == 0:
        raise ValueError("no sequences available for SSL pretraining")

    raw_clip_lo, raw_clip_hi, raw_mean, raw_std = _fit_raw_scaler(
        seqs=seqs_all,
        q_low=float(args.raw_clip_quantile_low),
        q_high=float(args.raw_clip_quantile_high),
    )
    y_dummy = np.zeros((len(seqs_all), len(CLASSES)), dtype=np.float32)
    ds_ssl = TrajectoryDataset(
        seqs=seqs_all,
        y=y_dummy,
        seq_len=int(args.seq_len),
        raw_clip_lo=raw_clip_lo,
        raw_clip_hi=raw_clip_hi,
        raw_mean=raw_mean,
        raw_std=raw_std,
        augment=True,
        time_crop_p=float(args.augment_time_crop_p),
        noise_std_xyz=float(args.augment_noise_std_xyz),
        rcs_drop_p=float(args.augment_rcs_drop_p),
        speed_drop_p=float(args.augment_speed_drop_p),
        sample_weights=None,
        soft_targets=None,
        pseudo_mask=None,
    )
    dl_ssl = DataLoader(
        ds_ssl,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = TwoStreamTransformerClassifier(
        n_classes=len(CLASSES),
        seq_len=int(args.seq_len),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        num_layers=int(args.num_layers),
        ffn_dim=int(args.ffn_dim),
        dropout=float(args.dropout),
        physics_token_mode=str(args.physics_token_mode),
        n_group_classes=0,
        use_multires=bool(int(args.transformer_multires)),
        multires_pool_factor=int(args.multires_pool_factor),
        aux_out_dim=4,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        eps=1e-4 if device.type == "mps" else 1e-8,
    )
    total_steps = max(1, int(args.epochs) * max(1, len(dl_ssl)))
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
    scheduler_name = str(args.lr_scheduler).strip().lower()
    if scheduler_name == "cosine_warmup":
        warmup_ratio = float(np.clip(args.warmup_ratio, 0.0, 0.95))
        warmup_steps = int(round(total_steps * warmup_ratio))
        min_lr_ratio = float(np.clip(args.min_lr_ratio, 1e-4, 1.0))

        def _lr_lambda(step: int) -> float:
            s = int(max(0, step))
            if warmup_steps > 0 and s < warmup_steps:
                return float((s + 1) / warmup_steps)
            if total_steps <= warmup_steps:
                return float(min_lr_ratio)
            progress = float(s - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = float(np.clip(progress, 0.0, 1.0))
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    aux_mask_ratio = float(args.aux_mask_ratio) if float(args.aux_mask_ratio) > 0.0 else 0.15
    aux_mask_ratio = float(np.clip(aux_mask_ratio, 0.01, 0.95))
    aux_channel_idx_raw = torch.tensor([0, 1, 2, 4], dtype=torch.long, device=device)
    best_state: dict[str, torch.Tensor] | None = None
    best_recon = float("inf")
    best_epoch = -1
    no_improve = 0
    global_step = 0
    non_finite_total = 0
    epoch_losses: list[float] = []

    for epoch in range(int(args.epochs)):
        model.train()
        loss_sum = 0.0
        loss_cnt = 0
        bad_batches = 0
        for xb, mb, *_ in dl_ssl:
            global_step += 1
            xb = xb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            rnd = torch.rand_like(mb)
            aux_mask = ((rnd < aux_mask_ratio) & (mb > 0.0)).to(xb.dtype)  # [B, L]
            if not torch.any(aux_mask > 0.0):
                continue

            xb_in = xb.clone()
            for j in [0, 1, 2, 4]:
                xb_in[:, j, :] = xb_in[:, j, :] * (1.0 - aux_mask)
                xb_in[:, 9 + j, :] = xb_in[:, 9 + j, :] * (1.0 - aux_mask)
            aux_target = xb[:, aux_channel_idx_raw, :].transpose(1, 2).contiguous()  # [B, L, 4]

            optimizer.zero_grad(set_to_none=True)
            out = model(xb_in, mb)
            _, _, aux_pred = _unpack_model_output(out)
            if aux_pred is None:
                raise RuntimeError("ssl pretrain requires model aux head output")
            denom = torch.clamp(aux_mask.sum() * aux_target.size(-1), min=1.0)
            sq = (aux_pred - aux_target) ** 2
            loss = (sq * aux_mask.unsqueeze(-1)).sum() / denom
            if not torch.isfinite(loss):
                bad_batches += 1
                non_finite_total += 1
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            loss_sum += float(loss.detach().cpu())
            loss_cnt += 1

        epoch_loss = float(loss_sum / max(1, loss_cnt))
        epoch_losses.append(epoch_loss)
        print(
            f"[ssl] epoch={epoch} recon_loss={epoch_loss:.6f} "
            f"batches={loss_cnt} bad_batches={bad_batches}",
            flush=True,
        )

        if epoch_loss < best_recon:
            best_recon = epoch_loss
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(args.patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = out_dir / "ssl_pretrained_encoder.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": str(args.model_type),
            "seq_len": int(args.seq_len),
            "d_model": int(args.d_model),
            "n_heads": int(args.n_heads),
            "num_layers": int(args.num_layers),
            "ffn_dim": int(args.ffn_dim),
            "transformer_multires": int(args.transformer_multires),
            "multires_pool_factor": int(args.multires_pool_factor),
            "physics_token_mode": str(args.physics_token_mode),
            "best_epoch": int(best_epoch),
            "best_recon_loss": float(best_recon),
        },
        ckpt_path,
    )

    summary = {
        "mode": "ssl_pretrain",
        "output_dir": str(out_dir),
        "ssl_checkpoint": str(ckpt_path.resolve()),
        "seed": int(args.seed),
        "device": str(device),
        "requested_device": str(args.device),
        "n_sequences_train": int(len(seqs_train)),
        "n_sequences_test": int(len(seqs_test)),
        "n_sequences_total": int(len(seqs_all)),
        "seq_len": int(args.seq_len),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "lr_scheduler": str(args.lr_scheduler),
        "warmup_ratio": float(args.warmup_ratio),
        "min_lr_ratio": float(args.min_lr_ratio),
        "aux_mask_ratio_ssl": float(aux_mask_ratio),
        "best_epoch": int(best_epoch),
        "best_recon_loss": float(best_recon),
        "epoch_recon_losses": [float(x) for x in epoch_losses],
        "non_finite_batches": int(non_finite_total),
    }
    dump_json(out_dir / "oof_summary.json", summary)
    print("=== SSL PRETRAIN COMPLETE ===", flush=True)
    print(f"output_dir={out_dir}", flush=True)
    print(f"ssl_checkpoint={ckpt_path}", flush=True)
    print(f"oof_summary={out_dir / 'oof_summary.json'}", flush=True)


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    for xb, mb, *_ in loader:
        xb = xb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)
        out_model = model(xb, mb)
        logits, _, _ = _unpack_model_output(out_model)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        preds.append(probs)
    if not preds:
        return np.empty((0, len(CLASSES)), dtype=np.float32)
    return np.concatenate(preds, axis=0)


def _fit_one_fold(
    seqs_train: list[np.ndarray],
    y_train: np.ndarray,
    seqs_val: list[np.ndarray],
    y_val: np.ndarray,
    seqs_test: list[np.ndarray],
    raw_clip_lo: np.ndarray,
    raw_clip_hi: np.ndarray,
    raw_mean: np.ndarray,
    raw_std: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    sample_weights_train: np.ndarray | None = None,
    teacher_soft_train: np.ndarray | None = None,
    pseudo_seqs_train: list[np.ndarray] | None = None,
    pseudo_y_train: np.ndarray | None = None,
    pseudo_weights_train: np.ndarray | None = None,
    pseudo_soft_train: np.ndarray | None = None,
    group_matrix: np.ndarray | None = None,
    multitask_group_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    seqs_train_all = list(seqs_train)
    y_train_all = np.asarray(y_train, dtype=np.float32)
    sw_train_all = (
        np.asarray(sample_weights_train, dtype=np.float32).reshape(-1)
        if sample_weights_train is not None
        else np.ones((len(seqs_train_all),), dtype=np.float32)
    )
    pseudo_mask_all = np.zeros((len(seqs_train_all),), dtype=np.float32)
    soft_train_all = (
        np.asarray(teacher_soft_train, dtype=np.float32)
        if teacher_soft_train is not None
        else None
    )

    if pseudo_seqs_train is not None and len(pseudo_seqs_train) > 0:
        py = np.asarray(pseudo_y_train, dtype=np.float32)
        if py.ndim != 2 or py.shape[1] != len(CLASSES):
            raise ValueError(f"pseudo_y_train bad shape: {py.shape}")
        pw = (
            np.asarray(pseudo_weights_train, dtype=np.float32).reshape(-1)
            if pseudo_weights_train is not None
            else np.ones((len(pseudo_seqs_train),), dtype=np.float32)
        )
        if len(pw) != len(pseudo_seqs_train):
            raise ValueError("pseudo_weights_train length mismatch")
        seqs_train_all.extend(pseudo_seqs_train)
        y_train_all = np.concatenate([y_train_all, py], axis=0).astype(np.float32, copy=False)
        sw_train_all = np.concatenate([sw_train_all, pw], axis=0).astype(np.float32, copy=False)
        pseudo_mask_all = np.concatenate(
            [pseudo_mask_all, np.ones((len(pseudo_seqs_train),), dtype=np.float32)],
            axis=0,
        ).astype(np.float32, copy=False)
        if soft_train_all is not None:
            psoft = (
                np.asarray(pseudo_soft_train, dtype=np.float32)
                if pseudo_soft_train is not None
                else py
            )
            if psoft.shape != py.shape:
                raise ValueError(f"pseudo_soft_train shape mismatch: {psoft.shape} vs {py.shape}")
            soft_train_all = np.concatenate([soft_train_all, psoft], axis=0).astype(np.float32, copy=False)

    ds_train = TrajectoryDataset(
        seqs=seqs_train_all,
        y=y_train_all,
        seq_len=int(args.seq_len),
        raw_clip_lo=raw_clip_lo,
        raw_clip_hi=raw_clip_hi,
        raw_mean=raw_mean,
        raw_std=raw_std,
        augment=True,
        time_crop_p=float(args.augment_time_crop_p),
        noise_std_xyz=float(args.augment_noise_std_xyz),
        rcs_drop_p=float(args.augment_rcs_drop_p),
        speed_drop_p=float(args.augment_speed_drop_p),
        sample_weights=sw_train_all,
        soft_targets=soft_train_all,
        pseudo_mask=pseudo_mask_all,
    )
    ds_val = TrajectoryDataset(
        seqs=seqs_val,
        y=y_val,
        seq_len=int(args.seq_len),
        raw_clip_lo=raw_clip_lo,
        raw_clip_hi=raw_clip_hi,
        raw_mean=raw_mean,
        raw_std=raw_std,
        augment=False,
        time_crop_p=0.0,
        noise_std_xyz=0.0,
        rcs_drop_p=0.0,
        speed_drop_p=0.0,
        sample_weights=None,
        soft_targets=None,
        pseudo_mask=None,
    )
    ds_test = TrajectoryDataset(
        seqs=seqs_test,
        y=np.zeros((len(seqs_test), len(CLASSES)), dtype=np.float32),
        seq_len=int(args.seq_len),
        raw_clip_lo=raw_clip_lo,
        raw_clip_hi=raw_clip_hi,
        raw_mean=raw_mean,
        raw_std=raw_std,
        augment=False,
        time_crop_p=0.0,
        noise_std_xyz=0.0,
        rcs_drop_p=0.0,
        speed_drop_p=0.0,
        sample_weights=None,
        soft_targets=None,
        pseudo_mask=None,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    dilations = [int(x.strip()) for x in str(args.dilations).split(",") if x.strip()]
    if not dilations:
        raise ValueError("empty --dilations list")

    if str(args.model_type).lower() == "transformer":
        n_group_classes = int(group_matrix.shape[1]) if group_matrix is not None else 0
        model = TwoStreamTransformerClassifier(
            n_classes=len(CLASSES),
            seq_len=int(args.seq_len),
            d_model=int(args.d_model),
            n_heads=int(args.n_heads),
            num_layers=int(args.num_layers),
            ffn_dim=int(args.ffn_dim),
            dropout=float(args.dropout),
            physics_token_mode=str(args.physics_token_mode),
            n_group_classes=n_group_classes,
            use_multires=bool(int(args.transformer_multires)),
            multires_pool_factor=int(args.multires_pool_factor),
            aux_out_dim=4 if float(args.aux_lambda) > 0.0 and float(args.aux_mask_ratio) > 0.0 else 0,
        ).to(device)
    else:
        model = TwoStreamTCNClassifier(
            n_classes=len(CLASSES),
            stem_channels=int(args.stem_channels),
            kernel_size=int(args.kernel_size),
            dilations=dilations,
            dropout=float(args.dropout),
        ).to(device)
    if str(args.finetune_from_ssl).strip():
        _load_pretrained_encoder_weights(model=model, ckpt_path=str(args.finetune_from_ssl))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        eps=1e-4 if device.type == "mps" else 1e-8,
    )
    total_steps = max(1, int(args.epochs) * max(1, len(dl_train)))
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
    scheduler_name = str(args.lr_scheduler).strip().lower()
    if scheduler_name == "cosine_warmup":
        warmup_ratio = float(np.clip(args.warmup_ratio, 0.0, 0.95))
        warmup_steps = int(round(total_steps * warmup_ratio))
        min_lr_ratio = float(np.clip(args.min_lr_ratio, 1e-4, 1.0))

        def _lr_lambda(step: int) -> float:
            s = int(max(0, step))
            if warmup_steps > 0 and s < warmup_steps:
                return float((s + 1) / warmup_steps)
            if total_steps <= warmup_steps:
                return float(min_lr_ratio)
            progress = float(s - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = float(np.clip(progress, 0.0, 1.0))
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    if bool(args.use_pos_weight):
        pos = y_train_all.sum(axis=0).astype(np.float32)
        mode = str(args.pos_weight_mode).lower()
        if mode == "inv_sqrt_freq":
            freq = np.clip(pos / max(float(len(y_train_all)), 1.0), 1e-6, 1.0)
            pw = (1.0 / np.sqrt(freq)).astype(np.float32)
        else:
            neg = float(len(y_train_all)) - pos
            pw = ((neg + 1.0) / (pos + 1.0)).astype(np.float32)
        pos_weight = torch.from_numpy(pw.astype(np.float32)).to(device)
    else:
        pos_weight = None
    criterion_hard = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    criterion_soft = nn.BCEWithLogitsLoss(reduction="none")
    criterion_group = nn.BCEWithLogitsLoss(reduction="none")
    group_matrix_t: torch.Tensor | None = None
    if group_matrix is not None and group_matrix.size > 0 and float(multitask_group_weight) > 0.0:
        group_matrix_t = torch.from_numpy(group_matrix.astype(np.float32)).to(device)
    distill_teacher_weight = float(max(0.0, args.distill_teacher_weight))
    distill_label_weight = float(max(0.0, args.distill_label_weight))
    distill_temperature = float(max(1e-6, args.distill_temperature))
    distill_warmup_epochs = int(max(0, args.distill_warmup_epochs))
    label_smoothing = float(np.clip(args.label_smoothing, 0.0, 0.3))
    logit_reg_lambda = float(max(0.0, args.logit_reg_lambda))
    logit_reg_threshold = float(max(0.0, args.logit_reg_threshold))
    distill_class_w = torch.from_numpy(_parse_class_downweight(args.distill_class_downweight)).to(device)
    hard_loss_class_w = torch.from_numpy(_parse_hard_loss_class_weight(args.hard_loss_class_weight)).to(device)
    lambda_hard = float(args.distill_lambda_hard) if float(args.distill_lambda_hard) >= 0.0 else float(distill_label_weight)
    lambda_soft = float(args.distill_lambda_soft) if float(args.distill_lambda_soft) >= 0.0 else float(distill_teacher_weight)
    distill_enabled = teacher_soft_train is not None and lambda_soft > 0.0
    consistency_lambda = float(max(0.0, args.consistency_lambda))
    aux_mask_ratio = float(np.clip(args.aux_mask_ratio, 0.0, 0.95))
    aux_lambda = float(max(0.0, args.aux_lambda))
    aux_ramp_ratio = float(np.clip(args.aux_ramp_ratio, 0.0, 1.0))
    aux_ramp_steps = int(round(total_steps * aux_ramp_ratio))
    pseudo_ramp_ratio = float(np.clip(args.pseudo_ramp_ratio, 0.0, 1.0))
    pseudo_ramp_steps = int(round(total_steps * pseudo_ramp_ratio))
    aux_channel_idx_raw = torch.tensor([0, 1, 2, 4], dtype=torch.long, device=device)
    global_step = 0
    if distill_enabled:
        print(
            f"[distill] enabled soft_w={lambda_soft:.3f} "
            f"hard_w={lambda_hard:.3f} warmup={distill_warmup_epochs} "
            f"temp={distill_temperature:.3f}",
            flush=True,
        )

    best_state: dict[str, torch.Tensor] | None = None
    best_map = -1.0
    best_epoch = -1
    no_improve = 0
    non_finite_total = 0

    for epoch in range(int(args.epochs)):
        model.train()
        bad_batches = 0
        total_batches = 0
        for xb, mb, yb, wb, ysb, is_pseudo in dl_train:
            total_batches += 1
            global_step += 1
            xb = xb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            ysb = ysb.to(device, non_blocking=True)
            is_pseudo = is_pseudo.to(device, non_blocking=True)
            if pseudo_ramp_steps > 0 and global_step <= pseudo_ramp_steps:
                pseudo_mult = float(global_step) / float(max(1, pseudo_ramp_steps))
            else:
                pseudo_mult = 1.0
            wb_eff = wb * (1.0 - is_pseudo + is_pseudo * pseudo_mult)

            optimizer.zero_grad(set_to_none=True)
            xb_in = xb
            aux_target: torch.Tensor | None = None
            aux_mask: torch.Tensor | None = None
            if aux_lambda > 0.0 and aux_mask_ratio > 0.0:
                rnd = torch.rand_like(mb)
                aux_mask = ((rnd < aux_mask_ratio) & (mb > 0.0)).to(xb.dtype)  # [B, L]
                if torch.any(aux_mask > 0.0):
                    xb_in = xb.clone()
                    for j in [0, 1, 2, 4]:
                        xb_in[:, j, :] = xb_in[:, j, :] * (1.0 - aux_mask)
                        xb_in[:, 9 + j, :] = xb_in[:, 9 + j, :] * (1.0 - aux_mask)
                    aux_target = xb[:, aux_channel_idx_raw, :].transpose(1, 2).contiguous()  # [B, L, 4]

            out = model(xb_in, mb)
            logits, logits_group, aux_pred = _unpack_model_output(out)
            logits = logits.clamp(-20.0, 20.0)
            if not torch.isfinite(logits).all():
                bad_batches += 1
                non_finite_total += 1
                continue
            yb_h = yb
            if label_smoothing > 0.0:
                yb_h = yb * (1.0 - label_smoothing) + 0.5 * label_smoothing
            loss_hard_raw = criterion_hard(logits, yb_h)  # [B, C]
            loss_hard_raw = loss_hard_raw * hard_loss_class_w.unsqueeze(0)
            loss_hard = (loss_hard_raw.mean(dim=1) * wb_eff).mean()

            if distill_enabled:
                if epoch < distill_warmup_epochs:
                    cur_teacher_w = 1.0
                    cur_label_w = 0.0
                else:
                    cur_teacher_w = lambda_soft
                    cur_label_w = lambda_hard

                if cur_teacher_w > 0.0:
                    loss_soft_raw = criterion_soft(logits / distill_temperature, ysb)
                    loss_soft_raw = loss_soft_raw * distill_class_w.unsqueeze(0)
                    loss_soft = (loss_soft_raw.mean(dim=1) * wb_eff).mean() * (distill_temperature ** 2)
                else:
                    loss_soft = torch.zeros((), device=device, dtype=loss_hard.dtype)

                loss = cur_label_w * loss_hard + cur_teacher_w * loss_soft
            else:
                loss = loss_hard

            if (
                logits_group is not None
                and group_matrix_t is not None
                and float(multitask_group_weight) > 0.0
            ):
                y_group = torch.matmul(yb, group_matrix_t).clamp(0.0, 1.0)
                loss_group_raw = criterion_group(logits_group, y_group)
                loss_group = (loss_group_raw.mean(dim=1) * wb_eff).mean()
                loss = loss + float(multitask_group_weight) * loss_group

            if consistency_lambda > 0.0:
                out2 = model(xb, mb)
                logits2, _, _ = _unpack_model_output(out2)
                logits2 = logits2.clamp(-20.0, 20.0)
                p1 = torch.sigmoid(logits)
                p2 = torch.sigmoid(logits2)
                loss_cons = torch.mean((p1 - p2) ** 2)
                loss = loss + consistency_lambda * loss_cons

            if aux_lambda > 0.0 and aux_pred is not None and aux_target is not None and aux_mask is not None:
                denom = torch.clamp(aux_mask.sum() * aux_target.size(-1), min=1.0)
                sq = (aux_pred - aux_target) ** 2
                loss_aux = (sq * aux_mask.unsqueeze(-1)).sum() / denom
                if torch.isfinite(loss_aux):
                    if aux_ramp_steps > 0 and global_step <= aux_ramp_steps:
                        aux_lambda_eff = aux_lambda * float(global_step) / float(max(1, aux_ramp_steps))
                    else:
                        aux_lambda_eff = aux_lambda
                    loss = loss + aux_lambda_eff * loss_aux

            if logit_reg_lambda > 0.0:
                loss_reg = torch.relu(torch.abs(logits) - logit_reg_threshold).mean()
                loss = loss + logit_reg_lambda * loss_reg

            if not torch.isfinite(loss):
                bad_batches += 1
                non_finite_total += 1
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        if bad_batches > 0:
            print(
                f"[warn] non-finite batches epoch={epoch} bad={bad_batches}/{max(total_batches,1)} device={device}",
                flush=True,
            )
        if bad_batches == total_batches and total_batches > 0:
            raise RuntimeError(f"NONFINITE_ALL_BATCHES device={device}")

        val_pred = _predict(model=model, loader=dl_val, device=device)
        val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        val_map = float(macro_map_score(y_val, val_pred))
        pred_mean = float(np.mean(val_pred))
        pred_std = float(np.std(val_pred))
        pred_p95 = float(np.quantile(val_pred, 0.95))
        print(
            f"[diag] epoch={epoch} val_map={val_map:.6f} pred_mean={pred_mean:.6f} pred_std={pred_std:.6f} pred_p95={pred_p95:.6f}",
            flush=True,
        )

        if val_map > best_map:
            best_map = val_map
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(args.patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred = _predict(model=model, loader=dl_val, device=device)
    test_pred = _predict(model=model, loader=dl_test, device=device)
    val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    return val_pred, test_pred, float(best_map), int(best_epoch), int(non_finite_total)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = _resolve_device(str(args.device))

    out_dir = Path(args.out_dir).resolve()
    art_dir = out_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    sample_df = pd.read_csv(args.sample_submission, usecols=[str(args.id_col)])

    required_train_cols = {str(args.id_col), str(args.time_col), str(args.group_col), "bird_group"}
    required_test_cols = {str(args.id_col)}
    missing_train = [c for c in required_train_cols if c not in train_df.columns]
    missing_test = [c for c in required_test_cols if c not in test_df.columns]
    if missing_train:
        raise ValueError(f"train csv missing columns: {missing_train}")
    if missing_test:
        raise ValueError(f"test csv missing columns: {missing_test}")

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")

    train_ids = train_df[str(args.id_col)].to_numpy(dtype=np.int64)
    test_ids = test_df[str(args.id_col)].to_numpy(dtype=np.int64)
    np.save(out_dir / "train_track_ids.npy", train_ids)
    np.save(out_dir / "test_track_ids.npy", test_ids)

    seqs_train: list[np.ndarray] = []
    for tid in train_ids.tolist():
        item = train_cache[int(tid)]
        raw = np.asarray(item["raw_features"], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 9:
            raise ValueError(f"track_id={tid} raw_features bad shape={raw.shape}, expected [T,9]")
        seqs_train.append(_normalize_raw_sequence(raw, args.trajectory_normalization))

    seqs_test: list[np.ndarray] = []
    for tid in test_ids.tolist():
        item = test_cache[int(tid)]
        raw = np.asarray(item["raw_features"], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 9:
            raise ValueError(f"test track_id={tid} raw_features bad shape={raw.shape}, expected [T,9]")
        seqs_test.append(_normalize_raw_sequence(raw, args.trajectory_normalization))

    if int(args.pretrain_ssl) == 1:
        _pretrain_ssl(
            seqs_train=seqs_train,
            seqs_test=seqs_test,
            args=args,
            device=device,
            out_dir=out_dir,
        )
        return

    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = np.zeros((len(train_ids), len(CLASSES)), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    np.save(out_dir / "oof_targets.npy", y.astype(np.float32))
    group_matrix, group_names = _build_group_matrix(args.multitask_group_scheme)
    if str(args.model_type).lower() != "transformer":
        group_matrix = np.zeros((len(CLASSES), 0), dtype=np.float32)
        group_names = []

    teacher_probs_train: np.ndarray | None = None
    teacher_ensemble_used = False
    if (
        str(args.teacher_ensemble_fr_oof_csv).strip()
        and str(args.teacher_ensemble_tcn_oof_npy).strip()
        and str(args.teacher_ensemble_tcn_track_ids_npy).strip()
        and str(args.teacher_ensemble_trf_oof_npy).strip()
        and str(args.teacher_ensemble_trf_track_ids_npy).strip()
    ):
        p_fr = _align_teacher_probs(str(args.teacher_ensemble_fr_oof_csv), train_ids, id_col=str(args.id_col))
        p_tcn = _align_probs_from_npy(
            str(args.teacher_ensemble_tcn_oof_npy),
            str(args.teacher_ensemble_tcn_track_ids_npy),
            train_ids,
        )
        p_trf = _align_probs_from_npy(
            str(args.teacher_ensemble_trf_oof_npy),
            str(args.teacher_ensemble_trf_track_ids_npy),
            train_ids,
        )
        te_w = _parse_teacher_ensemble_weights(args.teacher_ensemble_weights)
        teacher_probs_train = _build_teacher_ensemble_probs(
            p_fr=p_fr,
            p_tcn=p_tcn,
            p_trf=p_trf,
            weights=te_w,
            teacher_temp=float(args.teacher_temp),
            logit_clip=float(args.teacher_logit_clip),
        )
        np.save(out_dir / "teacher_ensemble_oof.npy", teacher_probs_train.astype(np.float32))
        teacher_ensemble_used = True
    elif str(args.teacher_oof_csv).strip():
        teacher_probs_train = _align_teacher_probs(str(args.teacher_oof_csv), train_ids, id_col=str(args.id_col))

    distill_soft_global = (
        float(args.distill_lambda_soft)
        if float(args.distill_lambda_soft) >= 0.0
        else float(args.distill_teacher_weight)
    )
    if distill_soft_global > 0.0 and teacher_probs_train is None:
        raise ValueError("distillation soft loss > 0 requires teacher soft targets (--teacher-oof-csv or teacher ensemble args)")

    sample_weights: np.ndarray | None = None
    if str(args.sample_weights_npy).strip():
        sw = np.load(str(args.sample_weights_npy))
        sw = np.asarray(sw, dtype=np.float32).reshape(-1)
        if len(sw) != len(train_ids):
            raise ValueError(
                f"sample_weights length mismatch: got {len(sw)}, expected {len(train_ids)}"
            )
        sample_weights = sw

    pseudo_seqs: list[np.ndarray] = []
    pseudo_y: np.ndarray | None = None
    pseudo_w: np.ndarray | None = None
    pseudo_soft: np.ndarray | None = None
    if str(args.pseudo_label_csv).strip():
        pseudo_seqs, pseudo_y, pseudo_w, pseudo_soft = _load_pseudo_labels(
            pseudo_csv=str(args.pseudo_label_csv),
            test_ids=test_ids,
            seqs_test=seqs_test,
            min_prob=float(args.pseudo_min_prob),
            base_weight=float(args.pseudo_weight),
            prob_power=float(args.pseudo_prob_power),
            id_col=str(args.id_col),
        )
        print(
            f"[pseudo] loaded n={len(pseudo_seqs)} min_prob={float(args.pseudo_min_prob):.3f} "
            f"w_mean={float(np.mean(pseudo_w)) if pseudo_w is not None and len(pseudo_w)>0 else 0.0:.4f}",
            flush=True,
        )

    cv_df = pd.DataFrame(
        {
            "_cv_ts": train_df[str(args.time_col)],
            "_cv_group": train_df[str(args.group_col)].astype(np.int64),
        }
    )
    cv_direction = str(args.cv_direction).strip().lower()
    if cv_direction == "reverse":
        ts = pd.to_datetime(cv_df["_cv_ts"], errors="coerce", utc=True)
        if ts.isna().any():
            bad = int(ts.isna().sum())
            raise ValueError(f"timestamp parsing failed for {bad} rows in reverse cv")
        # Reverse temporal order while preserving relative spacing.
        ts_ns = ts.astype("int64")
        rev_ns = int(ts_ns.max()) + int(ts_ns.min()) - ts_ns
        cv_df["_cv_ts"] = pd.to_datetime(rev_ns, utc=True)
    folds = make_forward_temporal_group_folds(
        cv_df,
        timestamp_col="_cv_ts",
        group_col="_cv_group",
        n_splits=int(args.n_splits),
    )
    if len(folds) == 0:
        raise RuntimeError("no folds created")

    oof = np.zeros((len(train_ids), len(CLASSES)), dtype=np.float32)
    covered_mask = np.zeros((len(train_ids),), dtype=bool)
    test_accum = np.zeros((len(test_ids), len(CLASSES)), dtype=np.float32)
    fold_scores: list[float] = []
    fold_best_maps: list[float] = []
    fold_best_epochs: list[int] = []
    fold_non_finite_batches: list[int] = []

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        print(f"[fold] {fold_id + 1}/{len(folds)} train={len(tr_idx)} val={len(va_idx)}", flush=True)
        seq_tr = [seqs_train[int(i)] for i in tr_idx]
        seq_va = [seqs_train[int(i)] for i in va_idx]
        seq_scaler = seq_tr + (pseudo_seqs if len(pseudo_seqs) > 0 else [])
        raw_clip_lo, raw_clip_hi, raw_mean, raw_std = _fit_raw_scaler(
            seqs=seq_scaler,
            q_low=float(args.raw_clip_quantile_low),
            q_high=float(args.raw_clip_quantile_high),
        )
        try:
            val_pred, test_pred, best_map, best_epoch, non_finite_batches_fold = _fit_one_fold(
                seqs_train=seq_tr,
                y_train=y[tr_idx],
                seqs_val=seq_va,
                y_val=y[va_idx],
                seqs_test=seqs_test,
                raw_clip_lo=raw_clip_lo,
                raw_clip_hi=raw_clip_hi,
                raw_mean=raw_mean,
                raw_std=raw_std,
                args=args,
                device=device,
                sample_weights_train=(sample_weights[tr_idx] if sample_weights is not None else None),
                teacher_soft_train=(teacher_probs_train[tr_idx] if teacher_probs_train is not None else None),
                pseudo_seqs_train=pseudo_seqs,
                pseudo_y_train=pseudo_y,
                pseudo_weights_train=pseudo_w,
                pseudo_soft_train=pseudo_soft,
                group_matrix=group_matrix,
                multitask_group_weight=float(args.multitask_group_weight),
            )
        except RuntimeError as exc:
            if device.type == "mps" and "NONFINITE" in str(exc):
                print(
                    f"[warn] fold={fold_id + 1} encountered non-finite training on MPS, retrying on CPU",
                    flush=True,
                )
                val_pred, test_pred, best_map, best_epoch, non_finite_batches_fold = _fit_one_fold(
                    seqs_train=seq_tr,
                    y_train=y[tr_idx],
                    seqs_val=seq_va,
                    y_val=y[va_idx],
                    seqs_test=seqs_test,
                    raw_clip_lo=raw_clip_lo,
                    raw_clip_hi=raw_clip_hi,
                    raw_mean=raw_mean,
                    raw_std=raw_std,
                    args=args,
                    device=torch.device("cpu"),
                    sample_weights_train=(sample_weights[tr_idx] if sample_weights is not None else None),
                    teacher_soft_train=(teacher_probs_train[tr_idx] if teacher_probs_train is not None else None),
                    pseudo_seqs_train=pseudo_seqs,
                    pseudo_y_train=pseudo_y,
                    pseudo_weights_train=pseudo_w,
                    pseudo_soft_train=pseudo_soft,
                    group_matrix=group_matrix,
                    multitask_group_weight=float(args.multitask_group_weight),
                )
            else:
                raise

        oof[va_idx] = val_pred
        covered_mask[va_idx] = True
        test_accum += test_pred / float(len(folds))
        fold_score = float(macro_map_score(y[va_idx], val_pred))
        fold_scores.append(fold_score)
        fold_best_maps.append(float(best_map))
        fold_best_epochs.append(int(best_epoch))
        fold_non_finite_batches.append(int(non_finite_batches_fold))

    art_prefix = "transformer" if str(args.model_type).lower() == "transformer" else "tcn"
    oof_path = art_dir / f"{art_prefix}_oof.npy"
    test_path = art_dir / f"{art_prefix}_test.npy"
    np.save(oof_path, oof.astype(np.float32))
    np.save(test_path, test_accum.astype(np.float32))

    covered_idx = np.where(covered_mask)[0].astype(np.int64)
    uncovered_idx = np.where(~covered_mask)[0].astype(np.int64)
    np.save(out_dir / "oof_covered_idx.npy", covered_idx)
    if len(covered_idx) == 0:
        raise RuntimeError("no covered validation rows produced by forward CV")

    macro_raw_full = float(macro_map_score(y, oof))
    macro_covered = float(macro_map_score(y[covered_idx], oof[covered_idx]))
    per_class_covered = per_class_average_precision(y[covered_idx], oof[covered_idx])

    teacher_diag: dict[str, float] = {}
    oof_full_filled = oof.copy()
    macro_full_filled_teacher = 0.0
    if teacher_probs_train is not None:
        teacher = teacher_probs_train
        oof_full_filled[uncovered_idx] = teacher[uncovered_idx]
        macro_full_filled_teacher = float(macro_map_score(y, oof_full_filled))
        teacher_diag = _teacher_diag(
            y_true=y[covered_idx],
            teacher=teacher[covered_idx],
            model=oof[covered_idx],
        )
        _save_per_class_delta_vs_teacher(
            y_true=y[covered_idx],
            teacher_prob=teacher[covered_idx],
            model_prob=oof[covered_idx],
            out_csv=out_dir / "diagnostics" / "per_class_ap_delta_vs_teacher.csv",
        )
    np.save(out_dir / "oof_forward_cv_complete.npy", oof_full_filled.astype(np.float32))

    sample_ids = sample_df[str(args.id_col)].to_numpy(dtype=np.int64)
    if not np.array_equal(sample_ids, test_ids):
        pos = {int(tid): i for i, tid in enumerate(test_ids.tolist())}
        miss = [int(tid) for tid in sample_ids.tolist() if int(tid) not in pos]
        if miss:
            raise ValueError(f"sample_submission ids missing in test predictions; first={miss[:10]}")
        take = np.array([pos[int(tid)] for tid in sample_ids.tolist()], dtype=np.int64)
        pred_sub = test_accum[take]
    else:
        pred_sub = test_accum

    sub = pd.DataFrame({str(args.id_col): sample_ids})
    for i, cls in enumerate(CLASSES):
        sub[cls] = np.clip(pred_sub[:, i], 0.0, 1.0).astype(np.float32)
    sub_path = out_dir / str(args.out_name)
    sub.to_csv(sub_path, index=False)

    summary: dict[str, Any] = {
        "output_dir": str(out_dir),
        "seed": int(args.seed),
        "device": str(device),
        "requested_device": str(args.device),
        "n_splits": int(len(folds)),
        "cv_direction": str(args.cv_direction),
        "seq_len": int(args.seq_len),
        "pretrain_ssl": int(args.pretrain_ssl),
        "finetune_from_ssl": str(args.finetune_from_ssl),
        "model_type": str(args.model_type),
        "stem_channels": int(args.stem_channels),
        "kernel_size": int(args.kernel_size),
        "dilations": [int(x.strip()) for x in str(args.dilations).split(",") if x.strip()],
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "num_layers": int(args.num_layers),
        "ffn_dim": int(args.ffn_dim),
        "transformer_multires": bool(int(args.transformer_multires)),
        "multires_pool_factor": int(args.multires_pool_factor),
        "physics_token_mode": str(args.physics_token_mode),
        "multitask_group_scheme": str(args.multitask_group_scheme),
        "multitask_group_weight": float(args.multitask_group_weight),
        "multitask_group_names": [str(x) for x in group_names],
        "dropout": float(args.dropout),
        "label_smoothing": float(args.label_smoothing),
        "logit_reg_lambda": float(args.logit_reg_lambda),
        "logit_reg_threshold": float(args.logit_reg_threshold),
        "pos_weight_mode": str(args.pos_weight_mode),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "lr_scheduler": str(args.lr_scheduler),
        "warmup_ratio": float(args.warmup_ratio),
        "min_lr_ratio": float(args.min_lr_ratio),
        "grad_clip": float(args.grad_clip),
        "sample_weights_npy": str(args.sample_weights_npy),
        "pseudo_label_csv": str(args.pseudo_label_csv),
        "pseudo_weight": float(args.pseudo_weight),
        "pseudo_min_prob": float(args.pseudo_min_prob),
        "pseudo_prob_power": float(args.pseudo_prob_power),
        "pseudo_ramp_ratio": float(args.pseudo_ramp_ratio),
        "hard_loss_class_weight": str(args.hard_loss_class_weight),
        "distill_teacher_weight": float(args.distill_teacher_weight),
        "distill_label_weight": float(args.distill_label_weight),
        "distill_lambda_hard": float(args.distill_lambda_hard),
        "distill_lambda_soft": float(args.distill_lambda_soft),
        "distill_warmup_epochs": int(args.distill_warmup_epochs),
        "distill_temperature": float(args.distill_temperature),
        "distill_class_downweight": str(args.distill_class_downweight),
        "consistency_lambda": float(args.consistency_lambda),
        "aux_mask_ratio": float(args.aux_mask_ratio),
        "aux_lambda": float(args.aux_lambda),
        "aux_ramp_ratio": float(args.aux_ramp_ratio),
        "teacher_ensemble_used": bool(teacher_ensemble_used),
        "teacher_ensemble_fr_oof_csv": str(args.teacher_ensemble_fr_oof_csv),
        "teacher_ensemble_tcn_oof_npy": str(args.teacher_ensemble_tcn_oof_npy),
        "teacher_ensemble_tcn_track_ids_npy": str(args.teacher_ensemble_tcn_track_ids_npy),
        "teacher_ensemble_trf_oof_npy": str(args.teacher_ensemble_trf_oof_npy),
        "teacher_ensemble_trf_track_ids_npy": str(args.teacher_ensemble_trf_track_ids_npy),
        "teacher_ensemble_weights": str(args.teacher_ensemble_weights),
        "teacher_temp": float(args.teacher_temp),
        "teacher_logit_clip": float(args.teacher_logit_clip),
        "patience": int(args.patience),
        "n_train": int(len(train_ids)),
        "n_test": int(len(test_ids)),
        "n_pseudo_train": int(len(pseudo_seqs)),
        "macro_map_raw_full": float(macro_raw_full),
        "macro_map_covered": float(macro_covered),
        "macro_map_full_filled_teacher": float(macro_full_filled_teacher),
        "covered_ratio": float(len(covered_idx) / len(train_ids)),
        "n_covered": int(len(covered_idx)),
        "n_uncovered": int(len(uncovered_idx)),
        "per_class_ap_covered": {k: float(v) for k, v in per_class_covered.items()},
        "fold_scores": [float(v) for v in fold_scores],
        "fold_best_maps": [float(v) for v in fold_best_maps],
        "fold_best_epochs": [int(v) for v in fold_best_epochs],
        "fold_non_finite_batches": [int(v) for v in fold_non_finite_batches],
        "non_finite_batches": int(np.sum(fold_non_finite_batches)),
        "fold_mean": float(np.mean(fold_scores) if fold_scores else 0.0),
        "fold_worst": float(np.min(fold_scores) if fold_scores else 0.0),
        "fold_best": float(np.max(fold_scores) if fold_scores else 0.0),
        "teacher_diag": teacher_diag,
        "models": {
            f"{art_prefix}_trajectory": {
                "type": f"{art_prefix}_trajectory",
                "oof_path": str(oof_path.resolve()),
                "test_path": str(test_path.resolve()),
                "macro_map_raw_full": float(macro_raw_full),
                "macro_map_covered": float(macro_covered),
                "macro_map_full_filled_teacher": float(macro_full_filled_teacher),
                "fold_scores": [float(v) for v in fold_scores],
            }
        },
        "track_ids_train_path": str((out_dir / "train_track_ids.npy").resolve()),
        "track_ids_test_path": str((out_dir / "test_track_ids.npy").resolve()),
        "submission_path": str(sub_path.resolve()),
    }
    if sample_weights is not None:
        summary["sample_weights_stats"] = {
            "min": float(np.min(sample_weights)),
            "max": float(np.max(sample_weights)),
            "mean": float(np.mean(sample_weights)),
            "p95": float(np.quantile(sample_weights, 0.95)),
        }
    if pseudo_w is not None and len(pseudo_w) > 0:
        summary["pseudo_weights_stats"] = {
            "min": float(np.min(pseudo_w)),
            "max": float(np.max(pseudo_w)),
            "mean": float(np.mean(pseudo_w)),
            "p95": float(np.quantile(pseudo_w, 0.95)),
        }
    dump_json(out_dir / "oof_summary.json", summary)

    print(f"=== {art_prefix.upper()} TRAJECTORY TRAIN COMPLETE ===", flush=True)
    print(f"output_dir={out_dir}", flush=True)
    print(f"oof_summary={out_dir / 'oof_summary.json'}", flush=True)
    print(
        f"macro_covered={summary['macro_map_covered']:.6f} "
        f"full_filled={summary['macro_map_full_filled_teacher']:.6f} "
        f"fold_mean={summary['fold_mean']:.6f} fold_worst={summary['fold_worst']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
