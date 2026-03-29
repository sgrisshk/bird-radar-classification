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
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds
from src.metrics import macro_map_score, per_class_average_precision
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train spectrogram-CNN on trajectory signals with forward CV.")
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--test-csv", type=str, required=True)
    p.add_argument("--sample-submission", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--cache-dir", type=str, default="bird_radar/artifacts/cache")

    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-scheduler", type=str, default="cosine_warmup", choices=["none", "cosine_warmup"])
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["mps", "cuda", "cpu"])
    p.add_argument("--sample-weights-npy", type=str, default="")

    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--spec-channels", type=str, default="speed,turn_rate,altitude")
    p.add_argument("--stft-n-fft", type=int, default=64)
    p.add_argument("--stft-hop-length", type=int, default=16)
    p.add_argument("--stft-win-length", type=int, default=64)
    p.add_argument("--spec-log1p-eps", type=float, default=1e-6)
    p.add_argument("--arch", type=str, default="tinyconvnext", choices=["tinyconvnext"])
    p.add_argument("--dropout", type=float, default=0.15)

    p.add_argument("--teacher-oof-csv", type=str, default="")
    p.add_argument("--anchor-submission-csv", type=str, default="")
    p.add_argument("--id-col", type=str, default="track_id")
    p.add_argument("--time-col", type=str, default="timestamp_start_radar_utc")
    p.add_argument("--group-col", type=str, default="observation_id")
    return p.parse_args()


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
    return c if np.isfinite(c) else 0.0


def _safe_spearman_flat(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size == 0:
        return 0.0
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return _safe_corr_flat(xr, yr)


def _macro_map_weighted(y_true: np.ndarray, y_prob: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    vals: list[float] = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        if yt.sum() <= 0:
            vals.append(0.0)
        else:
            vals.append(float(average_precision_score(yt, yp, sample_weight=sample_weight)))
    return float(np.mean(vals))


def _teacher_diag_simple(y_true: np.ndarray, teacher: np.ndarray, model: np.ndarray) -> dict[str, float]:
    teacher_macro = float(macro_map_score(y_true, teacher))
    model_macro = float(macro_map_score(y_true, model))
    best = teacher_macro
    best_w = 1.0
    for w in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
        blend = np.clip(w * teacher + (1.0 - w) * model, 0.0, 1.0)
        m = float(macro_map_score(y_true, blend))
        if m > best:
            best = m
            best_w = float(w)
    return {
        "teacher_macro": teacher_macro,
        "model_macro": model_macro,
        "best_blend_macro": best,
        "best_blend_w_teacher": best_w,
        "best_blend_gain": float(best - teacher_macro),
        "corr_with_teacher": float(_safe_corr_flat(teacher, model)),
    }


def _teacher_blend_diag_by_folds(
    y: np.ndarray,
    teacher: np.ndarray,
    model: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    covered_mask: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    w_grid = np.asarray([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60], dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    all_blend_scores: list[list[float]] = []
    all_fold_weights: list[float] = []

    for fold_id, (_, va_idx) in enumerate(folds):
        idx = np.asarray(va_idx, dtype=np.int64)
        idx = idx[covered_mask[idx]]
        if len(idx) == 0:
            continue
        yt = y[idx]
        te = teacher[idx]
        md = model[idx]
        sw = sample_weights[idx] if sample_weights is not None else None

        teacher_m = _macro_map_weighted(yt, te, sample_weight=sw)
        model_m = _macro_map_weighted(yt, md, sample_weight=sw)
        corr = _safe_corr_flat(te, md)

        b_scores: list[float] = []
        best_blend = -1.0
        best_w = 1.0
        for w in w_grid.tolist():
            p = np.clip(w * te + (1.0 - w) * md, 0.0, 1.0)
            m = _macro_map_weighted(yt, p, sample_weight=sw)
            b_scores.append(float(m))
            if m > best_blend:
                best_blend = float(m)
                best_w = float(w)
        delta = float(best_blend - teacher_m)
        fold_weight = float(np.sum(sw)) if sw is not None else float(len(idx))
        all_fold_weights.append(fold_weight)
        all_blend_scores.append(b_scores)
        fold_rows.append(
            {
                "fold": int(fold_id),
                "n_rows": int(len(idx)),
                "teacher_macro": float(teacher_m),
                "model_macro": float(model_m),
                "best_blend_macro": float(best_blend),
                "best_w_teacher": float(best_w),
                "delta_best_minus_teacher": float(delta),
                "corr": float(corr),
                "fold_weight": float(fold_weight),
            }
        )

    if not fold_rows:
        return {
            "sample_weighted_metrics": bool(sample_weights is not None),
            "weights_grid": [float(w) for w in w_grid.tolist()],
            "teacher_mean": 0.0,
            "model_mean": 0.0,
            "best_blend_mean": 0.0,
            "best_blend_gain": 0.0,
            "best_w_teacher": 1.0,
            "corr_mean": 0.0,
            "min_fold_delta": 0.0,
            "folds": [],
        }

    fw = np.asarray(all_fold_weights, dtype=np.float64)
    fw = np.where(fw <= 0.0, 1.0, fw)
    t_mean = float(np.average([r["teacher_macro"] for r in fold_rows], weights=fw))
    m_mean = float(np.average([r["model_macro"] for r in fold_rows], weights=fw))
    c_mean = float(np.average([r["corr"] for r in fold_rows], weights=fw))
    min_delta = float(np.min([r["delta_best_minus_teacher"] for r in fold_rows]))
    blend_mat = np.asarray(all_blend_scores, dtype=np.float64)
    w_means = np.average(blend_mat, axis=0, weights=fw)
    best_i = int(np.argmax(w_means))
    best_blend_mean = float(w_means[best_i])
    best_w_teacher = float(w_grid[best_i])
    return {
        "sample_weighted_metrics": bool(sample_weights is not None),
        "weights_grid": [float(w) for w in w_grid.tolist()],
        "teacher_mean": float(t_mean),
        "model_mean": float(m_mean),
        "best_blend_mean": float(best_blend_mean),
        "best_blend_gain": float(best_blend_mean - t_mean),
        "best_w_teacher": float(best_w_teacher),
        "corr_mean": float(c_mean),
        "min_fold_delta": float(min_delta),
        "folds": fold_rows,
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
        rows.append({"class": c, "ap_teacher": at, "ap_model": am, "delta_model_vs_teacher": float(am - at)})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def _normalize01(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v.astype(np.float32)
    lo = float(np.min(v))
    hi = float(np.max(v))
    span = hi - lo
    if span < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return ((v - lo) / span).astype(np.float32)


def _parse_spec_channels(raw: str) -> list[str]:
    allowed = {"speed", "turn_rate", "altitude", "vertical_speed", "accel", "curvature", "rcs", "dt"}
    parts = [p.strip().lower() for p in str(raw).split(",") if p.strip()]
    if not parts:
        raise ValueError("spec-channels resolved to empty list")
    bad = [p for p in parts if p not in allowed]
    if bad:
        raise ValueError(f"unknown spec channels: {bad}; allowed={sorted(allowed)}")
    return parts


def _channels_for_spec(spec_channels: list[str]) -> int:
    return int(len(spec_channels))


def _estimate_turn_rate(x: np.ndarray, y: np.ndarray, dt: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    dx = np.diff(x, prepend=x[:1]).astype(np.float32)
    dy = np.diff(y, prepend=y[:1]).astype(np.float32)
    heading = np.arctan2(dy, dx).astype(np.float32)
    dh = np.diff(heading, prepend=heading[:1]).astype(np.float32)
    dh = np.arctan2(np.sin(dh), np.cos(dh)).astype(np.float32)
    dt_safe = np.clip(dt.astype(np.float32), 1e-3, None)
    return (dh / dt_safe).astype(np.float32)


def _robust_scale_signal(sig: np.ndarray) -> np.ndarray:
    x = np.asarray(sig, dtype=np.float32)
    if x.size == 0:
        return x
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    med = float(np.median(x))
    z = x - med
    s = float(np.quantile(np.abs(z), 0.90))
    if s < 1e-6:
        s = float(np.std(z))
    if s < 1e-6:
        s = 1.0
    z = np.clip(z / s, -8.0, 8.0)
    return z.astype(np.float32, copy=False)


def _extract_signal(
    seq: np.ndarray,
    chan: str,
    turn_rate: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((1,), dtype=np.float32)
    c = str(chan).lower()
    if c == "speed":
        return arr[:, 4].astype(np.float32, copy=False)
    if c == "turn_rate":
        return turn_rate.astype(np.float32, copy=False)
    if c == "altitude":
        return arr[:, 2].astype(np.float32, copy=False)
    if c == "vertical_speed":
        return arr[:, 5].astype(np.float32, copy=False) if arr.shape[1] > 5 else np.zeros((arr.shape[0],), dtype=np.float32)
    if c == "accel":
        return arr[:, 6].astype(np.float32, copy=False) if arr.shape[1] > 6 else np.zeros((arr.shape[0],), dtype=np.float32)
    if c == "curvature":
        return arr[:, 7].astype(np.float32, copy=False) if arr.shape[1] > 7 else np.zeros((arr.shape[0],), dtype=np.float32)
    if c == "rcs":
        return arr[:, 3].astype(np.float32, copy=False) if arr.shape[1] > 3 else np.zeros((arr.shape[0],), dtype=np.float32)
    if c == "dt":
        return arr[:, 8].astype(np.float32, copy=False) if arr.shape[1] > 8 else np.ones((arr.shape[0],), dtype=np.float32)
    return np.zeros((arr.shape[0],), dtype=np.float32)


def _spectrogram_track(
    seq: np.ndarray,
    image_size: int,
    spec_channels: list[str],
    stft_n_fft: int,
    stft_hop_length: int,
    stft_win_length: int,
    spec_log1p_eps: float,
    window: torch.Tensor,
) -> np.ndarray:
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((_channels_for_spec(spec_channels), image_size, image_size), dtype=np.float32)
    x = arr[:, 0]
    y = arr[:, 1]
    dt = arr[:, 8] if arr.shape[1] > 8 else np.ones((arr.shape[0],), dtype=np.float32)
    turn_rate = _estimate_turn_rate(x=x, y=y, dt=dt)

    spec_maps: list[torch.Tensor] = []
    for chan in spec_channels:
        sig = _extract_signal(arr, chan=chan, turn_rate=turn_rate)
        sig = _robust_scale_signal(sig)
        sig_t = torch.as_tensor(sig, dtype=torch.float32)
        if sig_t.numel() < 2:
            sp = torch.zeros((int(stft_n_fft // 2 + 1), 1), dtype=torch.float32)
        else:
            if sig_t.numel() < int(stft_win_length):
                pad = int(stft_win_length - sig_t.numel())
                sig_t = F.pad(sig_t, (0, pad))
            sp_c = torch.stft(
                sig_t,
                n_fft=int(stft_n_fft),
                hop_length=int(stft_hop_length),
                win_length=int(stft_win_length),
                window=window,
                center=False,
                return_complex=True,
            )
            sp = torch.abs(sp_c).to(dtype=torch.float32)
            sp = torch.log1p(torch.clamp(sp, min=float(spec_log1p_eps)))
            if sp.numel() == 0:
                sp = torch.zeros((int(stft_n_fft // 2 + 1), 1), dtype=torch.float32)
        sp_np = _normalize01(sp.cpu().numpy().astype(np.float32, copy=False))
        spec_maps.append(torch.from_numpy(sp_np))

    spec = torch.stack(spec_maps, dim=0).unsqueeze(0)
    spec = F.interpolate(spec, size=(int(image_size), int(image_size)), mode="bilinear", align_corners=False)
    return spec.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def _build_spectrogram_images(
    seqs: list[np.ndarray],
    image_size: int,
    spec_channels: list[str],
    stft_n_fft: int,
    stft_hop_length: int,
    stft_win_length: int,
    spec_log1p_eps: float,
) -> np.ndarray:
    ch = _channels_for_spec(spec_channels)
    imgs = np.zeros((len(seqs), ch, image_size, image_size), dtype=np.float16)
    window = torch.hann_window(int(stft_win_length), periodic=True, dtype=torch.float32)
    for i, seq in enumerate(seqs):
        imgs[i] = _spectrogram_track(
            seq,
            image_size=image_size,
            spec_channels=spec_channels,
            stft_n_fft=int(stft_n_fft),
            stft_hop_length=int(stft_hop_length),
            stft_win_length=int(stft_win_length),
            spec_log1p_eps=float(spec_log1p_eps),
            window=window,
        ).astype(np.float16)
    return imgs


class PathImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, sample_weights: np.ndarray | None = None) -> None:
        self.images = images
        self.labels = labels.astype(np.float32, copy=False)
        if sample_weights is None:
            self.sample_weights = np.ones((len(images),), dtype=np.float32)
        else:
            self.sample_weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.images[idx].astype(np.float32, copy=False))
        y = torch.from_numpy(self.labels[idx])
        w = torch.tensor(float(self.sample_weights[idx]), dtype=torch.float32)
        return x, y, w


class TinyConvBlock(nn.Module):
    def __init__(self, channels: int, drop: float) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.bn = nn.BatchNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, channels * 4, kernel_size=1)
        self.pw2 = nn.Conv2d(channels * 4, channels, kernel_size=1)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dw(x)
        z = self.bn(z)
        z = F.gelu(z)
        z = self.pw1(z)
        z = F.gelu(z)
        z = self.pw2(z)
        z = self.drop(z)
        return x + z


class TinyConvNeXtClassifier(nn.Module):
    def __init__(self, n_classes: int, dropout: float, in_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(int(in_channels), 32, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(TinyConvBlock(32, drop=dropout), TinyConvBlock(32, drop=dropout))
        self.down1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=2, stride=2), nn.BatchNorm2d(64), nn.GELU())
        self.stage2 = nn.Sequential(TinyConvBlock(64, drop=dropout), TinyConvBlock(64, drop=dropout))
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=2, stride=2), nn.BatchNorm2d(128), nn.GELU())
        self.stage3 = nn.Sequential(TinyConvBlock(128, drop=dropout), TinyConvBlock(128, drop=dropout))
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.stage1(z)
        z = self.down1(z)
        z = self.stage2(z)
        z = self.down2(z)
        z = self.stage3(z)
        z = F.adaptive_avg_pool2d(z, output_size=1).flatten(1)
        return self.head(z)


def _make_model(arch: str, n_classes: int, dropout: float, in_channels: int) -> nn.Module:
    _ = str(arch).lower()
    return TinyConvNeXtClassifier(n_classes=n_classes, dropout=dropout, in_channels=in_channels)


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    for xb, *_ in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).clamp(-20.0, 20.0)
        p = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        preds.append(p)
    if not preds:
        return np.empty((0, len(CLASSES)), dtype=np.float32)
    return np.concatenate(preds, axis=0)


def _fit_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    sample_weights_train: np.ndarray | None,
    in_channels: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    ds_train = PathImageDataset(x_train, y_train, sample_weights=sample_weights_train)
    ds_val = PathImageDataset(x_val, y_val, sample_weights=None)
    ds_test = PathImageDataset(x_test, np.zeros((len(x_test), len(CLASSES)), dtype=np.float32), sample_weights=None)

    dl_train = DataLoader(ds_train, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers), pin_memory=(device.type == "cuda"))
    dl_val = DataLoader(ds_val, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=(device.type == "cuda"))
    dl_test = DataLoader(ds_test, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=(device.type == "cuda"))

    model = _make_model(args.arch, n_classes=len(CLASSES), dropout=float(args.dropout), in_channels=int(in_channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    total_steps = max(1, int(args.epochs) * max(1, len(dl_train)))
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
    if str(args.lr_scheduler).lower() == "cosine_warmup":
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

    pos = y_train.sum(axis=0).astype(np.float32)
    neg = float(len(y_train)) - pos
    pos_weight = torch.from_numpy(((neg + 1.0) / (pos + 1.0)).astype(np.float32)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    best_state: dict[str, torch.Tensor] | None = None
    best_map = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(int(args.epochs)):
        model.train()
        bad_batches = 0
        for xb, yb, wb in dl_train:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb).clamp(-20.0, 20.0)
            if not torch.isfinite(logits).all():
                bad_batches += 1
                continue
            loss_raw = criterion(logits, yb).mean(dim=1)
            loss = (loss_raw * wb).mean()
            if not torch.isfinite(loss):
                bad_batches += 1
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        if bad_batches > 0:
            print(f"[warn] non-finite batches epoch={epoch} bad={bad_batches}", flush=True)

        val_pred = _predict(model, dl_val, device=device)
        val_map = float(macro_map_score(y_val, val_pred))
        print(
            f"[diag] epoch={epoch} val_map={val_map:.6f} pred_mean={float(np.mean(val_pred)):.6f} "
            f"pred_std={float(np.std(val_pred)):.6f} pred_p95={float(np.quantile(val_pred, 0.95)):.6f}",
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

    val_pred = _predict(model, dl_val, device=device)
    test_pred = _predict(model, dl_test, device=device)
    return val_pred, test_pred, float(best_map), int(best_epoch)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = _resolve_device(str(args.device))

    out_dir = Path(args.out_dir).resolve()
    art_dir = out_dir / "artifacts"
    diag_dir = out_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

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

    seqs_train = [np.asarray(train_cache[int(tid)]["raw_features"], dtype=np.float32) for tid in train_ids.tolist()]
    seqs_test = [np.asarray(test_cache[int(tid)]["raw_features"], dtype=np.float32) for tid in test_ids.tolist()]
    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = np.zeros((len(train_ids), len(CLASSES)), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    np.save(out_dir / "oof_targets.npy", y.astype(np.float32))

    spec_channels = _parse_spec_channels(str(args.spec_channels))
    print(
        f"[spec] building images n_train={len(seqs_train)} n_test={len(seqs_test)} "
        f"size={int(args.image_size)} channels={','.join(spec_channels)} "
        f"stft(n_fft={int(args.stft_n_fft)},hop={int(args.stft_hop_length)},win={int(args.stft_win_length)})",
        flush=True,
    )
    img_train = _build_spectrogram_images(
        seqs_train,
        image_size=int(args.image_size),
        spec_channels=spec_channels,
        stft_n_fft=int(args.stft_n_fft),
        stft_hop_length=int(args.stft_hop_length),
        stft_win_length=int(args.stft_win_length),
        spec_log1p_eps=float(args.spec_log1p_eps),
    )
    img_test = _build_spectrogram_images(
        seqs_test,
        image_size=int(args.image_size),
        spec_channels=spec_channels,
        stft_n_fft=int(args.stft_n_fft),
        stft_hop_length=int(args.stft_hop_length),
        stft_win_length=int(args.stft_win_length),
        spec_log1p_eps=float(args.spec_log1p_eps),
    )
    in_channels = int(img_train.shape[1]) if img_train.ndim == 4 else int(_channels_for_spec(spec_channels))

    sample_weights: np.ndarray | None = None
    if str(args.sample_weights_npy).strip():
        sw = np.asarray(np.load(str(args.sample_weights_npy)), dtype=np.float32).reshape(-1)
        if len(sw) != len(train_ids):
            raise ValueError(f"sample_weights length mismatch: got {len(sw)}, expected {len(train_ids)}")
        sample_weights = sw

    teacher_probs_train: np.ndarray | None = None
    if str(args.teacher_oof_csv).strip():
        teacher_probs_train = _align_teacher_probs(str(args.teacher_oof_csv), train_ids, id_col=str(args.id_col))

    cv_df = pd.DataFrame(
        {
            "_cv_ts": train_df[str(args.time_col)],
            "_cv_group": train_df[str(args.group_col)].astype(np.int64),
        }
    )
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

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        print(f"[fold] {fold_id + 1}/{len(folds)} train={len(tr_idx)} val={len(va_idx)}", flush=True)
        x_tr = img_train[tr_idx]
        y_tr = y[tr_idx]
        x_va = img_train[va_idx]
        y_va = y[va_idx]
        sw_tr = sample_weights[tr_idx] if sample_weights is not None else None
        val_pred, test_pred, best_map, best_epoch = _fit_one_fold(
            x_train=x_tr,
            y_train=y_tr,
            x_val=x_va,
            y_val=y_va,
            x_test=img_test,
            sample_weights_train=sw_tr,
            in_channels=in_channels,
            args=args,
            device=device,
        )
        oof[va_idx] = val_pred
        covered_mask[va_idx] = True
        test_accum += test_pred / float(len(folds))
        fold_score = float(macro_map_score(y_va, val_pred))
        fold_scores.append(fold_score)
        fold_best_maps.append(float(best_map))
        fold_best_epochs.append(int(best_epoch))

    oof_path = art_dir / "spectrogram_oof.npy"
    test_path = art_dir / "spectrogram_test.npy"
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
        teacher_diag = _teacher_diag_simple(
            y_true=y[covered_idx],
            teacher=teacher[covered_idx],
            model=oof[covered_idx],
        )
        _save_per_class_delta_vs_teacher(
            y_true=y[covered_idx],
            teacher_prob=teacher[covered_idx],
            model_prob=oof[covered_idx],
            out_csv=diag_dir / "per_class_ap_delta_vs_teacher.csv",
        )
        diag_unw = _teacher_blend_diag_by_folds(
            y=y,
            teacher=teacher,
            model=oof,
            folds=folds,
            covered_mask=covered_mask,
            sample_weights=None,
        )
        dump_json(diag_dir / "teacher_blend_diag_unweighted.json", diag_unw)
        diag_w = _teacher_blend_diag_by_folds(
            y=y,
            teacher=teacher,
            model=oof,
            folds=folds,
            covered_mask=covered_mask,
            sample_weights=sample_weights,
        )
        dump_json(diag_dir / "teacher_blend_diag_weighted.json", diag_w)
    else:
        dump_json(diag_dir / "teacher_blend_diag_unweighted.json", {"error": "teacher_oof_csv not provided"})
        dump_json(diag_dir / "teacher_blend_diag_weighted.json", {"error": "teacher_oof_csv not provided"})

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

    anchor_diag: dict[str, float] = {}
    if str(args.anchor_submission_csv).strip():
        a_df = pd.read_csv(str(args.anchor_submission_csv))
        a_ids = a_df[str(args.id_col)].to_numpy(dtype=np.int64)
        if not np.array_equal(a_ids, sample_ids):
            pos = {int(t): i for i, t in enumerate(a_ids.tolist())}
            miss = [int(tid) for tid in sample_ids.tolist() if int(tid) not in pos]
            if miss:
                raise ValueError(f"anchor_submission missing ids; first={miss[:10]}")
            take = np.asarray([pos[int(tid)] for tid in sample_ids.tolist()], dtype=np.int64)
            a_prob = a_df[CLASSES].to_numpy(dtype=np.float32)[take]
        else:
            a_prob = a_df[CLASSES].to_numpy(dtype=np.float32)
        anchor_diag = {
            "corr_vs_anchor": float(_safe_corr_flat(pred_sub, a_prob)),
            "spearman_vs_anchor": float(_safe_spearman_flat(pred_sub, a_prob)),
        }

    sub = pd.DataFrame({str(args.id_col): sample_ids})
    for i, cls in enumerate(CLASSES):
        sub[cls] = np.clip(pred_sub[:, i], 0.0, 1.0).astype(np.float32)
    sub_path = out_dir / "submission_spectrogram.csv"
    sub.to_csv(sub_path, index=False)

    summary: dict[str, Any] = {
        "output_dir": str(out_dir),
        "seed": int(args.seed),
        "device": str(device),
        "requested_device": str(args.device),
        "n_splits": int(len(folds)),
        "image_size": int(args.image_size),
        "input_channels": int(in_channels),
        "spec_channels": list(spec_channels),
        "stft_n_fft": int(args.stft_n_fft),
        "stft_hop_length": int(args.stft_hop_length),
        "stft_win_length": int(args.stft_win_length),
        "spec_log1p_eps": float(args.spec_log1p_eps),
        "arch": str(args.arch),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "lr_scheduler": str(args.lr_scheduler),
        "warmup_ratio": float(args.warmup_ratio),
        "min_lr_ratio": float(args.min_lr_ratio),
        "grad_clip": float(args.grad_clip),
        "sample_weights_npy": str(args.sample_weights_npy),
        "n_train": int(len(train_ids)),
        "n_test": int(len(test_ids)),
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
        "fold_mean": float(np.mean(fold_scores) if fold_scores else 0.0),
        "fold_worst": float(np.min(fold_scores) if fold_scores else 0.0),
        "fold_best": float(np.max(fold_scores) if fold_scores else 0.0),
        "teacher_diag": teacher_diag,
        "anchor_diag": anchor_diag,
        "models": {
            "spectrogram_trajectory": {
                "type": "spectrogram_trajectory",
                "oof_path": str(oof_path.resolve()),
                "test_path": str(test_path.resolve()),
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
    dump_json(out_dir / "oof_summary.json", summary)

    print("=== SPECTROGRAM-CNN TRAJECTORY TRAIN COMPLETE ===", flush=True)
    print(f"output_dir={out_dir}", flush=True)
    print(f"oof_summary={out_dir / 'oof_summary.json'}", flush=True)
    print(
        f"macro_covered={summary['macro_map_covered']:.6f} "
        f"full_filled={summary['macro_map_full_filled_teacher']:.6f} "
        f"fold_mean={summary['fold_mean']:.6f} fold_worst={summary['fold_worst']:.6f}",
        flush=True,
    )
    if anchor_diag:
        print(
            f"anchor_corr={anchor_diag['corr_vs_anchor']:.6f} "
            f"anchor_spearman={anchor_diag['spearman_vs_anchor']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
