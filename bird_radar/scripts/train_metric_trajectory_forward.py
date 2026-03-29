#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
from torch.utils.data import DataLoader, Dataset, Sampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds
from src.metrics import macro_map_score, per_class_average_precision
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train metric-learning trajectory model with forward CV (embedding + retrieval).")
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--test-csv", type=str, required=True)
    p.add_argument("--sample-submission", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--cache-dir", type=str, default="bird_radar/artifacts/cache")

    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--lr-scheduler", type=str, default="cosine_warmup", choices=["none", "cosine_warmup"])
    p.add_argument("--warmup-ratio", type=float, default=0.08)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["mps", "cuda", "cpu"])
    p.add_argument("--sample-weights-npy", type=str, default="")

    p.add_argument("--seq-len", type=int, default=320)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--ffn-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--transformer-multires", type=int, default=1)
    p.add_argument("--multires-pool-factor", type=int, default=2)

    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--classification-weight", type=float, default=0.5)
    p.add_argument("--supcon-weight", type=float, default=0.5)
    p.add_argument("--supcon-temperature", type=float, default=0.07)

    p.add_argument("--balanced-sampler", type=int, default=1, choices=[0, 1])
    p.add_argument("--samples-per-class", type=int, default=2)

    p.add_argument("--knn-k", type=int, default=50)
    p.add_argument("--knn-temperature", type=float, default=0.07)
    p.add_argument("--proto-temperature", type=float, default=0.10)
    p.add_argument("--proto-weight", type=float, default=0.7)
    p.add_argument("--knn-weight", type=float, default=0.3)

    p.add_argument("--augment-time-crop-p", type=float, default=0.6)
    p.add_argument("--augment-noise-std-xyz", type=float, default=0.004)
    p.add_argument("--augment-rcs-drop-p", type=float, default=0.03)
    p.add_argument("--augment-speed-drop-p", type=float, default=0.02)

    p.add_argument("--raw-clip-quantile-low", type=float, default=0.01)
    p.add_argument("--raw-clip-quantile-high", type=float, default=0.99)

    p.add_argument("--teacher-oof-csv", type=str, default="")
    p.add_argument("--anchor-submission-csv", type=str, default="")
    p.add_argument("--id-col", type=str, default="track_id")
    p.add_argument("--time-col", type=str, default="timestamp_start_radar_utc")
    p.add_argument("--group-col", type=str, default="observation_id")
    p.add_argument("--cv-direction", type=str, default="forward", choices=["forward", "reverse"])
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


def _fit_raw_scaler(
    seqs: list[np.ndarray],
    q_low: float,
    q_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cat = np.concatenate(seqs, axis=0).astype(np.float64, copy=False)
    lo = np.quantile(cat, q_low, axis=0).astype(np.float32)
    hi = np.quantile(cat, q_high, axis=0).astype(np.float32)
    span = hi - lo
    hi = np.where(span < 1e-6, lo + 1.0, hi)
    clipped = np.clip(cat, lo[None, :], hi[None, :])
    mean = np.mean(clipped, axis=0).astype(np.float32)
    std = np.std(clipped, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return lo, hi, mean, std


def _normalize_per_track(arr: np.ndarray, valid_len: int) -> np.ndarray:
    out = arr.copy()
    if valid_len <= 1:
        return out
    m = np.mean(out[:valid_len], axis=0, keepdims=True)
    s = np.std(out[:valid_len], axis=0, keepdims=True)
    s = np.where(s < 1e-5, 1.0, s)
    out[:valid_len] = (out[:valid_len] - m) / s
    return out


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        seqs: list[np.ndarray],
        y_idx: np.ndarray,
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
    ) -> None:
        self.seqs = seqs
        self.y_idx = y_idx.astype(np.int64, copy=False)
        self.n_classes = len(CLASSES)
        self.y = np.zeros((len(self.y_idx), self.n_classes), dtype=np.float32)
        self.y[np.arange(len(self.y_idx)), self.y_idx] = 1.0
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
            if len(sw) != len(self.seqs):
                raise ValueError(f"sample_weights length mismatch: {len(sw)} vs {len(self.seqs)}")
            self.sample_weights = sw

    def __len__(self) -> int:
        return len(self.seqs)

    def _time_transform(self, seq: np.ndarray) -> np.ndarray:
        n = int(seq.shape[0])
        if (not self.augment) or n <= self.seq_len:
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
            return out.astype(np.float32), mask, self.seq_len
        out = np.zeros((self.seq_len, c), dtype=np.float32)
        out[:n] = seq
        mask = np.zeros((self.seq_len,), dtype=np.float32)
        mask[:n] = 1.0
        return out, mask, n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self._time_transform(self.seqs[idx])
        arr, mask, valid_len = self._to_fixed_len(seq)

        if self.augment and valid_len > 0:
            if self.noise_std_xyz > 0.0:
                arr[:valid_len, 0:3] += np.random.normal(0.0, self.noise_std_xyz, size=(valid_len, 3)).astype(np.float32)
            if self.rcs_drop_p > 0.0 and random.random() < self.rcs_drop_p:
                arr[:valid_len, 3] = 0.0
            if self.speed_drop_p > 0.0 and random.random() < self.speed_drop_p:
                arr[:valid_len, 4] = 0.0

        raw = arr.copy()
        if valid_len > 0:
            raw[:valid_len] = np.clip(raw[:valid_len], self.raw_clip_lo[None, :], self.raw_clip_hi[None, :])
            raw[:valid_len] = (raw[:valid_len] - self.raw_mean[None, :]) / self.raw_std[None, :]
        zn = _normalize_per_track(arr.copy(), valid_len=valid_len)

        x18 = np.concatenate([raw, zn], axis=1).astype(np.float32, copy=False)
        x = torch.from_numpy(x18.T.copy())
        m = torch.from_numpy(mask)
        y_idx = torch.tensor(int(self.y_idx[idx]), dtype=torch.long)
        y = torch.from_numpy(self.y[idx])
        w = torch.tensor(float(self.sample_weights[idx]), dtype=torch.float32)
        return x, m, y_idx, y, w


class BalancedClassBatchSampler(Sampler[list[int]]):
    def __init__(self, labels_idx: np.ndarray, batch_size: int, samples_per_class: int) -> None:
        self.labels_idx = np.asarray(labels_idx, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.samples_per_class = int(max(1, samples_per_class))
        self.class_to_indices: dict[int, np.ndarray] = {}
        for c in range(len(CLASSES)):
            idx = np.where(self.labels_idx == c)[0].astype(np.int64)
            if len(idx) > 0:
                self.class_to_indices[int(c)] = idx
        if not self.class_to_indices:
            raise ValueError("BalancedClassBatchSampler: no classes found")
        self.classes = np.asarray(sorted(self.class_to_indices.keys()), dtype=np.int64)
        self.n = int(len(self.labels_idx))
        self.n_batches = int(math.ceil(self.n / float(self.batch_size)))

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        all_idx = np.arange(self.n, dtype=np.int64)
        classes_per_batch = max(1, self.batch_size // self.samples_per_class)
        for _ in range(self.n_batches):
            c_count = min(len(self.classes), classes_per_batch)
            replace_cls = len(self.classes) < c_count
            chosen_classes = np.random.choice(self.classes, size=c_count, replace=replace_cls)
            out: list[int] = []
            for c in chosen_classes.tolist():
                pool = self.class_to_indices[int(c)]
                rep = len(pool) < self.samples_per_class
                take = np.random.choice(pool, size=self.samples_per_class, replace=rep)
                out.extend([int(t) for t in take.tolist()])
            if len(out) < self.batch_size:
                fill = np.random.choice(all_idx, size=self.batch_size - len(out), replace=True)
                out.extend([int(t) for t in fill.tolist()])
            random.shuffle(out)
            yield out[: self.batch_size]


class SequenceAttentionPooling(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        s = self.score(x).squeeze(-1)
        s = s.masked_fill(mask <= 0.0, -1e4)
        a = torch.softmax(s, dim=1)
        a = a * mask
        a = a / (a.sum(dim=1, keepdim=True) + 1e-8)
        return torch.sum(x * a.unsqueeze(-1), dim=1)


class MetricTwoStreamTransformer(nn.Module):
    def __init__(
        self,
        n_classes: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float,
        embedding_dim: int,
        use_multires: bool,
        multires_pool_factor: int,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.use_multires = bool(use_multires)
        self.multires_pool_factor = int(max(2, multires_pool_factor))

        self.stem_raw = nn.Sequential(nn.Linear(9, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.stem_zn = nn.Sequential(nn.Linear(9, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        low_len = int(np.ceil(float(self.seq_len) / float(self.multires_pool_factor)))
        self.pos_embed_low = nn.Parameter(torch.zeros(1, low_len, d_model))
        nn.init.trunc_normal_(self.pos_embed_low, std=0.02)

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

        feat_dim = d_model * (2 if self.use_multires else 1)
        self.embedding_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, int(embedding_dim)),
        )
        self.class_head = nn.Linear(int(embedding_dim), n_classes)

    def _encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        xr = xt[:, :, :9]
        xz = xt[:, :, 9:]
        z = self.fuse(torch.cat([self.stem_raw(xr), self.stem_zn(xz)], dim=-1))

        z = z + self.pos_embed[:, : z.size(1), :]
        z = self.encoder(z, src_key_padding_mask=(mask <= 0.0))
        p_main = self.pool(z, mask)

        if not self.use_multires:
            return p_main

        z_low = F.avg_pool1d(
            self.fuse(torch.cat([self.stem_raw(xr), self.stem_zn(xz)], dim=-1)).transpose(1, 2),
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

        z_low = z_low + self.pos_embed_low[:, : z_low.size(1), :]
        z_low = self.encoder_low(z_low, src_key_padding_mask=(m_low <= 0.0))
        p_low = self.pool(z_low, m_low)

        return torch.cat([p_main, p_low], dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self._encode(x, mask)
        emb = self.embedding_head(feat)
        emb = F.normalize(emb, dim=1)
        logits = self.class_head(emb)
        return emb, logits


@torch.no_grad()
def _extract_embeddings_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    embs: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    for xb, mb, *_ in loader:
        xb = xb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)
        emb, logits = model(xb, mb)
        p = torch.sigmoid(logits)
        embs.append(emb.detach().cpu().numpy().astype(np.float32))
        probs.append(p.detach().cpu().numpy().astype(np.float32))
    if not embs:
        return np.empty((0, 1), dtype=np.float32), np.empty((0, len(CLASSES)), dtype=np.float32)
    return np.concatenate(embs, axis=0), np.concatenate(probs, axis=0)


def _supcon_loss(emb: torch.Tensor, labels_idx: torch.Tensor, temperature: float) -> torch.Tensor:
    z = F.normalize(emb, dim=1)
    sim = torch.matmul(z, z.T) / max(float(temperature), 1e-6)

    b = z.size(0)
    eye = torch.eye(b, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)

    lbl = labels_idx.view(-1, 1)
    pos_mask = (lbl == lbl.T) & (~eye)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if not torch.any(valid):
        return torch.zeros((), device=z.device, dtype=z.dtype)

    loss_i = -(log_prob * pos_mask.float()).sum(dim=1) / torch.clamp(pos_count.float(), min=1.0)
    return loss_i[valid].mean()


def _build_prototypes(emb: np.ndarray, y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    emb_n = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
    protos = np.zeros((n_classes, emb.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        idx = np.where(y_idx == c)[0]
        if len(idx) == 0:
            continue
        p = emb_n[idx].mean(axis=0)
        n = float(np.linalg.norm(p))
        if n > 1e-8:
            p = p / n
        protos[c] = p.astype(np.float32)
    return protos


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(np.clip(z, -50.0, 50.0))
    sm = ez / np.clip(np.sum(ez, axis=1, keepdims=True), 1e-12, None)
    return sm.astype(np.float32)


def _predict_prototype(emb: np.ndarray, protos: np.ndarray, temperature: float) -> np.ndarray:
    e = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
    p = protos / np.clip(np.linalg.norm(protos, axis=1, keepdims=True), 1e-8, None)
    sim = np.matmul(e, p.T)
    return _softmax_rows(sim / max(float(temperature), 1e-6))


def _predict_knn(
    emb_query: np.ndarray,
    emb_train: np.ndarray,
    y_train_idx: np.ndarray,
    n_classes: int,
    k: int,
    temperature: float,
) -> np.ndarray:
    q = emb_query / np.clip(np.linalg.norm(emb_query, axis=1, keepdims=True), 1e-8, None)
    t = emb_train / np.clip(np.linalg.norm(emb_train, axis=1, keepdims=True), 1e-8, None)
    sim = np.matmul(q, t.T)

    k_eff = int(min(max(1, k), sim.shape[1]))
    idx_top = np.argpartition(sim, -k_eff, axis=1)[:, -k_eff:]
    sim_top = np.take_along_axis(sim, idx_top, axis=1)

    w = np.exp(np.clip(sim_top / max(float(temperature), 1e-6), -40.0, 40.0)).astype(np.float32)
    out = np.zeros((sim.shape[0], n_classes), dtype=np.float32)

    for i in range(sim.shape[0]):
        cls_i = y_train_idx[idx_top[i]]
        np.add.at(out[i], cls_i, w[i])

    s = np.sum(out, axis=1, keepdims=True)
    out = out / np.clip(s, 1e-12, None)
    return out.astype(np.float32)


def _hybrid_probs(proto_prob: np.ndarray, knn_prob: np.ndarray, proto_w: float, knn_w: float) -> np.ndarray:
    p = float(max(0.0, proto_w))
    k = float(max(0.0, knn_w))
    if p + k <= 1e-12:
        p, k = 1.0, 0.0
    p = p / (p + k)
    k = k / (p + k) if (p + k) > 0 else 0.0
    out = p * proto_prob + k * knn_prob
    out = out / np.clip(np.sum(out, axis=1, keepdims=True), 1e-12, None)
    return out.astype(np.float32)


def _fit_one_fold(
    seqs_train: list[np.ndarray],
    y_idx_train: np.ndarray,
    seqs_val: list[np.ndarray],
    y_idx_val: np.ndarray,
    seqs_test: list[np.ndarray],
    raw_clip_lo: np.ndarray,
    raw_clip_hi: np.ndarray,
    raw_mean: np.ndarray,
    raw_std: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    sample_weights_train: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, int]:
    ds_train = TrajectoryDataset(
        seqs=seqs_train,
        y_idx=y_idx_train,
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
        sample_weights=sample_weights_train,
    )
    ds_train_eval = TrajectoryDataset(
        seqs=seqs_train,
        y_idx=y_idx_train,
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
        sample_weights=sample_weights_train,
    )
    ds_val = TrajectoryDataset(
        seqs=seqs_val,
        y_idx=y_idx_val,
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
    )
    ds_test = TrajectoryDataset(
        seqs=seqs_test,
        y_idx=np.zeros((len(seqs_test),), dtype=np.int64),
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
    )

    if bool(int(args.balanced_sampler)):
        batch_sampler = BalancedClassBatchSampler(
            labels_idx=y_idx_train,
            batch_size=int(args.batch_size),
            samples_per_class=int(args.samples_per_class),
        )
        dl_train = DataLoader(
            ds_train,
            batch_sampler=batch_sampler,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
        )
    else:
        dl_train = DataLoader(
            ds_train,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    dl_train_eval = DataLoader(ds_train_eval, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=(device.type == "cuda"))
    dl_val = DataLoader(ds_val, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=(device.type == "cuda"))
    dl_test = DataLoader(ds_test, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=(device.type == "cuda"))

    model = MetricTwoStreamTransformer(
        n_classes=len(CLASSES),
        seq_len=int(args.seq_len),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        num_layers=int(args.num_layers),
        ffn_dim=int(args.ffn_dim),
        dropout=float(args.dropout),
        embedding_dim=int(args.embedding_dim),
        use_multires=bool(int(args.transformer_multires)),
        multires_pool_factor=int(args.multires_pool_factor),
    ).to(device)

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

    pos = np.zeros((len(CLASSES),), dtype=np.float32)
    np.add.at(pos, y_idx_train, 1.0)
    neg = float(len(y_idx_train)) - pos
    pos_weight = torch.from_numpy(((neg + 1.0) / (pos + 1.0)).astype(np.float32)).to(device)
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    best_state: dict[str, torch.Tensor] | None = None
    best_map = -1.0
    best_epoch = -1
    no_improve = 0
    non_finite_batches = 0

    for epoch in range(int(args.epochs)):
        model.train()
        for xb, mb, y_idx_b, yb, wb in dl_train:
            xb = xb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            y_idx_b = y_idx_b.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            emb, logits = model(xb, mb)
            logits = logits.clamp(-20.0, 20.0)
            if not torch.isfinite(logits).all() or not torch.isfinite(emb).all():
                non_finite_batches += 1
                continue

            cls_raw = criterion_cls(logits, yb).mean(dim=1)
            loss_cls = (cls_raw * wb).mean()
            loss_sup = _supcon_loss(emb, y_idx_b, temperature=float(args.supcon_temperature))
            loss = float(args.classification_weight) * loss_cls + float(args.supcon_weight) * loss_sup

            if not torch.isfinite(loss):
                non_finite_batches += 1
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # select best by validation class-head macro (fast/stable)
        _, val_prob_cls = _extract_embeddings_logits(model, dl_val, device=device)
        val_map = float(macro_map_score(np.eye(len(CLASSES), dtype=np.float32)[y_idx_val], val_prob_cls))
        print(
            f"[diag] epoch={epoch} val_map_cls={val_map:.6f} pred_mean={float(np.mean(val_prob_cls)):.6f} "
            f"pred_std={float(np.std(val_prob_cls)):.6f} pred_p95={float(np.quantile(val_prob_cls, 0.95)):.6f}",
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

    train_emb, _ = _extract_embeddings_logits(model, dl_train_eval, device=device)
    val_emb, _ = _extract_embeddings_logits(model, dl_val, device=device)
    test_emb, _ = _extract_embeddings_logits(model, dl_test, device=device)

    prototypes = _build_prototypes(train_emb, y_idx_train, n_classes=len(CLASSES))
    val_proto = _predict_prototype(val_emb, prototypes, temperature=float(args.proto_temperature))
    test_proto = _predict_prototype(test_emb, prototypes, temperature=float(args.proto_temperature))

    val_knn = _predict_knn(
        emb_query=val_emb,
        emb_train=train_emb,
        y_train_idx=y_idx_train,
        n_classes=len(CLASSES),
        k=int(args.knn_k),
        temperature=float(args.knn_temperature),
    )
    test_knn = _predict_knn(
        emb_query=test_emb,
        emb_train=train_emb,
        y_train_idx=y_idx_train,
        n_classes=len(CLASSES),
        k=int(args.knn_k),
        temperature=float(args.knn_temperature),
    )

    val_pred = _hybrid_probs(val_proto, val_knn, proto_w=float(args.proto_weight), knn_w=float(args.knn_weight))
    test_pred = _hybrid_probs(test_proto, test_knn, proto_w=float(args.proto_weight), knn_w=float(args.knn_weight))

    return val_pred, test_pred, val_emb, test_emb, train_emb, float(best_map), int(best_epoch), int(non_finite_batches)


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

    sample_weights: np.ndarray | None = None
    if str(args.sample_weights_npy).strip():
        sw = np.asarray(np.load(str(args.sample_weights_npy)), dtype=np.float32).reshape(-1)
        if len(sw) != len(train_ids):
            raise ValueError(f"sample_weights length mismatch: got {len(sw)}, expected {len(train_ids)}")
        sample_weights = sw

    teacher_probs_train: np.ndarray | None = None
    if str(args.teacher_oof_csv).strip():
        teacher_probs_train = _align_teacher_probs(str(args.teacher_oof_csv), train_ids, id_col=str(args.id_col))

    cv_df = pd.DataFrame({"_cv_ts": train_df[str(args.time_col)], "_cv_group": train_df[str(args.group_col)].astype(np.int64)})
    folds = make_forward_temporal_group_folds(
        cv_df,
        timestamp_col="_cv_ts",
        group_col="_cv_group",
        n_splits=int(args.n_splits),
    )
    if str(args.cv_direction).lower() == "reverse":
        folds = list(reversed(folds))
    if len(folds) == 0:
        raise RuntimeError("no folds created")

    oof = np.zeros((len(train_ids), len(CLASSES)), dtype=np.float32)
    oof_emb = np.zeros((len(train_ids), int(args.embedding_dim)), dtype=np.float32)
    covered_mask = np.zeros((len(train_ids),), dtype=bool)
    test_accum = np.zeros((len(test_ids), len(CLASSES)), dtype=np.float32)
    test_emb_accum = np.zeros((len(test_ids), int(args.embedding_dim)), dtype=np.float32)

    fold_scores: list[float] = []
    fold_best_maps: list[float] = []
    fold_best_epochs: list[int] = []
    non_finite_total = 0

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        print(f"[fold] {fold_id + 1}/{len(folds)} train={len(tr_idx)} val={len(va_idx)}", flush=True)
        seqs_tr = [seqs_train[int(i)] for i in tr_idx.tolist()]
        seqs_va = [seqs_train[int(i)] for i in va_idx.tolist()]
        y_tr_idx = y_idx[tr_idx]
        y_va_idx = y_idx[va_idx]
        sw_tr = sample_weights[tr_idx] if sample_weights is not None else None

        clip_lo, clip_hi, raw_mean, raw_std = _fit_raw_scaler(
            seqs=seqs_tr,
            q_low=float(args.raw_clip_quantile_low),
            q_high=float(args.raw_clip_quantile_high),
        )

        val_pred, test_pred, val_emb, test_emb, train_emb_fold, best_map, best_epoch, non_finite = _fit_one_fold(
            seqs_train=seqs_tr,
            y_idx_train=y_tr_idx,
            seqs_val=seqs_va,
            y_idx_val=y_va_idx,
            seqs_test=seqs_test,
            raw_clip_lo=clip_lo,
            raw_clip_hi=clip_hi,
            raw_mean=raw_mean,
            raw_std=raw_std,
            args=args,
            device=device,
            sample_weights_train=sw_tr,
        )

        oof[va_idx] = val_pred
        oof_emb[va_idx] = val_emb
        covered_mask[va_idx] = True
        test_accum += test_pred / float(len(folds))
        test_emb_accum += test_emb / float(len(folds))

        y_va_oh = np.zeros((len(y_va_idx), len(CLASSES)), dtype=np.float32)
        y_va_oh[np.arange(len(y_va_idx)), y_va_idx] = 1.0
        fold_score = float(macro_map_score(y_va_oh, val_pred))
        fold_scores.append(fold_score)
        fold_best_maps.append(float(best_map))
        fold_best_epochs.append(int(best_epoch))
        non_finite_total += int(non_finite)

    oof_path = art_dir / "metric_oof.npy"
    test_path = art_dir / "metric_test.npy"
    train_emb_path = art_dir / "train_embeddings.npy"
    test_emb_path = art_dir / "test_embeddings.npy"

    np.save(oof_path, oof.astype(np.float32))
    np.save(test_path, test_accum.astype(np.float32))
    np.save(train_emb_path, oof_emb.astype(np.float32))
    np.save(test_emb_path, test_emb_accum.astype(np.float32))

    covered_idx = np.where(covered_mask)[0].astype(np.int64)
    uncovered_idx = np.where(~covered_mask)[0].astype(np.int64)
    np.save(out_dir / "oof_covered_idx.npy", covered_idx)
    if len(covered_idx) == 0:
        raise RuntimeError("no covered validation rows produced by forward CV")

    macro_raw_full = float(macro_map_score(y, oof))
    macro_covered = float(macro_map_score(y[covered_idx], oof[covered_idx]))
    per_class_covered = per_class_average_precision(y[covered_idx], oof[covered_idx])

    oof_full_filled = oof.copy()
    macro_full_filled_teacher = 0.0
    teacher_diag: dict[str, float] = {}

    if teacher_probs_train is not None:
        teacher = np.clip(teacher_probs_train.astype(np.float32), 1e-6, 1.0 - 1e-6)
        oof_full_filled[uncovered_idx] = teacher[uncovered_idx]
        macro_full_filled_teacher = float(macro_map_score(y, oof_full_filled))

        teacher_macro = float(macro_map_score(y[covered_idx], teacher[covered_idx]))
        model_macro = float(macro_map_score(y[covered_idx], oof[covered_idx]))
        best = teacher_macro
        best_w = 1.0
        for w in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
            blend = np.clip(w * teacher[covered_idx] + (1.0 - w) * oof[covered_idx], 0.0, 1.0)
            m = float(macro_map_score(y[covered_idx], blend))
            if m > best:
                best = m
                best_w = float(w)
        teacher_diag = {
            "teacher_macro": teacher_macro,
            "model_macro": model_macro,
            "best_blend_macro": best,
            "best_blend_w_teacher": best_w,
            "best_blend_gain": float(best - teacher_macro),
            "corr_with_teacher": float(_safe_corr_flat(teacher[covered_idx], oof[covered_idx])),
        }

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
    sub_path = out_dir / "submission_metric.csv"
    sub.to_csv(sub_path, index=False)

    summary: dict[str, Any] = {
        "output_dir": str(out_dir),
        "seed": int(args.seed),
        "device": str(device),
        "requested_device": str(args.device),
        "n_splits": int(len(folds)),
        "seq_len": int(args.seq_len),
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "num_layers": int(args.num_layers),
        "ffn_dim": int(args.ffn_dim),
        "dropout": float(args.dropout),
        "transformer_multires": bool(int(args.transformer_multires)),
        "multires_pool_factor": int(args.multires_pool_factor),
        "embedding_dim": int(args.embedding_dim),
        "classification_weight": float(args.classification_weight),
        "supcon_weight": float(args.supcon_weight),
        "supcon_temperature": float(args.supcon_temperature),
        "balanced_sampler": bool(int(args.balanced_sampler)),
        "samples_per_class": int(args.samples_per_class),
        "knn_k": int(args.knn_k),
        "knn_temperature": float(args.knn_temperature),
        "proto_temperature": float(args.proto_temperature),
        "proto_weight": float(args.proto_weight),
        "knn_weight": float(args.knn_weight),
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
        "non_finite_batches": int(non_finite_total),
        "models": {
            "metric_trajectory": {
                "type": "metric_trajectory",
                "oof_path": str(oof_path.resolve()),
                "test_path": str(test_path.resolve()),
                "train_embeddings_path": str(train_emb_path.resolve()),
                "test_embeddings_path": str(test_emb_path.resolve()),
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

    print("=== METRIC TRAJECTORY TRAIN COMPLETE ===", flush=True)
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
