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


GROUP_TO_CLASSES: dict[str, list[str]] = {
    "waterbirds": ["Ducks", "Geese", "Gulls", "Waders", "Cormorants"],
    "raptors": ["Birds of Prey"],
    "smallbirds": ["Songbirds", "Pigeons"],
    "clutter": ["Clutter"],
}
GROUP_NAMES: list[str] = ["waterbirds", "raptors", "smallbirds", "clutter"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train hierarchical trajectory model with forward CV.")
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--test-csv", type=str, required=True)
    p.add_argument("--sample-submission", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--cache-dir", type=str, default="bird_radar/artifacts/cache")

    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--lr-scheduler", type=str, default="none", choices=["none", "cosine_warmup"])
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["mps", "cuda", "cpu"])
    p.add_argument("--sample-weights-npy", type=str, default="")

    p.add_argument("--seq-len", type=int, default=160)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--ffn-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--transformer-multires", type=int, default=1)
    p.add_argument("--multires-pool-factor", type=int, default=2)
    p.add_argument("--physics-token-mode", type=str, default="v1", choices=["none", "v1"])

    p.add_argument("--group-loss-weight", type=float, default=0.5)
    p.add_argument("--flat-loss-weight", type=float, default=0.7)
    p.add_argument("--blend-alpha", type=float, default=0.7)
    p.add_argument("--blend-alpha-by-class", type=str, default="")
    p.add_argument("--flat-loss-class-weight", type=str, default="")
    p.add_argument("--augment-time-crop-p", type=float, default=0.6)
    p.add_argument("--augment-noise-std-xyz", type=float, default=0.004)
    p.add_argument("--augment-rcs-drop-p", type=float, default=0.03)
    p.add_argument("--augment-speed-drop-p", type=float, default=0.02)
    p.add_argument("--raw-clip-quantile-low", type=float, default=0.01)
    p.add_argument("--raw-clip-quantile-high", type=float, default=0.99)

    p.add_argument("--teacher-oof-csv", type=str, default="")
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


def _parse_class_scalar_map(text: str, default: float) -> np.ndarray:
    arr = np.full((len(CLASSES),), float(default), dtype=np.float32)
    s = str(text).strip()
    if not s:
        return arr
    parts = [x.strip() for x in s.split(",") if x.strip()]
    for part in parts:
        if ":" not in part:
            raise ValueError(f"bad class map item '{part}', expected Class:value")
        name, val = [x.strip() for x in part.split(":", 1)]
        if name not in CLASS_TO_INDEX:
            raise ValueError(f"unknown class in map: {name}")
        arr[int(CLASS_TO_INDEX[name])] = float(val)
    return arr


def _normalize_per_track(arr: np.ndarray, valid_len: int) -> np.ndarray:
    out = arr.copy()
    if valid_len <= 1:
        return out
    m = np.mean(out[:valid_len], axis=0, keepdims=True)
    s = np.std(out[:valid_len], axis=0, keepdims=True)
    s = np.where(s < 1e-5, 1.0, s)
    out[:valid_len] = (out[:valid_len] - m) / s
    return out


def _fit_raw_scaler(seqs: list[np.ndarray], q_low: float, q_high: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


class HierTrajectoryDataset(Dataset):
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
        self.y_idx = np.asarray(y_idx, dtype=np.int64).reshape(-1)
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
            self.sample_weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)

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
            return out.astype(np.float32), mask, self.seq_len
        out = np.zeros((self.seq_len, c), dtype=np.float32)
        out[:n] = seq
        mask = np.zeros((self.seq_len,), dtype=np.float32)
        mask[:n] = 1.0
        return out, mask, n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self._time_transform(self.seqs[idx])
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

        raw = arr.copy()
        if valid_len > 0:
            raw[:valid_len] = np.clip(raw[:valid_len], self.raw_clip_lo[None, :], self.raw_clip_hi[None, :])
            raw[:valid_len] = (raw[:valid_len] - self.raw_mean[None, :]) / self.raw_std[None, :]

        zn = _normalize_per_track(arr.copy(), valid_len=valid_len)
        x18 = np.concatenate([raw, zn], axis=1).astype(np.float32, copy=False)

        x = torch.from_numpy(x18.T.copy())
        m = torch.from_numpy(mask)
        y = torch.tensor(int(self.y_idx[idx]), dtype=torch.long)
        w = torch.tensor(float(self.sample_weights[idx]), dtype=torch.float32)
        return x, m, y, w


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


class HierarchicalTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float,
        group_class_indices: list[list[int]],
        physics_token_mode: str = "none",
        use_multires: bool = True,
        multires_pool_factor: int = 2,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.group_class_indices = [list(v) for v in group_class_indices]
        self.n_groups = len(self.group_class_indices)
        self.n_classes = sum(len(v) for v in self.group_class_indices)
        self.use_multires = bool(use_multires)
        self.multires_pool_factor = int(max(2, multires_pool_factor))
        self.physics_token_mode = str(physics_token_mode).strip().lower()
        self.n_physics_tokens = 6 if self.physics_token_mode == "v1" else 0

        self.stem_raw = nn.Sequential(nn.Linear(9, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.stem_zn = nn.Sequential(nn.Linear(9, d_model), nn.LayerNorm(d_model), nn.GELU())
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

        pooled_dim = d_model * (2 if self.use_multires else 1)
        self.group_head = nn.Linear(pooled_dim, self.n_groups)
        self.class_heads = nn.ModuleList([nn.Linear(pooled_dim, len(idx)) for idx in self.group_class_indices])
        self.flat_head = nn.Linear(pooled_dim, len(CLASSES))

    @staticmethod
    def _masked_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        den = w.sum(dim=1).clamp_min(1e-6)
        return (x * w).sum(dim=1) / den

    @staticmethod
    def _masked_std(x: torch.Tensor, w: torch.Tensor, mean: torch.Tensor | None = None) -> torch.Tensor:
        if mean is None:
            mean = HierarchicalTransformer._masked_mean(x, w)
        xc = x - mean.unsqueeze(1)
        den = w.sum(dim=1).clamp_min(1e-6)
        var = ((xc * xc) * w).sum(dim=1) / den
        return torch.sqrt(var.clamp_min(1e-8))

    def _build_physics_tokens(self, xt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
            return torch.where(w_part.sum(dim=1, keepdim=True) > 0.0, w_part, m)

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

    def _compose_log_probs(
        self,
        group_logits: torch.Tensor,
        cond_logits: list[torch.Tensor],
    ) -> torch.Tensor:
        bsz = group_logits.size(0)
        log_pg = F.log_softmax(group_logits, dim=1)
        out = group_logits.new_full((bsz, len(CLASSES)), -1e9)
        for gi, cls_idx in enumerate(self.group_class_indices):
            if len(cls_idx) == 1:
                out[:, cls_idx[0]] = log_pg[:, gi]
            else:
                log_pc = F.log_softmax(cond_logits[gi], dim=1)
                out[:, cls_idx] = log_pg[:, gi : gi + 1] + log_pc
        return out

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xt = x.transpose(1, 2)
        xr = xt[:, :, :9]
        xz = xt[:, :, 9:]
        z_base = self.fuse(torch.cat([self.stem_raw(xr), self.stem_zn(xz)], dim=-1))

        full_mask = mask
        z = z_base
        if self.n_physics_tokens > 0:
            phys_tokens = self.physics_proj(self._build_physics_tokens(xt, mask))
            z = torch.cat([phys_tokens, z], dim=1)
            tok = mask.new_ones((mask.size(0), self.n_physics_tokens))
            full_mask = torch.cat([tok, mask], dim=1)
        z = z + self.pos_embed[:, : z.size(1), :]
        z = self.encoder(z, src_key_padding_mask=(full_mask <= 0.0))
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
            z_low = self.encoder_low(z_low, src_key_padding_mask=(full_mask_low <= 0.0))
            p_low = self.pool(z_low, full_mask_low)
            pooled = torch.cat([p_main, p_low], dim=1)
        else:
            pooled = p_main

        group_logits = self.group_head(pooled)
        cond_logits = [head(pooled) for head in self.class_heads]
        class_log_probs = self._compose_log_probs(group_logits, cond_logits)
        flat_logits = self.flat_head(pooled)
        return class_log_probs, group_logits, flat_logits


@torch.no_grad()
def _predict_heads(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    alpha_vec: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds_final: list[np.ndarray] = []
    preds_hier: list[np.ndarray] = []
    preds_flat: list[np.ndarray] = []
    for xb, mb, *_ in loader:
        xb = xb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)
        logp_hier, _, flat_logits = model(xb, mb)
        p_hier = torch.exp(logp_hier).clamp(1e-6, 1.0 - 1e-6)
        logit_hier = torch.log(p_hier / (1.0 - p_hier))
        p_flat = torch.sigmoid(flat_logits).clamp(1e-6, 1.0 - 1e-6)
        logits_final = alpha_vec.view(1, -1) * logit_hier + (1.0 - alpha_vec.view(1, -1)) * flat_logits
        p_final = torch.sigmoid(logits_final).clamp(1e-6, 1.0 - 1e-6)
        preds_final.append(p_final.cpu().numpy().astype(np.float32))
        preds_hier.append(p_hier.cpu().numpy().astype(np.float32))
        preds_flat.append(p_flat.cpu().numpy().astype(np.float32))
    if not preds_final:
        empty = np.empty((0, len(CLASSES)), dtype=np.float32)
        return empty, empty, empty
    return (
        np.concatenate(preds_final, axis=0),
        np.concatenate(preds_hier, axis=0),
        np.concatenate(preds_flat, axis=0),
    )


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


def _fit_one_fold(
    seqs_train: list[np.ndarray],
    y_train_idx: np.ndarray,
    seqs_val: list[np.ndarray],
    y_val_onehot: np.ndarray,
    seqs_test: list[np.ndarray],
    raw_clip_lo: np.ndarray,
    raw_clip_hi: np.ndarray,
    raw_mean: np.ndarray,
    raw_std: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    class_to_group: np.ndarray,
    sample_weights_train: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    ds_train = HierTrajectoryDataset(
        seqs=seqs_train,
        y_idx=y_train_idx,
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
    ds_val = HierTrajectoryDataset(
        seqs=seqs_val,
        y_idx=np.argmax(y_val_onehot, axis=1),
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
    ds_test = HierTrajectoryDataset(
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

    group_class_indices = [[CLASS_TO_INDEX[c] for c in GROUP_TO_CLASSES[g]] for g in GROUP_NAMES]
    model = HierarchicalTransformer(
        seq_len=int(args.seq_len),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        num_layers=int(args.num_layers),
        ffn_dim=int(args.ffn_dim),
        dropout=float(args.dropout),
        group_class_indices=group_class_indices,
        physics_token_mode=str(args.physics_token_mode),
        use_multires=bool(int(args.transformer_multires)),
        multires_pool_factor=int(args.multires_pool_factor),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    total_steps = max(1, int(args.epochs) * max(1, len(dl_train)))
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
    if str(args.lr_scheduler).strip().lower() == "cosine_warmup":
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

    class_to_group_t = torch.from_numpy(class_to_group.astype(np.int64)).to(device)
    group_loss_weight = float(max(0.0, args.group_loss_weight))
    flat_loss_weight = float(max(0.0, args.flat_loss_weight))
    alpha_global = float(np.clip(args.blend_alpha, 0.0, 1.0))
    alpha_by_class = _parse_class_scalar_map(args.blend_alpha_by_class, default=alpha_global)
    alpha_by_class = np.clip(alpha_by_class, 0.0, 1.0).astype(np.float32)
    alpha_vec = torch.from_numpy(alpha_by_class).to(device)
    flat_class_w = _parse_class_scalar_map(args.flat_loss_class_weight, default=1.0).astype(np.float32)
    flat_class_w_t = torch.from_numpy(flat_class_w).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_map = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(int(args.epochs)):
        model.train()
        bad_batches = 0
        total_batches = 0
        for xb, mb, yb, wb in dl_train:
            total_batches += 1
            xb = xb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logp_hier, group_logits, flat_logits = model(xb, mb)
            if (
                (not torch.isfinite(logp_hier).all())
                or (not torch.isfinite(group_logits).all())
                or (not torch.isfinite(flat_logits).all())
            ):
                bad_batches += 1
                continue

            y_group = class_to_group_t[yb]
            loss_class = F.nll_loss(logp_hier, yb, reduction="none")
            loss_group = F.cross_entropy(group_logits, y_group, reduction="none")
            yb_onehot = F.one_hot(yb, num_classes=len(CLASSES)).to(torch.float32)
            loss_flat_raw = F.binary_cross_entropy_with_logits(flat_logits, yb_onehot, reduction="none")
            loss_flat = (loss_flat_raw * flat_class_w_t.view(1, -1)).mean(dim=1)
            loss = ((loss_class + group_loss_weight * loss_group + flat_loss_weight * loss_flat) * wb).mean()
            if not torch.isfinite(loss):
                bad_batches += 1
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

        val_pred, _, _ = _predict_heads(model=model, loader=dl_val, device=device, alpha_vec=alpha_vec)
        val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        val_map = float(macro_map_score(y_val_onehot, val_pred))
        print(
            f"[diag] epoch={epoch} val_map={val_map:.6f} pred_mean={float(np.mean(val_pred)):.6f} "
            f"pred_std={float(np.std(val_pred)):.6f} pred_p95={float(np.quantile(val_pred,0.95)):.6f}",
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

    val_pred, val_hier, val_flat = _predict_heads(model=model, loader=dl_val, device=device, alpha_vec=alpha_vec)
    test_pred, test_hier, test_flat = _predict_heads(model=model, loader=dl_test, device=device, alpha_vec=alpha_vec)
    val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    val_hier = np.nan_to_num(val_hier, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    val_flat = np.nan_to_num(val_flat, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    test_hier = np.nan_to_num(test_hier, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    test_flat = np.nan_to_num(test_flat, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    return val_pred, test_pred, val_hier, test_hier, val_flat, test_flat, float(best_map), int(best_epoch)


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

    seqs_train: list[np.ndarray] = []
    for tid in train_ids.tolist():
        raw = np.asarray(train_cache[int(tid)]["raw_features"], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 9:
            raise ValueError(f"track_id={tid} raw_features bad shape={raw.shape}, expected [T,9]")
        seqs_train.append(raw)

    seqs_test: list[np.ndarray] = []
    for tid in test_ids.tolist():
        raw = np.asarray(test_cache[int(tid)]["raw_features"], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 9:
            raise ValueError(f"test track_id={tid} raw_features bad shape={raw.shape}, expected [T,9]")
        seqs_test.append(raw)

    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = np.zeros((len(train_ids), len(CLASSES)), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    np.save(out_dir / "oof_targets.npy", y.astype(np.float32))

    class_to_group = np.full((len(CLASSES),), -1, dtype=np.int64)
    for gi, gname in enumerate(GROUP_NAMES):
        for cname in GROUP_TO_CLASSES[gname]:
            class_to_group[int(CLASS_TO_INDEX[cname])] = gi
    if np.any(class_to_group < 0):
        missing_idx = np.where(class_to_group < 0)[0].tolist()
        missing_classes = [CLASSES[i] for i in missing_idx]
        raise ValueError(f"class->group mapping incomplete: {missing_classes}")

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
    oof_hier = np.zeros((len(train_ids), len(CLASSES)), dtype=np.float32)
    oof_flat = np.zeros((len(train_ids), len(CLASSES)), dtype=np.float32)
    covered_mask = np.zeros((len(train_ids),), dtype=bool)
    test_accum = np.zeros((len(test_ids), len(CLASSES)), dtype=np.float32)
    test_accum_hier = np.zeros((len(test_ids), len(CLASSES)), dtype=np.float32)
    test_accum_flat = np.zeros((len(test_ids), len(CLASSES)), dtype=np.float32)
    fold_scores: list[float] = []
    fold_best_maps: list[float] = []
    fold_best_epochs: list[int] = []

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        print(f"[fold] {fold_id + 1}/{len(folds)} train={len(tr_idx)} val={len(va_idx)}", flush=True)
        seq_tr = [seqs_train[int(i)] for i in tr_idx]
        seq_va = [seqs_train[int(i)] for i in va_idx]
        raw_clip_lo, raw_clip_hi, raw_mean, raw_std = _fit_raw_scaler(
            seqs=seq_tr,
            q_low=float(args.raw_clip_quantile_low),
            q_high=float(args.raw_clip_quantile_high),
        )
        val_pred, test_pred, val_hier, test_hier, val_flat, test_flat, best_map, best_epoch = _fit_one_fold(
            seqs_train=seq_tr,
            y_train_idx=y_idx[tr_idx],
            seqs_val=seq_va,
            y_val_onehot=y[va_idx],
            seqs_test=seqs_test,
            raw_clip_lo=raw_clip_lo,
            raw_clip_hi=raw_clip_hi,
            raw_mean=raw_mean,
            raw_std=raw_std,
            args=args,
            device=device,
            class_to_group=class_to_group,
            sample_weights_train=(sample_weights[tr_idx] if sample_weights is not None else None),
        )
        oof[va_idx] = val_pred
        oof_hier[va_idx] = val_hier
        oof_flat[va_idx] = val_flat
        covered_mask[va_idx] = True
        test_accum += test_pred / float(len(folds))
        test_accum_hier += test_hier / float(len(folds))
        test_accum_flat += test_flat / float(len(folds))
        fold_score = float(macro_map_score(y[va_idx], val_pred))
        fold_scores.append(fold_score)
        fold_best_maps.append(float(best_map))
        fold_best_epochs.append(int(best_epoch))

    oof_path = art_dir / "hier_blend_oof.npy"
    test_path = art_dir / "hier_blend_test.npy"
    np.save(oof_path, oof.astype(np.float32))
    np.save(test_path, test_accum.astype(np.float32))
    np.save(art_dir / "hier_oof.npy", oof_hier.astype(np.float32))
    np.save(art_dir / "hier_test.npy", test_accum_hier.astype(np.float32))
    np.save(art_dir / "flat_oof.npy", oof_flat.astype(np.float32))
    np.save(art_dir / "flat_test.npy", test_accum_flat.astype(np.float32))

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
    sub_path = out_dir / "submission_hier.csv"
    sub.to_csv(sub_path, index=False)

    summary: dict[str, Any] = {
        "output_dir": str(out_dir),
        "seed": int(args.seed),
        "device": str(device),
        "requested_device": str(args.device),
        "n_splits": int(len(folds)),
        "seq_len": int(args.seq_len),
        "model_type": "hierarchical_transformer",
        "group_names": GROUP_NAMES,
        "group_to_classes": GROUP_TO_CLASSES,
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "num_layers": int(args.num_layers),
        "ffn_dim": int(args.ffn_dim),
        "transformer_multires": bool(int(args.transformer_multires)),
        "multires_pool_factor": int(args.multires_pool_factor),
        "physics_token_mode": str(args.physics_token_mode),
        "dropout": float(args.dropout),
        "group_loss_weight": float(args.group_loss_weight),
        "flat_loss_weight": float(args.flat_loss_weight),
        "blend_alpha": float(args.blend_alpha),
        "blend_alpha_by_class": str(args.blend_alpha_by_class),
        "flat_loss_class_weight": str(args.flat_loss_class_weight),
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
        "models": {
            "hierarchical_trajectory": {
                "type": "hierarchical_trajectory",
                "oof_path": str(oof_path.resolve()),
                "test_path": str(test_path.resolve()),
                "hier_oof_path": str((art_dir / "hier_oof.npy").resolve()),
                "flat_oof_path": str((art_dir / "flat_oof.npy").resolve()),
                "hier_test_path": str((art_dir / "hier_test.npy").resolve()),
                "flat_test_path": str((art_dir / "flat_test.npy").resolve()),
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

    print("=== HIERARCHICAL TRAJECTORY TRAIN COMPLETE ===", flush=True)
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
