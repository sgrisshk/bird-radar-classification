from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceConfig:
    seq_len: int = 128
    time_crop_min: float = 0.6
    p_time_reverse: float = 0.5
    norm_mode: str = "global_robust"  # one of: global_robust, per_track_robust, none
    global_median: np.ndarray | None = None  # shape [C]
    global_iqr: np.ndarray | None = None  # shape [C]
    clip_abs: float | None = 30.0
    keep_raw_channels: tuple[int, ...] | None = None  # select from [x,y,z,rcs,speed,vs,accel,curv,dt]
    robust_channels: tuple[int, ...] | None = None  # apply per-track robust norm only for selected channels
    delta_channels: tuple[str, ...] | None = None  # drcs, dspeed, dlogdt
    channel_dropout_p: float = 0.0
    time_dropout_p: float = 0.0
    channel_dropout_candidates: tuple[int, ...] | None = (3, 4, 6, 8)
    group_dropout_channels: tuple[tuple[int, ...], ...] | None = None
    group_dropout_probs: tuple[float, ...] | None = None
    log_dt: bool = False
    dt_eps: float = 1e-6
    delta_clip_abs: float | None = 10.0


def _uniform_sample_indices(n: int, L: int) -> np.ndarray:
    if n <= L:
        return np.arange(n, dtype=np.int64)
    idx = np.linspace(0, n - 1, num=L, dtype=np.float64)
    idx = np.round(idx).astype(np.int64)
    idx = np.maximum.accumulate(idx)
    idx[-1] = n - 1
    return idx


def _build_seq_no_interp(seq: np.ndarray, t: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(t)
    if n < 2:
        seq_out = np.zeros((9, L), dtype=np.float32)
        t_out = np.linspace(0.0, 1.0, L, dtype=np.float32)
        mask = np.zeros((1, L), dtype=np.float32)
        return seq_out, t_out, mask

    idx = _uniform_sample_indices(n, L)
    seq_s = seq[idx].astype(np.float32)
    t_s = t[idx].astype(np.float32)

    m = len(t_s)
    seq_out = np.zeros((L, 9), dtype=np.float32)
    t_out = np.zeros((L,), dtype=np.float32)
    mask = np.zeros((L,), dtype=np.float32)

    seq_out[:m] = seq_s
    t_out[:m] = t_s
    mask[:m] = 1.0

    t_real = t_out[:m]
    if t_real[-1] <= t_real[0]:
        t_norm = np.linspace(0.0, 1.0, m, dtype=np.float32)
    else:
        t_norm = (t_real - t_real[0]) / (t_real[-1] - t_real[0])
    t_out[:m] = t_norm
    if m < L:
        t_out[m:] = 0.0

    return seq_out.T.astype(np.float32), t_out.astype(np.float32), mask.reshape(1, -1).astype(np.float32)


def _prepare_track_arrays(cache_item: dict[str, Any], cfg: SequenceConfig) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(cache_item["raw_features"], dtype=np.float32)
    t = np.asarray(cache_item["times"], dtype=np.float32)
    n = int(min(len(raw), len(t)))
    raw = raw[:n]
    t = t[:n]
    if n < 2:
        raise ValueError("track has fewer than 2 points")

    # raw_features = [x, y, z, rcs, speed, vertical_speed, acceleration, curvature, dt]
    seq = raw[:, :9].astype(np.float32)
    if bool(cfg.log_dt):
        seq[:, 8] = np.log(np.clip(seq[:, 8], float(cfg.dt_eps), None)).astype(np.float32)
    keep = cfg.keep_raw_channels
    if keep is not None:
        if len(keep) == 0:
            raise ValueError("keep_raw_channels cannot be empty")
        keep_idx = np.array(sorted(set(int(v) for v in keep)), dtype=np.int64)
        if np.any(keep_idx < 0) or np.any(keep_idx >= seq.shape[1]):
            raise ValueError(f"keep_raw_channels contains invalid indices: {keep_idx.tolist()}")
        gate = np.zeros((seq.shape[1],), dtype=np.float32)
        gate[keep_idx] = 1.0
        seq = seq * gate[None, :]
    return seq, t


def _robust_norm_1d(x: np.ndarray) -> np.ndarray:
    med = float(np.median(x))
    q25 = float(np.quantile(x, 0.25))
    q75 = float(np.quantile(x, 0.75))
    iqr = q75 - q25
    if iqr < 1e-6:
        iqr = 1.0
    return ((x - med) / iqr).astype(np.float32)


def _apply_per_track_robust_channels(seq: np.ndarray, mask: np.ndarray, channels: np.ndarray) -> np.ndarray:
    out = seq.copy()
    valid_idx = np.where(mask[0] > 0.0)[0]
    if len(valid_idx) < 2:
        return out
    for ch in channels:
        vals = out[ch, valid_idx]
        out[ch, valid_idx] = _robust_norm_1d(vals)
    return out


def _apply_time_dropout(seq: np.ndarray, mask: np.ndarray, p: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if p <= 0.0:
        return seq, mask
    out_seq = seq.copy()
    out_mask = mask.copy()
    valid = np.where(out_mask[0] > 0.0)[0]
    if len(valid) < 6:
        return out_seq, out_mask
    drop = rng.random(len(valid)) < p
    if int(drop.sum()) >= len(valid) - 2:
        return out_seq, out_mask
    idx = valid[drop]
    if len(idx) == 0:
        return out_seq, out_mask
    out_seq[:, idx] = 0.0
    out_mask[0, idx] = 0.0
    return out_seq, out_mask


def _apply_channel_dropout(seq: np.ndarray, p: float, candidates: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if p <= 0.0 or len(candidates) == 0:
        return seq
    out = seq.copy()
    if float(rng.random()) < p:
        ch = int(candidates[int(rng.integers(0, len(candidates)))])
        out[ch, :] = 0.0
    return out


def _apply_group_channel_dropout(
    seq: np.ndarray,
    groups: tuple[tuple[int, ...], ...],
    probs: tuple[float, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    if not groups or not probs:
        return seq
    out = seq.copy()
    for g, p in zip(groups, probs):
        p_f = float(p)
        if p_f <= 0.0:
            continue
        if float(rng.random()) >= p_f:
            continue
        if not g:
            continue
        ch_idx = np.array([int(v) for v in g], dtype=np.int64)
        ch_idx = ch_idx[(ch_idx >= 0) & (ch_idx < out.shape[0])]
        if len(ch_idx) == 0:
            continue
        out[ch_idx, :] = 0.0
    return out


def _build_delta_channels(
    seq: np.ndarray,
    mask: np.ndarray,
    delta_names: tuple[str, ...],
    delta_clip_abs: float | None,
) -> np.ndarray:
    if not delta_names:
        return np.zeros((0, seq.shape[1]), dtype=np.float32)
    valid_idx = np.where(mask[0] > 0.0)[0]
    deltas: list[np.ndarray] = []

    for name in delta_names:
        if name == "drcs":
            base = seq[3]
            d = np.diff(base, prepend=base[:1]).astype(np.float32)
        elif name == "dspeed":
            base = seq[4]
            d = np.diff(base, prepend=base[:1]).astype(np.float32)
        elif name == "dlogdt":
            base = np.log(np.clip(seq[8], 1e-6, None))
            d = np.diff(base, prepend=base[:1]).astype(np.float32)
        else:
            raise ValueError(f"unknown delta channel: {name}")

        if len(valid_idx) >= 2:
            d_valid = d[valid_idx]
            d[valid_idx] = _robust_norm_1d(d_valid)
        d[mask[0] <= 0.0] = 0.0
        if delta_clip_abs is not None and float(delta_clip_abs) > 0:
            d = np.clip(d, -float(delta_clip_abs), float(delta_clip_abs)).astype(np.float32)
        deltas.append(d)
    return np.stack(deltas, axis=0).astype(np.float32)


def _apply_time_crop(seq: np.ndarray, t: np.ndarray, cfg: SequenceConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = len(t)
    if n < 4:
        return seq, t
    min_len = max(3, int(n * cfg.time_crop_min))
    if min_len >= n:
        return seq, t
    length = int(rng.integers(min_len, n + 1))
    start_max = n - length
    start = int(rng.integers(0, start_max + 1)) if start_max > 0 else 0
    end = start + length
    return seq[start:end], t[start:end]


def _robust_norm(seq_new: np.ndarray) -> np.ndarray:
    # seq_new: [C, L]
    med = np.median(seq_new, axis=1, keepdims=True)
    q25 = np.quantile(seq_new, 0.25, axis=1, keepdims=True)
    q75 = np.quantile(seq_new, 0.75, axis=1, keepdims=True)
    iqr = q75 - q25
    iqr = np.where(iqr < 1e-6, 1.0, iqr)
    return ((seq_new - med) / iqr).astype(np.float32)


def _global_robust_norm(seq_new: np.ndarray, cfg: SequenceConfig) -> np.ndarray:
    med = cfg.global_median
    iqr = cfg.global_iqr
    if med is None or iqr is None:
        return _robust_norm(seq_new)
    med = np.asarray(med, dtype=np.float32).reshape(-1, 1)
    iqr = np.asarray(iqr, dtype=np.float32).reshape(-1, 1)
    iqr = np.where(iqr < 1e-6, 1.0, iqr)
    return ((seq_new - med) / iqr).astype(np.float32)


def build_sequence_tensor(
    cache_item: dict[str, Any],
    cfg: SequenceConfig,
    augment: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    seq, t = _prepare_track_arrays(cache_item, cfg)

    if augment:
        seq, t = _apply_time_crop(seq, t, cfg, rng)
        if cfg.p_time_reverse > 0 and float(rng.random()) < cfg.p_time_reverse:
            seq = seq[::-1].copy()
            t = t[::-1].copy()

    seq_base, t_new, mask = _build_seq_no_interp(seq, t, L=int(cfg.seq_len))
    if augment:
        seq_base, mask = _apply_time_dropout(seq_base, mask, float(cfg.time_dropout_p), rng)
        cand = np.array(cfg.channel_dropout_candidates or [], dtype=np.int64)
        cand = cand[(cand >= 0) & (cand < seq_base.shape[0])]
        seq_base = _apply_channel_dropout(seq_base, float(cfg.channel_dropout_p), cand, rng)
        groups = tuple(cfg.group_dropout_channels or ())
        probs = tuple(cfg.group_dropout_probs or ())
        if groups and probs:
            seq_base = _apply_group_channel_dropout(seq_base, groups=groups, probs=probs, rng=rng)

    norm_mode = str(getattr(cfg, "norm_mode", "global_robust")).lower()
    if norm_mode == "global_robust":
        seq_main = _global_robust_norm(seq_base, cfg)
    elif norm_mode == "per_track_robust":
        seq_main = _robust_norm(seq_base)
    elif norm_mode == "none":
        seq_main = seq_base.astype(np.float32)
    else:
        raise ValueError(f"unknown sequence norm_mode: {norm_mode}")

    robust_channels = np.array(cfg.robust_channels or [], dtype=np.int64)
    robust_channels = robust_channels[(robust_channels >= 0) & (robust_channels < seq_main.shape[0])]
    if len(robust_channels) > 0:
        seq_main = _apply_per_track_robust_channels(seq_main, mask, robust_channels)

    delta_names = tuple(cfg.delta_channels or ())
    seq_delta = _build_delta_channels(seq_base, mask, delta_names, cfg.delta_clip_abs)
    if seq_delta.shape[0] > 0:
        seq_main = np.concatenate([seq_main, seq_delta], axis=0)

    clip_abs = getattr(cfg, "clip_abs", None)
    if clip_abs is not None and float(clip_abs) > 0:
        seq_main = np.clip(seq_main, -float(clip_abs), float(clip_abs)).astype(np.float32)

    seq_with_mask = np.concatenate([seq_main, mask], axis=0)
    return seq_with_mask.astype(np.float32), t_new.astype(np.float32)


class RadarHybridDataset(Dataset):
    def __init__(
        self,
        track_ids: np.ndarray,
        tabular: np.ndarray,
        cache: dict[int, dict[str, Any]],
        targets: np.ndarray | None,
        domain_label: int,
        seq_cfg: SequenceConfig,
        augment: bool,
        seed: int,
    ) -> None:
        self.track_ids = track_ids.astype(np.int64)
        self.tabular = tabular.astype(np.float32)
        self.cache = cache
        self.targets = targets.astype(np.float32) if targets is not None else None
        self.domain_label = float(domain_label)
        self.seq_cfg = seq_cfg
        self.augment = augment
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        track_id = int(self.track_ids[idx])
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch + idx)
        seq, t_norm = build_sequence_tensor(
            self.cache[track_id],
            cfg=self.seq_cfg,
            augment=self.augment,
            rng=rng,
        )

        item = {
            "track_id": torch.tensor(track_id, dtype=torch.long),
            "seq": torch.from_numpy(seq),
            "time_norm": torch.from_numpy(t_norm.astype(np.float32)),
            "tab": torch.from_numpy(self.tabular[idx]),
            "domain": torch.tensor(self.domain_label, dtype=torch.float32),
        }
        if self.targets is not None:
            item["target"] = torch.from_numpy(self.targets[idx])
        return item
