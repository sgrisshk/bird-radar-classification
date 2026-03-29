from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.preprocessing import build_track_cache, load_track_cache, save_track_cache
from src.redesign.dataset import SequenceConfig, build_sequence_tensor
from src.redesign.model import ARCH_SPECS, TransformerBranch
from src.redesign.utils import dump_json, set_seed


def _resolve_device(requested: str) -> tuple[torch.device, str]:
    req = str(requested).lower().strip()
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), req
        return torch.device("cpu"), req
    if req == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps"), req
        return torch.device("cpu"), req
    return torch.device("cpu"), req


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict[str, Any]]:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


_SEQ_CHANNEL_NAME_TO_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
    "rcs": 3,
    "speed": 4,
    "vertical_speed": 5,
    "accel": 6,
    "acceleration": 6,
    "curvature": 7,
    "dt": 8,
}


def _parse_seq_channel_indices(values: Any) -> tuple[int, ...]:
    if values is None:
        return tuple()
    if isinstance(values, (str, int)):
        values = [values]
    out: list[int] = []
    for v in values:
        if isinstance(v, str):
            key = v.strip().lower()
            if key in _SEQ_CHANNEL_NAME_TO_INDEX:
                out.append(int(_SEQ_CHANNEL_NAME_TO_INDEX[key]))
            else:
                out.append(int(key))
        else:
            out.append(int(v))
    return tuple(sorted(set(out)))


def _parse_delta_channels(values: Any) -> tuple[str, ...]:
    if values is None:
        return tuple()
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    for v in values:
        k = str(v).strip().lower()
        if k:
            out.append(k)
    return tuple(out)


def _parse_group_dropout(
    cfg_sequence: dict[str, Any],
) -> tuple[tuple[tuple[int, ...], ...], tuple[float, ...]]:
    groups_raw = cfg_sequence.get("group_dropout_channels", [])
    probs_raw = cfg_sequence.get("group_dropout_probs", [])
    groups: list[tuple[int, ...]] = []
    probs: list[float] = []
    if groups_raw and probs_raw:
        for g_raw, p_raw in zip(groups_raw, probs_raw):
            g_idx = _parse_seq_channel_indices(g_raw)
            if len(g_idx) == 0:
                continue
            p = float(p_raw)
            if p <= 0.0:
                continue
            groups.append(g_idx)
            probs.append(p)
    return tuple(groups), tuple(probs)


def _compute_global_seq_stats(
    cache: dict[int, dict[str, Any]],
    channels: int = 9,
    log_dt: bool = False,
    dt_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    for item in cache.values():
        raw = np.asarray(item.get("raw_features", []), dtype=np.float32)
        if raw.ndim != 2 or raw.shape[0] == 0:
            continue
        c = min(int(channels), int(raw.shape[1]))
        chunks.append(raw[:, :c])
    if not chunks:
        median = np.zeros((channels,), dtype=np.float32)
        iqr = np.ones((channels,), dtype=np.float32)
        return median, iqr
    x = np.concatenate(chunks, axis=0)
    if x.shape[1] < channels:
        pad = np.zeros((x.shape[0], channels - x.shape[1]), dtype=np.float32)
        x = np.concatenate([x, pad], axis=1)
    if bool(log_dt) and x.shape[1] > 8:
        x[:, 8] = np.log(np.clip(x[:, 8], float(dt_eps), None)).astype(np.float32)
    median = np.median(x, axis=0).astype(np.float32)
    q25 = np.quantile(x, 0.25, axis=0).astype(np.float32)
    q75 = np.quantile(x, 0.75, axis=0).astype(np.float32)
    iqr = (q75 - q25).astype(np.float32)
    iqr = np.where(iqr < 1e-6, 1.0, iqr).astype(np.float32)
    return median, iqr


class SSLSequenceDataset(Dataset):
    def __init__(
        self,
        track_ids: np.ndarray,
        cache: dict[int, dict[str, Any]],
        seq_cfg: SequenceConfig,
        augment: bool,
        seed: int,
    ) -> None:
        self.track_ids = track_ids.astype(np.int64)
        self.cache = cache
        self.seq_cfg = seq_cfg
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tid = int(self.track_ids[idx])
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch + idx)
        seq, t_norm = build_sequence_tensor(
            self.cache[tid],
            cfg=self.seq_cfg,
            augment=self.augment,
            rng=rng,
        )
        return {
            "seq": torch.from_numpy(seq),
            "time_norm": torch.from_numpy(t_norm.astype(np.float32)),
        }


class MaskedReconTransformer(nn.Module):
    def __init__(
        self,
        arch_name: str,
        seq_in_ch: int,
        dropout: float,
        pooling: str = "mean",
        gem_p: float = 3.0,
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
        transformer_ffn_dim: int | None = None,
    ) -> None:
        super().__init__()
        spec = ARCH_SPECS[arch_name]
        self.seq_in_ch = int(seq_in_ch)
        self.trm = TransformerBranch(
            in_ch=self.seq_in_ch,
            d_model=spec.transformer_dim,
            nhead=int(transformer_nhead),
            num_layers=int(transformer_num_layers),
            dropout=float(dropout),
            ffn_dim=transformer_ffn_dim,
            pooling=pooling,
            gem_p=gem_p,
        )
        self.recon_head = nn.Sequential(
            nn.Linear(spec.transformer_dim, spec.transformer_dim),
            nn.GELU(),
            nn.Linear(spec.transformer_dim, self.seq_in_ch),
        )

    def forward(self, seq: torch.Tensor, time_norm: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, float]:
        # seq: [B, C+1, L]
        seq_feat = seq[:, : self.seq_in_ch, :]
        valid_mask = seq[:, self.seq_in_ch : self.seq_in_ch + 1, :]
        bsz, _, seqlen = seq_feat.shape

        valid = (valid_mask.squeeze(1) > 0.0)  # [B, L]
        rnd = torch.rand((bsz, seqlen), device=seq.device)
        masked_pos = (rnd < float(mask_ratio)) & valid
        # Ensure at least one masked valid position per row when possible.
        for b in range(bsz):
            if bool(valid[b].any().item()) and (not bool(masked_pos[b].any().item())):
                idx = torch.where(valid[b])[0]
                j = torch.randint(0, int(idx.numel()), (1,), device=seq.device)
                masked_pos[b, idx[j]] = True

        x_in = seq_feat.transpose(1, 2).contiguous()  # [B, L, C]
        x_in = x_in.clone()
        x_in[masked_pos] = 0.0
        x_in = x_in.transpose(1, 2).contiguous()  # [B, C, L]

        enc = self.trm.encode_sequence(x_in, time_norm, valid_mask)  # [B, L, D]
        recon = self.recon_head(enc)  # [B, L, C]
        target = seq_feat.transpose(1, 2).contiguous()  # [B, L, C]

        mask_f = masked_pos.unsqueeze(-1).to(dtype=recon.dtype)  # [B, L, 1]
        err = (recon - target) ** 2
        denom = torch.clamp(mask_f.sum() * float(self.seq_in_ch), min=1.0)
        loss = (err * mask_f).sum() / denom
        masked_frac = float(masked_pos.float().mean().detach().cpu().item())
        return loss, masked_frac


@dataclass
class SSLResult:
    summary: dict[str, Any]


def run_ssl_pretrain(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(cfg["seed"]))
    root = Path(cfg["paths"]["project_root"]).resolve()
    data_dir = Path(cfg["paths"]["data_dir"]).resolve()
    cache_dir = Path(cfg["paths"]["cache_dir"]).resolve()
    out_dir = Path(cfg["paths"]["output_dir"]).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")
    full_cache: dict[int, dict[str, Any]] = {}
    full_cache.update(train_cache)
    full_cache.update(test_cache)

    log_dt = bool(cfg.get("sequence", {}).get("log_dt", False))
    dt_eps = float(cfg.get("sequence", {}).get("dt_eps", 1e-6))
    seq_median, seq_iqr = _compute_global_seq_stats(full_cache, channels=9, log_dt=log_dt, dt_eps=dt_eps)
    group_dropout_channels, group_dropout_probs = _parse_group_dropout(cfg.get("sequence", {}))
    seq_cfg = SequenceConfig(
        seq_len=int(cfg["sequence"]["seq_len"]),
        time_crop_min=float(cfg["aug"]["time_crop_min"]),
        p_time_reverse=float(cfg["aug"]["time_reverse_prob"]),
        norm_mode=str(cfg["sequence"].get("norm_mode", "global_robust")),
        global_median=seq_median,
        global_iqr=seq_iqr,
        clip_abs=float(cfg["sequence"].get("clip_abs", 30.0)),
        keep_raw_channels=_parse_seq_channel_indices(cfg["sequence"].get("keep_raw_channels", [])) or None,
        robust_channels=_parse_seq_channel_indices(cfg["sequence"].get("robust_channels", [])) or None,
        delta_channels=_parse_delta_channels(cfg["sequence"].get("delta_channels", [])) or None,
        channel_dropout_p=float(cfg["sequence"].get("channel_dropout_p", 0.0)),
        time_dropout_p=float(cfg["sequence"].get("time_dropout_p", 0.0)),
        channel_dropout_candidates=_parse_seq_channel_indices(
            cfg["sequence"].get("channel_dropout_candidates", [3, 4, 6, 8])
        )
        or None,
        group_dropout_channels=group_dropout_channels or None,
        group_dropout_probs=group_dropout_probs or None,
        log_dt=log_dt,
        dt_eps=dt_eps,
        delta_clip_abs=float(cfg["sequence"].get("delta_clip_abs", 10.0)),
    )
    seq_in_ch = 9 + len(seq_cfg.delta_channels or ())

    ssl_cfg = cfg.get("ssl", {}) or {}
    requested_device = str(cfg.get("train", {}).get("device", "cpu"))
    device, requested = _resolve_device(requested_device)
    if str(device) != requested:
        print(
            f"[ssl] requested device='{requested}' unavailable, resolved='{device}' (fallback)",
            flush=True,
        )

    arch_name = str(cfg.get("model", {}).get("architectures", ["transformer_heavy"])[0])
    if arch_name not in ARCH_SPECS:
        raise ValueError(f"unknown architecture '{arch_name}'")
    model = MaskedReconTransformer(
        arch_name=arch_name,
        seq_in_ch=seq_in_ch,
        dropout=float(cfg.get("model", {}).get("seq_dropout", 0.2)),
        pooling=str(cfg.get("model", {}).get("pooling", "mean")),
        gem_p=float(cfg.get("model", {}).get("gem_p", 3.0)),
        transformer_nhead=int(cfg.get("model", {}).get("transformer_nhead", 4)),
        transformer_num_layers=int(cfg.get("model", {}).get("transformer_num_layers", 2)),
        transformer_ffn_dim=(
            int(cfg.get("model", {}).get("transformer_ffn_dim"))
            if cfg.get("model", {}).get("transformer_ffn_dim", None) is not None
            else None
        ),
    ).to(device)

    ids_all = np.concatenate(
        [
            train_df["track_id"].to_numpy(dtype=np.int64),
            test_df["track_id"].to_numpy(dtype=np.int64),
        ],
        axis=0,
    )
    ds = SSLSequenceDataset(
        track_ids=ids_all,
        cache=full_cache,
        seq_cfg=seq_cfg,
        augment=bool(ssl_cfg.get("augment", True)),
        seed=int(cfg["seed"]),
    )
    dl = DataLoader(
        ds,
        batch_size=int(ssl_cfg.get("batch_size", cfg.get("train", {}).get("batch_size", 64))),
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(ssl_cfg.get("lr", 2e-4)),
        weight_decay=float(ssl_cfg.get("weight_decay", 0.05)),
    )
    mask_ratio = float(ssl_cfg.get("mask_ratio", 0.4))
    epochs = int(ssl_cfg.get("epochs", 8))
    grad_clip = float(ssl_cfg.get("grad_clip", cfg.get("train", {}).get("grad_clip", 1.0)))

    loss_hist: list[float] = []
    for epoch in range(epochs):
        ds.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        fracs: list[float] = []
        bad = 0
        for batch in dl:
            seq = batch["seq"].to(device)
            time_norm = batch["time_norm"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, frac = model(seq, time_norm, mask_ratio=mask_ratio)
            if not torch.isfinite(loss):
                bad += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            fracs.append(float(frac))
        mean_loss = float(np.mean(losses)) if losses else float("inf")
        mean_frac = float(np.mean(fracs)) if fracs else 0.0
        loss_hist.append(mean_loss)
        print(
            f"[ssl] epoch={epoch} recon_loss={mean_loss:.6f} masked_frac={mean_frac:.4f} bad_batches={bad}",
            flush=True,
        )

    ckpt_path = out_dir / "ssl_pretrained_encoder.pt"
    torch.save(
        {
            "arch_name": arch_name,
            "seq_in_ch": int(seq_in_ch),
            "trm_state_dict": model.trm.state_dict(),
            "ssl_cfg": {
                "mask_ratio": mask_ratio,
                "epochs": epochs,
                "lr": float(ssl_cfg.get("lr", 2e-4)),
            },
        },
        ckpt_path,
    )

    summary: dict[str, Any] = {
        "task": "pretrain_masked_recon",
        "project_root": str(root),
        "output_dir": str(out_dir),
        "requested_device": requested,
        "device": str(device),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_total_tracks": int(len(ids_all)),
        "seq_len": int(seq_cfg.seq_len),
        "seq_in_channels": int(seq_in_ch),
        "arch_name": arch_name,
        "transformer_nhead": int(cfg.get("model", {}).get("transformer_nhead", 4)),
        "transformer_num_layers": int(cfg.get("model", {}).get("transformer_num_layers", 2)),
        "transformer_ffn_dim": (
            int(cfg.get("model", {}).get("transformer_ffn_dim"))
            if cfg.get("model", {}).get("transformer_ffn_dim", None) is not None
            else None
        ),
        "ssl": {
            "epochs": epochs,
            "batch_size": int(ssl_cfg.get("batch_size", cfg.get("train", {}).get("batch_size", 64))),
            "lr": float(ssl_cfg.get("lr", 2e-4)),
            "weight_decay": float(ssl_cfg.get("weight_decay", 0.05)),
            "mask_ratio": mask_ratio,
            "augment": bool(ssl_cfg.get("augment", True)),
            "loss_history": [float(v) for v in loss_hist],
            "final_recon_loss": float(loss_hist[-1]) if loss_hist else None,
        },
        "checkpoint_path": str(ckpt_path.resolve()),
    }
    dump_json(out_dir / "oof_summary.json", summary)
    print("=== SSL PRETRAIN COMPLETE ===", flush=True)
    print(f"output_dir: {out_dir}", flush=True)
    print(f"checkpoint: {ckpt_path}", flush=True)
    print(f"oof_summary: {out_dir / 'oof_summary.json'}", flush=True)
    return summary
