from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lam * grad_output, None


def grad_reverse(x: torch.Tensor, lam: float) -> torch.Tensor:
    return GradientReversalFn.apply(x, lam)


def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    w = mask.clamp_min(0.0)
    return (x * w).sum(dim=dim) / (w.sum(dim=dim) + 1e-6)


def _masked_max_pool(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    valid = mask > 0.0
    x_masked = x.masked_fill(~valid, float("-inf"))
    pooled = x_masked.max(dim=dim).values
    pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
    return pooled


def _masked_meanmax_pool(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mean = _masked_mean_pool(x, mask, dim=dim)
    mx = _masked_max_pool(x, mask, dim=dim)
    return torch.cat([mean, mx], dim=-1)


def _masked_gem_pool(
    x: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
    p: torch.Tensor | float,
    eps: float = 1e-6,
) -> torch.Tensor:
    w = mask.clamp_min(0.0)
    if isinstance(p, torch.Tensor):
        p_t = p.to(dtype=x.dtype, device=x.device)
    else:
        p_t = torch.tensor(float(p), dtype=x.dtype, device=x.device)
    # Keep p > 1 for numerical stability and smoother gradients.
    p_pos = F.softplus(p_t) + 1.0
    x_abs = x.abs().clamp_min(eps)
    x_pow = torch.pow(x_abs, p_pos)
    pooled_mag = (x_pow * w).sum(dim=dim) / (w.sum(dim=dim) + eps)
    pooled_mag = torch.pow(pooled_mag.clamp_min(eps), 1.0 / p_pos)
    pooled_sign = torch.sign(_masked_mean_pool(x, w, dim=dim))
    return pooled_sign * pooled_mag


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, groups: int = 8) -> None:
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation)
        self.gn1 = nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation)
        self.gn2 = nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h = self.drop(h)
        return F.gelu(h + x)


class TemporalCNNBranch(nn.Module):
    def __init__(
        self,
        in_ch: int,
        channels: int,
        blocks: int = 5,
        kernel_size: int = 5,
        dropout: float = 0.2,
        pooling: str = "mean",
        gem_p: float = 3.0,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_ch, channels, kernel_size=1)
        self.pooling = str(pooling).lower().strip()
        self.gem_p = nn.Parameter(torch.tensor(float(gem_p), dtype=torch.float32))
        layers = []
        for i in range(blocks):
            layers.append(
                ResidualTCNBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, seq9: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(seq9)
        x = self.net(x)
        if self.pooling == "max":
            return _masked_max_pool(x, mask, dim=-1)
        if self.pooling == "meanmax":
            return _masked_meanmax_pool(x, mask, dim=-1)
        if self.pooling == "gem":
            return _masked_gem_pool(x, mask, dim=-1, p=self.gem_p)
        return _masked_mean_pool(x, mask, dim=-1)


class TransformerBranch(nn.Module):
    def __init__(
        self,
        in_ch: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.2,
        ffn_dim: int | None = None,
        pooling: str = "mean",
        gem_p: float = 3.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_ch, d_model)
        self.pooling = str(pooling).lower().strip()
        self.gem_p = nn.Parameter(torch.tensor(float(gem_p), dtype=torch.float32))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(ffn_dim) if (ffn_dim is not None and int(ffn_dim) > 0) else d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)

    def _time_encoding(self, t: torch.Tensor, d_model: int) -> torch.Tensor:
        # t: [B, L], normalized to [0,1]
        half = d_model // 2
        div = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=t.device, dtype=t.dtype)
            / max(half, 1)
        )
        x = t.unsqueeze(-1) * div.unsqueeze(0).unsqueeze(0)
        sin = torch.sin(x)
        cos = torch.cos(x)
        pos = torch.cat([sin, cos], dim=-1)
        if pos.size(-1) < d_model:
            pos = F.pad(pos, (0, d_model - pos.size(-1)))
        return pos[:, :, :d_model]

    def encode_sequence(self, seq9: torch.Tensor, time_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # seq9: [B, C, L]
        x = seq9.transpose(1, 2)  # [B, L, 9]
        x = self.input_proj(x)
        x = x + self._time_encoding(time_norm, x.shape[-1])
        key_padding_mask = (mask.squeeze(1) <= 0.0)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x

    def pool_encoded(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask.transpose(1, 2).clamp_min(0.0)  # [B, L, 1]
        if self.pooling == "max":
            return _masked_max_pool(x, w, dim=1)
        if self.pooling == "meanmax":
            return _masked_meanmax_pool(x, w, dim=1)
        if self.pooling == "gem":
            return _masked_gem_pool(x, w, dim=1, p=self.gem_p)
        return _masked_mean_pool(x, w, dim=1)

    def forward(self, seq9: torch.Tensor, time_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.encode_sequence(seq9, time_norm, mask)
        return self.pool_encoded(x, mask)


class TabularMLPBranch(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ArchitectureSpec:
    name: str
    tcn_channels: int
    transformer_dim: int
    fusion_dim: int


ARCH_SPECS: dict[str, ArchitectureSpec] = {
    "tcn_heavy": ArchitectureSpec(name="tcn_heavy", tcn_channels=256, transformer_dim=128, fusion_dim=256),
    "transformer_heavy": ArchitectureSpec(name="transformer_heavy", tcn_channels=128, transformer_dim=192, fusion_dim=256),
    "transformer_tiny": ArchitectureSpec(name="transformer_tiny", tcn_channels=64, transformer_dim=64, fusion_dim=128),
}


class MultiBranchRadarModel(nn.Module):
    def __init__(
        self,
        tab_dim: int,
        num_classes: int,
        arch_name: str = "tcn_heavy",
        seq_in_ch: int = 9,
        seq_dropout: float = 0.2,
        tab_dropout: float = 0.3,
        pooling: str = "mean",
        gem_p: float = 3.0,
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
        transformer_ffn_dim: int | None = None,
    ) -> None:
        super().__init__()
        if arch_name not in ARCH_SPECS:
            raise ValueError(f"unknown architecture: {arch_name}")
        spec = ARCH_SPECS[arch_name]
        self.seq_in_ch = int(seq_in_ch)
        pool_mult = 2 if str(pooling).lower().strip() == "meanmax" else 1
        self.tcn_dim = int(spec.tcn_channels) * pool_mult
        self.trm_dim = int(spec.transformer_dim) * pool_mult
        self.tab_dim = 64

        self.tcn = TemporalCNNBranch(
            in_ch=seq_in_ch,
            channels=spec.tcn_channels,
            blocks=5,
            kernel_size=5,
            dropout=seq_dropout,
            pooling=pooling,
            gem_p=gem_p,
        )
        self.trm = TransformerBranch(
            in_ch=seq_in_ch,
            d_model=spec.transformer_dim,
            nhead=int(transformer_nhead),
            num_layers=int(transformer_num_layers),
            dropout=seq_dropout,
            ffn_dim=transformer_ffn_dim,
            pooling=pooling,
            gem_p=gem_p,
        )
        self.tab = TabularMLPBranch(in_dim=tab_dim, dropout=tab_dropout)

        fusion_in = self.tcn_dim + self.trm_dim + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, spec.fusion_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.cls_head = nn.Linear(spec.fusion_dim, num_classes)

        self.domain_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self.domain_head_fused = nn.Sequential(
            nn.Linear(spec.fusion_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        seq: torch.Tensor,
        tab: torch.Tensor,
        time_norm: torch.Tensor,
        grl_lambda: float = 0.0,
        use_seq: bool = True,
        use_tab: bool = True,
    ) -> dict[str, torch.Tensor]:
        # seq: [B, 10, L], first 9 channels + mask channel
        seq9 = seq[:, : self.seq_in_ch, :]
        mask = seq[:, self.seq_in_ch : self.seq_in_ch + 1, :]
        bsz = seq.shape[0]

        if use_seq:
            tcn_emb = self.tcn(seq9, mask)
            trm_emb = self.trm(seq9, time_norm, mask)
        else:
            tcn_emb = torch.zeros((bsz, self.tcn_dim), dtype=seq.dtype, device=seq.device)
            trm_emb = torch.zeros((bsz, self.trm_dim), dtype=seq.dtype, device=seq.device)

        if use_tab:
            tab_emb = self.tab(tab)
        else:
            tab_emb = torch.zeros((bsz, self.tab_dim), dtype=tab.dtype, device=tab.device)

        fused = torch.cat([tcn_emb, trm_emb, tab_emb], dim=1)
        fused = self.fusion(fused)
        logits = self.cls_head(fused)

        rev_tab = grad_reverse(tab_emb, grl_lambda)
        rev_fused = grad_reverse(fused, grl_lambda)
        domain_logits_tab = self.domain_head(rev_tab).squeeze(-1)
        domain_logits_fused = self.domain_head_fused(rev_fused).squeeze(-1)

        return {
            "logits": logits,
            "domain_logits_tab": domain_logits_tab,
            "domain_logits_fused": domain_logits_fused,
            "tab_emb": tab_emb,
            "fused": fused,
        }


class SequenceOnlyRadarModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        arch_name: str = "tcn_heavy",
        seq_in_ch: int = 9,
        seq_dropout: float = 0.2,
        pooling: str = "mean",
        seq_branch: str = "tcn_trm",
        gem_p: float = 3.0,
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
        transformer_ffn_dim: int | None = None,
    ) -> None:
        super().__init__()
        if arch_name not in ARCH_SPECS:
            raise ValueError(f"unknown architecture: {arch_name}")
        spec = ARCH_SPECS[arch_name]
        self.seq_in_ch = int(seq_in_ch)
        self.branch_mode = str(seq_branch).lower().strip()
        valid_modes = {"tcn_only", "transformer_only", "tcn_trm"}
        if self.branch_mode not in valid_modes:
            raise ValueError(f"unknown seq_branch: {self.branch_mode}; expected one of {sorted(valid_modes)}")
        self.use_tcn = self.branch_mode in {"tcn_only", "tcn_trm"}
        self.use_transformer = self.branch_mode in {"transformer_only", "tcn_trm"}
        pool_mult = 2 if str(pooling).lower().strip() == "meanmax" else 1
        self.tcn_out_dim = (int(spec.tcn_channels) * pool_mult) if self.use_tcn else 0
        self.trm_out_dim = (int(spec.transformer_dim) * pool_mult) if self.use_transformer else 0

        self.tcn = None
        if self.use_tcn:
            self.tcn = TemporalCNNBranch(
                in_ch=seq_in_ch,
                channels=spec.tcn_channels,
                blocks=5,
                kernel_size=5,
                dropout=seq_dropout,
                pooling=pooling,
                gem_p=gem_p,
            )
        self.trm = None
        if self.use_transformer:
            self.trm = TransformerBranch(
                in_ch=seq_in_ch,
                d_model=spec.transformer_dim,
                nhead=int(transformer_nhead),
                num_layers=int(transformer_num_layers),
                dropout=seq_dropout,
                ffn_dim=transformer_ffn_dim,
                pooling=pooling,
                gem_p=gem_p,
            )
        fusion_in = self.tcn_out_dim + self.trm_out_dim
        if fusion_in <= 0:
            raise ValueError("sequence-only model has zero fusion input; enable at least one seq branch")
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, spec.fusion_dim),
            nn.GELU(),
            nn.Dropout(seq_dropout),
        )
        self.cls_head = nn.Linear(spec.fusion_dim, num_classes)

    def forward(
        self,
        seq: torch.Tensor,
        tab: torch.Tensor,  # kept for API compatibility
        time_norm: torch.Tensor,
        grl_lambda: float = 0.0,  # kept for API compatibility
        use_seq: bool = True,
        use_tab: bool = False,
    ) -> dict[str, torch.Tensor]:
        del tab, grl_lambda, use_tab
        seq9 = seq[:, : self.seq_in_ch, :]
        mask = seq[:, self.seq_in_ch : self.seq_in_ch + 1, :]
        bsz = seq.shape[0]
        parts: list[torch.Tensor] = []
        if self.use_tcn:
            if use_seq and self.tcn is not None:
                parts.append(self.tcn(seq9, mask))
            else:
                parts.append(torch.zeros((bsz, self.tcn_out_dim), dtype=seq.dtype, device=seq.device))
        if self.use_transformer:
            if use_seq and self.trm is not None:
                parts.append(self.trm(seq9, time_norm, mask))
            else:
                parts.append(torch.zeros((bsz, self.trm_out_dim), dtype=seq.dtype, device=seq.device))
        fused_in = torch.cat(parts, dim=1)
        fused = self.fusion(fused_in)
        logits = self.cls_head(fused)
        dummy_domain = torch.zeros((bsz,), dtype=logits.dtype, device=logits.device)
        return {
            "logits": logits,
            "domain_logits_tab": dummy_domain,
            "domain_logits_fused": dummy_domain,
            "tab_emb": dummy_domain.unsqueeze(1),
            "fused": fused,
        }
