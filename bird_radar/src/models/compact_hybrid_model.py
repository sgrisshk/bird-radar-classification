from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn = self.score(x).squeeze(-1)
        if padding_mask is not None:
            attn = attn.masked_fill(padding_mask, float("-inf"))
            all_pad = padding_mask.all(dim=1)
            if all_pad.any():
                attn[all_pad] = 0.0
        w = torch.softmax(attn, dim=1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)


class ConvBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)
        y = self.conv(y)
        y = self.act(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)
        return self.norm(residual + y)


class CompactHybridModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 9,
        d_model: int = 96,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 192,
        dropout: float = 0.3,
        num_classes: int = 9,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv_block = ConvBlock(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = AttentionPooling(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.conv_block(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        pooled = self.pool(x, padding_mask=padding_mask)
        return self.head(pooled)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

