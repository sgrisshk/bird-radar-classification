from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceSample:
    x: torch.Tensor
    y: torch.Tensor | None
    track_id: int
    length: int


class TrackSequenceDataset(Dataset[SequenceSample]):
    def __init__(
        self,
        df: pd.DataFrame,
        track_cache: dict[int, dict[str, Any]],
        y: np.ndarray | None = None,
        max_len: int = 512,
        training: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.track_cache = track_cache
        self.y = y
        self.max_len = int(max_len)
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.df)

    def _crop(self, x: np.ndarray) -> np.ndarray:
        if len(x) <= self.max_len:
            return x
        if self.training:
            start = np.random.randint(0, len(x) - self.max_len + 1)
        else:
            start = (len(x) - self.max_len) // 2
        return x[start : start + self.max_len]

    def __getitem__(self, idx: int) -> SequenceSample:
        row = self.df.iloc[idx]
        track_id = int(row["track_id"])
        x_np = self.track_cache[track_id]["features"].astype(np.float32)
        x_np = self._crop(x_np)
        x = torch.from_numpy(x_np)
        y = None
        if self.y is not None:
            y = torch.from_numpy(self.y[idx].astype(np.float32))
        return SequenceSample(x=x, y=y, track_id=track_id, length=x.shape[0])


def collate_sequence_batch(batch: list[SequenceSample]) -> dict[str, torch.Tensor]:
    max_len = max(item.length for item in batch)
    feat_dim = batch[0].x.shape[-1]
    batch_size = len(batch)
    x = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    track_ids = torch.zeros(batch_size, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    has_targets = batch[0].y is not None
    y = None
    if has_targets:
        y = torch.zeros(batch_size, batch[0].y.shape[-1], dtype=torch.float32)

    for i, item in enumerate(batch):
        n = item.length
        x[i, :n] = item.x
        padding_mask[i, :n] = False
        track_ids[i] = item.track_id
        lengths[i] = n
        if has_targets and y is not None and item.y is not None:
            y[i] = item.y

    out: dict[str, torch.Tensor] = {
        "x": x,
        "padding_mask": padding_mask,
        "track_id": track_ids,
        "length": lengths,
    }
    if y is not None:
        out["y"] = y
    return out

