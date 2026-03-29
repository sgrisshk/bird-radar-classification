from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from src.metrics import macro_map_score, per_class_average_precision, sigmoid_numpy


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass


def compute_class_weights(y_onehot: np.ndarray) -> np.ndarray:
    pos = y_onehot.sum(axis=0).astype(np.float64)
    neg = len(y_onehot) - pos
    weights = neg / np.clip(pos, 1.0, None)
    weights = weights / (weights.mean() + 1e-8)
    return weights.astype(np.float32)


def smooth_labels(y: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0:
        return y
    num_classes = y.shape[-1]
    return y * (1.0 - eps) + eps / num_classes


class BinaryFocalWithLogits(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal = (1.0 - pt).pow(self.gamma) * bce
        return focal.mean()


class HybridLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None, label_smoothing: float = 0.05) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.focal = BinaryFocalWithLogits(gamma=2.0, pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = smooth_labels(targets, self.label_smoothing)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="mean",
            pos_weight=self.pos_weight,
        )
        focal = self.focal(logits, targets)
        return 0.8 * bce + 0.2 * focal


@dataclass
class TrainState:
    epoch: int
    best_score: float
    best_epoch: int
    patience_left: int


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max") -> None:
        self.patience = int(patience)
        self.mode = mode
        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.best_epoch = -1
        self.counter = 0

    def step(self, score: float, epoch: int) -> bool:
        improved = score > self.best_score if self.mode == "max" else score < self.best_score
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    losses: list[float] = []
    amp_enabled = device.type == "cuda"

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            logits = model(x, padding_mask=padding_mask)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    model.eval()
    logits_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    track_ids: list[np.ndarray] = []
    amp_enabled = device.type == "cuda"

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        with autocast(enabled=amp_enabled):
            logits = model(x, padding_mask=padding_mask)
        logits_list.append(logits.detach().cpu().numpy())
        track_ids.append(batch["track_id"].detach().cpu().numpy())
        if "y" in batch:
            y_list.append(batch["y"].detach().cpu().numpy())

    logits_np = np.concatenate(logits_list, axis=0) if logits_list else np.empty((0, 0), dtype=np.float32)
    y_np = np.concatenate(y_list, axis=0) if y_list else None
    track_np = np.concatenate(track_ids, axis=0) if track_ids else np.empty((0,), dtype=np.int64)
    return logits_np, y_np, track_np


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, object]:
    logits, y_true, _ = predict_logits(model, loader, device)
    if y_true is None:
        raise ValueError("Validation loader must contain labels")
    probs = sigmoid_numpy(logits)
    class_scores = per_class_average_precision(y_true, probs)
    return {
        "macro_map": macro_map_score(y_true, probs),
        "per_class_map": class_scores,
        "probs": probs,
        "logits": logits,
        "y_true": y_true,
    }


def save_checkpoint(path: str | Path, model: nn.Module, extra: dict | None = None) -> None:
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: nn.Module, map_location: str | torch.device = "cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(payload["state_dict"])
    return payload


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
