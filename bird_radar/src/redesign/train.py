from __future__ import annotations

import copy
import itertools
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import CLASSES
from src.redesign.dataset import RadarHybridDataset, SequenceConfig
from src.redesign.model import MultiBranchRadarModel, SequenceOnlyRadarModel
from src.redesign.utils import macro_map, per_class_ap, topk_weak_classes


@dataclass
class FoldResult:
    oof_pred: np.ndarray
    test_pred: np.ndarray
    oof_emb: np.ndarray | None
    test_emb: np.ndarray | None
    val_idx: np.ndarray
    best_score: float
    train_score: float | None
    val_pred_diag: dict[str, float] | None
    per_class: dict[str, float]
    weak_classes: list[str]


class HybridLoss(torch.nn.Module):
    def __init__(
        self,
        class_weights: np.ndarray,
        focal_gamma: float = 1.5,
        pos_weight: np.ndarray | None = None,
        bce_weight: float = 0.8,
        focal_weight: float = 0.2,
        ranking_weight: float = 0.0,
        ranking_margin: float = 0.0,
        ranking_max_pairs: int = 0,
        listwise_weight: float = 0.0,
        listwise_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", w)
        if pos_weight is None:
            self.pos_weight = None
        else:
            pw = torch.tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", pw)
        self.focal_gamma = float(focal_gamma)
        self.bce_weight = float(bce_weight)
        self.focal_weight = float(focal_weight)
        self.ranking_weight = float(ranking_weight)
        self.ranking_margin = float(ranking_margin)
        self.ranking_max_pairs = int(ranking_max_pairs)
        self.listwise_weight = float(listwise_weight)
        self.listwise_temperature = float(max(1e-6, listwise_temperature))

    def _pairwise_rank_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Pairwise AUC-style surrogate: encourage positive logits > negative logits per class.
        losses: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        num_classes = int(logits.shape[1])
        for c in range(num_classes):
            t = targets[:, c]
            pos = logits[t > 0.5, c]
            neg = logits[t <= 0.5, c]
            if pos.numel() == 0 or neg.numel() == 0:
                continue

            if self.ranking_max_pairs > 0 and int(pos.numel() * neg.numel()) > self.ranking_max_pairs:
                pos_idx = torch.randint(0, int(pos.numel()), (self.ranking_max_pairs,), device=logits.device)
                neg_idx = torch.randint(0, int(neg.numel()), (self.ranking_max_pairs,), device=logits.device)
                diff = neg[neg_idx] - pos[pos_idx] + self.ranking_margin
            else:
                diff = neg.unsqueeze(0) - pos.unsqueeze(1) + self.ranking_margin
            # softplus is smooth hinge: log(1 + exp(neg-pos+margin)).
            c_loss = F.softplus(diff).mean()
            losses.append(c_loss)
            weights.append(self.class_weights[c])

        if not losses:
            return logits.new_zeros(())
        loss_t = torch.stack(losses)
        w_t = torch.stack(weights).to(device=logits.device, dtype=loss_t.dtype)
        return (loss_t * w_t).sum() / torch.clamp(w_t.sum(), min=1e-8)

    def _listwise_rank_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # ListNet-style loss over samples for each class.
        # Positives define a target distribution; model predicts softmax over logits.
        losses: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        num_classes = int(logits.shape[1])
        temp = float(self.listwise_temperature)
        for c in range(num_classes):
            t = targets[:, c]
            pos_sum = torch.sum(t)
            if float(pos_sum.detach().item()) <= 0.0:
                continue
            p_target = t / torch.clamp(pos_sum, min=1e-8)
            log_p_pred = F.log_softmax(logits[:, c] / temp, dim=0)
            c_loss = -(p_target * log_p_pred).sum()
            losses.append(c_loss)
            weights.append(self.class_weights[c])

        if not losses:
            return logits.new_zeros(())
        loss_t = torch.stack(losses)
        w_t = torch.stack(weights).to(device=logits.device, dtype=loss_t.dtype)
        return (loss_t * w_t).sum() / torch.clamp(w_t.sum(), min=1e-8)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_w = self.pos_weight.view(1, -1) if self.pos_weight is not None else None
        bce_raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_w)
        w = self.class_weights.view(1, -1)
        bce = (bce_raw * w).mean()

        p = torch.sigmoid(logits)
        pt = p * targets + (1.0 - p) * (1.0 - targets)
        focal = ((1.0 - pt).clamp(min=1e-6) ** self.focal_gamma) * bce_raw * w
        focal = focal.mean()

        total = self.bce_weight * bce + self.focal_weight * focal
        if self.ranking_weight > 0.0:
            total = total + self.ranking_weight * self._pairwise_rank_loss(logits, targets)
        if self.listwise_weight > 0.0:
            total = total + self.listwise_weight * self._listwise_rank_loss(logits, targets)
        return total


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        out[k] = v.to(device)
    return out


def _autocast_ctx(amp_enabled: bool):
    if amp_enabled:
        return autocast(device_type="cuda")
    return nullcontext()


def _ablation_flags(mode: str) -> tuple[bool, bool]:
    m = str(mode).lower().strip()
    if m == "tab_only":
        return False, True
    if m == "seq_only":
        return True, False
    return True, True


def _snapshot_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _ema_update_state(
    ema_state: dict[str, torch.Tensor],
    model: torch.nn.Module,
    decay: float,
) -> None:
    cur = model.state_dict()
    d = float(np.clip(decay, 0.0, 1.0))
    keep = d
    add = 1.0 - d
    for k, v in cur.items():
        vv = v.detach()
        if (k not in ema_state) or (not torch.is_floating_point(vv)):
            ema_state[k] = vv.clone()
            continue
        ema_state[k].mul_(keep).add_(vv, alpha=add)


def _swa_update_state(
    swa_state: dict[str, torch.Tensor],
    model: torch.nn.Module,
    n_models: int,
) -> int:
    cur = model.state_dict()
    n = int(max(0, n_models))
    coeff_old = float(n) / float(n + 1)
    coeff_new = 1.0 / float(n + 1)
    for k, v in cur.items():
        vv = v.detach()
        if (k not in swa_state) or (not torch.is_floating_point(vv)):
            swa_state[k] = vv.clone()
            continue
        swa_state[k].mul_(coeff_old).add_(vv, alpha=coeff_new)
    return n + 1


def _predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    ablation_mode: str = "full",
    tta_runs: int = 0,
    tta_disable_dropout: bool = True,
    tta_time_crop_min: float | None = None,
    residual_teacher_map: dict[int, np.ndarray] | None = None,
    return_fused: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    model.eval()
    ds = loader.dataset
    original_aug = getattr(ds, "augment", False)
    seq_cfg = getattr(ds, "seq_cfg", None)
    orig_ch_drop = float(getattr(seq_cfg, "channel_dropout_p", 0.0)) if seq_cfg is not None else None
    orig_time_drop = float(getattr(seq_cfg, "time_dropout_p", 0.0)) if seq_cfg is not None else None
    orig_crop_min = float(getattr(seq_cfg, "time_crop_min", 1.0)) if seq_cfg is not None else None
    views = int(max(0, tta_runs))
    use_seq, use_tab = _ablation_flags(ablation_mode)
    all_preds: list[np.ndarray] = []
    all_fused: list[np.ndarray] = []
    with torch.no_grad():
        for view_id in range(views + 1):
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(10_000 + view_id)
            if hasattr(ds, "augment"):
                ds.augment = bool(view_id > 0 and views > 0)
            if seq_cfg is not None:
                if view_id > 0 and views > 0:
                    if tta_disable_dropout:
                        seq_cfg.channel_dropout_p = 0.0
                        seq_cfg.time_dropout_p = 0.0
                    if tta_time_crop_min is not None:
                        seq_cfg.time_crop_min = float(np.clip(float(tta_time_crop_min), 0.0, 1.0))
                else:
                    if orig_ch_drop is not None:
                        seq_cfg.channel_dropout_p = float(orig_ch_drop)
                    if orig_time_drop is not None:
                        seq_cfg.time_dropout_p = float(orig_time_drop)
                    if orig_crop_min is not None:
                        seq_cfg.time_crop_min = float(orig_crop_min)

            preds: list[np.ndarray] = []
            fused_preds: list[np.ndarray] = []
            for batch in loader:
                b = _to_device(batch, device)
                with _autocast_ctx(amp_enabled):
                    out = model(
                        b["seq"],
                        b["tab"],
                        b["time_norm"],
                        grl_lambda=0.0,
                        use_seq=use_seq,
                        use_tab=use_tab,
                    )
                    logits = out["logits"]
                    if residual_teacher_map is not None:
                        teacher_prob = _teacher_probs_from_track_ids(
                            b["track_id"],
                            teacher_map=residual_teacher_map,
                            device=device,
                        )
                        teacher_logit = torch.logit(torch.clamp(teacher_prob, 1e-6, 1.0 - 1e-6))
                        logits = logits + teacher_logit
                    prob = torch.sigmoid(logits)
                preds.append(prob.detach().cpu().numpy().astype(np.float32))
                if return_fused:
                    fused = out.get("fused", None)
                    if fused is None:
                        raise ValueError("model output missing 'fused' while return_fused=True")
                    fused_preds.append(fused.detach().cpu().numpy().astype(np.float32))
            all_preds.append(np.concatenate(preds, axis=0))
            if return_fused:
                all_fused.append(np.concatenate(fused_preds, axis=0))

    if hasattr(ds, "augment"):
        ds.augment = original_aug
    if seq_cfg is not None:
        if orig_ch_drop is not None:
            seq_cfg.channel_dropout_p = float(orig_ch_drop)
        if orig_time_drop is not None:
            seq_cfg.time_dropout_p = float(orig_time_drop)
        if orig_crop_min is not None:
            seq_cfg.time_crop_min = float(orig_crop_min)

    pred_out = np.mean(all_preds, axis=0).astype(np.float32)
    if return_fused:
        fused_out = np.mean(all_fused, axis=0).astype(np.float32)
        return pred_out, fused_out
    return pred_out


def _print_batch_debug(prefix: str, batch: dict[str, torch.Tensor]) -> None:
    seq = batch["seq"]
    y = batch.get("target")
    mask = seq[:, -1, :]
    zeros = float((seq.abs() < 1e-8).float().mean().item())
    msg = [
        f"[debug:{prefix}] seq_shape={tuple(seq.shape)}",
        f"seq_mean={float(seq.mean().item()):.6f}",
        f"seq_std={float(seq.std(unbiased=False).item()):.6f}",
        f"seq_zero_frac={zeros:.6f}",
        f"mask_zero_frac={float((mask <= 0).float().mean().item()):.6f}",
    ]
    if y is not None:
        ysum_row = y.sum(dim=1)
        ysum_cls = y.sum(dim=0)
        msg.append(f"y_row_sum_minmax=({float(ysum_row.min().item()):.3f},{float(ysum_row.max().item()):.3f})")
        msg.append(f"y_class_sum={ysum_cls.detach().cpu().numpy().round(1).tolist()}")
    print(" | ".join(msg), flush=True)


def _mixup(
    seq: torch.Tensor,
    tab: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not enabled or alpha <= 0.0 or seq.size(0) < 2:
        return seq, tab, y
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(seq.size(0), device=seq.device)
    seq2 = lam * seq + (1.0 - lam) * seq[idx]
    tab2 = lam * tab + (1.0 - lam) * tab[idx]
    y2 = lam * y + (1.0 - lam) * y[idx]
    return seq2, tab2, y2


def _consistency_augment_seq(
    seq: torch.Tensor,
    seq_in_ch: int,
    time_mask_p: float = 0.0,
    channel_mask_p: float = 0.0,
    noise_std: float = 0.0,
) -> torch.Tensor:
    # Augment only feature channels; keep validity mask channel untouched.
    out = seq.clone()
    feat = out[:, :seq_in_ch, :]
    if time_mask_p > 0.0:
        keep_t = (torch.rand((feat.size(0), 1, feat.size(2)), device=feat.device) > time_mask_p).to(feat.dtype)
        feat = feat * keep_t
    if channel_mask_p > 0.0:
        keep_c = (torch.rand((feat.size(0), feat.size(1), 1), device=feat.device) > channel_mask_p).to(feat.dtype)
        feat = feat * keep_c
    if noise_std > 0.0:
        feat = feat + torch.randn_like(feat) * float(noise_std)
    out[:, :seq_in_ch, :] = feat
    return out


def _pred_diag(pred: np.ndarray) -> dict[str, float]:
    flat = pred.astype(np.float32).reshape(-1)
    return {
        "pred_mean": float(np.mean(flat)),
        "pred_std": float(np.std(flat)),
        "pred_p95": float(np.quantile(flat, 0.95)),
        "pred_frac_gt_0p2": float(np.mean(flat > 0.2)),
        "pred_frac_gt_0p5": float(np.mean(flat > 0.5)),
    }


def _teacher_probs_from_track_ids(
    track_ids: torch.Tensor,
    teacher_map: dict[int, np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    ids = track_ids.detach().cpu().numpy().astype(np.int64)
    rows: list[np.ndarray] = []
    for tid in ids:
        arr = teacher_map.get(int(tid))
        if arr is None:
            raise KeyError(f"missing distill teacher probabilities for track_id={int(tid)}")
        rows.append(arr.astype(np.float32))
    out = np.stack(rows, axis=0).astype(np.float32)
    return torch.from_numpy(out).to(device)


def _build_weighted_sampler_from_targets(
    y: np.ndarray,
    clip_min: float = 0.25,
    clip_max: float = 8.0,
    uniform_mix: float = 0.0,
    power: float = 1.0,
) -> WeightedRandomSampler:
    # y: [N, C] one-hot/multi-label train targets
    pos = y.sum(axis=0).astype(np.float64)
    inv = 1.0 / (pos + 1.0)
    inv = inv / max(np.mean(inv), 1e-12)
    sample_w = (y.astype(np.float64) * inv[None, :]).sum(axis=1)
    sample_w = np.where(sample_w > 0.0, sample_w, float(np.mean(inv)))
    if power != 1.0:
        sample_w = np.power(sample_w, float(power))
    sample_w = np.clip(sample_w, float(clip_min), float(clip_max))
    mix = float(np.clip(uniform_mix, 0.0, 1.0))
    if mix > 0.0:
        sample_w = (1.0 - mix) * sample_w + mix * np.ones_like(sample_w)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=int(len(sample_w)),
        replacement=True,
    )


def _sample_weights_from_targets(
    y: np.ndarray,
    clip_min: float = 0.25,
    clip_max: float = 8.0,
    uniform_mix: float = 0.0,
    power: float = 1.0,
) -> np.ndarray:
    pos = y.sum(axis=0).astype(np.float64)
    inv = 1.0 / (pos + 1.0)
    inv = inv / max(np.mean(inv), 1e-12)
    sample_w = (y.astype(np.float64) * inv[None, :]).sum(axis=1)
    sample_w = np.where(sample_w > 0.0, sample_w, float(np.mean(inv)))
    if power != 1.0:
        sample_w = np.power(sample_w, float(power))
    sample_w = np.clip(sample_w, float(clip_min), float(clip_max))
    mix = float(np.clip(uniform_mix, 0.0, 1.0))
    if mix > 0.0:
        sample_w = (1.0 - mix) * sample_w + mix * np.ones_like(sample_w)
    return sample_w.astype(np.float64)


def _build_weighted_sampler_from_weights(sample_w: np.ndarray) -> WeightedRandomSampler:
    w = np.asarray(sample_w, dtype=np.float64)
    w = np.clip(w, 1e-8, np.inf)
    return WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=int(len(w)),
        replacement=True,
    )


def _hard_scores_from_probs(y_true: np.ndarray, p_pred: np.ndarray) -> np.ndarray:
    # Per-row BCE averaged over classes.
    p = np.clip(p_pred.astype(np.float32), 1e-6, 1.0 - 1e-6)
    y = y_true.astype(np.float32)
    bce = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return bce.mean(axis=1).astype(np.float32)


def _pos_weight_from_targets(y: np.ndarray, clip_max: float = 50.0) -> np.ndarray:
    # Standard BCEWithLogits pos_weight = neg/pos for each class.
    pos = y.sum(axis=0).astype(np.float64)
    total = float(y.shape[0])
    neg = total - pos
    pw = (neg + 1.0) / (pos + 1.0)
    pw = np.clip(pw, 1.0, float(clip_max))
    return pw.astype(np.float32)


def train_one_fold(
    train_track_ids: np.ndarray,
    val_track_ids: np.ndarray,
    train_tab: np.ndarray,
    val_tab: np.ndarray,
    test_track_ids: np.ndarray,
    test_tab: np.ndarray,
    train_y: np.ndarray,
    val_y: np.ndarray,
    cache_train: dict[int, dict[str, Any]],
    cache_test: dict[int, dict[str, Any]],
    seq_cfg: SequenceConfig,
    arch_name: str,
    seed: int,
    cfg: dict[str, Any],
    class_weights: np.ndarray,
    device: torch.device,
    train_time_groups: np.ndarray | None = None,
    distill_cfg: dict[str, Any] | None = None,
    distill_teacher_map: dict[int, np.ndarray] | None = None,
    distill_teacher_test_map: dict[int, np.ndarray] | None = None,
) -> FoldResult:
    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"
    ablation_mode = str(cfg["train"].get("ablation_mode", "full")).lower()
    use_seq, use_tab = _ablation_flags(ablation_mode)
    use_domain = float(cfg["train"].get("domain_loss_weight", 0.0)) > 0.0
    model_mode = str(cfg.get("model", {}).get("mode", "multibranch")).lower().strip()
    distill_enabled = bool(distill_cfg is not None and bool(distill_cfg.get("enabled", False)))
    if distill_enabled and not distill_teacher_map:
        raise ValueError("distill is enabled but distill_teacher_map is empty")
    if distill_enabled and bool(cfg["aug"].get("mixup", False)):
        raise ValueError("distill+mixup is unsupported; disable aug.mixup or distill.enabled")
    distill_temperature = float((distill_cfg or {}).get("temperature", 2.0))
    if distill_temperature <= 0.0:
        raise ValueError("distill temperature must be > 0")
    distill_alpha_true = float((distill_cfg or {}).get("alpha_true", 0.3))
    distill_alpha_soft = float((distill_cfg or {}).get("alpha_soft", 0.7))
    distill_soft_loss = str((distill_cfg or {}).get("soft_loss", "bce")).lower().strip()
    distill_warmup_epochs = int((distill_cfg or {}).get("warmup_soft_epochs", 0))
    distill_residual_mode = bool((distill_cfg or {}).get("residual_mode", False))
    distill_delta_l2 = float((distill_cfg or {}).get("delta_l2", 0.0))
    distill_selective_conf_enabled = bool((distill_cfg or {}).get("selective_conf_enabled", False))
    distill_selective_conf_threshold = float((distill_cfg or {}).get("selective_conf_threshold", 0.8))
    distill_selective_conf_mode = str((distill_cfg or {}).get("selective_conf_mode", "max")).lower().strip()
    distill_pseudo_target_enabled = bool((distill_cfg or {}).get("pseudo_target_enabled", False))
    distill_alpha_pseudo = float((distill_cfg or {}).get("alpha_pseudo", 0.0))
    distill_pseudo_hi = float((distill_cfg or {}).get("pseudo_threshold_high", 0.95))
    distill_pseudo_lo = float((distill_cfg or {}).get("pseudo_threshold_low", 0.05))
    distill_pseudo_conf_power = float((distill_cfg or {}).get("pseudo_conf_power", 1.0))
    distill_consistency_lambda = float((distill_cfg or {}).get("consistency_lambda", 0.0))
    distill_cons_time_mask_p = float((distill_cfg or {}).get("consistency_time_mask_p", 0.0))
    distill_cons_channel_mask_p = float((distill_cfg or {}).get("consistency_channel_mask_p", 0.0))
    distill_cons_noise_std = float((distill_cfg or {}).get("consistency_noise_std", 0.0))
    if distill_enabled and distill_soft_loss not in {"bce"}:
        raise ValueError(f"unsupported distill soft_loss='{distill_soft_loss}', only 'bce' is supported")
    if distill_enabled and distill_selective_conf_mode not in {"max", "margin"}:
        raise ValueError(
            f"unsupported distill selective_conf_mode='{distill_selective_conf_mode}', expected one of: max, margin"
        )
    if distill_enabled and (distill_selective_conf_threshold < 0.0 or distill_selective_conf_threshold > 1.0):
        raise ValueError("distill selective_conf_threshold must be in [0,1]")
    if distill_residual_mode and not distill_enabled:
        raise ValueError("distill.residual_mode=true requires distill.enabled=true")
    if distill_residual_mode and distill_teacher_test_map is None:
        raise ValueError("distill residual mode requires distill_teacher_test_map for test inference")
    if distill_pseudo_target_enabled:
        if not distill_enabled:
            raise ValueError("distill.pseudo_target_enabled=true requires distill.enabled=true")
        if distill_teacher_test_map is None:
            raise ValueError("distill.pseudo_target_enabled=true requires distill.teacher_test_csv")
        if not (0.0 <= distill_pseudo_lo < distill_pseudo_hi <= 1.0):
            raise ValueError("distill pseudo thresholds must satisfy 0 <= low < high <= 1")
        if distill_alpha_pseudo < 0.0:
            raise ValueError("distill.alpha_pseudo must be >= 0")
        if distill_pseudo_conf_power <= 0.0:
            raise ValueError("distill.pseudo_conf_power must be > 0")
    if distill_consistency_lambda < 0.0:
        raise ValueError("distill.consistency_lambda must be >= 0")

    ds_train = RadarHybridDataset(
        track_ids=train_track_ids,
        tabular=train_tab,
        cache=cache_train,
        targets=train_y,
        domain_label=0,
        seq_cfg=seq_cfg,
        augment=True,
        seed=seed,
    )
    ds_val = RadarHybridDataset(
        track_ids=val_track_ids,
        tabular=val_tab,
        cache=cache_train,
        targets=val_y,
        domain_label=0,
        seq_cfg=seq_cfg,
        augment=False,
        seed=seed,
    )
    ds_train_eval = RadarHybridDataset(
        track_ids=train_track_ids,
        tabular=train_tab,
        cache=cache_train,
        targets=train_y,
        domain_label=0,
        seq_cfg=seq_cfg,
        augment=False,
        seed=seed,
    )
    ds_test = RadarHybridDataset(
        track_ids=test_track_ids,
        tabular=test_tab,
        cache=cache_test,
        targets=None,
        domain_label=1,
        seq_cfg=seq_cfg,
        augment=False,
        seed=seed,
    )
    ds_target = None
    use_target_loader = use_domain or distill_pseudo_target_enabled or (distill_consistency_lambda > 0.0)
    if use_target_loader:
        ds_target = RadarHybridDataset(
            track_ids=test_track_ids,
            tabular=test_tab,
            cache=cache_test,
            targets=None,
            domain_label=1,
            seq_cfg=seq_cfg,
            augment=bool(cfg["train"].get("domain_target_augment", False)),
            seed=seed + 97,
        )

    use_weighted_sampler = bool(cfg.get("train", {}).get("use_weighted_sampler", False))
    sampler_clip_min = float(cfg.get("train", {}).get("sampler_clip_min", 0.25))
    sampler_clip_max = float(cfg.get("train", {}).get("sampler_clip_max", 8.0))
    sampler_uniform_mix = float(cfg.get("train", {}).get("sampler_uniform_mix", 0.0))
    sampler_power = float(cfg.get("train", {}).get("sampler_power", 1.0))
    base_sample_weights = (
        _sample_weights_from_targets(
            train_y,
            clip_min=sampler_clip_min,
            clip_max=sampler_clip_max,
            uniform_mix=sampler_uniform_mix,
            power=sampler_power,
        )
        if use_weighted_sampler
        else np.ones((len(train_y),), dtype=np.float64)
    )
    # Hard-example mining sampler mix:
    # new_w = (1-mix)*base_w + mix*(base_w*boost on hard rows, base_w otherwise)
    hard_mining_enabled = bool(cfg.get("train", {}).get("hard_mining_enabled", False))
    hard_mining_start_epoch = int(cfg.get("train", {}).get("hard_mining_start_epoch", 1))
    hard_mining_fraction = float(cfg.get("train", {}).get("hard_mining_fraction", 0.2))
    hard_mining_mix = float(cfg.get("train", {}).get("hard_mining_mix", 0.4))
    hard_mining_boost = float(cfg.get("train", {}).get("hard_mining_boost", 3.0))
    hard_mining_clip_min = float(cfg.get("train", {}).get("hard_mining_clip_min", sampler_clip_min))
    hard_mining_clip_max = float(cfg.get("train", {}).get("hard_mining_clip_max", sampler_clip_max * 2.0))
    hard_mining_fraction = float(np.clip(hard_mining_fraction, 0.0, 0.9))
    hard_mining_mix = float(np.clip(hard_mining_mix, 0.0, 1.0))
    hard_mining_boost = float(max(1.0, hard_mining_boost))
    hard_mining_group_enabled = bool(cfg.get("train", {}).get("hard_mining_group_enabled", False))
    hard_mining_group_topk = int(max(1, cfg.get("train", {}).get("hard_mining_group_topk", 1)))
    hard_mining_group_focus = float(np.clip(cfg.get("train", {}).get("hard_mining_group_focus", 1.0), 0.0, 1.0))
    if train_time_groups is not None:
        train_time_groups = np.asarray(train_time_groups, dtype=np.int64)
        if len(train_time_groups) != len(train_y):
            raise ValueError(
                f"train_time_groups length mismatch: len(groups)={len(train_time_groups)} len(train_y)={len(train_y)}"
            )
    else:
        hard_mining_group_enabled = False
    current_sample_weights: np.ndarray | None = (
        base_sample_weights.copy() if (use_weighted_sampler or hard_mining_enabled) else None
    )

    def _build_train_loader(weights: np.ndarray | None) -> DataLoader:
        if weights is None:
            return DataLoader(
                ds_train,
                batch_size=int(cfg["train"]["batch_size"]),
                shuffle=True,
                drop_last=False,
                num_workers=0,
            )
        sampler = _build_weighted_sampler_from_weights(weights)
        return DataLoader(
            ds_train,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            num_workers=0,
        )

    dl_train = _build_train_loader(current_sample_weights)
    dl_val = DataLoader(ds_val, batch_size=int(cfg["train"]["eval_batch_size"]), shuffle=False, drop_last=False, num_workers=0)
    dl_train_eval = DataLoader(
        ds_train_eval,
        batch_size=int(cfg["train"]["eval_batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    dl_test = DataLoader(ds_test, batch_size=int(cfg["train"]["eval_batch_size"]), shuffle=False, drop_last=False, num_workers=0)
    dl_target = None
    if ds_target is not None:
        dl_target = DataLoader(
            ds_target,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

    seq_dropout = float(cfg.get("model", {}).get("seq_dropout", 0.2))
    tab_dropout = float(cfg.get("model", {}).get("tab_dropout", 0.3))
    pooling = str(cfg.get("model", {}).get("pooling", "mean"))
    gem_p = float(cfg.get("model", {}).get("gem_p", 3.0))
    transformer_nhead = int(cfg.get("model", {}).get("transformer_nhead", 4))
    transformer_num_layers = int(cfg.get("model", {}).get("transformer_num_layers", 2))
    transformer_ffn_dim = cfg.get("model", {}).get("transformer_ffn_dim", None)
    if transformer_ffn_dim is not None:
        transformer_ffn_dim = int(transformer_ffn_dim)
    seq_in_ch = int(cfg.get("sequence", {}).get("seq_in_channels", 9))
    seq_branch = str(cfg.get("model", {}).get("seq_branch", "tcn_trm"))
    if model_mode == "seq_only":
        model = SequenceOnlyRadarModel(
            num_classes=len(CLASSES),
            arch_name=arch_name,
            seq_in_ch=seq_in_ch,
            seq_dropout=seq_dropout,
            pooling=pooling,
            seq_branch=seq_branch,
            gem_p=gem_p,
            transformer_nhead=transformer_nhead,
            transformer_num_layers=transformer_num_layers,
            transformer_ffn_dim=transformer_ffn_dim,
        ).to(device)
    else:
        model = MultiBranchRadarModel(
            tab_dim=train_tab.shape[1],
            num_classes=len(CLASSES),
            arch_name=arch_name,
            seq_in_ch=seq_in_ch,
            seq_dropout=seq_dropout,
            tab_dropout=tab_dropout,
            pooling=pooling,
            gem_p=gem_p,
            transformer_nhead=transformer_nhead,
            transformer_num_layers=transformer_num_layers,
            transformer_ffn_dim=transformer_ffn_dim,
        ).to(device)

    pretrained_encoder_path = str(cfg.get("train", {}).get("pretrained_encoder_path", "")).strip()
    if pretrained_encoder_path:
        p = Path(pretrained_encoder_path).expanduser()
        if not p.is_absolute():
            proj_root = Path(str(cfg.get("paths", {}).get("project_root", ""))).expanduser()
            cand = (proj_root / p).resolve() if str(proj_root) else p.resolve()
            if cand.exists():
                p = cand
            else:
                p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"pretrained_encoder_path not found: {p}")
        if not hasattr(model, "trm") or getattr(model, "trm") is None:
            print(
                f"[pretrain] checkpoint provided but model has no transformer branch; skip load path={p}",
                flush=True,
            )
        else:
            ckpt = torch.load(p, map_location="cpu")
            if isinstance(ckpt, dict) and "trm_state_dict" in ckpt:
                state = ckpt["trm_state_dict"]
            elif isinstance(ckpt, dict):
                state = ckpt
            else:
                raise ValueError(f"unsupported checkpoint format in {p}")
            if any(str(k).startswith("trm.") for k in state.keys()):
                state = {
                    str(k)[4:]: v
                    for k, v in state.items()
                    if str(k).startswith("trm.")
                }
            missing, unexpected = model.trm.load_state_dict(state, strict=False)
            print(
                f"[pretrain] loaded encoder from {p} missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )

    optimizer = AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg["train"]["epochs"]))
    scaler = GradScaler(device="cuda", enabled=amp_enabled)

    train_cfg = cfg.get("train", {})
    epochs_total = int(cfg["train"]["epochs"])
    ema_enabled = bool(train_cfg.get("ema_enabled", False))
    ema_decay = float(train_cfg.get("ema_decay", 0.999))
    ema_start_epoch = int(max(0, train_cfg.get("ema_start_epoch", 0)))
    ema_use_for_eval = bool(train_cfg.get("ema_use_for_eval", False))
    ema_use_for_final = bool(train_cfg.get("ema_use_for_final", ema_enabled))
    ema_state: dict[str, torch.Tensor] | None = _snapshot_state_dict(model) if ema_enabled else None
    ema_started = (ema_state is not None) and (ema_start_epoch <= 0)

    swa_enabled = bool(train_cfg.get("swa_enabled", False))
    swa_start_epoch = int(train_cfg.get("swa_start_epoch", -1))
    if swa_start_epoch < 0:
        swa_start_ratio = float(train_cfg.get("swa_start_ratio", 0.8))
        swa_start_epoch = int(np.floor(max(0.0, min(0.99, swa_start_ratio)) * epochs_total))
    swa_freq = int(max(1, train_cfg.get("swa_freq", 1)))
    swa_use_for_eval = bool(train_cfg.get("swa_use_for_eval", False))
    swa_use_for_final = bool(train_cfg.get("swa_use_for_final", swa_enabled))
    swa_state: dict[str, torch.Tensor] | None = _snapshot_state_dict(model) if swa_enabled else None
    swa_n_models = 0

    use_pos_weight = bool(cfg.get("train", {}).get("use_pos_weight", False))
    pos_weight_clip_max = float(cfg.get("train", {}).get("pos_weight_clip_max", 50.0))
    pos_weight = _pos_weight_from_targets(train_y, clip_max=pos_weight_clip_max) if use_pos_weight else None
    sup_loss_fn = HybridLoss(
        class_weights=class_weights,
        focal_gamma=float(cfg["train"]["focal_gamma"]),
        pos_weight=pos_weight,
        bce_weight=float(cfg.get("train", {}).get("loss_bce_weight", 0.8)),
        focal_weight=float(cfg.get("train", {}).get("loss_focal_weight", 0.2)),
        ranking_weight=float(cfg.get("train", {}).get("loss_ranking_weight", 0.0)),
        ranking_margin=float(cfg.get("train", {}).get("ranking_margin", 0.0)),
        ranking_max_pairs=int(cfg.get("train", {}).get("ranking_max_pairs", 0)),
        listwise_weight=float(cfg.get("train", {}).get("loss_listwise_weight", 0.0)),
        listwise_temperature=float(cfg.get("train", {}).get("listwise_temperature", 1.0)),
    ).to(device)
    domain_loss_fn = torch.nn.BCEWithLogitsLoss()

    best_score = -1.0e18
    best_state: dict[str, torch.Tensor] | None = None
    best_per_class: dict[str, float] = {c: 0.0 for c in CLASSES}
    no_improve = 0
    target_iter = itertools.cycle(dl_target) if dl_target is not None else None
    selection_worst_weight = float(train_cfg.get("selection_worst_weight", 0.0))
    selection_min_worst = train_cfg.get("selection_min_worst", None)
    if selection_min_worst is not None:
        selection_min_worst = float(selection_min_worst)

    for epoch in range(epochs_total):
        # Rebuild sampler-driven loader each epoch if hard mining updates weights.
        if hard_mining_enabled or use_weighted_sampler:
            dl_train = _build_train_loader(current_sample_weights)
        if ema_state is not None and (not ema_started) and epoch >= ema_start_epoch:
            ema_state = _snapshot_state_dict(model)
            ema_started = True
        model.train()
        warmup = int(cfg["train"].get("domain_warmup_epochs", 3))
        total_epochs = int(cfg["train"]["epochs"])
        if (not use_domain) or epoch < warmup:
            lam_grl = 0.0
            domain_w = 0.0
        else:
            phase = float(epoch - warmup) / max(1, total_epochs - warmup - 1)
            lam_grl = float(np.clip(phase, 0.0, 1.0))
            domain_w = float(cfg["train"]["domain_loss_weight"]) * lam_grl
        ds_train.set_epoch(epoch)
        if ds_target is not None:
            ds_target.set_epoch(epoch)

        printed_train_debug = False
        for batch in dl_train:
            src = _to_device(batch, device)
            if bool(cfg["train"].get("debug_batch_stats", False)) and (not printed_train_debug):
                _print_batch_debug("train", src)
                printed_train_debug = True
            tgt = None
            if domain_w > 0.0 and target_iter is not None:
                tgt_raw = next(target_iter)
                tgt = _to_device(tgt_raw, device)
            elif target_iter is not None and (distill_pseudo_target_enabled or distill_consistency_lambda > 0.0):
                tgt_raw = next(target_iter)
                tgt = _to_device(tgt_raw, device)

            seq_src, tab_src, y_src = src["seq"], src["tab"], src["target"]
            teacher_src = None
            if distill_enabled:
                teacher_src = _teacher_probs_from_track_ids(
                    src["track_id"],
                    teacher_map=distill_teacher_map or {},
                    device=device,
                )
            seq_src, tab_src, y_src = _mixup(
                seq_src,
                tab_src,
                y_src,
                alpha=float(cfg["aug"]["mixup_alpha"]),
                enabled=bool(cfg["aug"]["mixup"]),
            )

            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx(amp_enabled):
                out_src = model(
                    seq_src,
                    tab_src,
                    src["time_norm"],
                    grl_lambda=lam_grl,
                    use_seq=use_seq,
                    use_tab=use_tab,
                )
                sup_loss = sup_loss_fn(out_src["logits"], y_src)
                if distill_enabled and teacher_src is not None:
                    if distill_residual_mode:
                        teacher_logit = torch.logit(torch.clamp(teacher_src, 1e-6, 1.0 - 1e-6))
                        fused_logits = out_src["logits"] + teacher_logit
                        true_loss = sup_loss_fn(fused_logits, y_src)
                        student_soft = torch.sigmoid(fused_logits / distill_temperature)
                        soft_raw = F.binary_cross_entropy(student_soft, teacher_src, reduction="none")
                        if distill_selective_conf_enabled:
                            if distill_selective_conf_mode == "margin":
                                top2 = torch.topk(teacher_src, k=2, dim=1).values
                                conf = top2[:, 0] - top2[:, 1]
                            else:
                                conf = torch.max(teacher_src, dim=1).values
                            m = conf >= distill_selective_conf_threshold
                            if bool(torch.any(m).item()):
                                soft_loss = soft_raw[m].mean()
                            else:
                                soft_loss = soft_raw.mean() * 0.0
                        else:
                            soft_loss = soft_raw.mean()
                        if epoch < distill_warmup_epochs:
                            alpha_true_ep = 0.0
                            alpha_soft_ep = 1.0
                        else:
                            alpha_true_ep = distill_alpha_true
                            alpha_soft_ep = distill_alpha_soft
                        delta_reg = (out_src["logits"] ** 2).mean()
                        sup_total = (alpha_true_ep * true_loss + alpha_soft_ep * soft_loss) + distill_delta_l2 * delta_reg
                    else:
                        student_soft = torch.sigmoid(out_src["logits"] / distill_temperature)
                        soft_raw = F.binary_cross_entropy(student_soft, teacher_src, reduction="none")
                        if distill_selective_conf_enabled:
                            if distill_selective_conf_mode == "margin":
                                top2 = torch.topk(teacher_src, k=2, dim=1).values
                                conf = top2[:, 0] - top2[:, 1]
                            else:
                                conf = torch.max(teacher_src, dim=1).values
                            m = conf >= distill_selective_conf_threshold
                            if bool(torch.any(m).item()):
                                soft_loss = soft_raw[m].mean()
                            else:
                                soft_loss = soft_raw.mean() * 0.0
                        else:
                            soft_loss = soft_raw.mean()
                        if epoch < distill_warmup_epochs:
                            alpha_true_ep = 0.0
                            alpha_soft_ep = 1.0
                        else:
                            alpha_true_ep = distill_alpha_true
                            alpha_soft_ep = distill_alpha_soft
                        sup_total = alpha_true_ep * sup_loss + alpha_soft_ep * soft_loss
                else:
                    sup_total = sup_loss

                need_tgt_forward = tgt is not None and (
                    domain_w > 0.0 or distill_pseudo_target_enabled or distill_consistency_lambda > 0.0
                )
                if need_tgt_forward:
                    out_tgt = model(
                        tgt["seq"],
                        tgt["tab"],
                        tgt["time_norm"],
                        grl_lambda=lam_grl,
                        use_seq=use_seq,
                        use_tab=use_tab,
                    )
                total = sup_total
                if distill_pseudo_target_enabled and tgt is not None:
                    teacher_tgt = _teacher_probs_from_track_ids(
                        tgt["track_id"],
                        teacher_map=distill_teacher_test_map or {},
                        device=device,
                    )
                    conf = torch.abs(teacher_tgt - 0.5) * 2.0
                    pseudo_mask = ((teacher_tgt >= distill_pseudo_hi) | (teacher_tgt <= distill_pseudo_lo)).to(
                        teacher_tgt.dtype
                    )
                    pseudo_w = (conf.clamp(min=0.0, max=1.0) ** distill_pseudo_conf_power) * pseudo_mask
                    pseudo_raw = F.binary_cross_entropy_with_logits(out_tgt["logits"], teacher_tgt, reduction="none")
                    denom = torch.clamp(pseudo_w.sum(), min=1e-8)
                    pseudo_loss = (pseudo_raw * pseudo_w).sum() / denom
                    total = total + distill_alpha_pseudo * pseudo_loss

                if distill_consistency_lambda > 0.0 and tgt is not None:
                    seq_tgt_aug = _consistency_augment_seq(
                        tgt["seq"],
                        seq_in_ch=seq_in_ch,
                        time_mask_p=distill_cons_time_mask_p,
                        channel_mask_p=distill_cons_channel_mask_p,
                        noise_std=distill_cons_noise_std,
                    )
                    out_tgt_aug = model(
                        seq_tgt_aug,
                        tgt["tab"],
                        tgt["time_norm"],
                        grl_lambda=0.0,
                        use_seq=use_seq,
                        use_tab=use_tab,
                    )
                    p1 = torch.sigmoid(out_tgt["logits"])
                    p2 = torch.sigmoid(out_tgt_aug["logits"])
                    consistency_loss = F.mse_loss(p1, p2)
                    total = total + distill_consistency_lambda * consistency_loss

                if domain_w > 0.0 and tgt is not None:
                    d_src_tab = out_src["domain_logits_tab"]
                    d_tgt_tab = out_tgt["domain_logits_tab"]
                    d_src_fused = out_src["domain_logits_fused"]
                    d_tgt_fused = out_tgt["domain_logits_fused"]

                    d_lbl_src_tab = torch.zeros_like(d_src_tab)
                    d_lbl_tgt_tab = torch.ones_like(d_tgt_tab)
                    d_lbl_src_fused = torch.zeros_like(d_src_fused)
                    d_lbl_tgt_fused = torch.ones_like(d_tgt_fused)

                    d_loss_tab = 0.5 * (
                        domain_loss_fn(d_src_tab, d_lbl_src_tab) + domain_loss_fn(d_tgt_tab, d_lbl_tgt_tab)
                    )
                    d_loss_fused = 0.5 * (
                        domain_loss_fn(d_src_fused, d_lbl_src_fused) + domain_loss_fn(d_tgt_fused, d_lbl_tgt_fused)
                    )
                    d_loss = 0.5 * d_loss_tab + 0.5 * d_loss_fused
                    total = total + domain_w * d_loss

            scaler.scale(total).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            if ema_state is not None and ema_started:
                _ema_update_state(ema_state, model, decay=ema_decay)

        if swa_state is not None and epoch >= swa_start_epoch and ((epoch - swa_start_epoch) % swa_freq == 0):
            swa_n_models = _swa_update_state(swa_state, model, n_models=swa_n_models)

        scheduler.step()

        eval_state: dict[str, torch.Tensor] | None = None
        if ema_state is not None and ema_use_for_eval and ema_started:
            eval_state = ema_state
        elif swa_state is not None and swa_use_for_eval and swa_n_models > 0:
            eval_state = swa_state
        restore_state: dict[str, torch.Tensor] | None = None
        if eval_state is not None:
            restore_state = _snapshot_state_dict(model)
            model.load_state_dict(eval_state, strict=True)

        val_pred = _predict_loader(
            model,
            dl_val,
            device=device,
            amp_enabled=amp_enabled,
            ablation_mode=ablation_mode,
            tta_runs=int(cfg["train"].get("tta_val_runs", 0)),
            tta_disable_dropout=bool(cfg["train"].get("tta_disable_dropout", True)),
            tta_time_crop_min=cfg["train"].get("tta_time_crop_min", None),
            residual_teacher_map=distill_teacher_map if distill_residual_mode else None,
        )
        candidate_best_state = _snapshot_state_dict(model)
        val_diag = _pred_diag(val_pred)
        if bool(cfg["train"].get("log_pred_diag", True)):
            print(
                "[diag] "
                f"epoch={epoch} "
                f"val_map={float(macro_map(val_y, val_pred)):.6f} "
                f"pred_mean={val_diag['pred_mean']:.6f} "
                f"pred_std={val_diag['pred_std']:.6f} "
                f"pred_p95={val_diag['pred_p95']:.6f} "
                f"gt0p2={val_diag['pred_frac_gt_0p2']:.6f} "
                f"gt0p5={val_diag['pred_frac_gt_0p5']:.6f}",
                flush=True,
            )
        val_score = macro_map(val_y, val_pred)
        val_per = per_class_ap(val_y, val_pred)
        val_worst = float(min(val_per.values())) if len(val_per) > 0 else 0.0
        selection_score = float(val_score) + selection_worst_weight * val_worst
        passes_worst_gate = (selection_min_worst is None) or (val_worst >= selection_min_worst)

        if passes_worst_gate and selection_score > best_score:
            best_score = float(selection_score)
            best_per_class = val_per
            best_state = candidate_best_state
            no_improve = 0
        else:
            no_improve += 1

        if restore_state is not None:
            model.load_state_dict(restore_state, strict=True)

        if no_improve >= int(cfg["train"]["patience"]):
            break

        if hard_mining_enabled and (epoch >= int(hard_mining_start_epoch) - 1):
            train_pred_hm = _predict_loader(
                model,
                dl_train_eval,
                device=device,
                amp_enabled=amp_enabled,
                ablation_mode=ablation_mode,
                tta_runs=0,
                residual_teacher_map=distill_teacher_map if distill_residual_mode else None,
            )
            hard_scores = _hard_scores_from_probs(train_y, train_pred_hm)
            n_hard = int(max(1, np.floor(len(hard_scores) * hard_mining_fraction)))
            if hard_mining_group_enabled and train_time_groups is not None and len(train_time_groups) == len(hard_scores):
                uniq_groups = np.unique(train_time_groups)
                group_means: list[tuple[int, float]] = []
                for gid in uniq_groups.tolist():
                    g_mask = train_time_groups == int(gid)
                    if not np.any(g_mask):
                        continue
                    group_means.append((int(gid), float(np.mean(hard_scores[g_mask]))))
                group_means.sort(key=lambda x: x[1], reverse=True)
                worst_groups = np.array([g for g, _ in group_means[:hard_mining_group_topk]], dtype=np.int64)
                worst_mask = np.isin(train_time_groups, worst_groups)
                worst_idx = np.where(worst_mask)[0]
                n_focus = int(max(0, np.floor(n_hard * hard_mining_group_focus)))
                n_focus = int(min(n_focus, len(worst_idx)))
                hard_parts: list[np.ndarray] = []
                if n_focus > 0:
                    focus_scores = hard_scores[worst_idx]
                    focus_take = np.argpartition(focus_scores, -n_focus)[-n_focus:]
                    hard_focus = worst_idx[focus_take]
                    hard_parts.append(hard_focus.astype(np.int64))
                n_remaining = int(max(0, n_hard - sum(len(x) for x in hard_parts)))
                if n_remaining > 0:
                    rem_scores = hard_scores.copy()
                    if len(hard_parts) > 0:
                        rem_scores[np.concatenate(hard_parts)] = -np.inf
                    hard_rem = np.argpartition(rem_scores, -n_remaining)[-n_remaining:]
                    hard_parts.append(hard_rem.astype(np.int64))
                hard_idx = np.unique(np.concatenate(hard_parts)) if len(hard_parts) > 0 else np.array([], dtype=np.int64)
                if len(hard_idx) < n_hard:
                    extra = n_hard - len(hard_idx)
                    rem_scores = hard_scores.copy()
                    rem_scores[hard_idx] = -np.inf
                    add_idx = np.argpartition(rem_scores, -extra)[-extra:]
                    hard_idx = np.unique(np.concatenate([hard_idx, add_idx.astype(np.int64)]))
            else:
                hard_idx = np.argpartition(hard_scores, -n_hard)[-n_hard:]
            hard_mask = np.zeros((len(hard_scores),), dtype=np.float64)
            hard_mask[hard_idx] = 1.0

            boosted = base_sample_weights.copy()
            boosted[hard_mask > 0] *= hard_mining_boost

            base_norm = base_sample_weights / max(np.mean(base_sample_weights), 1e-12)
            boosted_norm = boosted / max(np.mean(boosted), 1e-12)
            mixed = (1.0 - hard_mining_mix) * base_norm + hard_mining_mix * boosted_norm
            mixed = np.clip(mixed, hard_mining_clip_min, hard_mining_clip_max)
            current_sample_weights = mixed.astype(np.float64)
            if bool(cfg["train"].get("log_pred_diag", True)):
                msg = (
                    "[hard] "
                    f"epoch={epoch} "
                    f"n_hard={int(n_hard)} "
                    f"hard_mean_loss={float(np.mean(hard_scores[hard_idx])):.6f} "
                    f"all_mean_loss={float(np.mean(hard_scores)):.6f} "
                    f"w_mean={float(np.mean(current_sample_weights)):.6f} "
                    f"w_max={float(np.max(current_sample_weights)):.6f}"
                )
                if hard_mining_group_enabled and train_time_groups is not None:
                    uniq_groups = np.unique(train_time_groups)
                    group_means: list[tuple[int, float]] = []
                    for gid in uniq_groups.tolist():
                        g_mask = train_time_groups == int(gid)
                        if np.any(g_mask):
                            group_means.append((int(gid), float(np.mean(hard_scores[g_mask]))))
                    group_means.sort(key=lambda x: x[1], reverse=True)
                    worst_groups = [g for g, _ in group_means[:hard_mining_group_topk]]
                    msg += f" worst_groups={worst_groups}"
                print(
                    msg,
                    flush=True,
                )

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    used_avg_for_eval = (
        (ema_state is not None and ema_use_for_eval and ema_started)
        or (swa_state is not None and swa_use_for_eval and swa_n_models > 0)
    )
    if not used_avg_for_eval:
        if swa_state is not None and swa_use_for_final and swa_n_models > 0:
            model.load_state_dict(swa_state, strict=True)
        elif ema_state is not None and ema_use_for_final and ema_started:
            model.load_state_dict(ema_state, strict=True)

    # Optional weak-class focused short finetune
    if bool(cfg["train"].get("class_specialized_heads", False)):
        weak = topk_weak_classes(best_per_class, k=int(cfg["train"].get("weak_k", 3)))
        weak_idx = [CLASSES.index(c) for c in weak]
        extra_epochs = int(cfg["train"].get("weak_finetune_epochs", 2))
        boost = float(cfg["train"].get("weak_boost", 1.5))
        custom_w = class_weights.copy()
        for wi in weak_idx:
            custom_w[wi] *= boost
        weak_loss = HybridLoss(
            class_weights=custom_w,
            focal_gamma=float(cfg["train"]["focal_gamma"]),
            pos_weight=pos_weight,
            bce_weight=float(cfg.get("train", {}).get("loss_bce_weight", 0.8)),
            focal_weight=float(cfg.get("train", {}).get("loss_focal_weight", 0.2)),
            ranking_weight=float(cfg.get("train", {}).get("loss_ranking_weight", 0.0)),
            ranking_margin=float(cfg.get("train", {}).get("ranking_margin", 0.0)),
            ranking_max_pairs=int(cfg.get("train", {}).get("ranking_max_pairs", 0)),
            listwise_weight=float(cfg.get("train", {}).get("loss_listwise_weight", 0.0)),
            listwise_temperature=float(cfg.get("train", {}).get("listwise_temperature", 1.0)),
        ).to(device)

        for _ in range(extra_epochs):
            model.train()
            for batch in dl_train:
                src = _to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with _autocast_ctx(amp_enabled):
                    out = model(
                        src["seq"],
                        src["tab"],
                        src["time_norm"],
                        grl_lambda=0.0,
                        use_seq=use_seq,
                        use_tab=use_tab,
                    )
                    loss = weak_loss(out["logits"], src["target"]) 
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()

    save_embeddings = bool(cfg["train"].get("save_embeddings", False))
    oof_emb: np.ndarray | None = None
    test_emb: np.ndarray | None = None
    if save_embeddings:
        oof_pred, oof_emb = _predict_loader(
            model,
            dl_val,
            device=device,
            amp_enabled=amp_enabled,
            ablation_mode=ablation_mode,
            tta_runs=int(cfg["train"].get("tta_val_runs", 0)),
            tta_disable_dropout=bool(cfg["train"].get("tta_disable_dropout", True)),
            tta_time_crop_min=cfg["train"].get("tta_time_crop_min", None),
            residual_teacher_map=distill_teacher_map if distill_residual_mode else None,
            return_fused=True,
        )
        test_pred, test_emb = _predict_loader(
            model,
            dl_test,
            device=device,
            amp_enabled=amp_enabled,
            ablation_mode=ablation_mode,
            tta_runs=int(cfg["train"].get("tta_test_runs", 2)),
            tta_disable_dropout=bool(cfg["train"].get("tta_disable_dropout", True)),
            tta_time_crop_min=cfg["train"].get("tta_time_crop_min", None),
            residual_teacher_map=distill_teacher_test_map if distill_residual_mode else None,
            return_fused=True,
        )
    else:
        oof_pred = _predict_loader(
            model,
            dl_val,
            device=device,
            amp_enabled=amp_enabled,
            ablation_mode=ablation_mode,
            tta_runs=int(cfg["train"].get("tta_val_runs", 0)),
            tta_disable_dropout=bool(cfg["train"].get("tta_disable_dropout", True)),
            tta_time_crop_min=cfg["train"].get("tta_time_crop_min", None),
            residual_teacher_map=distill_teacher_map if distill_residual_mode else None,
        )
        test_pred = _predict_loader(
            model,
            dl_test,
            device=device,
            amp_enabled=amp_enabled,
            ablation_mode=ablation_mode,
            tta_runs=int(cfg["train"].get("tta_test_runs", 2)),
            tta_disable_dropout=bool(cfg["train"].get("tta_disable_dropout", True)),
            tta_time_crop_min=cfg["train"].get("tta_time_crop_min", None),
            residual_teacher_map=distill_teacher_test_map if distill_residual_mode else None,
        )

    final_per = per_class_ap(val_y, oof_pred)
    weak_classes = topk_weak_classes(final_per, k=3)
    val_pred_diag = _pred_diag(oof_pred)
    train_score: float | None = None
    if bool(cfg["train"].get("report_train_metric", False)):
        train_pred = _predict_loader(
            model,
            dl_train_eval,
            device=device,
            amp_enabled=amp_enabled,
            ablation_mode=ablation_mode,
            tta_runs=0,
            residual_teacher_map=distill_teacher_map if distill_residual_mode else None,
        )
        train_score = float(macro_map(train_y, train_pred))

    return FoldResult(
        oof_pred=oof_pred,
        test_pred=test_pred,
        oof_emb=oof_emb,
        test_emb=test_emb,
        val_idx=np.arange(len(val_track_ids), dtype=np.int64),
        best_score=float(macro_map(val_y, oof_pred)),
        train_score=train_score,
        val_pred_diag=val_pred_diag,
        per_class=final_per,
        weak_classes=weak_classes,
    )
