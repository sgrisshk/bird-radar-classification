from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASSES
from src.cv import make_forward_temporal_group_folds
from src.metrics import macro_map_score, per_class_average_precision
from src.redesign.utils import dump_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 2D CNN on track spectrogram tensors with forward CV.")
    p.add_argument("--spec-dir", type=str, required=True)
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--sample-submission", type=str, default="")
    p.add_argument("--teacher-oof-csv", type=str, default="")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"])
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--split-mode", type=str, default="forward", choices=["forward"])
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--use-pos-weight", type=int, default=1)
    p.add_argument("--specaug", type=int, default=1)
    p.add_argument("--time-mask-p", type=float, default=0.5)
    p.add_argument("--freq-mask-p", type=float, default=0.5)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--id-col", type=str, default="track_id")
    p.add_argument("--out-name", type=str, default="submission_spec_cnn.csv")
    return p.parse_args()


def _resolve_device(name: str) -> torch.device:
    n = str(name).strip().lower()
    if n == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if n == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")


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
    if not np.isfinite(c):
        return 0.0
    return c


def _teacher_diag(y_true: np.ndarray, teacher: np.ndarray, model: np.ndarray) -> dict[str, float]:
    teacher = np.clip(teacher.astype(np.float32), 1e-6, 1.0 - 1e-6)
    model = np.clip(model.astype(np.float32), 1e-6, 1.0 - 1e-6)
    teacher_macro = float(macro_map_score(y_true, teacher))
    model_macro = float(macro_map_score(y_true, model))
    best = teacher_macro
    best_w = 1.0
    for w in [0.95, 0.90, 0.85, 0.80, 0.70]:
        blend = np.clip(w * teacher + (1.0 - w) * model, 0.0, 1.0)
        m = float(macro_map_score(y_true, blend))
        if m > best:
            best = m
            best_w = float(w)
    return {
        "teacher_macro": teacher_macro,
        "model_macro": model_macro,
        "corr_with_teacher": float(_safe_corr_flat(teacher, model)),
        "best_blend_macro": best,
        "best_blend_w_teacher": best_w,
        "best_blend_gain": float(best - teacher_macro),
    }


class SpectrogramDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        augment: bool,
        time_mask_p: float,
        freq_mask_p: float,
        noise_std: float,
    ) -> None:
        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        self.augment = bool(augment)
        self.time_mask_p = float(time_mask_p)
        self.freq_mask_p = float(freq_mask_p)
        self.noise_std = float(noise_std)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def _augment(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        _, f, t = out.shape
        if self.time_mask_p > 0.0 and random.random() < self.time_mask_p and t >= 8:
            width = random.randint(1, max(1, int(0.12 * t)))
            start = random.randint(0, max(0, t - width))
            out[:, :, start : start + width] = 0.0
        if self.freq_mask_p > 0.0 and random.random() < self.freq_mask_p and f >= 8:
            width = random.randint(1, max(1, int(0.12 * f)))
            start = random.randint(0, max(0, f - width))
            out[:, start : start + width, :] = 0.0
        if self.noise_std > 0.0:
            out = out + np.random.normal(0.0, self.noise_std, size=out.shape).astype(np.float32)
        return out

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[idx]
        y = self.y[idx]
        if self.augment:
            x = self._augment(x)
        return torch.from_numpy(x), torch.from_numpy(y)


def _build_model(arch: str, in_channels: int, n_classes: int) -> nn.Module:
    if arch == "resnet18":
        model = torchvision.models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model
    if arch == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights=None)
        old = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
        return model
    raise ValueError(f"unknown arch: {arch}")


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        out.append(probs)
    if not out:
        return np.empty((0, len(CLASSES)), dtype=np.float32)
    return np.concatenate(out, axis=0)


def _diagnostic_line(epoch: int, pred: np.ndarray) -> str:
    p = pred.astype(np.float32)
    mean = float(np.mean(p))
    std = float(np.std(p))
    p95 = float(np.quantile(p, 0.95))
    gt02 = float(np.mean(p > 0.2))
    gt05 = float(np.mean(p > 0.5))
    return (
        f"[diag] epoch={epoch} val_map={{val_map:.6f}} pred_mean={mean:.6f} "
        f"pred_std={std:.6f} pred_p95={p95:.6f} gt0p2={gt02:.6f} gt0p5={gt05:.6f}"
    )


def _fit_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    x_val = np.nan_to_num(x_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    ds_train = SpectrogramDataset(
        x=x_train,
        y=y_train,
        augment=bool(args.specaug),
        time_mask_p=float(args.time_mask_p),
        freq_mask_p=float(args.freq_mask_p),
        noise_std=float(args.noise_std),
    )
    ds_val = SpectrogramDataset(
        x=x_val,
        y=y_val,
        augment=False,
        time_mask_p=0.0,
        freq_mask_p=0.0,
        noise_std=0.0,
    )
    ds_test = SpectrogramDataset(
        x=x_test,
        y=np.zeros((len(x_test), len(CLASSES)), dtype=np.float32),
        augment=False,
        time_mask_p=0.0,
        freq_mask_p=0.0,
        noise_std=0.0,
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

    model = _build_model(arch=str(args.arch), in_channels=x_train.shape[1], n_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        eps=1e-4 if device.type == "mps" else 1e-8,
    )

    if bool(args.use_pos_weight):
        pos = y_train.sum(axis=0).astype(np.float32)
        neg = float(len(y_train)) - pos
        pw = (neg + 1.0) / (pos + 1.0)
        pos_weight = torch.from_numpy(pw.astype(np.float32)).to(device)
    else:
        pos_weight = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(int(args.epochs)):
        model.train()
        bad_batches = 0
        total_batches = 0
        for xb, yb in dl_train:
            total_batches += 1
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            if not torch.isfinite(logits).all():
                bad_batches += 1
                continue
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                bad_batches += 1
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if bad_batches > 0:
            print(
                f"[warn] non-finite batches epoch={epoch} bad={bad_batches}/{max(total_batches,1)} device={device}",
                flush=True,
            )
        if bad_batches == total_batches and total_batches > 0:
            raise RuntimeError(f"NONFINITE_ALL_BATCHES device={device}")

        val_pred = _predict(model=model, loader=dl_val, device=device)
        val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        val_map = float(macro_map_score(y_val, val_pred))
        print(_diagnostic_line(epoch, val_pred).format(val_map=val_map), flush=True)

        if val_map > best_score:
            best_score = val_map
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(args.patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred = _predict(model=model, loader=dl_val, device=device)
    test_pred = _predict(model=model, loader=dl_test, device=device)
    val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    return val_pred, test_pred, float(best_score), int(best_epoch)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = _resolve_device(str(args.device))

    spec_dir = Path(args.spec_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    artifacts_dir = out_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    x_train = np.load(spec_dir / "train_spec.npy").astype(np.float32)
    x_test = np.load(spec_dir / "test_spec.npy").astype(np.float32)
    y_train = np.load(spec_dir / "train_y.npy").astype(np.float32)
    train_ids_raw = np.load(spec_dir / "train_track_ids.npy").astype(np.int64)
    test_ids = np.load(spec_dir / "test_track_ids.npy").astype(np.int64)

    if y_train.shape != (len(x_train), len(CLASSES)):
        raise ValueError(f"train_y shape mismatch: got {y_train.shape}, expected {(len(x_train), len(CLASSES))}")

    meta_df = pd.read_csv(
        args.train_csv,
        usecols=[str(args.id_col), "timestamp_start_radar_utc", "observation_id"],
    )
    meta_ids = meta_df[str(args.id_col)].to_numpy(dtype=np.int64)

    pos = {int(tid): i for i, tid in enumerate(train_ids_raw.tolist())}
    missing = [int(tid) for tid in meta_ids.tolist() if int(tid) not in pos]
    if missing:
        raise ValueError(f"spec train ids missing {len(missing)} rows from train.csv; first={missing[:10]}")
    take = np.array([pos[int(tid)] for tid in meta_ids.tolist()], dtype=np.int64)
    x_train = x_train[take]
    y_train = y_train[take]
    train_ids = train_ids_raw[take]

    cv_df = pd.DataFrame(
        {
            "_cv_ts": meta_df["timestamp_start_radar_utc"],
            "_cv_group": meta_df["observation_id"].astype(np.int64),
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

    oof = np.zeros((len(x_train), len(CLASSES)), dtype=np.float32)
    covered_mask = np.zeros((len(x_train),), dtype=bool)
    test_accum = np.zeros((len(x_test), len(CLASSES)), dtype=np.float32)
    fold_scores: list[float] = []
    fold_best_epochs: list[int] = []
    fold_best_maps: list[float] = []

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        print(f"[fold] {fold_id + 1}/{len(folds)} train={len(tr_idx)} val={len(va_idx)}", flush=True)
        try:
            val_pred, test_pred, best_map, best_epoch = _fit_one_fold(
                x_train=x_train[tr_idx],
                y_train=y_train[tr_idx],
                x_val=x_train[va_idx],
                y_val=y_train[va_idx],
                x_test=x_test,
                args=args,
                device=device,
            )
        except RuntimeError as exc:
            if device.type == "mps" and "NONFINITE" in str(exc):
                print(
                    f"[warn] fold={fold_id + 1} encountered non-finite training on MPS, retrying on CPU",
                    flush=True,
                )
                val_pred, test_pred, best_map, best_epoch = _fit_one_fold(
                    x_train=x_train[tr_idx],
                    y_train=y_train[tr_idx],
                    x_val=x_train[va_idx],
                    y_val=y_train[va_idx],
                    x_test=x_test,
                    args=args,
                    device=torch.device("cpu"),
                )
            else:
                raise
        oof[va_idx] = val_pred
        covered_mask[va_idx] = True
        test_accum += test_pred / float(len(folds))
        fold_score = float(macro_map_score(y_train[va_idx], val_pred))
        fold_scores.append(fold_score)
        fold_best_maps.append(float(best_map))
        fold_best_epochs.append(int(best_epoch))

    model_name = f"deep_spec_{args.arch}_seed{args.seed}"
    oof_path = artifacts_dir / f"{model_name}_oof.npy"
    test_path = artifacts_dir / f"{model_name}_test.npy"
    np.save(oof_path, oof)
    np.save(test_path, test_accum)
    np.save(out_dir / "train_track_ids.npy", train_ids)
    np.save(out_dir / "test_track_ids.npy", test_ids)
    np.save(out_dir / "oof_targets.npy", y_train)
    covered_idx = np.where(covered_mask)[0].astype(np.int64)
    uncovered_idx = np.where(~covered_mask)[0].astype(np.int64)
    np.save(out_dir / "oof_covered_idx.npy", covered_idx)

    if len(covered_idx) == 0:
        raise RuntimeError("no covered validation rows produced by forward CV")

    macro_raw_full = float(macro_map_score(y_train, oof))
    macro_covered = float(macro_map_score(y_train[covered_idx], oof[covered_idx]))
    per_class_covered = per_class_average_precision(y_train[covered_idx], oof[covered_idx])
    teacher_diag: dict[str, float] = {}
    oof_full_filled = oof.copy()
    macro_full_filled_teacher = 0.0
    if str(args.teacher_oof_csv).strip():
        teacher_df = pd.read_csv(args.teacher_oof_csv, usecols=[str(args.id_col), *CLASSES])
        tpos = {int(v): i for i, v in enumerate(teacher_df[str(args.id_col)].to_numpy(dtype=np.int64).tolist())}
        miss = [int(tid) for tid in train_ids.tolist() if int(tid) not in tpos]
        if miss:
            raise ValueError(f"teacher_oof_csv missing {len(miss)} train ids; first={miss[:10]}")
        tarr = teacher_df[CLASSES].to_numpy(dtype=np.float32)
        teacher = np.stack([tarr[tpos[int(tid)]] for tid in train_ids.tolist()], axis=0).astype(np.float32)
        oof_full_filled[uncovered_idx] = teacher[uncovered_idx]
        macro_full_filled_teacher = float(macro_map_score(y_train, oof_full_filled))
        teacher_diag = _teacher_diag(
            y_true=y_train[covered_idx],
            teacher=teacher[covered_idx],
            model=oof[covered_idx],
        )
    np.save(out_dir / "oof_forward_cv_complete.npy", oof_full_filled.astype(np.float32))

    summary: dict[str, Any] = {
        "output_dir": str(out_dir),
        "spec_dir": str(spec_dir),
        "device": str(device),
        "requested_device": str(args.device),
        "arch": str(args.arch),
        "seed": int(args.seed),
        "split_mode": str(args.split_mode),
        "n_splits": int(len(folds)),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "macro_map_raw_full": float(macro_raw_full),
        "macro_map_covered": float(macro_covered),
        "macro_map_full_filled_teacher": float(macro_full_filled_teacher),
        "covered_ratio": float(len(covered_idx) / len(x_train)),
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
            model_name: {
                "type": "spec_cnn",
                "architecture": str(args.arch),
                "seed": int(args.seed),
                "oof_path": str(oof_path),
                "test_path": str(test_path),
                "macro_map_raw_full": float(macro_raw_full),
                "macro_map_covered": float(macro_covered),
                "macro_map_full_filled_teacher": float(macro_full_filled_teacher),
                "per_class_ap_covered": {k: float(v) for k, v in per_class_covered.items()},
                "fold_scores": [float(v) for v in fold_scores],
            }
        },
        "track_ids_train_path": str((out_dir / "train_track_ids.npy").resolve()),
        "track_ids_test_path": str((out_dir / "test_track_ids.npy").resolve()),
    }
    dump_json(out_dir / "oof_summary.json", summary)

    if str(args.sample_submission).strip():
        sample_df = pd.read_csv(args.sample_submission, usecols=[str(args.id_col)])
        sample_ids = sample_df[str(args.id_col)].to_numpy(dtype=np.int64)
        if not np.array_equal(sample_ids, test_ids):
            idx = {int(tid): i for i, tid in enumerate(test_ids.tolist())}
            miss = [int(tid) for tid in sample_ids.tolist() if int(tid) not in idx]
            if miss:
                raise ValueError(f"sample_submission ids missing in test predictions; first={miss[:10]}")
            take_test = np.array([idx[int(tid)] for tid in sample_ids.tolist()], dtype=np.int64)
            pred_sub = test_accum[take_test]
        else:
            pred_sub = test_accum
        sub = pd.DataFrame({str(args.id_col): sample_ids})
        for i, cls in enumerate(CLASSES):
            sub[cls] = np.clip(pred_sub[:, i], 0.0, 1.0).astype(np.float32)
        sub_path = out_dir / str(args.out_name)
        sub.to_csv(sub_path, index=False)
        summary["submission_path"] = str(sub_path)
        with (out_dir / "oof_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2)

    print("=== SPEC CNN TRAIN COMPLETE ===", flush=True)
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
