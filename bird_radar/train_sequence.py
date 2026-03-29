from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CACHE_DIR,
    CLASSES,
    CLASS_TO_INDEX,
    DEFAULT_DATA_DIR,
    MAX_LEN,
    N_FOLDS,
    SEQUENCE_DIR,
    SEQUENCE_EPOCHS,
    SEQUENCE_LR,
    SEQUENCE_PATIENCE,
    SEQUENCE_SEEDS,
    SEQUENCE_WEIGHT_DECAY,
    ensure_dirs,
)
from src.calibration import apply_temperature_scaling, fit_temperature_scaling
from src.dataset import TrackSequenceDataset, collate_sequence_batch
from src.metrics import macro_map_score, one_hot_labels, per_class_average_precision
from src.models.compact_hybrid_model import CompactHybridModel, count_trainable_parameters
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache
from src.training import (
    EarlyStopping,
    HybridLoss,
    compute_class_weights,
    evaluate_model,
    load_checkpoint,
    predict_logits,
    save_checkpoint,
    save_json,
    seed_everything,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(SEQUENCE_DIR))
    parser.add_argument("--cache-dir", type=str, default=str(CACHE_DIR))
    parser.add_argument("--epochs", type=int, default=SEQUENCE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=SEQUENCE_WEIGHT_DECAY)
    parser.add_argument("--lr", type=float, default=SEQUENCE_LR)
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in SEQUENCE_SEEDS),
        help="Comma-separated list of seeds",
    )
    return parser.parse_args()


def _load_or_build_cache(df: pd.DataFrame, cache_path: Path) -> dict:
    if cache_path.exists():
        return load_track_cache(cache_path)
    cache = build_track_cache(df)
    save_track_cache(cache, cache_path)
    return cache


def _dataloader(
    df: pd.DataFrame,
    track_cache: dict,
    y: np.ndarray | None,
    batch_size: int,
    max_len: int,
    training: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    ds = TrackSequenceDataset(df=df, track_cache=track_cache, y=y, max_len=max_len, training=training)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_sequence_batch,
        drop_last=False,
    )


def main() -> None:
    args = parse_args()
    ensure_dirs()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    if not run_seeds:
        raise RuntimeError("No valid seeds provided via --seeds")

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    if "observation_id" not in train_df.columns:
        raise RuntimeError("train.csv must contain observation_id for StratifiedGroupKFold grouping")
    if StratifiedGroupKFold is None:
        raise RuntimeError("scikit-learn StratifiedGroupKFold is required for sequence training")

    labels_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y_all = one_hot_labels(labels_idx, len(CLASSES))
    group_col = "observation_id"
    groups = train_df[group_col].to_numpy()

    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")

    device = torch.device(args.device)
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=2026)
    cv_splits = list(cv.split(train_df, labels_idx, groups=groups))

    aggregate_oof: list[np.ndarray] = []
    aggregate_test: list[np.ndarray] = []

    for seed in run_seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(seed)

        oof_probs = np.zeros((len(train_df), len(CLASSES)), dtype=np.float32)
        oof_logits = np.zeros_like(oof_probs)
        test_fold_probs: list[np.ndarray] = []
        fold_logs: list[dict] = []

        for fold, (tr_idx, va_idx) in enumerate(cv_splits):
            tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
            va_df = train_df.iloc[va_idx].reset_index(drop=True)
            y_tr = y_all[tr_idx]
            y_va = y_all[va_idx]

            train_loader = _dataloader(
                tr_df, train_cache, y_tr, args.batch_size, args.max_len, True, args.num_workers, device
            )
            valid_loader = _dataloader(
                va_df, train_cache, y_va, args.batch_size, args.max_len, False, args.num_workers, device
            )
            test_loader = _dataloader(
                test_df, test_cache, None, args.batch_size, args.max_len, False, args.num_workers, device
            )

            model = CompactHybridModel(
                input_dim=9,
                d_model=96,
                n_heads=4,
                num_layers=2,
                dim_feedforward=192,
                dropout=args.dropout,
                num_classes=len(CLASSES),
            ).to(device)
            n_params = count_trainable_parameters(model)
            if n_params > 3_000_000:
                raise RuntimeError(f"Model exceeds parameter budget: {n_params}")

            class_weights = compute_class_weights(y_tr)
            pos_weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
            criterion = HybridLoss(pos_weight=pos_weight, label_smoothing=0.05)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            total_steps = max(1, args.epochs * max(1, len(train_loader)))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
            scaler = GradScaler(enabled=device.type == "cuda")
            early_stopping = EarlyStopping(patience=SEQUENCE_PATIENCE, mode="max")
            ckpt_path = seed_dir / f"fold_{fold}.pt"

            best_metric = -1.0
            for epoch in range(args.epochs):
                train_loss = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    device=device,
                    scaler=scaler,
                    grad_clip=1.0,
                )
                val_info = evaluate_model(model, valid_loader, device)
                val_map = float(val_info["macro_map"])
                if early_stopping.step(val_map, epoch):
                    best_metric = val_map
                    save_checkpoint(
                        ckpt_path,
                        model,
                        extra={"epoch": epoch, "best_macro_map": val_map, "seed": seed, "fold": fold},
                    )
                print(
                    f"[sequence][seed={seed}][fold={fold}] epoch={epoch+1}/{args.epochs} "
                    f"train_loss={train_loss:.5f} val_macro_map={val_map:.5f}",
                    flush=True,
                )
                if early_stopping.should_stop:
                    break

            load_checkpoint(ckpt_path, model, map_location=device)

            va_logits, _, va_track_ids = predict_logits(model, valid_loader, device)
            va_probs = 1.0 / (1.0 + np.exp(-np.clip(va_logits, -40.0, 40.0)))
            expected_va_ids = va_df["track_id"].to_numpy()
            if not np.array_equal(va_track_ids.astype(expected_va_ids.dtype), expected_va_ids):
                raise RuntimeError("Validation track_id order mismatch")

            te_logits, _, te_track_ids = predict_logits(model, test_loader, device)
            te_probs = 1.0 / (1.0 + np.exp(-np.clip(te_logits, -40.0, 40.0)))
            expected_te_ids = test_df["track_id"].to_numpy()
            if not np.array_equal(te_track_ids.astype(expected_te_ids.dtype), expected_te_ids):
                raise RuntimeError("Test track_id order mismatch")

            oof_probs[va_idx] = va_probs
            oof_logits[va_idx] = va_logits
            test_fold_probs.append(te_probs)

            fold_class_scores = per_class_average_precision(y_va, va_probs)
            fold_log = {
                "seed": seed,
                "fold": fold,
                "best_val_macro_map": float(best_metric),
                "val_macro_map_recomputed": float(macro_map_score(y_va, va_probs)),
                "per_class_map": fold_class_scores,
                "train_size": int(len(tr_idx)),
                "valid_size": int(len(va_idx)),
            }
            fold_logs.append(fold_log)
            print(
                f"[sequence][seed={seed}][fold={fold}] best_macro_map={fold_log['val_macro_map_recomputed']:.5f}",
                flush=True,
            )

        test_probs = np.mean(np.stack(test_fold_probs, axis=0), axis=0)
        temps = fit_temperature_scaling(y_all, oof_probs)
        oof_probs_cal = apply_temperature_scaling(oof_probs, temps)
        test_probs_cal = apply_temperature_scaling(test_probs, temps)

        overall_class_map = per_class_average_precision(y_all, oof_probs_cal)
        overall_macro = macro_map_score(y_all, oof_probs_cal)
        fold_scores = [float(f["val_macro_map_recomputed"]) for f in fold_logs]
        fold_mean = float(np.mean(fold_scores)) if fold_scores else 0.0
        fold_std = float(np.std(fold_scores)) if fold_scores else 0.0
        fold_spread = float(np.max(fold_scores) - np.min(fold_scores)) if fold_scores else 0.0
        summary = {
            "seed": seed,
            "group_column": group_col,
            "cv_splitter": "StratifiedGroupKFold",
            "hyperparams": {
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "max_len": args.max_len,
                "epochs": args.epochs,
            },
            "macro_map_oof_raw": float(macro_map_score(y_all, oof_probs)),
            "macro_map_oof_calibrated": float(overall_macro),
            "fold_macro_map_mean": fold_mean,
            "fold_macro_map_std": fold_std,
            "fold_macro_map_spread": fold_spread,
            "fold_macro_map_spread_lt_0_02": bool(fold_spread < 0.02),
            "per_class_map_calibrated": overall_class_map,
            "folds": fold_logs,
        }

        np.save(seed_dir / "oof_probs.npy", oof_probs)
        np.save(seed_dir / "oof_logits.npy", oof_logits)
        np.save(seed_dir / "oof_calibrated.npy", oof_probs_cal)
        np.save(seed_dir / "test_probs.npy", test_probs)
        np.save(seed_dir / "test_calibrated.npy", test_probs_cal)
        np.save(seed_dir / "temperatures.npy", temps.astype(np.float32))
        save_json(seed_dir / "scores.json", summary)

        aggregate_oof.append(oof_probs_cal)
        aggregate_test.append(test_probs_cal)

        print(f"[sequence][seed={seed}] OOF macro mAP (calibrated) = {overall_macro:.5f}", flush=True)
        print(
            f"[sequence][seed={seed}] fold_mean={fold_mean:.5f} fold_std={fold_std:.5f} fold_spread={fold_spread:.5f}",
            flush=True,
        )
        for cls, score in overall_class_map.items():
            print(f"[sequence][seed={seed}] {cls}: {score:.5f}", flush=True)

    mean_oof = np.mean(np.stack(aggregate_oof, axis=0), axis=0)
    mean_test = np.mean(np.stack(aggregate_test, axis=0), axis=0)
    np.save(output_dir / "oof_sequence_mean_calibrated.npy", mean_oof.astype(np.float32))
    np.save(output_dir / "test_sequence_mean_calibrated.npy", mean_test.astype(np.float32))
    seed_scores = [float(macro_map_score(y_all, p)) for p in aggregate_oof]
    save_json(
        output_dir / "summary.json",
        {
            "seeds": SEQUENCE_SEEDS,
            "run_seeds": run_seeds,
            "group_column": group_col,
            "cv_splitter": "StratifiedGroupKFold",
            "macro_map_oof_sequence_mean": float(macro_map_score(y_all, mean_oof)),
            "seed_macro_map_oof": seed_scores,
            "seed_macro_map_std": float(np.std(seed_scores)) if seed_scores else 0.0,
            "seed_macro_map_spread": float(np.max(seed_scores) - np.min(seed_scores)) if seed_scores else 0.0,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
