"""
Species Auxiliary Model
=======================
Train species-level OvR model (68 classes) on the same engineered features.
Save OOF/test species probabilities for later join into group-level model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Make local project imports work from repo root
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "bird_radar"))
sys.path.insert(0, str(REPO_ROOT / "bird_radar" / "src"))

from feature_engineering import build_feature_frame  # type: ignore  # noqa: E402
from preprocessing import build_track_cache, load_track_cache  # type: ignore  # noqa: E402
from cv import make_forward_temporal_group_folds  # type: ignore  # noqa: E402


CLASSES_GROUP = [
    "Clutter",
    "Cormorants",
    "Pigeons",
    "Ducks",
    "Geese",
    "Gulls",
    "Birds of Prey",
    "Waders",
    "Songbirds",
]
GROUP_TO_IDX = {c: i for i, c in enumerate(CLASSES_GROUP)}

LGB_CFG = dict(
    objective="binary",
    metric="average_precision",
    n_estimators=800,
    learning_rate=0.02,
    num_leaves=47,
    colsample_bytree=0.7,
    subsample=0.8,
    subsample_freq=1,
    reg_alpha=0.1,
    reg_lambda=0.4,
    min_child_samples=10,
    n_jobs=-1,
    verbosity=-1,
)

N_FOLDS = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--cv-mode", type=str, default="stratified", choices=["stratified", "forward"])
    p.add_argument("--forward-splits", type=int, default=5)
    p.add_argument("--forward-complete", type=str, default="backcast_last", choices=["off", "backcast_last", "backcast_all"])
    return p.parse_args()


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict]:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    import pickle

    with path.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cache


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    print(f"Train: {len(train_df)} tracks, Test: {len(test_df)} tracks")
    return train_df, test_df


def build_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cache_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\nBuilding feature matrix...")
    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")

    feat_train = build_feature_frame(train_df, track_cache=train_cache)
    feat_test = build_feature_frame(test_df, track_cache=test_cache)
    print(f"Features before blacklist: {feat_train.shape[1]} cols")

    blacklist_patterns = [
        "observer_pos",
        "hours_since_sunrise",
        "frac_of_day",
        "day_length_hours",
        "hour_utc",
        "is_daytime",
    ]
    drop_cols = [c for c in feat_train.columns if any(p in c for p in blacklist_patterns)]
    feat_train = feat_train.drop(columns=drop_cols, errors="ignore")
    feat_test = feat_test.drop(columns=drop_cols, errors="ignore")
    print(f"After blacklist: {feat_train.shape[1]} cols, dropped {len(drop_cols)}")
    return feat_train, feat_test


def _macro_ap_per_species(probs: np.ndarray, labels: np.ndarray, n_species: int) -> float:
    aps: list[float] = []
    for ci in range(n_species):
        y = (labels == ci).astype(int)
        if y.sum() == 0:
            continue
        aps.append(float(average_precision_score(y, probs[:, ci])))
    return float(np.mean(aps)) if aps else 0.0


def train_species_ovr(
    X_train: pd.DataFrame,
    y_species: np.ndarray,
    X_test: pd.DataFrame,
    le: LabelEncoder,
    seed: int,
    cv_mode: str,
    cv_meta: pd.DataFrame | None = None,
    forward_splits: int = 5,
    forward_complete: str = "backcast_last",
) -> tuple[np.ndarray, np.ndarray]:
    n_species = len(le.classes_)
    oof_probs = np.zeros((len(X_train), n_species), dtype=np.float32)
    test_preds: list[np.ndarray] = []
    covered = np.zeros(len(X_train), dtype=bool)

    def _fit_fold(tr_idx: np.ndarray, va_idx: np.ndarray, fold_seed: int) -> np.ndarray:
        X_tr = X_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        fold_test_preds = np.zeros((len(X_test), n_species), dtype=np.float32)
        for ci, _sp_name in enumerate(le.classes_):
            ytr = (y_species[tr_idx] == ci).astype(np.int32)
            yva = (y_species[va_idx] == ci).astype(np.int32)
            pos = int(ytr.sum())
            neg = int(len(ytr) - pos)
            if pos < 2:
                continue

            model = lgb.LGBMClassifier(
                **LGB_CFG,
                scale_pos_weight=float(max(neg, 1) / max(pos, 1)),
                random_state=fold_seed + ci,
            )
            model.fit(
                X_tr,
                ytr,
                eval_set=[(X_va, yva)],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
            )
            oof_probs[va_idx, ci] = model.predict_proba(X_va)[:, 1].astype(np.float32)
            fold_test_preds[:, ci] = model.predict_proba(X_test)[:, 1].astype(np.float32)
        return fold_test_preds

    if cv_mode == "stratified":
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_species), start=1):
            print(f"\n  Fold {fold}/{N_FOLDS} — train={len(tr_idx)}, val={len(va_idx)}")
            fold_test_preds = _fit_fold(tr_idx, va_idx, seed + fold * 1000)
            covered[va_idx] = True
            test_preds.append(fold_test_preds)
            fold_macro = _macro_ap_per_species(oof_probs[va_idx], y_species[va_idx], n_species)
            print(f"    fold macro-AP (species): {fold_macro:.4f}")
    else:
        if cv_meta is None:
            raise ValueError("cv_meta is required for forward cv mode")
        folds = make_forward_temporal_group_folds(
            cv_meta,
            timestamp_col="_cv_ts",
            group_col="_cv_group",
            n_splits=max(int(forward_splits), 2),
        )
        print(f"\nForward CV folds: {len(folds)}")
        for fold_id, (tr_idx, va_idx) in enumerate(folds, start=1):
            print(f"\n  Forward fold {fold_id}/{len(folds)} — train={len(tr_idx)}, val={len(va_idx)}")
            fold_test_preds = _fit_fold(tr_idx, va_idx, seed + fold_id * 1000)
            covered[va_idx] = True
            test_preds.append(fold_test_preds)
            fold_macro = _macro_ap_per_species(oof_probs[va_idx], y_species[va_idx], n_species)
            print(f"    fold macro-AP (species): {fold_macro:.4f}")

        if forward_complete != "off":
            missing_idx = np.where(~covered)[0]
            if len(missing_idx) > 0 and len(folds) > 0:
                warmup_idx = np.array(folds[0][0], dtype=np.int64)
                if forward_complete == "backcast_last":
                    backcast_train = np.setdiff1d(np.array(folds[-1][0], dtype=np.int64), warmup_idx, assume_unique=False)
                else:
                    backcast_train = np.setdiff1d(np.arange(len(X_train), dtype=np.int64), warmup_idx, assume_unique=False)
                if len(backcast_train) > 0:
                    print(
                        f"\nForward complete: mode={forward_complete} train={len(backcast_train)} "
                        f"missing={len(missing_idx)}"
                    )
                    _ = _fit_fold(backcast_train, missing_idx, seed + 900000)
                    covered[missing_idx] = True

    test_probs = np.mean(np.stack(test_preds, axis=0), axis=0).astype(np.float32) if test_preds else np.zeros((len(X_test), n_species), dtype=np.float32)
    print(f"\nOOF coverage: {int(covered.sum())}/{len(covered)} ({covered.mean():.3f})")
    return oof_probs, test_probs


def save_species_features(
    track_ids: np.ndarray,
    probs: np.ndarray,
    le: LabelEncoder,
    path: Path,
    prefix: str = "sp_",
) -> pd.DataFrame:
    cols = {}
    for ci, sp in enumerate(le.classes_):
        col = f"{prefix}{str(sp).replace(' ', '_').replace('-', '_')}"
        cols[col] = probs[:, ci]
    df = pd.DataFrame(cols)
    df.insert(0, "track_id", track_ids)
    df.to_parquet(path, index=False)
    print(f"Saved {path} shape={df.shape}")
    return df


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed)

    train_df, test_df = load_data(data_dir)

    le = LabelEncoder()
    y_species = le.fit_transform(train_df["bird_species"].values)
    print(f"\nSpecies: {len(le.classes_)} unique")
    species_counts = pd.Series(y_species).value_counts()
    print(f"Rarest species counts: {species_counts.tail(5).to_dict()}")

    feat_train, feat_test = build_features(train_df, test_df, cache_dir)
    drop_targets = [c for c in ("bird_group", "bird_species", "track_id") if c in feat_train.columns]
    X_train = feat_train.drop(columns=drop_targets, errors="ignore")
    X_test = feat_test.drop(columns=drop_targets, errors="ignore")

    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols].astype(np.float32)
    X_test = X_test[common_cols].astype(np.float32)
    print(f"\nFeature matrix: train={X_train.shape}, test={X_test.shape}")

    print(f"\nTraining species OvR ({len(le.classes_)} classes, cv_mode={args.cv_mode})...")
    cv_meta = None
    if args.cv_mode == "forward":
        cv_meta = pd.DataFrame(
            {
                "_cv_ts": pd.to_datetime(train_df["timestamp_start_radar_utc"], utc=True, errors="coerce"),
                "_cv_group": train_df["observation_id"].to_numpy(),
            }
        )
    oof_probs, test_probs = train_species_ovr(
        X_train,
        y_species,
        X_test,
        le,
        seed,
        cv_mode=args.cv_mode,
        cv_meta=cv_meta,
        forward_splits=int(args.forward_splits),
        forward_complete=str(args.forward_complete),
    )
    species_macro = _macro_ap_per_species(oof_probs, y_species, len(le.classes_))
    print(f"\nOOF macro-AP (species level): {species_macro:.4f}")

    # species -> group mapping is unique (verified earlier), use first observed mapping
    sp_to_group = train_df.drop_duplicates("bird_species").set_index("bird_species")["bird_group"].to_dict()
    group_from_species = np.array([GROUP_TO_IDX[sp_to_group[str(sp)]] for sp in le.classes_], dtype=np.int32)

    n_groups = len(CLASSES_GROUP)
    oof_group_from_sp = np.zeros((len(oof_probs), n_groups), dtype=np.float32)
    test_group_from_sp = np.zeros((len(test_probs), n_groups), dtype=np.float32)
    for ci, gi in enumerate(group_from_species):
        oof_group_from_sp[:, gi] = np.maximum(oof_group_from_sp[:, gi], oof_probs[:, ci])
        test_group_from_sp[:, gi] = np.maximum(test_group_from_sp[:, gi], test_probs[:, ci])

    y_group = train_df["bird_group"].map(GROUP_TO_IDX).to_numpy(dtype=np.int32)
    group_aps: dict[str, float] = {}
    for gi, gname in enumerate(CLASSES_GROUP):
        y_g = (y_group == gi).astype(np.int32)
        if y_g.sum() == 0:
            continue
        group_aps[gname] = float(average_precision_score(y_g, oof_group_from_sp[:, gi]))

    macro_group = float(np.mean(list(group_aps.values())))
    print(f"\nGroup-level AP from species aggregation: {macro_group:.4f}")
    for gname, ap in sorted(group_aps.items(), key=lambda x: -x[1]):
        print(f"  {gname:<22} {ap:.4f}")

    train_track_ids = train_df["track_id"].to_numpy()
    test_track_ids = test_df["track_id"].to_numpy()
    oof_path = output_dir / "species_oof.parquet"
    test_path = output_dir / "species_test.parquet"
    save_species_features(train_track_ids, oof_probs, le, oof_path, prefix="sp_")
    save_species_features(test_track_ids, test_probs, le, test_path, prefix="sp_")

    oof_group_df = pd.DataFrame(oof_group_from_sp, columns=[f"spg_{c}" for c in CLASSES_GROUP])
    test_group_df = pd.DataFrame(test_group_from_sp, columns=[f"spg_{c}" for c in CLASSES_GROUP])
    oof_group_df.insert(0, "track_id", train_track_ids)
    test_group_df.insert(0, "track_id", test_track_ids)
    oof_group_df.to_parquet(output_dir / "species_group_oof.parquet", index=False)
    test_group_df.to_parquet(output_dir / "species_group_test.parquet", index=False)
    print("Saved species_group_oof.parquet + species_group_test.parquet")

    report = {
        "species_oof_macro_ap": species_macro,
        "group_from_species_macro_ap": macro_group,
        "group_aps": group_aps,
        "n_species": int(len(le.classes_)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(common_cols)),
        "seed": int(seed),
        "cv_mode": str(args.cv_mode),
        "forward_splits": int(args.forward_splits),
        "forward_complete": str(args.forward_complete),
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {output_dir}/report.json")
    print("\nDONE.")
    print(f"  --external-train-parquet {oof_path}")
    print(f"  --external-test-parquet  {test_path}")
    print("  OR use species_group_oof/species_group_test parquet (9 features).")


if __name__ == "__main__":
    main()
