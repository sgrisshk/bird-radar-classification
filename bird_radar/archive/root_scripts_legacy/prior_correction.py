from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EBIRD_ABUNDANCE = {
    "Cormorants": {1: 0.9, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.3, 6: 0.2, 7: 0.3, 8: 0.4, 9: 0.5, 10: 0.7, 11: 0.9, 12: 1.0},
    "Ducks": {1: 1.0, 2: 0.9, 3: 0.7, 4: 0.5, 5: 0.3, 6: 0.2, 7: 0.3, 8: 0.5, 9: 0.7, 10: 0.9, 11: 1.0, 12: 1.0},
    "Geese": {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.4, 5: 0.1, 6: 0.05, 7: 0.05, 8: 0.2, 9: 0.6, 10: 0.9, 11: 1.0, 12: 1.0},
    "Gulls": {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.5, 6: 0.5, 7: 0.6, 8: 0.7, 9: 0.8, 10: 0.9, 11: 0.9, 12: 1.0},
    "Waders": {1: 0.4, 2: 0.4, 3: 0.6, 4: 0.9, 5: 1.0, 6: 0.7, 7: 0.8, 8: 1.0, 9: 0.9, 10: 0.6, 11: 0.4, 12: 0.3},
    "Birds of Prey": {1: 0.7, 2: 0.7, 3: 0.8, 4: 0.9, 5: 0.8, 6: 0.6, 7: 0.6, 8: 0.7, 9: 0.9, 10: 1.0, 11: 0.8, 12: 0.7},
    "Songbirds": {1: 0.3, 2: 0.3, 3: 0.5, 4: 0.8, 5: 1.0, 6: 0.9, 7: 0.8, 8: 0.9, 9: 1.0, 10: 0.7, 11: 0.4, 12: 0.3},
    "Clutter": {m: 1.0 for m in range(1, 13)},
    "Pigeons": {m: 0.8 for m in range(1, 13)},
}


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-9)


def get_timestamp_col(df: pd.DataFrame) -> pd.Series:
    for col in [
        "timestamp_start_radar_utc",
        "start_time",
        "timestamp",
        "time",
        "date",
        "datetime",
    ]:
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce", utc=True)
    raise ValueError(f"No timestamp column found. Columns: {list(df.columns)}")


def compute_month_dist(df: pd.DataFrame) -> dict[int, float]:
    ts = get_timestamp_col(df)
    months = ts.dt.month.dropna().astype(int)
    vc = months.value_counts(normalize=True).sort_index()
    return {int(k): float(v) for k, v in vc.items()}


def compute_abundance_prior(month_dist: dict[int, float], classes: list[str]) -> np.ndarray:
    priors = []
    for cls in classes:
        abund = EBIRD_ABUNDANCE.get(cls)
        if abund is None:
            expected = 0.5
        else:
            expected = sum(month_dist.get(m, 0.0) * abund.get(m, 0.5) for m in range(1, 13))
        priors.append(expected)
    p = np.asarray(priors, dtype=np.float64)
    p = p / (p.sum() + 1e-9)
    return p


def apply_prior_correction(
    probs: np.ndarray,
    prior_train: np.ndarray,
    prior_test: np.ndarray,
    temperature: float,
    mode: str,
) -> np.ndarray:
    ratio = prior_test / (prior_train + 1e-9)
    if mode == "multiplicative":
        ratio_t = ratio ** float(temperature)
        corrected = probs * ratio_t[np.newaxis, :]
        corrected = corrected / (corrected.sum(axis=1, keepdims=True) + 1e-9)
        return corrected
    if mode == "additive":
        logits = np.log(np.clip(probs, 1e-9, 1.0))
        shift = np.log(ratio + 1e-9) * float(temperature)
        return _softmax(logits + shift[np.newaxis, :])
    raise ValueError(f"unknown mode: {mode}")


def run_prior_correction(
    submission_path: str,
    train_csv_path: str,
    test_csv_path: str,
    output_path: str,
    temperature: float,
    mode: str,
) -> pd.DataFrame:
    print("=" * 64)
    print("Bayesian Prior Correction")
    print(f"submission : {submission_path}")
    print(f"train_csv   : {train_csv_path}")
    print(f"test_csv    : {test_csv_path}")
    print(f"temperature : {temperature:.2f}")
    print(f"mode        : {mode}")
    print("=" * 64)

    sub = pd.read_csv(submission_path)
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    species_cols = [c for c in sub.columns if c != "track_id"]
    probs = sub[species_cols].to_numpy(dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-9)

    train_months = compute_month_dist(train_df)
    test_months = compute_month_dist(test_df)

    prior_train = compute_abundance_prior(train_months, species_cols)
    prior_test = compute_abundance_prior(test_months, species_cols)
    ratio = prior_test / (prior_train + 1e-9)

    print("\nTrain month distribution:")
    print({k: f"{v:.3f}" for k, v in sorted(train_months.items())})
    print("Test month distribution:")
    print({k: f"{v:.3f}" for k, v in sorted(test_months.items())})

    print("\nPrior comparison (train -> test):")
    for cls, p_tr, p_te, r in zip(species_cols, prior_train, prior_test, ratio):
        mark = "↑" if r > 1.10 else ("↓" if r < 0.90 else "~")
        print(f"  {cls:<20} train={p_tr:.4f}  test={p_te:.4f}  ratio={r:.3f} {mark}")

    print("\nTemperature scan:")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        c = apply_prior_correction(probs, prior_train, prior_test, t, mode)
        mean_entropy = float((-c * np.log(c + 1e-9)).sum(axis=1).mean())
        mean_max = float(c.max(axis=1).mean())
        print(f"  t={t:.2f}: mean_entropy={mean_entropy:.4f}  mean_maxprob={mean_max:.4f}")

    corrected = apply_prior_correction(probs, prior_train, prior_test, temperature, mode)

    out = sub.copy()
    out[species_cols] = corrected
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"\nSaved: {output}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--mode", choices=["multiplicative", "additive"], default="multiplicative")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prior_correction(
        submission_path=args.submission,
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv,
        output_path=args.output,
        temperature=args.temperature,
        mode=args.mode,
    )
