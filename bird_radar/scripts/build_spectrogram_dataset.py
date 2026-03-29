from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import stft

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_TO_INDEX, CLASSES
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache


CHANNEL_TO_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
    "altitude": 2,
    "rcs": 3,
    "speed": 4,
    "vertical_speed": 5,
    "vs": 5,
    "acceleration": 6,
    "accel": 6,
    "curvature": 7,
    "dt": 8,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build fixed-size STFT spectrogram tensors for train/test tracks.")
    p.add_argument("--data-dir", type=str, required=True, help="Directory containing train.csv and test.csv")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory")
    p.add_argument("--cache-dir", type=str, default="", help="Cache dir for track_cache pkl files")
    p.add_argument("--channels", type=str, default="rcs,speed", help="Comma-separated sequence channels")
    p.add_argument("--stft-nperseg", type=int, default=64)
    p.add_argument("--stft-hop", type=int, default=16)
    p.add_argument("--time-len", type=int, default=256, help="Output spectrogram time bins after pad/crop")
    p.add_argument("--norm-mode", type=str, default="robust", choices=["robust", "zscore", "none"])
    p.add_argument("--crop-mode", type=str, default="center", choices=["center", "left", "right"])
    p.add_argument("--rebuild-cache", action="store_true")
    return p.parse_args()


def _load_or_build_cache(df: pd.DataFrame, cache_path: Path, rebuild: bool) -> dict[int, dict[str, Any]]:
    if cache_path.exists() and not rebuild:
        return load_track_cache(cache_path)
    cache = build_track_cache(df)
    save_track_cache(cache, cache_path)
    return cache


def _parse_channels(raw: str) -> tuple[list[str], list[int]]:
    names = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not names:
        raise ValueError("no channels specified")
    idxs: list[int] = []
    out_names: list[str] = []
    for name in names:
        if name not in CHANNEL_TO_INDEX:
            raise ValueError(f"unknown channel: {name}")
        idxs.append(int(CHANNEL_TO_INDEX[name]))
        out_names.append(name)
    return out_names, idxs


def _norm_1d(x: np.ndarray, mode: str) -> np.ndarray:
    out = x.astype(np.float32, copy=False)
    if mode == "none":
        return out
    if mode == "zscore":
        mu = float(np.mean(out))
        sd = float(np.std(out))
        if sd < 1e-6:
            sd = 1.0
        return ((out - mu) / sd).astype(np.float32)
    med = float(np.median(out))
    q25 = float(np.quantile(out, 0.25))
    q75 = float(np.quantile(out, 0.75))
    iqr = q75 - q25
    if iqr < 1e-6:
        iqr = 1.0
    return ((out - med) / iqr).astype(np.float32)


def _pad_or_crop_time(spec_ft: np.ndarray, out_t: int, crop_mode: str) -> np.ndarray:
    f, t = spec_ft.shape
    if t == out_t:
        return spec_ft.astype(np.float32, copy=False)
    if t > out_t:
        if crop_mode == "left":
            return spec_ft[:, :out_t].astype(np.float32, copy=False)
        if crop_mode == "right":
            return spec_ft[:, -out_t:].astype(np.float32, copy=False)
        start = max(0, (t - out_t) // 2)
        return spec_ft[:, start : start + out_t].astype(np.float32, copy=False)
    out = np.zeros((f, out_t), dtype=np.float32)
    out[:, :t] = spec_ft.astype(np.float32, copy=False)
    return out


def _track_to_spec(
    raw_features: np.ndarray,
    channel_idxs: list[int],
    norm_mode: str,
    nperseg: int,
    hop: int,
    time_len: int,
    crop_mode: str,
) -> np.ndarray:
    specs: list[np.ndarray] = []
    noverlap = nperseg - hop
    if noverlap < 0:
        raise ValueError(f"stft-hop={hop} cannot be greater than stft-nperseg={nperseg}")

    for ch in channel_idxs:
        x = raw_features[:, ch].astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = _norm_1d(x, norm_mode)
        if len(x) < nperseg:
            pad = np.zeros((nperseg,), dtype=np.float32)
            pad[: len(x)] = x
            x = pad
        _, _, zxx = stft(
            x,
            fs=1.0,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None,
            padded=False,
        )
        mag = np.abs(zxx).astype(np.float32)
        log_mag = np.log1p(mag).astype(np.float32)
        fixed = _pad_or_crop_time(log_mag, out_t=time_len, crop_mode=crop_mode)
        specs.append(fixed)

    out = np.stack(specs, axis=0).astype(np.float32)
    return out


def _build_array(
    df: pd.DataFrame,
    cache: dict[int, dict[str, Any]],
    channel_idxs: list[int],
    norm_mode: str,
    nperseg: int,
    hop: int,
    time_len: int,
    crop_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    ids = df["track_id"].to_numpy(dtype=np.int64)
    arr_list: list[np.ndarray] = []
    for i, tid in enumerate(ids.tolist()):
        if int(tid) not in cache:
            raise KeyError(f"track_id={int(tid)} is missing in track cache")
        raw = np.asarray(cache[int(tid)]["raw_features"], dtype=np.float32)
        spec = _track_to_spec(
            raw_features=raw,
            channel_idxs=channel_idxs,
            norm_mode=norm_mode,
            nperseg=nperseg,
            hop=hop,
            time_len=time_len,
            crop_mode=crop_mode,
        )
        arr_list.append(spec)
        if (i + 1) % 500 == 0 or (i + 1) == len(ids):
            print(f"[build] processed {i + 1}/{len(ids)} tracks", flush=True)
    arr = np.stack(arr_list, axis=0).astype(np.float32)
    return arr, ids


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else (PROJECT_ROOT / "artifacts" / "cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    nperseg = int(args.stft_nperseg)
    hop = int(args.stft_hop)
    time_len = int(args.time_len)
    if nperseg < 8:
        raise ValueError("--stft-nperseg must be >= 8")
    if hop < 1:
        raise ValueError("--stft-hop must be >= 1")
    if time_len < 8:
        raise ValueError("--time-len must be >= 8")

    channel_names, channel_idxs = _parse_channels(args.channels)
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl", rebuild=bool(args.rebuild_cache))
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl", rebuild=bool(args.rebuild_cache))

    train_spec, train_ids = _build_array(
        df=train_df,
        cache=train_cache,
        channel_idxs=channel_idxs,
        norm_mode=str(args.norm_mode),
        nperseg=nperseg,
        hop=hop,
        time_len=time_len,
        crop_mode=str(args.crop_mode),
    )
    test_spec, test_ids = _build_array(
        df=test_df,
        cache=test_cache,
        channel_idxs=channel_idxs,
        norm_mode=str(args.norm_mode),
        nperseg=nperseg,
        hop=hop,
        time_len=time_len,
        crop_mode=str(args.crop_mode),
    )

    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    if np.any(y_idx < 0) or np.any(y_idx >= len(CLASSES)):
        raise ValueError("train.csv contains unknown bird_group labels")
    train_y = np.zeros((len(y_idx), len(CLASSES)), dtype=np.float32)
    train_y[np.arange(len(y_idx)), y_idx] = 1.0

    np.save(out_dir / "train_spec.npy", train_spec)
    np.save(out_dir / "test_spec.npy", test_spec)
    np.save(out_dir / "train_y.npy", train_y)
    np.save(out_dir / "train_track_ids.npy", train_ids)
    np.save(out_dir / "test_track_ids.npy", test_ids)

    meta = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "cache_dir": str(cache_dir),
        "channels": channel_names,
        "channel_indices": channel_idxs,
        "stft_nperseg": int(nperseg),
        "stft_hop": int(hop),
        "time_len": int(time_len),
        "norm_mode": str(args.norm_mode),
        "crop_mode": str(args.crop_mode),
        "n_train": int(train_spec.shape[0]),
        "n_test": int(test_spec.shape[0]),
        "shape_train": [int(v) for v in train_spec.shape],
        "shape_test": [int(v) for v in test_spec.shape],
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    print("=== SPECTROGRAM DATASET COMPLETE ===", flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print(f"train_shape={train_spec.shape} test_shape={test_spec.shape}", flush=True)
    print(f"meta_path={out_dir / 'meta.json'}", flush=True)


if __name__ == "__main__":
    main()
