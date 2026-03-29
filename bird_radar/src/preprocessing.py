from __future__ import annotations

import ast
import pickle
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from config import SEQUENCE_FEATURES

EPS = 1e-6


def _ctx(track_id: int | None) -> str:
    return f"track_id={track_id}" if track_id is not None else "track_id=unknown"


def parse_time_array(trajectory_time: str, track_id: int | None = None) -> np.ndarray:
    try:
        arr = np.asarray(ast.literal_eval(trajectory_time), dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"failed to parse trajectory_time for {_ctx(track_id)}") from exc
    if arr.ndim != 1:
        raise ValueError(f"trajectory_time must parse to a 1D array for {_ctx(track_id)}")
    if arr.size == 0:
        raise ValueError(f"trajectory_time is empty for {_ctx(track_id)}")
    if not np.isfinite(arr).all():
        raise ValueError(f"trajectory_time has non-finite values for {_ctx(track_id)}")
    return arr


def parse_ewkb_linestring_zm_hex(
    trajectory_hex: str,
    track_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(trajectory_hex, str) or not trajectory_hex:
        raise ValueError(f"trajectory hex is empty for {_ctx(track_id)}")

    try:
        blob = bytes.fromhex(trajectory_hex)
    except Exception as exc:
        raise ValueError(f"trajectory is not valid hex for {_ctx(track_id)}") from exc

    if len(blob) < 9:
        raise ValueError(f"trajectory blob too short for {_ctx(track_id)}")

    endian_flag = blob[0]
    if endian_flag == 0:
        endian = ">"
    elif endian_flag == 1:
        endian = "<"
    else:
        raise ValueError(f"invalid endian flag={endian_flag} for {_ctx(track_id)}")

    offset = 1
    geom_type = struct.unpack_from(f"{endian}I", blob, offset)[0]
    offset += 4

    has_z = bool(geom_type & 0x80000000)
    has_m = bool(geom_type & 0x40000000)
    has_srid = bool(geom_type & 0x20000000)
    has_ewkb_flags = bool(geom_type & 0xE0000000)

    if has_ewkb_flags:
        base_type = geom_type & 0xFF
    else:
        base_type = geom_type % 1000
        dim_code = geom_type // 1000
        if dim_code in (1, 3):
            has_z = True
        if dim_code in (2, 3):
            has_m = True

    if base_type != 2:
        raise ValueError(f"geometry type={base_type} is not LINESTRING for {_ctx(track_id)}")

    if has_srid:
        if len(blob) < offset + 4:
            raise ValueError(f"missing SRID payload for {_ctx(track_id)}")
        _ = struct.unpack_from(f"{endian}I", blob, offset)[0]
        offset += 4

    if len(blob) < offset + 4:
        raise ValueError(f"missing points count for {_ctx(track_id)}")
    n_points = struct.unpack_from(f"{endian}I", blob, offset)[0]
    offset += 4

    if n_points < 2:
        raise ValueError(f"trajectory has fewer than 2 points for {_ctx(track_id)}")

    dims = 2 + int(has_z) + int(has_m)
    if dims != 4:
        raise ValueError(f"trajectory is not ZM (dims={dims}) for {_ctx(track_id)}")

    values_count = int(n_points * dims)
    bytes_needed = values_count * 8
    if len(blob) < offset + bytes_needed:
        raise ValueError(f"trajectory payload truncated for {_ctx(track_id)}")

    coords = np.frombuffer(blob, dtype=np.dtype(f"{endian}f8"), count=values_count, offset=offset)
    coords = coords.reshape(n_points, dims)

    lon = coords[:, 0].astype(np.float64, copy=False)
    lat = coords[:, 1].astype(np.float64, copy=False)
    alt = coords[:, 2].astype(np.float64, copy=False)
    rcs = coords[:, 3].astype(np.float64, copy=False)

    if not (np.isfinite(lon).all() and np.isfinite(lat).all() and np.isfinite(alt).all() and np.isfinite(rcs).all()):
        raise ValueError(f"trajectory contains non-finite coordinates for {_ctx(track_id)}")

    return lon, lat, alt, rcs


@lru_cache(maxsize=128)
def _transformer_for_epsg(epsg: int) -> Transformer:
    return Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)


def _utm_epsg_from_lon_lat(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def _safe_diff(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    dv = np.diff(values, prepend=values[:1])
    dt = np.diff(times, prepend=times[:1])
    if len(dt) > 1:
        positive = dt[dt > 0]
        fill = float(np.median(positive)) if positive.size else 1.0
        dt[0] = fill
    dt = np.clip(dt, 1e-3, None)
    return dv / dt


def _discrete_curvature(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return np.zeros_like(x, dtype=np.float32)
    dx = np.gradient(x, t, edge_order=1)
    dy = np.gradient(y, t, edge_order=1)
    ddx = np.gradient(dx, t, edge_order=1)
    ddy = np.gradient(dy, t, edge_order=1)
    num = np.abs(dx * ddy - dy * ddx)
    den = np.power(dx * dx + dy * dy + EPS, 1.5)
    return (num / den).astype(np.float32)


def _clip_features(features: np.ndarray) -> np.ndarray:
    clips = np.array(
        [
            [-30000.0, 30000.0],  # x
            [-30000.0, 30000.0],  # y
            [-500.0, 5000.0],  # altitude
            [-80.0, 80.0],  # true rcs (dB m^2)
            [0.0, 200.0],  # speed
            [-100.0, 100.0],  # vertical speed
            [-100.0, 100.0],  # acceleration
            [0.0, 10.0],  # curvature
            [1e-3, 10.0],  # dt
        ],
        dtype=np.float32,
    )
    return np.clip(features, clips[:, 0], clips[:, 1])


def _z_norm_per_track(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-5, 1.0, std)
    return (features - mean) / std


def preprocess_track(
    trajectory_hex: str,
    trajectory_time: str,
    radar_bird_size: str | None = None,
    track_id: int | None = None,
) -> dict[str, np.ndarray]:
    del radar_bird_size

    lon, lat, alt_raw, rcs_raw = parse_ewkb_linestring_zm_hex(trajectory_hex=trajectory_hex, track_id=track_id)
    t = parse_time_array(trajectory_time, track_id=track_id)

    if len(lon) != len(t):
        raise ValueError(
            f"trajectory length mismatch for {_ctx(track_id)}: coords={len(lon)} vs time={len(t)}"
        )

    alt = alt_raw.astype(np.float32)
    rcs = rcs_raw.astype(np.float32)
    t = t.astype(np.float32)

    epsg = _utm_epsg_from_lon_lat(float(lon[0]), float(lat[0]))
    transformer = _transformer_for_epsg(epsg)
    x_m, y_m = transformer.transform(lon, lat)
    x = np.asarray(x_m, dtype=np.float32)
    y = np.asarray(y_m, dtype=np.float32)

    x = x - x[0]
    y = y - y[0]

    dt = np.diff(t, prepend=t[:1]).astype(np.float32)
    if len(dt) > 1:
        positive_dt = dt[1:][dt[1:] > 0]
        dt0 = float(np.median(positive_dt)) if positive_dt.size else 1.0
        dt[0] = dt0
    dt = np.clip(dt, 1e-3, 10.0).astype(np.float32)

    dx = np.diff(x, prepend=x[:1]).astype(np.float32)
    dy = np.diff(y, prepend=y[:1]).astype(np.float32)
    dz = np.diff(alt, prepend=alt[:1]).astype(np.float32)

    speed = np.sqrt(dx * dx + dy * dy) / dt
    vertical_speed = dz / dt
    acceleration = _safe_diff(speed.astype(np.float32), t).astype(np.float32)
    curvature = _discrete_curvature(x.astype(np.float64), y.astype(np.float64), t.astype(np.float64))

    raw_features = np.column_stack([x, y, alt, rcs, speed, vertical_speed, acceleration, curvature, dt]).astype(np.float32)
    raw_features = _clip_features(raw_features)
    norm_features = _z_norm_per_track(raw_features).astype(np.float32)

    return {
        "features": norm_features,
        "raw_features": raw_features,
        "times": t.astype(np.float32),
        "xyz": np.column_stack([x, y, alt]).astype(np.float32),
        "lonlat": np.column_stack([lon, lat]).astype(np.float32),
        "lonlat_start": np.asarray([lon[0], lat[0]], dtype=np.float32),
        "lonlat_end": np.asarray([lon[-1], lat[-1]], dtype=np.float32),
        "lonlat_center": np.asarray([float(np.mean(lon)), float(np.mean(lat))], dtype=np.float32),
        "rcs": rcs.astype(np.float32),
        "feature_names": np.asarray(SEQUENCE_FEATURES),
    }


def build_track_cache(df: pd.DataFrame) -> dict[int, dict[str, Any]]:
    cache: dict[int, dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        track_id = int(getattr(row, "track_id"))
        try:
            parsed = preprocess_track(
                trajectory_hex=getattr(row, "trajectory"),
                trajectory_time=getattr(row, "trajectory_time"),
                radar_bird_size=getattr(row, "radar_bird_size", None),
                track_id=track_id,
            )
        except Exception as exc:
            raise ValueError(f"failed to preprocess track_id={track_id}: {exc}") from exc

        parsed["track_id"] = track_id
        parsed["radar_bird_size"] = getattr(row, "radar_bird_size", None)
        parsed["airspeed"] = float(getattr(row, "airspeed", 0.0))
        parsed["min_z"] = float(getattr(row, "min_z", 0.0))
        parsed["max_z"] = float(getattr(row, "max_z", 0.0))
        cache[track_id] = parsed
    return cache


def save_track_cache(cache: dict[int, dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_track_cache(path: str | Path) -> dict[int, dict[str, Any]]:
    with Path(path).open("rb") as f:
        return pickle.load(f)
