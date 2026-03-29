from __future__ import annotations

import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import RADAR_BIRD_SIZE_CATEGORIES
from src.preprocessing import build_track_cache, parse_ewkb_linestring_zm_hex

EPS = 1e-8

try:
    from astral import LocationInfo
    from astral.sun import sun

    _ASTRAL_AVAILABLE = True
    _EEMSHAVEN = LocationInfo(
        "Eemshaven",
        "Netherlands",
        "Europe/Amsterdam",
        53.4386,
        6.8347,
    )
except Exception:
    _ASTRAL_AVAILABLE = False
    _EEMSHAVEN = None

try:
    from scipy.signal import lombscargle as _lombscargle

    _SCIPY_LOMB_AVAILABLE = True
except Exception:
    _lombscargle = None
    _SCIPY_LOMB_AVAILABLE = False


EEMSHAVEN_LAT = 53.4386
EEMSHAVEN_LON = 6.8347

# Static habitat anchors around Eemshaven (coastal Groningen).
# These are fixed geospatial proxies, independent of month labels.
HABITAT_ANCHORS: dict[str, tuple[float, float]] = {
    "port": (53.4386, 6.8347),          # Eemshaven harbor
    "open_water": (53.4780, 6.8600),    # Wadden/open-sea side (north)
    "tidal_flat": (53.4570, 6.9050),    # intertidal mudflat zone (east-northeast)
    "farmland": (53.4040, 6.8600),      # inland agricultural area (south)
}

# Fallback month mapping for months absent in train coverage.
# The goal is to avoid using global_center for OOD months and instead
# use the nearest known seasonal month center.
MONTH_CENTER_FALLBACK: dict[int, int] = {
    2: 1,
    3: 4,
    5: 4,
    6: 4,
    7: 9,
    8: 9,
    11: 10,
    12: 10,
}


def _signed_log1p(x: float) -> float:
    return float(np.sign(x) * np.log1p(abs(x)))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a + EPS))
    return float(r * c)


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dlmb = np.radians(lon2 - lon1)
    y = np.sin(dlmb) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlmb)
    brg = np.degrees(np.arctan2(y, x))
    return float((brg + 360.0) % 360.0)


def _track_lonlat(row: pd.Series, cache_item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lonlat = cache_item.get("lonlat")
    if isinstance(lonlat, np.ndarray) and lonlat.ndim == 2 and lonlat.shape[1] == 2 and len(lonlat) >= 2:
        return lonlat[:, 0].astype(np.float64), lonlat[:, 1].astype(np.float64)
    try:
        lon, lat, _, _ = parse_ewkb_linestring_zm_hex(str(row.get("trajectory", "")), track_id=int(row.get("track_id", -1)))
        return lon.astype(np.float64), lat.astype(np.float64)
    except Exception:
        return np.asarray([EEMSHAVEN_LON, EEMSHAVEN_LON], dtype=np.float64), np.asarray([EEMSHAVEN_LAT, EEMSHAVEN_LAT], dtype=np.float64)


def _row_month(row: pd.Series) -> int | None:
    ts = pd.to_datetime(row.get("timestamp_start_radar_utc"), errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    m = int(ts.month)
    if 1 <= m <= 12:
        return m
    return None


def _resolve_month_center(
    month: int | None,
    by_month: dict[int, tuple[float, float]],
    global_center: tuple[float, float],
) -> tuple[float, float]:
    if month is not None and month in by_month:
        return by_month[month]
    if month is not None:
        fallback_month = MONTH_CENTER_FALLBACK.get(month)
        if fallback_month is not None and fallback_month in by_month:
            return by_month[fallback_month]
    return global_center


def compute_monthly_track_centers(
    df: pd.DataFrame,
    track_cache: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if track_cache is None:
        track_cache = build_track_cache(df)

    rows: list[dict[str, float]] = []
    for _, row in df.iterrows():
        month = _row_month(row)
        if month is None:
            continue
        track_id = int(row["track_id"])
        lon, lat = _track_lonlat(row, track_cache[track_id])
        rows.append(
            {
                "month": float(month),
                "center_lat": float(np.mean(lat)),
                "center_lon": float(np.mean(lon)),
            }
        )

    if len(rows) == 0:
        return {
            "by_month": {},
            "global_center": (float(EEMSHAVEN_LAT), float(EEMSHAVEN_LON)),
        }

    tmp = pd.DataFrame(rows)
    monthly = (
        tmp.groupby("month", as_index=True)[["center_lat", "center_lon"]]
        .mean()
        .astype(float)
    )
    by_month: dict[int, tuple[float, float]] = {
        int(m): (float(v["center_lat"]), float(v["center_lon"]))
        for m, v in monthly.iterrows()
    }
    global_center = (float(tmp["center_lat"].mean()), float(tmp["center_lon"].mean()))
    return {"by_month": by_month, "global_center": global_center}


def _entropy(values: np.ndarray, bins: int = 16) -> float:
    if values.size == 0:
        return 0.0
    vmin, vmax = float(values.min()), float(values.max())
    if abs(vmax - vmin) < 1e-9:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=(vmin, vmax), density=False)
    p = hist.astype(np.float64)
    p = p / (p.sum() + EPS)
    p = p[p > 0]
    return float(-(p * np.log(p + EPS)).sum())


def _entropy_norm_10bins(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    vmin, vmax = float(values.min()), float(values.max())
    if abs(vmax - vmin) < 1e-9:
        return 0.0
    bins = 10
    hist, _ = np.histogram(values, bins=bins, range=(vmin, vmax), density=False)
    p = hist.astype(np.float64)
    p = p / (p.sum() + EPS)
    p = p[p > 0]
    h = float(-(p * np.log(p + EPS)).sum())
    return float(h / np.log(bins))


def _safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    var = float(np.var(xv))
    if var < 1e-12:
        return 0.0
    cov = float(np.mean((xv - xv.mean()) * (yv - yv.mean())))
    return cov / (var + EPS)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(xv) & np.isfinite(yv)
    if finite.sum() < 2:
        return 0.0
    xv = xv[finite]
    yv = yv[finite]
    sx = float(np.std(xv))
    sy = float(np.std(yv))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    c = np.corrcoef(xv, yv)[0, 1]
    if not np.isfinite(c):
        return 0.0
    return float(c)


def _safe_skew_kurtosis(values: np.ndarray) -> tuple[float, float]:
    """Return (skew, excess_kurtosis) without scipy dependency."""
    if values.size < 3:
        return 0.0, 0.0
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() < 3:
        return 0.0, 0.0
    x = x[finite]
    m = float(np.mean(x))
    s = float(np.std(x))
    if s < 1e-12:
        return 0.0, 0.0
    z = (x - m) / s
    skew = float(np.mean(z ** 3))
    kurt_excess = float(np.mean(z ** 4) - 3.0)
    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(kurt_excess):
        kurt_excess = 0.0
    return skew, kurt_excess


def _wingbeat_period_from_acf(
    rcs: np.ndarray,
    t: np.ndarray,
    max_lag_sec: float = 5.0,
) -> tuple[float, float]:
    """Estimate dominant oscillation period from RCS autocorrelation.

    Returns:
        (first_peak_lag_seconds, first_peak_strength)
    """
    if len(rcs) < 15 or len(t) < 15:
        return np.nan, np.nan

    rr = np.asarray(rcs, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64)
    finite = np.isfinite(rr) & np.isfinite(tt)
    if finite.sum() < 15:
        return np.nan, np.nan
    rr = rr[finite]
    tt = tt[finite]

    order = np.argsort(tt)
    rr = rr[order]
    tt = tt[order]

    dtt = np.diff(tt)
    dtt = dtt[np.isfinite(dtt) & (dtt > 1e-6)]
    if dtt.size == 0:
        return np.nan, np.nan
    dt_uniform = float(np.median(dtt))
    if not np.isfinite(dt_uniform) or dt_uniform <= 0:
        return np.nan, np.nan

    t_uniform = np.arange(float(tt[0]), float(tt[-1]) + 0.5 * dt_uniform, dt_uniform, dtype=np.float64)
    if t_uniform.size < 10:
        return np.nan, np.nan

    rcs_interp = np.interp(t_uniform, tt, rr)
    rcs_norm = rcs_interp - float(np.mean(rcs_interp))
    if float(np.std(rcs_norm)) < 1e-9:
        return np.nan, 0.0

    n = int(len(rcs_norm))
    fft = np.fft.rfft(rcs_norm, n=2 * n)
    acf = np.fft.irfft(fft * np.conj(fft))[:n]
    acf = np.asarray(acf, dtype=np.float64)
    acf0 = float(acf[0]) if n > 0 else 0.0
    acf = acf / (acf0 + EPS)

    max_lag_idx = int(max_lag_sec / dt_uniform)
    max_lag_idx = max(3, min(max_lag_idx, n - 2))
    if max_lag_idx <= 3:
        return np.nan, np.nan

    search = acf[2 : max_lag_idx + 1]
    if search.size < 3:
        return np.nan, np.nan

    mid = search[1:-1]
    left = search[:-2]
    right = search[2:]
    peak_rel = np.where((mid > left) & (mid >= right))[0]
    if peak_rel.size == 0:
        return np.nan, float(acf[1]) if np.isfinite(acf[1]) else 0.0

    first_peak_idx = int(peak_rel[0] + 3)  # +2 for search offset, +1 for mid offset
    first_peak_lag = float(first_peak_idx * dt_uniform)
    first_peak_strength = float(acf[first_peak_idx]) if first_peak_idx < len(acf) else 0.0
    if not np.isfinite(first_peak_lag):
        first_peak_lag = np.nan
    if not np.isfinite(first_peak_strength):
        first_peak_strength = 0.0
    return first_peak_lag, first_peak_strength


def _segment_masks(t: np.ndarray, num_segments: int = 5) -> list[np.ndarray]:
    if len(t) == 0:
        return [np.zeros((0,), dtype=bool) for _ in range(num_segments)]
    t0 = float(t[0])
    t1 = float(t[-1])
    if t1 <= t0:
        mask = np.ones(len(t), dtype=bool)
        return [mask] + [np.zeros(len(t), dtype=bool) for _ in range(num_segments - 1)]
    t_norm = (t - t0) / (t1 - t0 + EPS)
    masks: list[np.ndarray] = []
    for i in range(num_segments):
        lo = i / num_segments
        hi = (i + 1) / num_segments + 1e-9
        masks.append((t_norm >= lo) & (t_norm < hi if i < num_segments - 1 else t_norm <= hi))
    return masks


def _segment_stats(values: np.ndarray, t: np.ndarray, num_segments: int = 5) -> list[tuple[float, float, float]]:
    masks = _segment_masks(t=t, num_segments=num_segments)
    out: list[tuple[float, float, float]] = []
    global_mean = float(np.mean(values)) if len(values) else 0.0
    for mask in masks:
        if not np.any(mask):
            if out:
                out.append(out[-1])
            else:
                out.append((global_mean, 0.0, 0.0))
            continue
        seg_v = values[mask]
        seg_t = t[mask]
        mean_v = float(np.mean(seg_v))
        std_v = float(np.std(seg_v))
        slope_v = _safe_slope(seg_t, seg_v)
        out.append((mean_v, std_v, slope_v))
    return out


def _convex_hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    pts = np.unique(points.astype(np.float64), axis=0)
    if len(pts) < 3:
        return 0.0

    pts_sorted = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts_sorted:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in reversed(pts_sorted):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)


def _bbox_area(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float((x.max() - x.min()) * (y.max() - y.min()))


def _turn_angles_deg(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return np.zeros((0,), dtype=np.float32)
    p = np.column_stack([x, y]).astype(np.float64)
    v1 = p[1:-1] - p[:-2]
    v2 = p[2:] - p[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    valid = (n1 > 1e-9) & (n2 > 1e-9)
    angles = np.zeros(len(v1), dtype=np.float64)
    if np.any(valid):
        v1v = v1[valid] / n1[valid][:, None]
        v2v = v2[valid] / n2[valid][:, None]
        dot = np.sum(v1v * v2v, axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        angles_valid = np.degrees(np.arccos(dot))
        angles[valid] = angles_valid
    return angles.astype(np.float32)


def _quantiles(values: np.ndarray, probs: list[float]) -> list[float]:
    if values.size == 0:
        return [0.0] * len(probs)
    q = np.quantile(values.astype(np.float64), probs)
    return [float(v) for v in q]


def _parse_observer_position_z(hex_str: Any) -> tuple[float, float]:
    if not isinstance(hex_str, str) or not hex_str:
        return 0.0, 0.0
    try:
        blob = bytes.fromhex(hex_str)
    except Exception:
        return 0.0, 0.0
    if len(blob) < 9:
        return 0.0, 0.0

    endian_flag = blob[0]
    if endian_flag == 0:
        endian = ">"
    elif endian_flag == 1:
        endian = "<"
    else:
        return 0.0, 0.0

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

    if base_type != 1:  # POINT
        return 0.0, 0.0

    if has_srid:
        if len(blob) < offset + 4:
            return 0.0, 0.0
        offset += 4

    dims = 2 + int(has_z) + int(has_m)
    bytes_needed = int(dims * 8)
    if len(blob) < offset + bytes_needed:
        return 0.0, 0.0

    coords = np.frombuffer(blob, dtype=np.dtype(f"{endian}f8"), count=dims, offset=offset)
    z = float(coords[2]) if has_z and dims >= 3 else 0.0
    if not np.isfinite(z):
        return 0.0, 0.0
    return z, 1.0


@lru_cache(maxsize=4096)
def _sunrise_sunset_timestamps(date_key: str) -> tuple[float, float] | None:
    if not _ASTRAL_AVAILABLE or _EEMSHAVEN is None:
        return None
    try:
        d = pd.Timestamp(date_key).date()
        s = sun(_EEMSHAVEN.observer, date=d, tzinfo=_EEMSHAVEN.timezone)
        return float(s["sunrise"].timestamp()), float(s["sunset"].timestamp())
    except Exception:
        return None


def _observer_time_features(ts_raw: Any) -> dict[str, float]:
    ts = pd.to_datetime(ts_raw, errors="coerce", utc=True)
    if pd.isna(ts):
        return {
            "hour_utc": 0.0,
            "hours_since_sunrise": 0.0,
            "frac_of_day": 0.0,
            "is_daytime": 0.0,
            "day_length_hours": 0.0,
        }

    hour_utc = float(ts.hour + ts.minute / 60.0 + ts.second / 3600.0)
    if not _ASTRAL_AVAILABLE:
        return {
            "hour_utc": hour_utc,
            "hours_since_sunrise": 0.0,
            "frac_of_day": 0.0,
            "is_daytime": 0.0,
            "day_length_hours": 0.0,
        }

    ts_local = ts.tz_convert("Europe/Amsterdam")
    date_key = str(ts_local.date())
    sun_pair = _sunrise_sunset_timestamps(date_key)
    if sun_pair is None:
        return {
            "hour_utc": hour_utc,
            "hours_since_sunrise": 0.0,
            "frac_of_day": 0.0,
            "is_daytime": 0.0,
            "day_length_hours": 0.0,
        }

    sunrise, sunset = sun_pair
    t = float(ts_local.timestamp())
    day_len = max((sunset - sunrise) / 3600.0, 1e-6)
    hours_since_sunrise = (t - sunrise) / 3600.0
    frac_of_day = float(np.clip(hours_since_sunrise / day_len, 0.0, 1.0))
    is_daytime = float(sunrise < t < sunset)
    return {
        "hour_utc": hour_utc,
        "hours_since_sunrise": float(hours_since_sunrise),
        "frac_of_day": frac_of_day,
        "is_daytime": is_daytime,
        "day_length_hours": float(day_len),
    }


def _track_row_features(
    row: pd.Series,
    cache_item: dict[str, Any],
    monthly_centers: dict[str, Any] | None = None,
) -> dict[str, float]:
    raw = cache_item["raw_features"]
    t = cache_item["times"]
    xyz = cache_item["xyz"]
    rcs = cache_item.get("rcs", raw[:, 3])

    n = int(min(len(raw), len(t), len(xyz), len(rcs)))
    if n < 2:
        raise ValueError("track has fewer than 2 points after cache alignment")

    raw = raw[:n]
    t = t[:n]
    xyz = xyz[:n]
    rcs = np.asarray(rcs[:n], dtype=np.float32)

    x = xyz[:, 0]
    y = xyz[:, 1]
    alt = xyz[:, 2]
    lon, lat = _track_lonlat(row, cache_item)
    center_lon = float(np.mean(lon))
    center_lat = float(np.mean(lat))
    span_lon = float(np.max(lon) - np.min(lon))
    span_lat = float(np.max(lat) - np.min(lat))
    dist_to_port_km = _haversine_km(
        center_lat,
        center_lon,
        float(HABITAT_ANCHORS["port"][0]),
        float(HABITAT_ANCHORS["port"][1]),
    )
    dist_to_open_water_km = _haversine_km(
        center_lat,
        center_lon,
        float(HABITAT_ANCHORS["open_water"][0]),
        float(HABITAT_ANCHORS["open_water"][1]),
    )
    dist_to_tidal_flat_km = _haversine_km(
        center_lat,
        center_lon,
        float(HABITAT_ANCHORS["tidal_flat"][0]),
        float(HABITAT_ANCHORS["tidal_flat"][1]),
    )
    dist_to_farmland_km = _haversine_km(
        center_lat,
        center_lon,
        float(HABITAT_ANCHORS["farmland"][0]),
        float(HABITAT_ANCHORS["farmland"][1]),
    )
    habitat_water_minus_farmland_km = float(dist_to_open_water_km - dist_to_farmland_km)
    habitat_tidal_minus_port_km = float(dist_to_tidal_flat_km - dist_to_port_km)
    nearest_habitat_dist_km = float(
        min(dist_to_port_km, dist_to_open_water_km, dist_to_tidal_flat_km, dist_to_farmland_km)
    )
    month = _row_month(row)
    if monthly_centers is not None:
        by_month = monthly_centers.get("by_month", {})
        global_center = monthly_centers.get(
            "global_center",
            (float(EEMSHAVEN_LAT), float(EEMSHAVEN_LON)),
        )
        month_center = _resolve_month_center(month, by_month, global_center)
        month_center_lat = float(month_center[0])
        month_center_lon = float(month_center[1])
    else:
        month_center_lat = float(EEMSHAVEN_LAT)
        month_center_lon = float(EEMSHAVEN_LON)

    lat_deviation = float(center_lat - month_center_lat)
    lon_deviation = float(center_lon - month_center_lon)
    lat_deviation_km = float(lat_deviation * 111.0)
    lon_deviation_km = float(
        lon_deviation * (111.0 * np.cos(np.deg2rad(month_center_lat)))
    )
    dist_from_month_center_km = _haversine_km(
        month_center_lat, month_center_lon, center_lat, center_lon
    )
    month_center_bearing_deg = _bearing_deg(
        month_center_lat, month_center_lon, center_lat, center_lon
    )
    month_center_bearing_rad = np.deg2rad(month_center_bearing_deg)
    month_center_bearing_sin = float(np.sin(month_center_bearing_rad))
    month_center_bearing_cos = float(np.cos(month_center_bearing_rad))
    speed = raw[:, 4]
    vertical_speed = raw[:, 5]
    accel = raw[:, 6]
    curvature = raw[:, 7]
    dt = raw[:, 8]
    dt_valid = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt_valid) == 0:
        dt_valid = dt[np.isfinite(dt)]
    if len(dt_valid) == 0:
        dt_valid = np.asarray([0.0], dtype=np.float32)
    dt_mean = float(np.mean(dt_valid))
    dt_std = float(np.std(dt_valid))
    dt_max = float(np.max(dt_valid))
    dt_cv = float(dt_std / (dt_mean + EPS))
    dt_gap_ratio = float(np.mean(dt_valid > 2.0))
    dt_regularity = float(1.0 / (dt_std + 1e-6))

    dx = np.diff(x)
    dy = np.diff(y)
    segment_dist = np.sqrt(dx * dx + dy * dy)
    total_distance = float(segment_dist.sum())
    straight_distance = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))
    straightness_ratio = straight_distance / (total_distance + EPS)
    path_tortuosity = total_distance / (straight_distance + EPS)

    duration = float(max(t[-1] - t[0], 1e-3))
    altitude_range = float(alt.max() - alt.min()) if len(alt) else 0.0
    climb_rate = float((alt[-1] - alt[0]) / duration)
    z_p90 = float(np.quantile(alt, 0.90)) if len(alt) else 0.0
    z_p95 = float(np.quantile(alt, 0.95)) if len(alt) else 0.0
    pct_above_100 = float(np.mean(alt > 100.0)) if len(alt) else 0.0
    pct_above_150 = float(np.mean(alt > 150.0)) if len(alt) else 0.0
    pct_above_200 = float(np.mean(alt > 200.0)) if len(alt) else 0.0

    speed_seg_stats = _segment_stats(speed, t, num_segments=5)
    alt_seg_stats = _segment_stats(alt, t, num_segments=5)
    vs_seg_stats = _segment_stats(vertical_speed, t, num_segments=5)
    # Start/mid/end segment descriptors for trajectory consistency.
    rcs_seg3_stats = _segment_stats(rcs, t, num_segments=3)
    speed_seg3_stats = _segment_stats(speed, t, num_segments=3)
    alt_seg3_stats = _segment_stats(alt, t, num_segments=3)
    rcs_seg1_mean = float(rcs_seg3_stats[0][0])
    rcs_seg3_mean = float(rcs_seg3_stats[2][0])
    speed_seg3_means = np.asarray([s[0] for s in speed_seg3_stats], dtype=np.float64)
    alt_seg3_means = np.asarray([s[0] for s in alt_seg3_stats], dtype=np.float64)
    speed_consistency = float(np.std(speed_seg3_means))
    altitude_trend_consistency = float(
        _safe_corr(np.asarray([1.0, 2.0, 3.0], dtype=np.float64), alt_seg3_means)
    )

    turn_angles = _turn_angles_deg(x, y)
    max_turn_angle = float(np.max(turn_angles)) if turn_angles.size else 0.0
    turn_mean = float(np.mean(turn_angles)) if turn_angles.size else 0.0
    turn_std = float(np.std(turn_angles)) if turn_angles.size else 0.0
    turn_p95 = float(np.percentile(turn_angles, 95)) if turn_angles.size else 0.0
    pct_small_turns = float(np.mean(turn_angles < 10)) if turn_angles.size else 0.0
    pct_large_turns = float(np.mean(turn_angles > 45)) if turn_angles.size else 0.0
    turn_entropy = _entropy_norm_10bins(turn_angles) if turn_angles.size else 0.0
    alt_cv = float(np.std(alt) / (np.mean(alt) + 1e-6)) if len(alt) else 0.0
    alt_range_norm = float(altitude_range / (np.mean(alt) + 1e-6)) if len(alt) else 0.0
    alt_autocorr = (
        float(np.corrcoef(alt[:-1], alt[1:])[0, 1])
        if len(alt) > 5 and float(np.std(alt)) > 1e-9
        else 0.0
    )
    # Heading consistency from step directions (stable for straight, noisy for erratic paths).
    if len(dx) >= 1:
        _heading = np.arctan2(dy, dx).astype(np.float64)
        _heading_unwrapped = np.unwrap(_heading)
        _heading_step = np.diff(_heading_unwrapped)
        heading_step_std = float(np.std(_heading_step)) if len(_heading_step) >= 1 else 0.0
        heading_step_abs_mean = float(np.mean(np.abs(_heading_step))) if len(_heading_step) >= 1 else 0.0
        _cos_m = float(np.mean(np.cos(_heading)))
        _sin_m = float(np.mean(np.sin(_heading)))
        _r = float(np.clip(np.sqrt(_cos_m * _cos_m + _sin_m * _sin_m), 0.0, 1.0))
        heading_consistency = _r
        heading_circular_std = float(np.sqrt(max(0.0, -2.0 * np.log(_r + EPS))))
        if len(_heading_step) >= 1:
            _turns = ((_heading_step + np.pi) % (2.0 * np.pi)) - np.pi
            _abs_turns = np.abs(_turns)
            turn_abs_mean_rad = float(np.mean(_abs_turns))
            turn_abs_p90_rad = float(np.quantile(_abs_turns, 0.90))
            _turn_vec_mean = np.mean(np.exp(1j * _turns))
            turn_circular_var = float(1.0 - np.abs(_turn_vec_mean))
            turn_sharp_ratio_30deg = float(np.mean(_abs_turns > (np.pi / 6.0)))
            turn_reversal_ratio_90deg = float(np.mean(_abs_turns > (np.pi / 2.0)))
        else:
            turn_abs_mean_rad = 0.0
            turn_abs_p90_rad = 0.0
            turn_circular_var = 0.0
            turn_sharp_ratio_30deg = 0.0
            turn_reversal_ratio_90deg = 0.0
    else:
        heading_step_std = 0.0
        heading_step_abs_mean = 0.0
        heading_consistency = 0.0
        heading_circular_std = 0.0
        turn_abs_mean_rad = 0.0
        turn_abs_p90_rad = 0.0
        turn_circular_var = 0.0
        turn_sharp_ratio_30deg = 0.0
        turn_reversal_ratio_90deg = 0.0

    # Acceleration profile from speed deltas (requested explicit proxy).
    if len(speed) >= 2 and len(t) >= 2:
        _dt_speed = np.diff(t).astype(np.float64)
        _dt_speed = np.where(np.abs(_dt_speed) < 1e-6, 1e-6, _dt_speed)
        _speed_acc = np.diff(speed).astype(np.float64) / _dt_speed
        speed_delta_std = float(np.std(_speed_acc))
        speed_delta_abs_p90 = float(np.quantile(np.abs(_speed_acc), 0.90))
        speed_delta_mean = float(np.mean(_speed_acc))
    else:
        speed_delta_std = 0.0
        speed_delta_abs_p90 = 0.0
        speed_delta_mean = 0.0

    # Altitude profile shape: monotone up/down/flat/variable.
    alt_slope = _safe_slope(t, alt)
    alt_total_change = float(alt[-1] - alt[0]) if len(alt) else 0.0
    alt_monotonicity = float(abs(alt_total_change) / (altitude_range + EPS))
    _is_up = float(alt_total_change > 0.0 and alt_monotonicity >= 0.75)
    _is_down = float(alt_total_change < 0.0 and alt_monotonicity >= 0.75)
    _is_flat = float(altitude_range <= 10.0 or float(np.std(alt)) <= 4.0)
    _is_variable = float(1.0 - float(bool(_is_up or _is_down or _is_flat)))
    avg_curvature = float(np.mean(curvature)) if len(curvature) else 0.0
    time_weighted_mean_alt = float(np.sum(alt * dt) / (np.sum(dt) + EPS))

    q_probs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    speed_q = _quantiles(speed, q_probs)
    vs_q = _quantiles(vertical_speed, q_probs)
    rcs_q_full = _quantiles(rcs, q_probs)

    rcs_p10 = float(np.quantile(rcs, 0.10))
    rcs_p25 = float(np.quantile(rcs, 0.25))
    rcs_p50 = float(np.quantile(rcs, 0.50))
    rcs_p75 = float(np.quantile(rcs, 0.75))
    rcs_p90 = float(np.quantile(rcs, 0.90))
    rcs_mean = float(np.mean(rcs))
    # RCS spectral features (temporally stable)
    _rcs_detrended = rcs - np.mean(rcs)
    _rcs_spectrum = np.abs(np.fft.rfft(_rcs_detrended)) ** 2 if len(rcs) >= 8 else np.array([1.0])
    _rcs_spectrum[0] = 0.0
    _rcs_total_power = _rcs_spectrum.sum() + 1e-9
    _rcs_dom_freq_idx = float(np.argmax(_rcs_spectrum)) / len(_rcs_spectrum)
    _rcs_p = _rcs_spectrum / _rcs_total_power
    _rcs_p_nz = _rcs_p[_rcs_p > 0]
    _rcs_spectral_entropy = (
        float(-np.sum(_rcs_p_nz * np.log(_rcs_p_nz + 1e-9))) if len(_rcs_p_nz) > 0 else 0.0
    )
    _rcs_peak_power_ratio = float(_rcs_spectrum.max() / _rcs_total_power)
    _rcs_freq = np.linspace(0.0, 0.5, num=len(_rcs_spectrum), dtype=np.float64)
    _rcs_low_band_power = float(np.sum(_rcs_spectrum[_rcs_freq <= 0.15]) / _rcs_total_power)
    _rcs_mid_band_power = float(
        np.sum(_rcs_spectrum[(_rcs_freq > 0.15) & (_rcs_freq <= 0.35)]) / _rcs_total_power
    )
    _rcs_high_band_power = float(np.sum(_rcs_spectrum[_rcs_freq > 0.35]) / _rcs_total_power)

    _rcs_thr = float(np.quantile(_rcs_detrended, 0.75)) if len(_rcs_detrended) else 0.0
    _peak_mask = (_rcs_detrended[1:-1] > _rcs_detrended[:-2]) & (_rcs_detrended[1:-1] >= _rcs_detrended[2:])
    _peak_idx = np.where(_peak_mask & (_rcs_detrended[1:-1] > _rcs_thr))[0] + 1 if len(_rcs_detrended) >= 3 else np.array([], dtype=np.int64)
    _rcs_peak_count = float(len(_peak_idx))
    if len(_peak_idx) >= 2:
        _peak_int = np.diff(_peak_idx).astype(np.float64)
        _rcs_peak_interval_mean = float(np.mean(_peak_int))
        _rcs_peak_interval_std = float(np.std(_peak_int))
    else:
        _rcs_peak_interval_mean = 0.0
        _rcs_peak_interval_std = 0.0

    _win = max(5, min(21, len(rcs) // 8 if len(rcs) > 0 else 5))
    if len(rcs) >= _win:
        _local_std = np.array([np.std(rcs[i:i + _win]) for i in range(len(rcs) - _win + 1)], dtype=np.float64)
        _rcs_local_std_mean = float(np.mean(_local_std))
        _rcs_local_std_max = float(np.max(_local_std))
    else:
        _rcs_local_std_mean = float(np.std(rcs)) if len(rcs) else 0.0
        _rcs_local_std_max = float(np.std(rcs)) if len(rcs) else 0.0

    def _acf_lag(a: np.ndarray, lag: int) -> float:
        if len(a) <= lag + 1:
            return 0.0
        return float(_safe_corr(a[:-lag], a[lag:]))

    _rcs_ac2 = _acf_lag(rcs, 2)
    _rcs_ac3 = _acf_lag(rcs, 3)
    _rcs_ac5 = _acf_lag(rcs, 5)
    _rcs_ac10 = _acf_lag(rcs, 10)
    _acf_seq = [_acf_lag(rcs, lag) for lag in range(1, 11)]
    _rcs_periodicity_score = float(max(_acf_seq)) if _acf_seq else 0.0

    wingbeat_period_sec, wingbeat_acf_peak_strength = _wingbeat_period_from_acf(
        rcs=rcs,
        t=t,
        max_lag_sec=5.0,
    )

    # Short-lag autocorrelation profile (robust on 10+ point tracks).
    _acf_short = [_acf_lag(rcs, lag) for lag in range(1, 7)]
    rcs_acf_lag1 = float(_acf_short[0])
    rcs_acf_lag2 = float(_acf_short[1])
    rcs_acf_lag3 = float(_acf_short[2])
    rcs_acf_lag4 = float(_acf_short[3])
    rcs_acf_lag5 = float(_acf_short[4])
    rcs_acf_lag6 = float(_acf_short[5])
    rcs_acf_mean_lag1_3 = float(np.mean(_acf_short[:3]))
    rcs_acf_mean_lag4_6 = float(np.mean(_acf_short[3:6]))
    rcs_acf_decay_lag1_6 = float(rcs_acf_lag1 - rcs_acf_lag6)

    rcs_std = float(np.std(rcs))
    rcs_iqr = float(rcs_p75 - rcs_p25)
    rcs_range = float(np.max(rcs) - np.min(rcs))
    rcs_range_rel = float(rcs_range / (abs(rcs_mean) + EPS))
    rcs_cv = float(rcs_std / (abs(rcs_mean) + EPS))

    rcs_skew, rcs_kurtosis_excess = _safe_skew_kurtosis(rcs)
    # Bimodality coefficient proxy, clipped to stable numeric range.
    _n_rcs = len(rcs)
    _k_raw = float(rcs_kurtosis_excess + 3.0)
    if _n_rcs > 3:
        _k_adj = _k_raw + (3.0 * ((_n_rcs - 1.0) ** 2)) / (((_n_rcs - 2.0) * (_n_rcs - 3.0)) + EPS)
    else:
        _k_adj = _k_raw
    rcs_bimodality = float((rcs_skew * rcs_skew + 1.0) / (_k_adj + EPS))
    if not np.isfinite(rcs_bimodality):
        rcs_bimodality = 0.0

    rcs_slope_rel = float(_safe_slope(t, rcs) / (abs(rcs_mean) + EPS))
    rcs_entropy = _entropy_norm_10bins(rcs)
    rcs_vs_speed_corr = _safe_corr(rcs, speed)
    rcs_vs_z_corr = _safe_corr(rcs, alt)
    # Explicit RCS trend fit quality.
    if len(t) >= 2:
        _x = np.asarray(t, dtype=np.float64)
        _y = np.asarray(rcs, dtype=np.float64)
        _slope = _safe_slope(_x, _y)
        _intercept = float(np.mean(_y) - _slope * np.mean(_x))
        _yhat = _intercept + _slope * _x
        _ss_res = float(np.sum((_y - _yhat) ** 2))
        _ss_tot = float(np.sum((_y - np.mean(_y)) ** 2))
        rcs_trend_slope = float(_slope)
        rcs_trend_r2 = float(1.0 - (_ss_res / (_ss_tot + EPS)))
    else:
        rcs_trend_slope = 0.0
        rcs_trend_r2 = 0.0

    # Wingbeat proxies: only compute on sufficiently long tracks.
    # Short tracks (e.g. Clutter/Pigeons) produce unstable spectra and hurt ranking.
    _wingbeat_min_points = 30
    if len(rcs) >= _wingbeat_min_points and len(t) >= _wingbeat_min_points:
        _dt_med = float(np.median(np.diff(np.asarray(t, dtype=np.float64))))
        _dt_med = max(_dt_med, 1e-3)
        _fft_freq_hz = np.fft.rfftfreq(len(rcs), d=_dt_med)
        _fft_pow = np.abs(np.fft.rfft(_rcs_detrended)) ** 2
        if len(_fft_pow) > 0:
            _fft_pow[0] = 0.0
        _fft_total = float(np.sum(_fft_pow) + EPS)
        _fft_peak_idx = int(np.argmax(_fft_pow)) if len(_fft_pow) else 0
        wingbeat_fft_peak_freq_hz = float(_fft_freq_hz[_fft_peak_idx]) if len(_fft_freq_hz) else 0.0
        wingbeat_fft_peak_power_ratio = float((_fft_pow[_fft_peak_idx] / _fft_total)) if len(_fft_pow) else 0.0
        _wb_band = (_fft_freq_hz >= 0.2) & (_fft_freq_hz <= 6.0)
        wingbeat_fft_band_power = float(np.sum(_fft_pow[_wb_band]) / _fft_total) if np.any(_wb_band) else 0.0
    else:
        wingbeat_fft_peak_freq_hz = np.nan
        wingbeat_fft_peak_power_ratio = np.nan
        wingbeat_fft_band_power = np.nan

    if _SCIPY_LOMB_AVAILABLE and len(rcs) >= _wingbeat_min_points and len(t) >= _wingbeat_min_points:
        _tt = np.asarray(t, dtype=np.float64)
        _tt = _tt - float(_tt[0])
        _yy = np.asarray(_rcs_detrended, dtype=np.float64)
        _freq_grid = np.linspace(0.2, 6.0, num=96, dtype=np.float64)
        _omega = 2.0 * np.pi * _freq_grid
        _lomb = _lombscargle(_tt, _yy, _omega, normalize=True)
        _lomb = np.asarray(_lomb, dtype=np.float64)
        if len(_lomb) > 0 and np.isfinite(_lomb).any():
            _best = int(np.nanargmax(_lomb))
            wingbeat_lomb_peak_freq_hz = float(_freq_grid[_best])
            wingbeat_lomb_peak_power = float(np.nanmax(_lomb))
        else:
            wingbeat_lomb_peak_freq_hz = 0.0
            wingbeat_lomb_peak_power = 0.0
    else:
        wingbeat_lomb_peak_freq_hz = np.nan
        wingbeat_lomb_peak_power = np.nan
    # Train has observer_position while test does not. Using observer_pos_* features
    # creates hard train/test leakage. Keep columns for compatibility but force neutral values.
    observer_z, observer_has_z = 0.0, 0.0
    tod = _observer_time_features(row.get("timestamp_start_radar_utc"))
    airspeed_raw = float(row.get("airspeed", 0.0))
    min_z_row = float(row.get("min_z", 0.0))
    max_z_row = float(row.get("max_z", 0.0))
    size_label = str(row.get("radar_bird_size", ""))
    speed_per_altitude_den = float(np.clip(max_z_row + 1.0, 1.0, None))

    feats: dict[str, float] = {
        "num_points": float(len(raw)),
        "n_points": float(len(raw)),
        "duration": duration,
        "track_duration_sec": duration,
        "points_per_second": float(len(raw) / (duration + EPS)),
        "total_distance": total_distance,
        "straightness": float(np.clip(straightness_ratio, 0.0, 1.0)),
        "straightness_ratio": float(np.clip(straightness_ratio, 0.0, 1.0)),
        "convex_hull_area": _convex_hull_area(np.column_stack([x, y])),
        "bounding_box_area": _bbox_area(np.column_stack([x, y])),
        "path_tortuosity": float(path_tortuosity),
        "max_turn_angle": max_turn_angle,
        "turn_mean": turn_mean,
        "turn_std": turn_std,
        "turn_p95": turn_p95,
        "pct_small_turns": pct_small_turns,
        "pct_large_turns": pct_large_turns,
        "turn_entropy": turn_entropy,
        "heading_consistency": heading_consistency,
        "heading_circular_std": heading_circular_std,
        "heading_step_std": heading_step_std,
        "heading_step_abs_mean": heading_step_abs_mean,
        "turn_abs_mean_rad": turn_abs_mean_rad,
        "turn_abs_p90_rad": turn_abs_p90_rad,
        "turn_circular_var": turn_circular_var,
        "turn_sharp_ratio_30deg": turn_sharp_ratio_30deg,
        "turn_reversal_ratio_90deg": turn_reversal_ratio_90deg,
        "alt_cv": alt_cv,
        "alt_range_norm": alt_range_norm,
        "alt_autocorr": alt_autocorr,
        "alt_profile_slope": float(alt_slope),
        "alt_profile_monotonicity": alt_monotonicity,
        "alt_profile_monotone_up": _is_up,
        "alt_profile_monotone_down": _is_down,
        "alt_profile_flat": _is_flat,
        "alt_profile_variable": _is_variable,
        "avg_curvature": avg_curvature,
        "mean_speed": float(np.mean(speed)),
        "std_speed": float(np.std(speed)),
        "mean_vertical_speed": float(np.mean(vertical_speed)),
        "std_vertical_speed": float(np.std(vertical_speed)),
        "mean_acceleration": float(np.mean(accel)),
        "acceleration_variance": float(np.var(accel)),
        "speed_delta_mean": speed_delta_mean,
        "speed_delta_std": speed_delta_std,
        "speed_delta_abs_p90": speed_delta_abs_p90,
        "mean_alt": float(np.mean(alt)),
        "std_alt": float(np.std(alt)),
        "altitude_range": altitude_range,
        "time_weighted_mean_altitude": time_weighted_mean_alt,
        "climb_rate": climb_rate,
        "airspeed": airspeed_raw,
        "min_z": min_z_row,
        "max_z": max_z_row,
        "z_p90": z_p90,
        "z_p95": z_p95,
        "pct_above_100": pct_above_100,
        "pct_above_150": pct_above_150,
        "pct_above_200": pct_above_200,
        "is_high_altitude": float(max_z_row > 150.0),
        "max_z_log": float(np.log1p(max(max_z_row, 0.0))),
        "low_fast": float((max_z_row < 40.0) and (airspeed_raw > 17.0)),
        "speed_per_altitude": float(airspeed_raw / speed_per_altitude_den),
        "small_but_high_rcs": float((size_label == "Small bird") and (rcs_mean > -26.0)),
        "dt_mean": dt_mean,
        "dt_std": dt_std,
        "dt_max": dt_max,
        "dt_cv": dt_cv,
        "dt_gap_ratio": dt_gap_ratio,
        "dt_regularity": dt_regularity,
        "track_span_lat_deg": span_lat,
        "track_span_lon_deg": span_lon,
        "dist_to_port_km": dist_to_port_km,
        "dist_to_open_water_km": dist_to_open_water_km,
        "dist_to_tidal_flat_km": dist_to_tidal_flat_km,
        "dist_to_farmland_km": dist_to_farmland_km,
        "habitat_water_minus_farmland_km": habitat_water_minus_farmland_km,
        "habitat_tidal_minus_port_km": habitat_tidal_minus_port_km,
        "nearest_habitat_dist_km": nearest_habitat_dist_km,
        "lat_deviation_from_month_center": lat_deviation,
        "lon_deviation_from_month_center": lon_deviation,
        "lat_deviation_from_month_center_km": lat_deviation_km,
        "lon_deviation_from_month_center_km": lon_deviation_km,
        "dist_from_month_center_km": dist_from_month_center_km,
        "month_center_bearing_sin": month_center_bearing_sin,
        "month_center_bearing_cos": month_center_bearing_cos,
        "curvature_mean": float(np.mean(curvature)),
        "curvature_std": float(np.std(curvature)),
        "vertical_speed_mean": float(np.mean(vertical_speed)),
        "vertical_speed_std": float(np.std(vertical_speed)),
        "interaction_speed_altitude": float(np.mean(speed * alt)),
        "interaction_rcs_speed": float(np.mean(rcs * speed)),
        "interaction_altitude_range_over_duration": float(altitude_range / (duration + EPS)),
        "observer_pos_z": 0.0,
        "observer_pos_has_z": 0.0,
        "observer_pos_z_minus_mean_alt": 0.0,
        "observer_pos_z_minus_last_alt": 0.0,
        "hour_utc": float(tod["hour_utc"]),
        "hours_since_sunrise": float(tod["hours_since_sunrise"]),
        "frac_of_day": float(tod["frac_of_day"]),
        "is_daytime": float(tod["is_daytime"]),
        "day_length_hours": float(tod["day_length_hours"]),
        "rcs_mean": rcs_mean,
        "rcs_std": rcs_std,
        "rcs_iqr": rcs_iqr,
        "rcs_cv": rcs_cv,
        "rcs_p10": rcs_p10,
        "rcs_p25": rcs_p25,
        "rcs_p50": rcs_p50,
        "rcs_p75": rcs_p75,
        "rcs_p90": rcs_p90,
        "rcs_skew": rcs_skew,
        "rcs_kurtosis_excess": rcs_kurtosis_excess,
        "rcs_bimodality": rcs_bimodality,
        "rcs_slope_rel": rcs_slope_rel,
        "rcs_trend_slope": rcs_trend_slope,
        "rcs_trend_r2": rcs_trend_r2,
        "rcs_entropy": rcs_entropy,
        "rcs_vs_speed_corr": rcs_vs_speed_corr,
        "rcs_vs_z_corr": rcs_vs_z_corr,
        "rcs_range": rcs_range,
        "rcs_range_rel": rcs_range_rel,
        "mean_rcs": rcs_mean,
        "rcs_dom_freq_idx": _rcs_dom_freq_idx,
        "wingbeat_fft_peak_freq_hz": wingbeat_fft_peak_freq_hz,
        "wingbeat_fft_peak_power_ratio": wingbeat_fft_peak_power_ratio,
        "wingbeat_fft_band_power": wingbeat_fft_band_power,
        "wingbeat_lomb_peak_freq_hz": wingbeat_lomb_peak_freq_hz,
        "wingbeat_lomb_peak_power": wingbeat_lomb_peak_power,
        "rcs_spectral_entropy": _rcs_spectral_entropy,
        "rcs_peak_power_ratio": _rcs_peak_power_ratio,
        "rcs_low_band_power": _rcs_low_band_power,
        "rcs_mid_band_power": _rcs_mid_band_power,
        "rcs_high_band_power": _rcs_high_band_power,
        "rcs_peak_count": _rcs_peak_count,
        "rcs_peak_interval_mean": _rcs_peak_interval_mean,
        "rcs_peak_interval_std": _rcs_peak_interval_std,
        "rcs_local_std_mean": _rcs_local_std_mean,
        "rcs_local_std_max": _rcs_local_std_max,
        "rcs_periodicity_score": _rcs_periodicity_score,
        "wingbeat_period_sec": wingbeat_period_sec,
        "wingbeat_acf_peak_strength": wingbeat_acf_peak_strength,
        "rcs_ac2": _rcs_ac2,
        "rcs_ac3": _rcs_ac3,
        "rcs_ac5": _rcs_ac5,
        "rcs_ac10": _rcs_ac10,
        "rcs_acf_lag1": rcs_acf_lag1,
        "rcs_acf_lag2": rcs_acf_lag2,
        "rcs_acf_lag3": rcs_acf_lag3,
        "rcs_acf_lag4": rcs_acf_lag4,
        "rcs_acf_lag5": rcs_acf_lag5,
        "rcs_acf_lag6": rcs_acf_lag6,
        "rcs_acf_mean_lag1_3": rcs_acf_mean_lag1_3,
        "rcs_acf_mean_lag4_6": rcs_acf_mean_lag4_6,
        "rcs_acf_decay_lag1_6": rcs_acf_decay_lag1_6,
        "high_altitude_flag": float(min_z_row > 20.0),
        "mid_altitude_flag": float(min_z_row > 50.0),
        "very_high_altitude_flag": float(min_z_row > 100.0),
        "rcs_x_minz": float(rcs_mean * min_z_row),
        "rcs_x_maxz": float(rcs_mean * max_z_row),
        "minz_x_maxz": float(min_z_row * max_z_row),
        "std_rcs": rcs_std,
        "entropy_rcs": rcs_entropy,
    }

    airspeed_safe = float(np.clip(abs(airspeed_raw), 1e-6, None))
    speed_mean = float(feats["mean_speed"])
    speed_std = float(feats["std_speed"])
    speed_p10 = float(speed_q[1])
    speed_p25 = float(speed_q[2])
    speed_p50 = float(speed_q[3])
    speed_p75 = float(speed_q[4])
    speed_p90 = float(speed_q[5])

    feats["speed_mean_over_airspeed"] = float(speed_mean / airspeed_safe)
    feats["speed_p50_over_airspeed"] = float(speed_p50 / airspeed_safe)
    feats["speed_p90_over_airspeed"] = float(speed_p90 / airspeed_safe)
    feats["speed_p10_over_airspeed"] = float(speed_p10 / airspeed_safe)
    feats["speed_cv"] = float(speed_std / (abs(speed_mean) + 1e-6))
    feats["speed_iqr_rel"] = float((speed_p75 - speed_p25) / (abs(speed_mean) + 1e-6))
    feats["speed_tail_rel"] = float((speed_p90 - speed_p10) / (abs(speed_mean) + 1e-6))
    feats["speed_p90_minus_p50"] = float(speed_p90 - speed_p50)
    feats["speed_p10_minus_p50"] = float(speed_p10 - speed_p50)
    feats["rcs_mean_seg1_vs_seg3"] = float(rcs_seg1_mean / (rcs_seg3_mean + 1e-6))
    feats["speed_consistency"] = speed_consistency
    feats["altitude_trend_consistency"] = altitude_trend_consistency

    slog_keys = [
        "rcs_mean",
        "rcs_std",
        "rcs_iqr",
        "rcs_p10",
        "rcs_p25",
        "rcs_p50",
        "rcs_p75",
        "rcs_p90",
        "rcs_slope_rel",
        "rcs_entropy",
        "rcs_range",
        "rcs_range_rel",
    ]
    for key in slog_keys:
        feats[f"{key}_slog1p"] = _signed_log1p(float(feats[key]))

    q_names = ["p5", "p10", "p25", "p50", "p75", "p90", "p95"]
    for qn, qv in zip(q_names, speed_q):
        feats[f"speed_{qn}"] = float(qv)
    for qn, qv in zip(q_names, vs_q):
        feats[f"vertical_speed_{qn}"] = float(qv)
    for qn, qv in zip(q_names, rcs_q_full):
        feats[f"rcs_{qn}"] = float(qv)

    for i, (m, s, sl) in enumerate(speed_seg_stats, start=1):
        feats[f"speed_seg_{i}_mean"] = float(m)
        feats[f"speed_seg_{i}_std"] = float(s)
        feats[f"speed_seg_{i}_slope"] = float(sl)
        feats[f"speed_seg_{i}_mean_minus_global"] = float(m - speed_mean)
        feats[f"speed_seg_{i}_mean_over_airspeed"] = float(m / airspeed_safe)
    for i, (m, s, sl) in enumerate(alt_seg_stats, start=1):
        feats[f"altitude_seg_{i}_mean"] = float(m)
        feats[f"altitude_seg_{i}_std"] = float(s)
        feats[f"altitude_seg_{i}_slope"] = float(sl)
    for i, (m, s, sl) in enumerate(vs_seg_stats, start=1):
        feats[f"vertical_speed_seg_{i}_mean"] = float(m)
        feats[f"vertical_speed_seg_{i}_std"] = float(s)
        feats[f"vertical_speed_seg_{i}_slope"] = float(sl)

    size_code_map = {"Small bird": 0.0, "Medium bird": 1.0, "Large bird": 2.0, "Flock": 3.0}
    size_code = float(size_code_map.get(str(row.get("radar_bird_size", "")), 1.0))
    feats["radar_bird_size_code"] = size_code
    feats["rcs_mean_x_size"] = float(feats["rcs_mean"] * size_code)
    feats["mean_alt_x_size"] = float(feats["mean_alt"] * size_code)
    feats["airspeed_x_size"] = float(feats["airspeed"] * size_code)
    feats["n_points_x_size"] = float(feats["n_points"] * size_code)
    feats["rcs_std_x_size"] = float(feats["rcs_std"] * size_code)

    for cat in RADAR_BIRD_SIZE_CATEGORIES:
        feats[f"radar_bird_size__{cat.lower().replace(' ', '_')}"] = float(row.get("radar_bird_size", "") == cat)
    return feats


def build_feature_frame(
    df: pd.DataFrame,
    track_cache: dict[int, dict[str, Any]] | None = None,
    external_features: pd.DataFrame | None = None,
    monthly_centers: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if track_cache is None:
        track_cache = build_track_cache(df)

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        track_id = int(row["track_id"])
        feats = _track_row_features(row, track_cache[track_id], monthly_centers=monthly_centers)
        base = {
            "track_id": track_id,
            "observation_id": int(row["observation_id"]) if "observation_id" in row else -1,
            "primary_observation_id": int(row["primary_observation_id"]) if "primary_observation_id" in row else -1,
        }
        rows.append({**base, **feats})

    feat_df = pd.DataFrame(rows)
    if external_features is not None and len(external_features) > 0:
        ext = external_features.copy()
        if "track_id" in ext.columns:
            ext = ext.set_index("track_id")
        ext.index = pd.to_numeric(ext.index, errors="coerce")
        ext = ext[ext.index.notna()]
        ext.index = ext.index.astype(np.int64)
        ext = ext[~ext.index.duplicated(keep="first")]
        feat_df = feat_df.join(ext, on="track_id", how="left")

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in feat_df.columns:
        if col not in {"track_id", "observation_id", "primary_observation_id"}:
            feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce").fillna(0.0).astype(np.float32)
    return feat_df


def load_external_features(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"external features file not found: {p}")
    ext = pd.read_parquet(p)
    if "track_id" not in ext.columns:
        raise ValueError(f"external features must contain 'track_id' column: {p}")
    return ext
