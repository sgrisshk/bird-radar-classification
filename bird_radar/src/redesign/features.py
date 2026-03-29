from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import RADAR_BIRD_SIZE_CATEGORIES

EPS = 1e-8


def _safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    vx = float(np.var(xv))
    if vx < 1e-12:
        return 0.0
    cov = float(np.mean((xv - xv.mean()) * (yv - yv.mean())))
    return cov / (vx + EPS)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    m = np.isfinite(xv) & np.isfinite(yv)
    if m.sum() < 2:
        return 0.0
    xv = xv[m]
    yv = yv[m]
    sx = float(np.std(xv))
    sy = float(np.std(yv))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    c = np.corrcoef(xv, yv)[0, 1]
    if not np.isfinite(c):
        return 0.0
    return float(c)


def _entropy(values: np.ndarray, bins: int = 16) -> float:
    if values.size == 0:
        return 0.0
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if abs(vmax - vmin) < 1e-12:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=(vmin, vmax), density=False)
    p = hist.astype(np.float64)
    p = p / (p.sum() + EPS)
    p = p[p > 0]
    return float(-(p * np.log(p + EPS)).sum())


def _signed_log1p(x: float) -> float:
    return float(np.sign(x) * np.log1p(abs(x)))


def _convex_hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    pts = np.unique(points.astype(np.float64), axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


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
    m = (n1 > 1e-12) & (n2 > 1e-12)
    out = np.zeros(len(v1), dtype=np.float64)
    if np.any(m):
        a = v1[m] / n1[m][:, None]
        b = v2[m] / n2[m][:, None]
        dot = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
        out[m] = np.degrees(np.arccos(dot))
    return out.astype(np.float32)


def _fft_features(values: np.ndarray, prefix: str, n_bins: int = 10) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 4:
        return {f"{prefix}_fft_{i}": 0.0 for i in range(1, n_bins + 1)} | {
            f"{prefix}_fft_energy_low": 0.0,
            f"{prefix}_fft_energy_mid": 0.0,
            f"{prefix}_fft_energy_high": 0.0,
            f"{prefix}_fft_energy_ratio_lh": 0.0,
            f"{prefix}_fft_energy_ratio_mh": 0.0,
        }

    v = v - v.mean()
    mag = np.abs(np.fft.rfft(v))
    if len(mag) <= 1:
        mag = np.pad(mag, (0, max(0, n_bins + 1 - len(mag))), mode="constant")
    mag = mag[1:]  # skip DC
    if len(mag) < n_bins:
        mag = np.pad(mag, (0, n_bins - len(mag)), mode="constant")

    out: dict[str, float] = {}
    for i in range(n_bins):
        out[f"{prefix}_fft_{i+1}"] = float(mag[i])

    n = len(mag)
    low = float(np.sum(mag[: max(1, n // 3)] ** 2))
    mid = float(np.sum(mag[max(1, n // 3) : max(2, 2 * n // 3)] ** 2))
    high = float(np.sum(mag[max(2, 2 * n // 3) :] ** 2))
    out[f"{prefix}_fft_energy_low"] = low
    out[f"{prefix}_fft_energy_mid"] = mid
    out[f"{prefix}_fft_energy_high"] = high
    out[f"{prefix}_fft_energy_ratio_lh"] = float(low / (high + EPS))
    out[f"{prefix}_fft_energy_ratio_mh"] = float(mid / (high + EPS))
    return out


def _distribution_bins(values: np.ndarray, prefix: str) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return {f"{prefix}_bin_{i}": 0.0 for i in range(6)} | {f"{prefix}_pct_above_median": 0.0}
    med = float(np.median(v))
    q1 = float(np.quantile(v, 0.25))
    q3 = float(np.quantile(v, 0.75))
    iqr = q3 - q1
    z = (v - med) / (abs(iqr) + EPS)
    edges = np.array([-np.inf, -1.0, -0.5, 0.0, 0.5, 1.0, np.inf], dtype=np.float64)
    hist, _ = np.histogram(z, bins=edges)
    hist = hist.astype(np.float64) / (len(v) + EPS)
    out = {f"{prefix}_bin_{i}": float(hist[i]) for i in range(6)}
    out[f"{prefix}_pct_above_median"] = float(np.mean(v > med))
    return out


def _moment_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_range": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p25": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p75": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_iqr": 0.0,
            f"{prefix}_mad": 0.0,
            f"{prefix}_skew": 0.0,
            f"{prefix}_kurt": 0.0,
            f"{prefix}_mean_abs_diff": 0.0,
            f"{prefix}_sign_change_rate": 0.0,
            f"{prefix}_max_run_above_med_rel": 0.0,
            f"{prefix}_max_run_below_med_rel": 0.0,
            f"{prefix}_roll10_mean_std": 0.0,
            f"{prefix}_roll20_mean_std": 0.0,
        }

    mean = float(np.mean(v))
    std = float(np.std(v))
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    p10, p25, p50, p75, p90 = [float(np.quantile(v, q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90)]
    iqr = p75 - p25
    mad = float(np.median(np.abs(v - p50)))
    centered = v - mean
    m2 = float(np.mean(centered * centered))
    m3 = float(np.mean(centered * centered * centered))
    m4 = float(np.mean((centered * centered) ** 2))
    denom = (m2 ** 0.5) + EPS
    skew = float(m3 / (denom ** 3))
    kurt = float(m4 / ((m2 + EPS) ** 2) - 3.0)

    dv = np.diff(v)
    mean_abs_diff = float(np.mean(np.abs(dv))) if dv.size else 0.0
    sign = np.sign(v - p50)
    sign_change_rate = float(np.mean(sign[1:] != sign[:-1])) if sign.size > 1 else 0.0

    above = v > p50
    below = v < p50

    def _max_run(mask: np.ndarray) -> int:
        if mask.size == 0:
            return 0
        arr = mask.astype(np.int32)
        best = 0
        cur = 0
        for val in arr:
            if val:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        return int(best)

    max_run_above = float(_max_run(above) / max(1, len(v)))
    max_run_below = float(_max_run(below) / max(1, len(v)))

    def _roll_mean_std(x: np.ndarray, w: int) -> float:
        if x.size < w or w <= 1:
            return 0.0
        k = np.ones((w,), dtype=np.float64) / float(w)
        rm = np.convolve(x, k, mode="valid")
        return float(np.std(rm))

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_min": vmin,
        f"{prefix}_max": vmax,
        f"{prefix}_range": float(vmax - vmin),
        f"{prefix}_p10": p10,
        f"{prefix}_p25": p25,
        f"{prefix}_p50": p50,
        f"{prefix}_p75": p75,
        f"{prefix}_p90": p90,
        f"{prefix}_iqr": float(iqr),
        f"{prefix}_mad": mad,
        f"{prefix}_skew": skew,
        f"{prefix}_kurt": kurt,
        f"{prefix}_mean_abs_diff": mean_abs_diff,
        f"{prefix}_sign_change_rate": sign_change_rate,
        f"{prefix}_max_run_above_med_rel": max_run_above,
        f"{prefix}_max_run_below_med_rel": max_run_below,
        f"{prefix}_roll10_mean_std": _roll_mean_std(v, 10),
        f"{prefix}_roll20_mean_std": _roll_mean_std(v, 20),
    }


def _windowed_stats(values: np.ndarray, prefix: str, n_windows: int = 8) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        out = {}
        for i in range(n_windows):
            out[f"{prefix}_w{i}_mean"] = 0.0
            out[f"{prefix}_w{i}_std"] = 0.0
            out[f"{prefix}_w{i}_p10"] = 0.0
            out[f"{prefix}_w{i}_p90"] = 0.0
        out[f"{prefix}_w_delta_last_first"] = 0.0
        out[f"{prefix}_w_slope"] = 0.0
        out[f"{prefix}_w_max_jump"] = 0.0
        return out

    idx_splits = np.array_split(np.arange(v.size), n_windows)
    means: list[float] = []
    out: dict[str, float] = {}
    for i, idx in enumerate(idx_splits):
        if idx.size == 0:
            w = np.zeros((0,), dtype=np.float64)
        else:
            w = v[idx]
        if w.size == 0:
            m = s = p10 = p90 = 0.0
        else:
            m = float(np.mean(w))
            s = float(np.std(w))
            p10 = float(np.quantile(w, 0.10))
            p90 = float(np.quantile(w, 0.90))
        out[f"{prefix}_w{i}_mean"] = m
        out[f"{prefix}_w{i}_std"] = s
        out[f"{prefix}_w{i}_p10"] = p10
        out[f"{prefix}_w{i}_p90"] = p90
        means.append(m)

    mean_arr = np.asarray(means, dtype=np.float64)
    out[f"{prefix}_w_delta_last_first"] = float(mean_arr[-1] - mean_arr[0])
    out[f"{prefix}_w_slope"] = _safe_slope(np.arange(len(mean_arr), dtype=np.float64), mean_arr)
    out[f"{prefix}_w_max_jump"] = float(np.max(np.abs(np.diff(mean_arr)))) if mean_arr.size > 1 else 0.0
    return out


def _event_segment_stats(mask: np.ndarray, prefix: str) -> dict[str, float]:
    m = np.asarray(mask, dtype=bool)
    n = int(m.size)
    if n == 0:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_frac": 0.0,
            f"{prefix}_mean_len_rel": 0.0,
            f"{prefix}_max_len_rel": 0.0,
            f"{prefix}_t_first_rel": 1.0,
        }

    count = 0
    runs: list[int] = []
    t_first = -1
    i = 0
    while i < n:
        if not m[i]:
            i += 1
            continue
        if t_first < 0:
            t_first = i
        j = i
        while j < n and m[j]:
            j += 1
        runs.append(int(j - i))
        count += 1
        i = j

    if not runs:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_frac": float(np.mean(m)),
            f"{prefix}_mean_len_rel": 0.0,
            f"{prefix}_max_len_rel": 0.0,
            f"{prefix}_t_first_rel": 1.0,
        }

    return {
        f"{prefix}_count": float(count),
        f"{prefix}_frac": float(np.mean(m)),
        f"{prefix}_mean_len_rel": float(np.mean(runs) / n),
        f"{prefix}_max_len_rel": float(np.max(runs) / n),
        f"{prefix}_t_first_rel": float(max(0, t_first) / n),
    }


def _relative_speed_features(speed: np.ndarray, t: np.ndarray) -> dict[str, float]:
    s = np.asarray(speed, dtype=np.float64)
    mean = float(np.mean(s))
    std = float(np.std(s))
    q25, q50, q75, q90 = [float(np.quantile(s, q)) for q in (0.25, 0.50, 0.75, 0.90)]
    iqr = q75 - q25
    slope = _safe_slope(t, s)
    return {
        "speed_cv": float(std / (abs(mean) + EPS)),
        "speed_iqr_rel": float(iqr / (abs(mean) + EPS)),
        "speed_p25_rel": float(q25 / (abs(mean) + EPS)),
        "speed_p50_rel": float(q50 / (abs(mean) + EPS)),
        "speed_p75_rel": float(q75 / (abs(mean) + EPS)),
        "speed_p90_rel": float(q90 / (abs(mean) + EPS)),
        "speed_slope_rel": float(slope / (abs(mean) + EPS)),
    }


def _track_features(row: pd.Series, cache_item: dict[str, Any]) -> dict[str, float]:
    raw = np.asarray(cache_item["raw_features"], dtype=np.float32)
    t = np.asarray(cache_item["times"], dtype=np.float32)
    xyz = np.asarray(cache_item["xyz"], dtype=np.float32)
    rcs = np.asarray(cache_item.get("rcs", raw[:, 3]), dtype=np.float32)

    n = int(min(len(raw), len(t), len(xyz), len(rcs)))
    raw = raw[:n]
    t = t[:n]
    xyz = xyz[:n]
    rcs = rcs[:n]

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    speed = raw[:, 4]
    v_speed = raw[:, 5]
    accel = raw[:, 6]
    curvature = raw[:, 7]
    dt = raw[:, 8]
    time_diff = np.diff(t).astype(np.float64) if n > 1 else np.zeros((0,), dtype=np.float64)

    dx = np.diff(x)
    dy = np.diff(y)
    segment_dist = np.sqrt(dx * dx + dy * dy)
    path_len = float(np.sum(segment_dist))
    e2e = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))

    turn_angles = _turn_angles_deg(x, y)

    duration = float(max(t[-1] - t[0], 1e-3))
    z_range = float(np.max(z) - np.min(z))
    z_mean = float(np.mean(z))
    z_std = float(np.std(z))

    rcs_mean = float(np.mean(rcs))
    rcs_std = float(np.std(rcs))
    rcs_q10, rcs_q25, rcs_q50, rcs_q75, rcs_q90 = [float(np.quantile(rcs, q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90)]
    rcs_iqr = rcs_q75 - rcs_q25
    rcs_range = float(np.max(rcs) - np.min(rcs))
    rcs_slope = _safe_slope(t, rcs)

    # Turn-rate and jerk provide dynamics that track-level summary misses.
    if n >= 3:
        heading = np.unwrap(np.arctan2(dy, dx)).astype(np.float64)
        heading_delta = np.abs(np.diff(heading))
        dt_turn = np.clip(time_diff[1:], 1e-3, None)
        turn_rate = (heading_delta / dt_turn).astype(np.float32)
    else:
        turn_rate = np.zeros((0,), dtype=np.float32)

    if n >= 2:
        dt_safe = np.clip(time_diff, 1e-3, None)
        jerk = (np.diff(accel.astype(np.float64)) / dt_safe).astype(np.float32) if n >= 3 else np.zeros((0,), dtype=np.float32)
    else:
        jerk = np.zeros((0,), dtype=np.float32)

    feats: dict[str, float] = {
        "num_points": float(n),
        "duration": duration,
        "path_length": path_len,
        "straightness": float(e2e / (path_len + EPS)),
        "tortuosity": float(path_len / (e2e + EPS)),
        "convex_hull_area": _convex_hull_area(np.column_stack([x, y])),
        "bbox_area": _bbox_area(np.column_stack([x, y])),
        "max_turn_angle": float(np.max(turn_angles)) if len(turn_angles) else 0.0,
        "mean_curvature": float(np.mean(curvature)),
        "std_curvature": float(np.std(curvature)),
        "z_mean": z_mean,
        "z_std": z_std,
        "z_range": z_range,
        "z_range_rel": float(z_range / (abs(z_mean) + EPS)),
        "vertical_speed_cv": float(np.std(v_speed) / (abs(np.mean(v_speed)) + EPS)),
        "accel_cv": float(np.std(accel) / (abs(np.mean(accel)) + EPS)),
        "rcs_mean": rcs_mean,
        "rcs_std": rcs_std,
        "rcs_iqr": rcs_iqr,
        "rcs_p10": rcs_q10,
        "rcs_p25": rcs_q25,
        "rcs_p50": rcs_q50,
        "rcs_p75": rcs_q75,
        "rcs_p90": rcs_q90,
        "rcs_slope_rel": float(rcs_slope / (abs(rcs_mean) + EPS)),
        "rcs_entropy": float(_entropy(rcs, bins=10) / np.log(10.0)),
        "rcs_vs_speed_corr": _safe_corr(rcs, speed),
        "rcs_vs_z_corr": _safe_corr(rcs, z),
        "rcs_range": rcs_range,
        "rcs_range_rel": float(rcs_range / (abs(rcs_mean) + EPS)),
    }

    feats.update(_relative_speed_features(speed, t))
    feats.update(_distribution_bins(speed, "speed_dist"))
    feats.update(_distribution_bins(rcs, "rcs_dist"))
    feats.update(_distribution_bins(z, "z_dist"))
    feats.update(_moment_stats(speed, "speed"))
    feats.update(_moment_stats(v_speed, "v_speed"))
    feats.update(_moment_stats(accel, "accel"))
    feats.update(_moment_stats(curvature, "curvature"))
    feats.update(_moment_stats(z, "z"))
    feats.update(_moment_stats(rcs, "rcs_stats"))
    feats.update(_moment_stats(dt, "dt"))
    feats.update(_moment_stats(turn_rate, "turn_rate"))
    feats.update(_moment_stats(jerk, "jerk"))
    feats.update(_fft_features(rcs, "rcs", n_bins=10))
    feats.update(_fft_features(speed, "speed", n_bins=10))
    feats.update(_fft_features(accel, "accel", n_bins=8))
    feats.update(_fft_features(z, "z", n_bins=8))
    feats.update(_fft_features(turn_rate, "turn_rate", n_bins=6))
    feats.update(_fft_features(jerk, "jerk", n_bins=6))

    # Window-level dynamics: 8 temporal chunks per signal.
    feats.update(_windowed_stats(speed, "speed", n_windows=8))
    feats.update(_windowed_stats(accel, "accel", n_windows=8))
    feats.update(_windowed_stats(z, "z", n_windows=8))
    feats.update(_windowed_stats(turn_rate, "turn_rate", n_windows=8))
    feats.update(_windowed_stats(jerk, "jerk", n_windows=8))

    # Event segment features.
    speed_p10 = float(np.quantile(speed, 0.10)) if speed.size else 0.0
    accel_p95 = float(np.quantile(np.abs(accel), 0.95)) if accel.size else 0.0
    turn_p95 = float(np.quantile(turn_rate, 0.95)) if turn_rate.size else 0.0
    jerk_p95 = float(np.quantile(np.abs(jerk), 0.95)) if jerk.size else 0.0
    feats.update(_event_segment_stats(speed < speed_p10, "evt_speed_low"))
    feats.update(_event_segment_stats(np.abs(accel) > accel_p95, "evt_accel_spike"))
    feats.update(_event_segment_stats(turn_rate > turn_p95, "evt_turn_spike"))
    feats.update(_event_segment_stats(np.abs(jerk) > jerk_p95, "evt_jerk_spike"))

    for key in [
        "rcs_mean",
        "rcs_std",
        "rcs_iqr",
        "rcs_slope_rel",
        "rcs_range",
        "rcs_range_rel",
        "speed_cv",
        "speed_iqr_rel",
    ]:
        feats[f"{key}_slog1p"] = _signed_log1p(feats[key])

    for cat in RADAR_BIRD_SIZE_CATEGORIES:
        feats[f"radar_bird_size__{cat.lower().replace(' ', '_')}"] = float(row.get("radar_bird_size", "") == cat)

    return feats


def build_tabular_frame(df: pd.DataFrame, cache: dict[int, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        track_id = int(row["track_id"])
        feats = _track_features(row, cache[track_id])
        base = {
            "track_id": track_id,
            "observation_id": int(row["observation_id"]) if "observation_id" in row else -1,
            "timestamp_start_radar_utc": row.get("timestamp_start_radar_utc", None),
        }
        rows.append({**base, **feats})

    out = pd.DataFrame(rows)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    reserved = {"track_id", "observation_id", "timestamp_start_radar_utc"}
    for c in out.columns:
        if c not in reserved:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(np.float32)
    return out


def get_feature_columns(tab_df: pd.DataFrame) -> list[str]:
    reserved = {"track_id", "observation_id", "timestamp_start_radar_utc"}
    cols = [c for c in tab_df.columns if c not in reserved]
    return cols
