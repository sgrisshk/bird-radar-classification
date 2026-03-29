from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

EEMSHAVEN_LAT = 53.4386
EEMSHAVEN_LON = 6.8347

RWS_STATION = "DELFZL"
RWS_PARAM = "WATHTE"
RWS_BASE = (
    "https://waterwebservices.rijkswaterstaat.nl/"
    "ONLINEWAARNEMINGENSERVICES_DBO/OphalenWaarnemingen"
)

OM_BASE = "https://archive-api.open-meteo.com/v1/archive"
OM_VARS = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "precipitation",
    "cloud_cover",
    "visibility",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "weather_code",
]


def parse_track_timestamps(tracks_df: pd.DataFrame) -> pd.Series:
    for col in [
        "timestamp_start_radar_utc",
        "start_time",
        "timestamp",
        "time",
        "date",
        "datetime",
    ]:
        if col in tracks_df.columns:
            ts = pd.to_datetime(tracks_df[col], errors="coerce", utc=True)
            print(f"  Using column '{col}' as track timestamp.")
            return tracks_df[["track_id"]].assign(ts=ts.values).set_index("track_id")["ts"]

    raise ValueError(
        "No timestamp column found. Expected one of: "
        "timestamp_start_radar_utc,start_time,timestamp,time,date,datetime\n"
        f"Available columns: {list(tracks_df.columns)}"
    )


def floor_to_hour(ts: pd.Series) -> pd.Series:
    return ts.dt.floor("h")


def _as_naive_utc_index(idx_like: pd.Index | pd.Series) -> pd.DatetimeIndex:
    dt = pd.to_datetime(idx_like, errors="coerce", utc=True)
    return pd.DatetimeIndex(dt).tz_localize(None)


def fetch_openmeteo(date_min: str, date_max: str) -> pd.DataFrame:
    print(f"  Fetching Open-Meteo weather: {date_min} -> {date_max}")
    params = {
        "latitude": EEMSHAVEN_LAT,
        "longitude": EEMSHAVEN_LON,
        "start_date": date_min,
        "end_date": date_max,
        "hourly": ",".join(OM_VARS),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    for attempt in range(5):
        try:
            r = requests.get(OM_BASE, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as exc:
            print(f"    Attempt {attempt + 1} failed: {exc}")
            time.sleep(2**attempt)
    else:
        raise RuntimeError("Open-Meteo fetch failed after 5 attempts.")

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    wd_rad = np.deg2rad(df["wind_direction_10m"])
    df["wind_u"] = -df["wind_speed_10m"] * np.sin(wd_rad)
    df["wind_v"] = -df["wind_speed_10m"] * np.cos(wd_rad)

    df["wind_is_strong"] = (df["wind_speed_10m"] > 10).astype(int)
    df["is_gust_event"] = (df["wind_gusts_10m"] > 15).astype(int)
    df["is_raining"] = (df["precipitation"] > 0.1).astype(int)
    df["cloud_clear"] = (df["cloud_cover"] < 20).astype(int)
    df["cloud_overcast"] = (df["cloud_cover"] > 80).astype(int)
    df["low_visibility"] = (df["visibility"] < 5000).astype(int)

    wc = df["weather_code"].fillna(0).astype(int)
    df["weather_clear"] = (wc < 3).astype(int)
    df["weather_fog"] = ((wc >= 40) & (wc < 50)).astype(int)
    df["weather_rain"] = ((wc >= 60) & (wc < 70)).astype(int)
    df["weather_snow"] = ((wc >= 70) & (wc < 80)).astype(int)
    return df


def _iter_month_chunks(start_dt: datetime, end_dt: datetime) -> list[tuple[datetime, datetime]]:
    chunks: list[tuple[datetime, datetime]] = []
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + timedelta(days=10), end_dt)
        chunks.append((cur, nxt))
        cur = nxt + timedelta(seconds=1)
    return chunks


def fetch_rws_tide(date_min: str, date_max: str) -> pd.DataFrame:
    print(f"  Fetching RWS tide data: {date_min} -> {date_max}")

    start_dt = datetime.strptime(date_min, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(date_max, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    chunks_data: list[dict[str, object]] = []
    for chunk_start, chunk_end in _iter_month_chunks(start_dt, end_dt):
        payload = {
            "Locatie": {
                "Code": RWS_STATION,
                "X": 6.9344,
                "Y": 53.3339,
            },
            "AquoMetaData": {
                "Parameter": {
                    "Sleutelwaarde": RWS_PARAM,
                    "Code": RWS_PARAM,
                }
            },
            "Periode": {
                "Begindatumtijd": chunk_start.strftime("%Y-%m-%dT%H:%M:%S.000+00:00"),
                "Einddatumtijd": chunk_end.strftime("%Y-%m-%dT%H:%M:%S.000+00:00"),
            },
        }
        try:
            r = requests.post(RWS_BASE, json=payload, timeout=60)
            r.raise_for_status()
            rows = r.json().get("WaarnemingenLijst", [{}])[0].get("MetingenLijst", [])
            for row in rows:
                ts_str = row.get("Tijdstip", "")
                val = row.get("Meetwaarde", {}).get("Waarde_Numeriek")
                if ts_str and val is not None:
                    chunks_data.append({"time": ts_str, "tide_height_m": float(val) / 100.0})
        except Exception as exc:
            print(f"    RWS chunk {chunk_start.date()} failed: {exc} (will fallback if empty)")

    if chunks_data:
        df = pd.DataFrame(chunks_data)
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
        print(f"    Got {len(df)} tide observations from RWS API.")
        return df
    print("    RWS API returned no data; trying Open-Meteo Marine fallback.")
    return _marine_or_synthetic_tide(date_min, date_max)


def _marine_or_synthetic_tide(date_min: str, date_max: str) -> pd.DataFrame:
    try:
        params = {
            "latitude": EEMSHAVEN_LAT,
            "longitude": EEMSHAVEN_LON,
            "start_date": date_min,
            "end_date": date_max,
            "hourly": "sea_level_height_msl",
            "timezone": "UTC",
        }
        r = requests.get("https://marine-api.open-meteo.com/v1/marine", params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", {})
        if "time" in hourly and "sea_level_height_msl" in hourly:
            df = pd.DataFrame(
                {
                    "time": pd.to_datetime(hourly["time"], errors="coerce", utc=True),
                    "tide_height_m": pd.to_numeric(hourly["sea_level_height_msl"], errors="coerce"),
                }
            ).dropna()
            if len(df) > 0:
                print(f"    Open-Meteo Marine fallback rows: {len(df)}")
                return df
    except Exception as exc:
        print(f"    Open-Meteo Marine fallback failed: {exc}")

    print("    Using synthetic M2 tide fallback.")
    times = pd.date_range(date_min, date_max, freq="10min", tz="UTC")
    m2_period_hours = 12.4206
    ref = pd.Timestamp("2020-01-01 02:30:00", tz="UTC")
    elapsed_hours = (times - ref).total_seconds() / 3600.0
    tide = 1.7 * np.cos(2 * np.pi * elapsed_hours / m2_period_hours)
    return pd.DataFrame({"time": times, "tide_height_m": tide})


def add_tide_features(tide_df: pd.DataFrame) -> pd.DataFrame:
    df = tide_df.copy().sort_values("time")
    h = df["tide_height_m"].to_numpy(dtype=np.float64)

    dt_hours = np.diff(df["time"].astype("int64").to_numpy(dtype=np.int64)) / 1e9 / 3600.0
    rate = np.concatenate([[0.0], np.diff(h) / np.where(dt_hours > 0, dt_hours, 1.0)])
    df["tide_rate"] = rate
    df["tide_is_flood"] = (rate > 0.05).astype(int)
    df["tide_is_ebb"] = (rate < -0.05).astype(int)

    window = 18
    hs = pd.Series(h)
    roll_max = hs.rolling(window, center=True).max()
    roll_min = hs.rolling(window, center=True).min()
    df["is_high_tide"] = (hs >= (roll_max - 0.1)).astype(int)
    df["is_low_tide"] = (hs <= (roll_min + 0.1)).astype(int)

    roll_range = roll_max - roll_min
    df["tide_norm"] = np.where(roll_range > 0.2, (hs - roll_min) / roll_range, 0.5)
    return df


def resample_tide_hourly(tide_df: pd.DataFrame) -> pd.DataFrame:
    df = tide_df.set_index("time")
    num_cols = df.select_dtypes(include=[np.number]).columns
    return df[num_cols].resample("h").mean().reset_index()


def build_external_features(tracks_path: str, output_path: str) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print(f"Building external features for: {tracks_path}")
    print("=" * 60)

    print("\n[1/5] Loading tracks...")
    tracks = pd.read_csv(tracks_path)
    if "track_id" not in tracks.columns:
        raise ValueError("tracks file must contain 'track_id'")
    print(f"  {len(tracks)} tracks loaded.")

    timestamps = parse_track_timestamps(tracks)
    timestamps = timestamps.dropna()
    date_min = timestamps.min().strftime("%Y-%m-%d")
    date_max = timestamps.max().strftime("%Y-%m-%d")
    print(f"  Date range: {date_min} -> {date_max}")

    print("\n[2/5] Fetching weather from Open-Meteo...")
    weather_df = add_weather_features(fetch_openmeteo(date_min, date_max))
    print(f"  Got {len(weather_df)} weather rows.")

    print("\n[3/5] Fetching tide from Rijkswaterstaat...")
    tide_hourly = resample_tide_hourly(add_tide_features(fetch_rws_tide(date_min, date_max)))
    print(f"  Got {len(tide_hourly)} tide rows.")

    print("\n[4/5] Merging weather + tide...")
    ext = pd.merge(weather_df, tide_hourly, on="time", how="outer").sort_values("time").set_index("time")
    ext.index = _as_naive_utc_index(ext.index)

    print("\n[5/5] Joining to tracks...")
    ts_hourly = floor_to_hour(parse_track_timestamps(tracks))
    ts_hourly = pd.Series(_as_naive_utc_index(ts_hourly), index=ts_hourly.index)

    result_rows: list[dict[str, object]] = []
    for track_id, ts_h in ts_hourly.items():
        if pd.isna(ts_h):
            row = {}
        elif ts_h in ext.index:
            row = ext.loc[ts_h].to_dict()
        else:
            idx = ext.index.get_indexer([ts_h], method="nearest")[0]
            row = ext.iloc[idx].to_dict() if idx >= 0 else {}
        row["track_id"] = int(track_id)
        result_rows.append(row)

    result = pd.DataFrame(result_rows)
    result = result.drop(columns=["time"], errors="ignore")
    ext_cols = [c for c in result.columns if c != "track_id"]
    result = result.rename(columns={c: f"ext_{c}" for c in ext_cols})

    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_p, index=False)

    print(f"\nResult: {len(result)} rows x {len(result.columns)} cols")
    print(f"Saved: {output_p}")

    nan_pct = result.isnull().mean().sort_values(ascending=False)
    bad = nan_pct[nan_pct > 0.1]
    if len(bad):
        print("\nWarning: columns with >10% NaN")
        print(bad.to_string())
    else:
        print("No columns with >10% NaN")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch external weather+tide features")
    parser.add_argument("--tracks", required=True, help="Path to train.csv or test.csv")
    parser.add_argument("--output", required=True, help="Output parquet path")
    args = parser.parse_args()
    build_external_features(args.tracks, args.output)


if __name__ == "__main__":
    main()
