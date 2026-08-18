"""
Feature engineering stage.

Transforms raw per-trip TLC records into a zone-hour demand time series
with time, lag, rolling, and weather features. This is the exact feature
set that both the trainer (Phase 1) and the drift monitor (Phase 3) will
consume, so every feature added here has a corresponding entry that will
later show up in the reference distribution snapshot.

load_raw_trips, aggregate_zone_hour_demand, and add_weather_features all
accept optional `months`/`weather_filename` overrides so this same,
already-tested logic (including the non-contiguous-month grid-fill and
lag-leakage fixes) can be reused for the simulated production dataset
without duplicating it. Calling with no override reproduces the original
Phase 1 training behavior exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_trips(params: dict, months: list[str] | None = None) -> pd.DataFrame:
    """Load and concatenate the given months of raw trip Parquet files
    (defaults to training_months)."""
    raw_dir = Path(params["data"]["raw_dir"])
    target_months = months or params["data"]["training_months"]
    frames = []
    for month in target_months:
        path = raw_dir / f"yellow_tripdata_{month}.parquet"
        logger.info("Loading %s", path)
        df = pd.read_parquet(
            path,
            columns=["tpep_pickup_datetime", "PULocationID", "passenger_count", "trip_distance"],
        )
        frames.append(df)
    trips = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d raw trip records across %d months.", len(trips), len(frames))
    return trips


def aggregate_zone_hour_demand(trips: pd.DataFrame, params: dict, months: list[str] | None = None) -> pd.DataFrame:
    """
    Aggregate raw trips into a (zone_id, hour_timestamp) demand time series.

    Drops obviously invalid rows (null pickup time/zone, non-positive
    trip_distance) before aggregating — a light data-quality gate, not a
    full validation suite.
    """
    before = len(trips)
    trips = trips.dropna(subset=["tpep_pickup_datetime", "PULocationID"])
    trips = trips[trips["trip_distance"] > 0]
    logger.info("Dropped %d invalid rows during cleaning.", before - len(trips))

    if trips.empty:
        raise ValueError(
            "No valid trips remain after cleaning (null pickup fields or "
            "non-positive trip_distance removed every row). This likely "
            "indicates an upstream data quality issue, not a legitimate "
            "zero-demand period — investigate the raw source before proceeding."
        )

    trips["hour_ts"] = trips["tpep_pickup_datetime"].dt.floor("h")
    trips["zone_id"] = trips["PULocationID"].astype(int)

    demand = (
        trips.groupby(["zone_id", "hour_ts"])
        .size()
        .reset_index(name="trip_count")
    )

    # Build the zero-fill grid from the UNION of each sampled month's own
    # hour range — NOT a single min-to-max span across all trips combined.
    # target_months is often non-contiguous (e.g. Jan/Apr/Jul/Oct), and a
    # naive min-max range would fabricate "zero demand" for every month in
    # between that was never actually sampled, silently corrupting the
    # dataset with fake data for months we have no real signal for.
    target_months = months or params["data"]["training_months"]
    all_zones = demand["zone_id"].unique()
    month_hour_ranges = []
    for month in target_months:
        month_start = pd.Timestamp(f"{month}-01")
        month_end = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)
        month_hour_ranges.append(pd.date_range(month_start, month_end, freq="h"))
    full_hours = pd.DatetimeIndex(np.concatenate(month_hour_ranges)).unique().sort_values()

    grid = pd.MultiIndex.from_product([all_zones, full_hours], names=["zone_id", "hour_ts"])
    demand = (
        demand.set_index(["zone_id", "hour_ts"])
        .reindex(grid, fill_value=0)
        .reset_index()
    )

    min_volume = params["features"]["min_zone_hourly_volume"]
    zone_totals = demand.groupby("zone_id")["trip_count"].sum()
    keep_zones = zone_totals[zone_totals >= min_volume].index
    dropped = len(zone_totals) - len(keep_zones)
    if dropped:
        logger.info("Dropping %d low-volume zones (< %d total trips).", dropped, min_volume)
    demand = demand[demand["zone_id"].isin(keep_zones)]

    return demand.sort_values(["zone_id", "hour_ts"]).reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features derived purely from the timestamp (no leakage risk)."""
    df = df.copy()
    df["hour_of_day"] = df["hour_ts"].dt.hour
    df["day_of_week"] = df["hour_ts"].dt.dayofweek
    df["month"] = df["hour_ts"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Add per-zone lag and rolling-mean features on trip_count.

    Grouped by (zone_id, month_block) — not zone_id alone — because
    training_months can be non-contiguous (e.g. Jan/Apr/Jul/Oct). Grouping
    by zone_id alone would let `.shift()` pull a "lag" value across a
    2-month gap (e.g. April 1st's lag_1h silently reading late January's
    trip_count), which is wrong data dressed up as a real feature. Grouping
    by month block means each sampled month gets its own short warm-up
    period at the start (rows dropped later), which is the correct
    trade-off for non-contiguous sampling.

    All lag/rolling values are computed strictly from *past* hours relative
    to the row's own hour_ts, so there is no target leakage — this is what
    makes it valid to also use trip_count-derived features as model inputs.
    """
    df = df.sort_values(["zone_id", "hour_ts"]).copy()
    df["_month_block"] = df["hour_ts"].dt.to_period("M").astype(str)
    group_keys = ["zone_id", "_month_block"]

    for lag_h in params["features"]["lag_features"]["lag_hours"]:
        df[f"lag_{lag_h}h"] = df.groupby(group_keys)["trip_count"].shift(lag_h)

    for window_h in params["features"]["rolling_features"]["windows_hours"]:
        df[f"rolling_mean_{window_h}h"] = df.groupby(group_keys)["trip_count"].transform(
            lambda s, w=window_h: s.shift(1).rolling(window=w, min_periods=1).mean()
        )

    return df.drop(columns=["_month_block"])


def add_weather_features(df: pd.DataFrame, params: dict, weather_filename: str = "weather_hourly.csv") -> pd.DataFrame:
    """Join NYC-wide hourly weather onto the zone-hour demand table."""
    weather_path = Path(params["data"]["raw_dir"]) / weather_filename
    weather = pd.read_csv(weather_path, parse_dates=["time"])
    weather = weather.rename(columns={"time": "hour_ts"})

    df = df.merge(weather, on="hour_ts", how="left")

    weather_cols = params["data"]["weather"]["hourly_vars"]
    missing_frac = df[weather_cols].isna().mean()
    for col, frac in missing_frac.items():
        if frac > 0:
            logger.warning("Weather feature '%s' is %.1f%% null after join.", col, frac * 100)

    return df


def build_features(params_path: str = "params.yaml") -> pd.DataFrame:
    """Full feature engineering entry point: raw trips -> model-ready table.
    Uses training_months / weather_hourly.csv by default — unchanged from
    Phase 1."""
    params = load_params(params_path)

    trips = load_raw_trips(params)
    demand = aggregate_zone_hour_demand(trips, params)
    demand = add_time_features(demand)
    demand = add_lag_and_rolling_features(demand, params)
    demand = add_weather_features(demand, params)

    # Drop rows where the longest lag feature is still NaN (start of each
    # zone's series) — these rows can't be used for training or evaluation.
    max_lag = max(params["features"]["lag_features"]["lag_hours"])
    lag_col = f"lag_{max_lag}h"
    before = len(demand)
    demand = demand.dropna(subset=[lag_col])
    logger.info("Dropped %d warm-up rows without full lag history.", before - len(demand))

    out_path = Path(params["data"]["processed_dir"]) / "features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(out_path, index=False)
    logger.info("Saved %d feature rows to %s", len(demand), out_path)

    return demand


if __name__ == "__main__":
    build_features()