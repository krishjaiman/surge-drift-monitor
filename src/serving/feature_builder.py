"""
Assembles the full feature row a request needs, in the exact column order
the model was trained on. Pulls: calendar features (deterministic from
timestamp), lag/rolling (from HistoricalDemandStore), weather (from the
combined training+production weather cache built by build_weather_cache.py).
"""
from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache

import pandas as pd

from src.serving.historical_store import HistoricalDemandStore

logger = logging.getLogger(__name__)

# Must match the training-time feature order exactly.
FEATURE_ORDER = [
    "zone_id", "hour_of_day", "day_of_week", "month", "is_weekend",
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_3h", "rolling_mean_24h",
    "temperature_2m", "precipitation", "windspeed_10m",
]


@lru_cache(maxsize=1)
def _load_weather_cache(weather_cache_path: str) -> pd.DataFrame:
    weather = pd.read_parquet(weather_cache_path).set_index("hour_ts")
    logger.info("Loaded weather cache: %d unique hours", len(weather))
    return weather


class FeatureBuilder:
    def __init__(
        self,
        store: HistoricalDemandStore,
        weather_cache_path: str,
        lag_hours: list[int],
        rolling_windows: list[int],
        categorical_features: list[str],
    ) -> None:
        self.store = store
        self.weather_cache_path = weather_cache_path
        self.lag_hours = lag_hours
        self.rolling_windows = rolling_windows
        self.categorical_features = categorical_features

    def _weather_for(self, timestamp: datetime) -> dict[str, float] | None:
        weather = _load_weather_cache(self.weather_cache_path)
        ts_hour = pd.Timestamp(timestamp).floor("h")
        if ts_hour not in weather.index:
            logger.warning(
                "No weather data cached for %s - outside both training and "
                "production ingested ranges.", ts_hour
            )
            return None
        row = weather.loc[ts_hour]
        return {
            "temperature_2m": float(row["temperature_2m"]),
            "precipitation": float(row["precipitation"]),
            "windspeed_10m": float(row["windspeed_10m"]),
        }

    def build(self, zone_id: int, timestamp: datetime) -> pd.DataFrame | None:
        lag_roll = self.store.get_lag_and_rolling_features(
            zone_id, timestamp, self.lag_hours, self.rolling_windows
        )
        if lag_roll is None:
            return None

        weather = self._weather_for(timestamp)
        if weather is None:
            return None

        row = {
            "zone_id": zone_id,
            "hour_of_day": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "month": timestamp.month,
            "is_weekend": int(timestamp.weekday() >= 5),
            **lag_roll,
            **weather,
        }
        df = pd.DataFrame([row])[FEATURE_ORDER]

        # Must match train.py's dtype cast exactly -- LightGBM's Booster
        # stores the categorical schema from training (pandas_categorical)
        # and needs the incoming column to be category dtype to re-map
        # correctly. A plain int column here would either error (as it just
        # did) or, worse, silently be treated as a raw numeric split value.
        for col in self.categorical_features:
            df[col] = df[col].astype("category")

        return df
