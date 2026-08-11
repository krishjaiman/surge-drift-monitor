"""
Combines training + production weather CSVs into one deduped hourly
lookup table, so FeatureBuilder has a single real file to read regardless
of which period a request timestamp falls in.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .build_features import load_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_weather_cache(params_path: str = "params.yaml") -> None:
    params = load_params(params_path)
    raw_dir = Path(params["data"]["raw_dir"])
    raw_production_dir = Path(params["data"]["raw_production_dir"])

    frames = []
    for dir_path, filename in [
        (raw_dir, "weather_hourly.csv"),
        (raw_production_dir, "weather_hourly_production.csv"),
    ]:
        path = dir_path / filename
        if not path.exists():
            logger.warning("%s not found, skipping.", path)
            continue
        df = pd.read_csv(path, parse_dates=["time"]).rename(columns={"time": "hour_ts"})
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No weather CSVs found - run ingestion first.")

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="hour_ts")
        .sort_values("hour_ts")
        .reset_index(drop=True)
    )
    out_path = Path(params["data"]["processed_dir"]) / "weather_hourly.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info("Saved combined weather cache: %d hours -> %s", len(combined), out_path)


if __name__ == "__main__":
    build_weather_cache()