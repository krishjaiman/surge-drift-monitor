"""
Ingestion stage: download and schema-validate raw data sources.

Downloads:
  1. NYC TLC Yellow Taxi trip records (one Parquet file per configured month)
  2. NYC TLC Zone Lookup CSV (zone_id -> borough/zone name)
  3. NYC-wide hourly weather from the Open-Meteo archive API

Each downloaded file is schema-validated before being written to disk, so
a silently-changed upstream schema fails loudly here rather than corrupting
features three stages downstream. This validation gate is the ingestion
equivalent of the "reference snapshot" principle: fail fast, fail visibly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Columns the pipeline actually depends on. If any are missing after a
# download, the TLC schema has changed upstream and we must not proceed
# silently — this is exactly the "upstream pipeline drift" failure mode
# called out in the project's business context.
REQUIRED_TRIP_COLUMNS = {
    "tpep_pickup_datetime",
    "PULocationID",
    "passenger_count",
    "trip_distance",
}

REQUIRED_ZONE_COLUMNS = {"LocationID", "Borough", "Zone"}


def load_params(params_path: str = "params.yaml") -> dict:
    """Load the central params.yaml config."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def _download_file(url: str, dest_path: Path, chunk_size: int = 1 << 20) -> None:
    """Stream-download a file to disk, raising on any HTTP error."""
    logger.info("Downloading %s -> %s", url, dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)


def download_trip_month(month: str, params: dict) -> Path:
    """
    Download a single month of Yellow Taxi trip data.

    Args:
        month: "YYYY-MM" string, e.g. "2022-01".
        params: loaded params.yaml dict.

    Returns:
        Path to the downloaded Parquet file.

    Raises:
        ValueError: if the downloaded file is missing required columns.
    """
    url = params["data"]["tlc_base_url"].format(month=month)
    dest = Path(params["data"]["raw_dir"]) / f"yellow_tripdata_{month}.parquet"

    if dest.exists():
        logger.info("Already have %s, skipping download.", dest)
    else:
        _download_file(url, dest)

    _validate_trip_schema(dest, month)
    return dest


def _validate_trip_schema(path: Path, month: str) -> None:
    """Raise ValueError if the trip file is missing required columns."""
    # Read only the schema (no data) for speed on large files.
    import pyarrow.parquet as pq

    parquet_schema = pq.ParquetFile(path).schema_arrow
    available_cols = set(parquet_schema.names)

    missing = REQUIRED_TRIP_COLUMNS - available_cols
    if missing:
        raise ValueError(
            f"Trip data for {month} is missing required columns {missing}. "
            f"TLC schema may have changed upstream — do not proceed until "
            f"this is investigated and the pipeline updated."
        )
    logger.info("Schema OK for %s (%d columns).", month, len(available_cols))


def download_zone_lookup(params: dict) -> Path:
    """Download the TLC zone lookup CSV and validate its schema."""
    dest = Path(params["data"]["raw_dir"]) / "taxi_zone_lookup.csv"
    if not dest.exists():
        _download_file(params["data"]["zone_lookup_url"], dest)

    df = pd.read_csv(dest, nrows=5)
    missing = REQUIRED_ZONE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Zone lookup missing required columns {missing}.")
    logger.info("Zone lookup schema OK.")
    return dest


def download_weather(params: dict, months: list[str] | None = None, dest_filename: str = "weather_hourly.csv") -> Path:
    """
    Download NYC-wide hourly weather covering the given months (defaults to
    training_months) via the Open-Meteo archive API.
    """
    months = sorted(months or params["data"]["training_months"])
    start_date = f"{months[0]}-01"
    last_month = pd.Period(months[-1], freq="M")
    end_date = last_month.end_time.strftime("%Y-%m-%d")

    weather_cfg = params["data"]["weather"]
    dest = Path(params["data"]["raw_dir"]) / dest_filename

    if dest.exists():
        logger.info("Already have %s, skipping download.", dest)
        return dest

    resp = requests.get(
        weather_cfg["archive_base_url"],
        params={
            "latitude": weather_cfg["latitude"],
            "longitude": weather_cfg["longitude"],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(weather_cfg["hourly_vars"]),
            "timezone": "America/New_York",
        },
        timeout=120,
    )
    resp.raise_for_status()
    payload = resp.json()

    if "hourly" not in payload:
        raise ValueError(f"Unexpected Open-Meteo response shape: keys={list(payload.keys())}")

    weather_df = pd.DataFrame(payload["hourly"])
    weather_df.to_csv(dest, index=False)
    logger.info("Weather data saved to %s (%d rows).", dest, len(weather_df))
    return dest


def run_ingestion(params_path: str = "params.yaml", months: list[str] | None = None, weather_dest_filename: str = "weather_hourly.csv") -> None:
    """Entry point: download raw data for the given months (defaults to training_months)."""
    params = load_params(params_path)
    target_months = months or params["data"]["training_months"]

    for month in target_months:
        download_trip_month(month, params)

    download_zone_lookup(params)
    download_weather(params, months=target_months, dest_filename=weather_dest_filename)

    logger.info("Ingestion complete for months: %s", target_months)



if __name__ == "__main__":
    run_ingestion()
