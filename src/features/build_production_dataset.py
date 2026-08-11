"""..."""
from __future__ import annotations

import copy
import logging
from pathlib import Path

from.build_features import load_params, load_raw_trips, aggregate_zone_hour_demand, add_weather_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_production_dataset(params_path: str = "params.yaml") -> None:
    params = load_params(params_path)
    months = params["data"]["production_months"]

    # Production raw files live in their own directory, separate from
    # data/raw (which is a DVC pipeline output owned by the `ingest`
    # stage) -- swap raw_dir on a copy of params rather than touching
    # the shared training path.
    prod_params = copy.deepcopy(params)
    prod_params["data"]["raw_dir"] = params["data"]["raw_production_dir"]

    trips = load_raw_trips(prod_params, months=months)
    demand = aggregate_zone_hour_demand(trips, prod_params, months=months)
    demand = add_weather_features(demand, prod_params, weather_filename="weather_hourly_production.csv")

    out_path = Path(params["data"]["processed_dir"]) / "production_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(out_path, index=False)
    logger.info("Saved %d production rows to %s", len(demand), out_path)


if __name__ == "__main__":
    build_production_dataset()