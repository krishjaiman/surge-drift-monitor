"""
One-time (rerunnable) script to seed HistoricalDemandStore with raw
(zone_id, hour_ts, trip_count) rows from the Phase 1 processed dataset.

Deliberately ignores the pre-computed lag_*/rolling_* columns in
features.parquet -- HistoricalDemandStore recomputes those fresh at
request time from raw trip_count, so seeding with the baked-in values
would just be redundant (and risks drifting out of sync if the two
computations ever diverge).

Run from project root:
    python -m src.serving.seed_store
"""
from __future__ import annotations

import logging

import pandas as pd
import yaml

from src.serving.historical_store import HistoricalDemandStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    with open("params.yaml") as f:
        config = yaml.safe_load(f)

    seed_path = config["serving"]["seed_source_path"]
    sqlite_path = config["serving"]["historical_store"]["sqlite_path"]

    logger.info("Reading %s", seed_path)
    df = pd.read_parquet(seed_path, columns=["zone_id", "hour_ts", "trip_count"])
    df = df.rename(columns={"hour_ts": "timestamp"})

    store = HistoricalDemandStore(sqlite_path)
    n = store.seed_from_dataframe(df)
    store.close()

    logger.info("Seeded %d raw (zone_id, hour, trip_count) rows into %s", n, sqlite_path)


if __name__ == "__main__":
    main()