"""
Replay driver: walks the simulated production period (Jul-Dec 2024) hour
by hour against the live FastAPI service, to generate a realistic
prediction log for Phase 3 drift monitoring.

For each hour H:
  1. POST /predict/batch for H, using whatever lag/rolling history is
     currently in demand_history (strictly hours < H, no leakage).
  2. POST /actuals for hour H itself, immediately after prediction --
     see module history/docstring below for why this is NOT delayed to
     H+1.
  3. Sleep briefly (config-driven) so the run is watchable, if configured.

RESUME SUPPORT: on startup, the driver checks demand_history for the
latest hour already recorded within the production date range, and skips
straight to the hour after that. This makes an interrupted run (e.g.
laptop losing power) cheap to recover from -- you lose at most the
current in-flight hour's prediction log flush buffer, not hours of
already-completed work. Pass --restart to ignore existing progress and
start from hour 0 regardless (e.g. after intentionally clearing the
stores).

Expect the first ~168 hours (7 days) of production to show skipped/low
n_zones_scored on a truly fresh run: there is an 11-month gap between
training data (ending ~2023-07) and production data (starting 2024-07),
so lag_168h has nothing to look back at until a full week of production
history has accumulated. This is expected warm-up, not a bug.

This script assumes the FastAPI service (src.serving.app) is already
running and demand_history has already been seeded from training data
(python -m src.serving.seed_store) — it does not start or seed anything
itself.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_production_hours(params: dict) -> tuple[list[pd.Timestamp], dict[pd.Timestamp, pd.DataFrame]]:
    """
    Load production_features.parquet and index it by hour_ts, so each hour
    of replay can cheaply look up the (zone_id, trip_count) actuals to
    backfill.
    """
    path = Path(params["serving"]["production_source_path"])
    df = pd.read_parquet(path, columns=["zone_id", "hour_ts", "trip_count"])

    hours = sorted(df["hour_ts"].unique())
    by_hour = {h: g for h, g in df.groupby("hour_ts")}

    logger.info("Loaded %d production hours (%s to %s)", len(hours), hours[0], hours[-1])
    return hours, by_hour


def get_last_completed_production_hour(sqlite_path: str, production_start: pd.Timestamp) -> pd.Timestamp | None:
    """
    Returns the latest hour already recorded in demand_history at or after
    production_start, or None if no production hours have been recorded
    yet. Used to resume an interrupted replay without redoing completed
    hours.
    """
    if not Path(sqlite_path).exists():
        return None

    conn = sqlite3.connect(sqlite_path)
    try:
        row = conn.execute(
            "SELECT MAX(ts) FROM demand_history WHERE ts >= ?",
            (production_start.isoformat(),),
        ).fetchone()
    finally:
        conn.close()

    if row and row[0]:
        return pd.Timestamp(row[0])
    return None


def _predict_hour(base_url: str, timestamp: pd.Timestamp) -> tuple[int, int]:
    """Calls /predict/batch for one hour. Returns (n_scored, n_skipped)."""
    resp = requests.post(
        f"{base_url}/predict/batch",
        json={"timestamp": timestamp.isoformat()},
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()
    return body["n_zones_scored"], body["n_zones_skipped"]


def _record_actuals(base_url: str, timestamp: pd.Timestamp, actuals_df: pd.DataFrame) -> int:
    """Calls /actuals for every zone at one hour. Returns count recorded."""
    recorded = 0
    for row in actuals_df.itertuples(index=False):
        resp = requests.post(
            f"{base_url}/actuals",
            json={
                "zone_id": int(row.zone_id),
                "timestamp": timestamp.isoformat(),
                "actual_trip_count": float(row.trip_count),
            },
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning("Failed to record actual for zone %d at %s: %s", row.zone_id, timestamp, resp.text)
            continue
        recorded += 1
    return recorded


def run_replay(params_path: str = "params.yaml", max_hours: int | None = None, restart: bool = False) -> None:
    params = load_params(params_path)
    replay_cfg = params["serving"]["replay"]
    base_url = replay_cfg["base_url"]
    delay = replay_cfg["delay_seconds_per_hour"]
    log_every = replay_cfg["progress_log_every_n_hours"]
    sqlite_path = params["serving"]["historical_store"]["sqlite_path"]

    try:
        health = requests.get(f"{base_url}/health", timeout=5)
        health.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Could not reach serving API at {base_url}/health — is "
            f"'python -m uvicorn src.serving.app:app' running? ({e})"
        ) from e

    hours, by_hour = load_production_hours(params)

    if not restart:
        last_done = get_last_completed_production_hour(sqlite_path, hours[0])
        if last_done is not None:
            resume_from = last_done + pd.Timedelta(hours=1)
            original_count = len(hours)
            hours = [h for h in hours if h >= resume_from]
            logger.info(
                "Resuming: found completed production data up to %s. "
                "Skipping %d already-done hours, %d remaining.",
                last_done, original_count - len(hours), len(hours),
            )
            if not hours:
                logger.info("Nothing left to replay — production period already fully processed.")
                return
        else:
            logger.info("No prior production progress found — starting from hour 0.")

    if max_hours is not None:
        hours = hours[:max_hours]
        logger.info("Capped replay to first %d hours of this run.", max_hours)

    total_scored = 0
    total_skipped_hours = 0
    total_actuals_recorded = 0
    start_time = time.monotonic()

    for i, hour in enumerate(hours):
        try:
            n_scored, n_skipped_zones = _predict_hour(base_url, hour)
            if n_scored == 0:
                logger.warning("Hour %s: 0 zones scored (all skipped).", hour)
                total_skipped_hours += 1
            total_scored += n_scored
        except requests.RequestException as e:
            logger.warning("Hour %s: /predict/batch failed (%s) — skipping this hour.", hour, e)
            total_skipped_hours += 1

        actuals_df = by_hour.get(hour)
        if actuals_df is not None:
            total_actuals_recorded += _record_actuals(base_url, hour, actuals_df)

        if (i + 1) % log_every == 0 or i == len(hours) - 1:
            elapsed = time.monotonic() - start_time
            logger.info(
                "Progress: %d/%d hours (this run) | scored=%d | skipped_hours=%d | actuals=%d | elapsed=%.0fs",
                i + 1, len(hours), total_scored, total_skipped_hours, total_actuals_recorded, elapsed,
            )

        if delay:
            time.sleep(delay)

    elapsed = time.monotonic() - start_time
    logger.info(
        "Replay run complete: %d hours processed this run | %d predictions scored | "
        "%d hours skipped | %d actuals recorded | %.0fs total",
        len(hours), total_scored, total_skipped_hours, total_actuals_recorded, elapsed,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replay production data hour-by-hour against the serving API.")
    parser.add_argument("max_hours", nargs="?", type=int, default=None, help="Optional cap on hours to process this run.")
    parser.add_argument("--restart", action="store_true", help="Ignore existing progress and start from hour 0.")
    args = parser.parse_args()

    run_replay(max_hours=args.max_hours, restart=args.restart)