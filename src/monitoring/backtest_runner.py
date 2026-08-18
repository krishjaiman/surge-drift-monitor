"""Backtest entrypoint: replays the monitoring job hourly across the full
Jul-Dec 2024 production period, using data that already exists on disk from
the completed Phase 2 replay.

This is what backfills a realistic drift time series for Grafana (once
Docker is installed) and lets TTD be measured against the point where PSI
first crosses threshold, rather than only ever seeing a single "current"
reading.

Run: python -m src.monitoring.backtest_runner
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
import yaml

from src.monitoring.correlation_stability import seed_or_load_baseline
from src.monitoring.monitoring_job import run_monitoring_for_hour
from src.monitoring.output_writer import append_alert, write_metrics_row
from src.monitoring.prediction_log_reader import load_predictions_with_actuals
from src.monitoring.reference_loader import ReferenceSnapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_backtest(params: dict, start: str, end: str) -> None:
    mon = params["monitoring"]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    reference = ReferenceSnapshot.from_json(mon["reference_snapshot_path"])

    logger.info(
        "Loading full joined prediction+actuals dataset for backtest [%s, %s)...",
        start_ts, end_ts,
    )
    joined_df = load_predictions_with_actuals(
        mon["inputs"]["predictions_dir"], mon["inputs"]["sqlite_path"], start_ts, end_ts
    )
    if joined_df.empty:
        raise RuntimeError(
            "No joined prediction+actual data found for the backtest window. "
            "Confirm the Phase 2 replay completed and populated both "
            "data/predictions/ and demand_history before running Phase 3."
        )
    logger.info("Loaded %d joined rows across %d hours.", len(joined_df), joined_df["timestamp"].nunique())

    baseline_window_hours = mon["windows"]["correlation_baseline_window_hours"]
    baseline_end = start_ts + pd.Timedelta(hours=baseline_window_hours)
    seeding_df = joined_df[joined_df["timestamp"] <= baseline_end]
    correlation_baseline = seed_or_load_baseline(
        mon["prediction_correlation_baseline_path"],
        seeding_df,
        mon["drift_thresholds"]["psi_features"],
    )
    logger.info(
        "Correlation baseline seeded/loaded for %d features from first %d hours.",
        len(correlation_baseline), baseline_window_hours,
    )

    hours = pd.date_range(
        start_ts, end_ts,
        freq=f"{mon['cadence']['granularity_hours']}h",
        inclusive="left",
    )
    consecutive_bias_counts: dict[int, int] = {}
    n_trigger_fires = 0

    for i, as_of in enumerate(hours):
        alert, metrics_row, consecutive_bias_counts = run_monitoring_for_hour(
            as_of, reference, joined_df, correlation_baseline, params, consecutive_bias_counts
        )
        write_metrics_row(metrics_row, mon["outputs"]["metrics_timeseries_path"])
        append_alert(alert, mon["outputs"]["alerts_log_path"])
        if alert.retrain_trigger_fired:
            n_trigger_fires += 1
        if (i + 1) % 168 == 0:
            logger.info("Backtest progress: %d/%d hours (through %s)", i + 1, len(hours), as_of)

    logger.info(
        "Backtest complete: %d hours evaluated, %d retrain-trigger fires. "
        "Output written to %s and %s.",
        len(hours), n_trigger_fires,
        mon["outputs"]["metrics_timeseries_path"], mon["outputs"]["alerts_log_path"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3 monitoring backtest over historical production data."
    )
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--start", default="2024-07-01T00:00:00")
    parser.add_argument("--end", default="2025-01-01T00:00:00")
    args = parser.parse_args()

    run_backtest(load_params(args.params), args.start, args.end)