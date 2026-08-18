"""Scheduled production entrypoint: evaluates monitoring for a single hour,
as_of=now (or an explicit --as-of for manual reruns).

This is the real production job. Point Windows Task Scheduler (or cron, on
whatever the deployment target ends up being) at this script hourly. It
calls the exact same `run_monitoring_for_hour` the backtest used, so drift
math is never duplicated between the two run modes.

Unlike the backtest (one long-lived process, state kept in memory), each
scheduled invocation is a fresh process -- so the per-zone consecutive-bias
breach counter must be persisted to disk between runs rather than passed
in memory.

Run: python -m src.monitoring.scheduled_run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

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


def _load_bias_state(path: str | Path) -> dict[int, int]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _save_bias_state(state: dict[int, int], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f)


def run_scheduled(params: dict, as_of: pd.Timestamp) -> None:
    mon = params["monitoring"]
    reference = ReferenceSnapshot.from_json(mon["reference_snapshot_path"])

    # Lookback must cover the widest window run_monitoring_for_hour needs:
    # the 30-day RMSE window plus its variance historical lookback (also 30d).
    lookback_hours = max(mon["windows"]["rmse_windows_hours"]) + 24
    window_start = as_of - pd.Timedelta(hours=lookback_hours)

    joined_df = load_predictions_with_actuals(
        mon["inputs"]["predictions_dir"], mon["inputs"]["sqlite_path"], window_start, as_of
    )
    if joined_df.empty:
        logger.error(
            "No prediction/actual data in [%s, %s]. Skipping this run -- "
            "check the FastAPI service and SQLite store are being written to.",
            window_start, as_of,
        )
        return

    if not Path(mon["prediction_correlation_baseline_path"]).exists():
        logger.error(
            "No correlation baseline cached at %s. Run the backtest first, or seed it "
            "manually -- a scheduled run should never silently create a baseline from "
            "a single hour of data.", mon["prediction_correlation_baseline_path"],
        )
        return
    correlation_baseline = seed_or_load_baseline(
        mon["prediction_correlation_baseline_path"], joined_df,
        mon["drift_thresholds"]["psi_features"],
    )

    bias_state = _load_bias_state(mon["outputs"]["bias_state_path"])
    alert, metrics_row, updated_bias_state = run_monitoring_for_hour(
        as_of, reference, joined_df, correlation_baseline, params, bias_state
    )
    _save_bias_state(updated_bias_state, mon["outputs"]["bias_state_path"])

    write_metrics_row(metrics_row, mon["outputs"]["metrics_timeseries_path"])
    append_alert(alert, mon["outputs"]["alerts_log_path"])

    logger.info(
        "Monitoring run complete for %s: drift_breached=%s, performance_degraded=%s, "
        "retrain_trigger_fired=%s",
        as_of, alert.drift_breached, alert.performance_degraded, alert.retrain_trigger_fired,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 scheduled monitoring run.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--as-of", default=None,
        help="ISO timestamp to evaluate as-of. Defaults to now, floored to the hour.",
    )
    args = parser.parse_args()

    as_of_ts = (
        pd.Timestamp(args.as_of) if args.as_of
        else pd.Timestamp.now().floor("h")
    )
    run_scheduled(load_params(args.params), as_of_ts)