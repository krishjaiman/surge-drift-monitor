"""Integrated Phase 3 + Phase 4 backtest: monitoring detects drift, the
retrain orchestrator consumes the trigger, and on promotion the loop
hot-swaps its working state (reference snapshot, joined predictions
dataframe, correlation baseline, bias-streak counters) so monitoring
continues against the NEW baseline for the rest of the backtest.

This is what demonstrates the full TTD -> TTR story: quiet -> drift ramps
-> trigger fires -> retrain -> promote -> quiet again (if the promotion
actually helped).

Run: python -m src.retraining.backtest_runner_with_retraining
"""

from __future__ import annotations

import argparse
import logging

import mlflow
import pandas as pd
import yaml

from src.monitoring.correlation_stability import seed_or_load_baseline
from src.monitoring.monitoring_job import run_monitoring_for_hour
from src.monitoring.output_writer import append_alert, write_metrics_row
from src.monitoring.prediction_log_reader import load_predictions_with_actuals
from src.monitoring.reference_loader import ReferenceSnapshot
from src.retraining.retrain_orchestrator import (
    append_retrain_event,
    run_retrain_cycle,
    should_attempt_retrain,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_feature_names(model) -> list[str]:
    """mlflow.lightgbm.load_model can return either a raw Booster or an
    sklearn-wrapper depending on how the model was logged -- handle both
    rather than assume one.
    """
    if hasattr(model, "feature_name"):
        return list(model.feature_name())
    if hasattr(model, "booster_"):
        return list(model.booster_.feature_name())
    raise TypeError(
        f"Could not determine feature names from champion model of type {type(model)}."
    )


def run_backtest_with_retraining(params: dict, start: str, end: str) -> None:
    mon = params["monitoring"]
    retrain_cfg = params["retraining"]
    mlflow_cfg = params["mlflow"]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    champion_model = mlflow.lightgbm.load_model(
        f"models:/{mlflow_cfg['registered_model_name']}@{mlflow_cfg['champion_alias']}"
    )
    feature_names = _get_feature_names(champion_model)
    logger.info("Loaded champion feature schema: %d features.", len(feature_names))

    reference = ReferenceSnapshot.from_json(mon["reference_snapshot_path"])

    # Load a lookback BUFFER before start_ts, not just [start_ts, end_ts).
    # Without this, any retrain attempt near the beginning of the backtest
    # window finds a 30-day lookback that's mostly empty (clipped at
    # start_ts) -- producing a "successful" retrain trained on a few hours
    # of data instead of 30 days, which is worse than no retrain at all.
    # Monitoring still only EVALUATES hours in [start_ts, end_ts) below;
    # this buffer exists purely so retrain attempts have real history.
    data_load_start = start_ts - pd.Timedelta(days=retrain_cfg["lookback_days"])
    logger.info(
        "Loading joined prediction+actuals dataset with lookback buffer: "
        "[%s, %s) (monitoring evaluates [%s, %s))...",
        data_load_start, end_ts, start_ts, end_ts,
    )
    joined_df = load_predictions_with_actuals(
        mon["inputs"]["predictions_dir"], mon["inputs"]["sqlite_path"], data_load_start, end_ts
    )
    if joined_df.empty:
        raise RuntimeError("No joined prediction+actual data found for the backtest window.")
    logger.info("Loaded %d joined rows (including lookback buffer).", len(joined_df))

    baseline_window_hours = mon["windows"]["correlation_baseline_window_hours"]
    baseline_end = start_ts + pd.Timedelta(hours=baseline_window_hours)
    seeding_df = joined_df[joined_df["timestamp"] <= baseline_end]
    correlation_baseline = seed_or_load_baseline(
        mon["prediction_correlation_baseline_path"], seeding_df,
        mon["drift_thresholds"]["psi_features"],
    )

    hours = pd.date_range(
        start_ts, end_ts, freq=f"{mon['cadence']['granularity_hours']}h", inclusive="left"
    )
    consecutive_bias_counts: dict[int, int] = {}
    last_promotion_as_of: pd.Timestamp | None = None
    last_attempt_as_of: pd.Timestamp | None = None
    n_trigger_fires = 0
    n_retrain_attempts = 0
    n_promotions = 0

    for i, as_of in enumerate(hours):
        alert, metrics_row, consecutive_bias_counts = run_monitoring_for_hour(
            as_of, reference, joined_df, correlation_baseline, params, consecutive_bias_counts
        )
        write_metrics_row(metrics_row, mon["outputs"]["metrics_timeseries_path"])
        append_alert(alert, mon["outputs"]["alerts_log_path"])

        if alert.retrain_trigger_fired:
            n_trigger_fires += 1

        should_attempt, gate_reason = should_attempt_retrain(
            alert.retrain_trigger_fired, as_of, last_promotion_as_of, last_attempt_as_of,
            promotion_cooldown_days=retrain_cfg["cooldown_days"],
            failed_attempt_cooldown_days=retrain_cfg["failed_attempt_cooldown_days"],
        )

        if should_attempt:
            n_retrain_attempts += 1
            last_attempt_as_of = as_of
            logger.warning("Attempting retrain at %s (%s)", as_of, gate_reason)
            outcome, updated_joined_df = run_retrain_cycle(
                as_of, joined_df, feature_names, params
            )
            append_retrain_event(outcome, as_of, retrain_cfg["outputs"]["retrain_events_path"])

            if outcome.promoted:
                n_promotions += 1
                last_promotion_as_of = as_of

                # Hot-swap every piece of state monitoring depends on, so
                # the rest of the loop evaluates against the NEW baseline.
                joined_df = updated_joined_df
                reference = ReferenceSnapshot.from_json(mon["reference_snapshot_path"])
                consecutive_bias_counts = {}

                reseed_end = as_of + pd.Timedelta(hours=baseline_window_hours)
                reseed_df = joined_df[
                    (joined_df["timestamp"] > as_of) & (joined_df["timestamp"] <= reseed_end)
                ]
                correlation_baseline = seed_or_load_baseline(
                    mon["prediction_correlation_baseline_path"], reseed_df,
                    mon["drift_thresholds"]["psi_features"],
                )
                logger.warning(
                    "State hot-swapped after promotion at %s: reference reloaded, "
                    "correlation baseline reseeded, bias counters reset.", as_of,
                )

        if (i + 1) % 168 == 0:
            logger.info(
                "Progress: %d/%d hours | triggers=%d | retrain attempts=%d | promotions=%d",
                i + 1, len(hours), n_trigger_fires, n_retrain_attempts, n_promotions,
            )

    logger.info(
        "Backtest+retraining complete: %d hours, %d trigger fires, "
        "%d retrain attempts, %d promotions.",
        len(hours), n_trigger_fires, n_retrain_attempts, n_promotions,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3+4 integrated backtest.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--start", default="2024-07-01T00:00:00")
    parser.add_argument("--end", default="2025-01-01T00:00:00")
    args = parser.parse_args()

    run_backtest_with_retraining(load_params(args.params), args.start, args.end)