"""Core monitoring evaluation: a single as-of-timestamp function.

This is the one function that both the backtest runner (looping `as_of`
over Jul-Dec 2024 history) and the scheduled production entrypoint
(`as_of=now`) call. Keeping it a pure function of (as_of, data, config) ->
results is what avoids building two parallel implementations of the same
drift math -- the design decision locked in before this file was written.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.monitoring.alerting import AlertSummary, evaluate_alerts
from src.monitoring.correlation_stability import compute_correlation_stability
from src.monitoring.drift_metrics import (
    compute_feature_psi,
    compute_label_kl_divergence,
    compute_null_rate_drift,
)
from src.monitoring.performance_metrics import (
    compute_calibration_error,
    compute_prediction_variance,
    compute_rolling_error,
    compute_zone_bias,
)
from src.monitoring.reference_loader import ReferenceSnapshot

logger = logging.getLogger(__name__)


def run_monitoring_for_hour(
    as_of: pd.Timestamp,
    reference: ReferenceSnapshot,
    joined_df: pd.DataFrame,
    correlation_baseline: dict[str, float],
    params: dict,
    prior_consecutive_bias_counts: dict[int, int],
) -> tuple[AlertSummary, dict, dict[int, int]]:
    """Evaluate every Phase 3 metric as of a single hour.

    `joined_df` must already cover enough trailing history ending at `as_of`
    to satisfy the widest window this function needs (the 30-day RMSE
    window plus the variance historical lookback) -- callers own loading a
    sufficiently wide slice; this function does not re-query storage.

    Returns (alert_summary, flattened_metrics_row_for_parquet,
    updated_consecutive_bias_counts_for_next_call).
    """
    mon = params["monitoring"]
    drift_window_hours = mon["windows"]["drift_window_hours"]
    drift_window_start = as_of - pd.Timedelta(hours=drift_window_hours)
    drift_window_df = joined_df[
        (joined_df["timestamp"] > drift_window_start) & (joined_df["timestamp"] <= as_of)
    ]

    metrics_row: dict = {"as_of": as_of.isoformat()}

    # --- Drift metrics (all against the frozen Phase 1 reference snapshot) ---
    psi_results, null_results, correlation_results = [], [], []
    label_kl = float("nan")

    if not drift_window_df.empty:
        psi_results = compute_feature_psi(
            drift_window_df, reference,
            mon["drift_thresholds"]["psi_features"],
            mon["drift_thresholds"]["psi_alert_threshold"],
            threshold_overrides=mon["drift_thresholds"].get("psi_alert_threshold_overrides", {}),
        )
        try:
            label_kl = compute_label_kl_divergence(drift_window_df["trip_count"], reference)
        except ValueError as exc:
            logger.warning("Skipping label KL at %s: %s", as_of, exc)

        null_results = compute_null_rate_drift(
            drift_window_df, reference,
            mon["drift_thresholds"]["psi_features"],
            mon["drift_thresholds"]["null_rate_multiplier_threshold"],
        )
        correlation_results = compute_correlation_stability(
            drift_window_df, correlation_baseline,
            mon["drift_thresholds"]["psi_features"],
            mon["drift_thresholds"]["spearman_drop_pct_threshold"],
        )
    else:
        logger.warning(
            "No data in drift window ending %s; drift metrics skipped this hour.", as_of
        )

    for r in psi_results:
        metrics_row[f"psi__{r.feature}"] = r.psi
    metrics_row["label_kl"] = label_kl
    for r in null_results:
        metrics_row[f"null_rate_ratio__{r.feature}"] = r.ratio
    for r in correlation_results:
        metrics_row[f"corr_drop_pct__{r.feature}"] = r.relative_drop_pct

    # --- Performance metrics (plain aggregation over predictions + actuals) ---
    for window_hours in mon["windows"]["rmse_windows_hours"]:
        err = compute_rolling_error(joined_df, as_of, window_hours)
        metrics_row[f"rmse_{window_hours}h"] = err.rmse
        metrics_row[f"mae_{window_hours}h"] = err.mae
        metrics_row[f"n_preds_{window_hours}h"] = err.n_predictions

    zone_bias_results, updated_counts = compute_zone_bias(
        joined_df, as_of,
        mon["performance_thresholds"]["zone_bias_pct_threshold"],
        mon["performance_thresholds"]["zone_bias_consecutive_hours"],
        prior_consecutive_bias_counts,
    )
    metrics_row["zones_with_bias_alert"] = sum(r.alert_fired for r in zone_bias_results)

    calibration_results = compute_calibration_error(
        joined_df, as_of,
        window_hours=mon["windows"]["rmse_windows_hours"][0],  # 24h
        calibration_threshold=mon["performance_thresholds"]["calibration_error_threshold"],
    )
    metrics_row["zones_with_calibration_alert"] = sum(r.breached for r in calibration_results)

    variance_result = compute_prediction_variance(
        joined_df, as_of,
        current_window_hours=24,
        historical_lookback_hours=mon["windows"]["rmse_windows_hours"][-1],  # 30d
        variance_multiplier=mon["performance_thresholds"]["prediction_variance_multiplier"],
    )
    metrics_row["prediction_variance_current"] = variance_result.current_variance
    metrics_row["prediction_variance_breached"] = variance_result.breached

    # --- Alert rollup / two-stage retrain trigger ---
    alert = evaluate_alerts(
        as_of_iso=as_of.isoformat(),
        psi_results=psi_results,
        label_kl=label_kl if label_kl == label_kl else 0.0,  # NaN-safe: missing KL doesn't breach
        label_kl_threshold=mon["drift_thresholds"]["label_kl_alert_threshold"],
        null_rate_results=null_results,
        correlation_results=correlation_results,
        zone_bias_results=zone_bias_results,
        calibration_results=calibration_results,
        variance_result=variance_result,
        require_drift_breach=mon["retrain_trigger"]["require_drift_breach"],
        require_performance_degradation=mon["retrain_trigger"]["require_performance_degradation"],
        psi_trigger_exempt=mon["drift_thresholds"].get("psi_trigger_exempt", []),
    )
    metrics_row["drift_breached"] = alert.drift_breached
    metrics_row["performance_degraded"] = alert.performance_degraded
    metrics_row["retrain_trigger_fired"] = alert.retrain_trigger_fired

    return alert, metrics_row, updated_counts