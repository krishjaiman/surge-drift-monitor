"""Model performance metrics computed on predictions joined to actuals.

Unlike drift_metrics.py, none of this compares against the frozen reference
snapshot -- it's plain aggregation over (prediction, actual) pairs. Historical
comparison points (e.g. "2x historical variance") are computed from the
prediction log's own past, not from Phase 1 training data, since prediction
variance at serving time isn't a quantity the training-time snapshot captured.

Reading on why RMSE/MAE alone hide zone-level failure and per-segment bias
tracking matters for marketplace pricing models:
https://www.evidentlyai.com/ml-in-production/model-monitoring
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RollingErrorResult:
    window_hours: int
    rmse: float
    mae: float
    n_predictions: int


@dataclass(frozen=True)
class ZoneBiasResult:
    zone_id: int
    bias_pct: float
    breached_this_hour: bool
    consecutive_breach_hours: int
    alert_fired: bool


@dataclass(frozen=True)
class CalibrationResult:
    zone_id: int
    calibration_error: float
    breached: bool


@dataclass(frozen=True)
class VarianceResult:
    breached: bool
    current_variance: float
    historical_lower_bound: float
    historical_upper_bound: float


def compute_rolling_error(
    joined_df: pd.DataFrame,
    as_of: pd.Timestamp,
    window_hours: int,
) -> RollingErrorResult:
    """RMSE/MAE over predictions in (as_of - window_hours, as_of]."""
    window_start = as_of - pd.Timedelta(hours=window_hours)
    window_df = joined_df[
        (joined_df["timestamp"] > window_start) & (joined_df["timestamp"] <= as_of)
    ]
    if window_df.empty:
        return RollingErrorResult(window_hours=window_hours, rmse=float("nan"),
                                   mae=float("nan"), n_predictions=0)
    errors = window_df["predicted_demand"] - window_df["trip_count"]
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    return RollingErrorResult(
        window_hours=window_hours, rmse=rmse, mae=mae, n_predictions=len(window_df)
    )


def compute_zone_bias(
    joined_df: pd.DataFrame,
    as_of: pd.Timestamp,
    bias_pct_threshold: float,
    consecutive_hours_threshold: int,
    prior_consecutive_counts: dict[int, int],
) -> tuple[list[ZoneBiasResult], dict[int, int]]:
    """Per-zone bias for the single hour `as_of`, plus consecutive-breach
    tracking carried across calls via `prior_consecutive_counts` (an
    in/out dict: {zone_id: consecutive_hours_currently_breaching}).

    Bias here is mean signed percentage error: (pred - actual) / actual,
    averaged per zone for that hour. Positive = over-predicting demand.
    Zones with near-zero actuals are excluded from bias for that hour rather
    than producing an exploding percentage on a tiny denominator.
    """
    hour_df = joined_df[joined_df["timestamp"] == as_of].copy()
    results: list[ZoneBiasResult] = []
    new_counts: dict[int, int] = {}

    # Raised from 1.0 to 5.0: backtest diagnostics showed zones averaging
    # 1.5-6 trips/hour flagged in 50-68% of ALL hours individually -- pure
    # percentage-noise on tiny counts, not real per-hour miscalibration.
    MIN_ACTUAL_FOR_PCT_BIAS = 5.0

    for zone_id, group in hour_df.groupby("zone_id"):
        actual = group["trip_count"].iloc[0]
        predicted = group["predicted_demand"].iloc[0]
        if actual < MIN_ACTUAL_FOR_PCT_BIAS:
            continue
        bias_pct = float((predicted - actual) / actual)
        breached_this_hour = abs(bias_pct) > bias_pct_threshold

        prior = prior_consecutive_counts.get(int(zone_id), 0)
        consecutive = prior + 1 if breached_this_hour else 0
        new_counts[int(zone_id)] = consecutive

        results.append(
            ZoneBiasResult(
                zone_id=int(zone_id),
                bias_pct=bias_pct,
                breached_this_hour=breached_this_hour,
                consecutive_breach_hours=consecutive,
                alert_fired=consecutive >= consecutive_hours_threshold,
            )
        )
    return results, new_counts


def compute_calibration_error(
    joined_df: pd.DataFrame,
    as_of: pd.Timestamp,
    window_hours: int,
    calibration_threshold: float,
) -> list[CalibrationResult]:
    """Zone-level calibration error over a trailing window: normalized gap
    between mean predicted and mean actual demand. Distinct from hourly bias
    -- this smooths over a window to catch persistent directional skew that
    a single noisy hour wouldn't reveal, per-zone as required by the
    zone-level-metrics principle.
    """
    window_start = as_of - pd.Timedelta(hours=window_hours)
    window_df = joined_df[
        (joined_df["timestamp"] > window_start) & (joined_df["timestamp"] <= as_of)
    ]
    results: list[CalibrationResult] = []

    # Same floor as compute_zone_bias, for the same reason: a zone averaging
    # 1-2 trips/hour will blow past a 10% error threshold on pure noise, not
    # real miscalibration. Confirmed via backtest diagnostics -- without this
    # floor, over half of all 261 zones were alerting on most hours.
    # Same reasoning and same raised value as compute_zone_bias's floor --
    # confirmed via the same backtest diagnostic (zone 97, avg 2.5 trips/hr,
    # flagged in 2,905/4,248 hours -- ~68% of the entire backtest).
    MIN_ACTUAL_FOR_CALIBRATION = 5.0

    for zone_id, group in window_df.groupby("zone_id"):
        mean_actual = group["trip_count"].mean()
        mean_predicted = group["predicted_demand"].mean()
        if mean_actual < MIN_ACTUAL_FOR_CALIBRATION:
            continue
        calibration_error = float(abs(mean_predicted - mean_actual) / mean_actual)
        results.append(
            CalibrationResult(
                zone_id=int(zone_id),
                calibration_error=calibration_error,
                breached=calibration_error > calibration_threshold,
            )
        )
    return results


def compute_prediction_variance(
    joined_df: pd.DataFrame,
    as_of: pd.Timestamp,
    current_window_hours: int,
    historical_lookback_hours: int,
    variance_multiplier: float,
) -> VarianceResult:
    """Flags if current prediction variance falls outside
    [historical_mean_variance / multiplier, historical_mean_variance * multiplier].

    'Historical' here means the prediction log's own recent past (a rolling
    lookback), not the Phase 1 training period -- serving-time prediction
    variance has no equivalent quantity in the training-time snapshot.
    """
    current_start = as_of - pd.Timedelta(hours=current_window_hours)
    current_df = joined_df[
        (joined_df["timestamp"] > current_start) & (joined_df["timestamp"] <= as_of)
    ]
    hist_start = as_of - pd.Timedelta(hours=historical_lookback_hours)
    hist_df = joined_df[
        (joined_df["timestamp"] > hist_start) & (joined_df["timestamp"] <= current_start)
    ]

    if current_df.empty or hist_df.empty:
        return VarianceResult(
            breached=False, current_variance=float("nan"),
            historical_lower_bound=float("nan"), historical_upper_bound=float("nan"),
        )

    current_variance = float(current_df["predicted_demand"].var())
    historical_variance = float(hist_df["predicted_demand"].var())
    lower = historical_variance / variance_multiplier
    upper = historical_variance * variance_multiplier
    breached = not (lower <= current_variance <= upper)

    return VarianceResult(
        breached=breached,
        current_variance=current_variance,
        historical_lower_bound=lower,
        historical_upper_bound=upper,
    )