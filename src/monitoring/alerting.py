"""Aggregates individual metric breaches into alerts, and evaluates the
two-stage retraining trigger (drift breach AND performance degradation,
confirmed decision from earlier phases -- drift alone must not fire it).

Phase 3 only decides and logs whether the trigger condition is met. It does
not retrain anything -- Phase 4 is the consumer of `retrain_trigger_fired`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.monitoring.correlation_stability import CorrelationStabilityResult
from src.monitoring.drift_metrics import FeaturePSIResult, NullRateResult
from src.monitoring.performance_metrics import (
    CalibrationResult,
    VarianceResult,
    ZoneBiasResult,
)


@dataclass(frozen=True)
class AlertSummary:
    as_of: str  # ISO timestamp, kept as str for clean JSON serialization
    drift_breached: bool
    performance_degraded: bool
    retrain_trigger_fired: bool
    breach_details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "drift_breached": self.drift_breached,
            "performance_degraded": self.performance_degraded,
            "retrain_trigger_fired": self.retrain_trigger_fired,
            "breach_details": self.breach_details,
        }


def evaluate_alerts(
    as_of_iso: str,
    psi_results: list[FeaturePSIResult],
    label_kl: float,
    label_kl_threshold: float,
    null_rate_results: list[NullRateResult],
    correlation_results: list[CorrelationStabilityResult],
    zone_bias_results: list[ZoneBiasResult],
    calibration_results: list[CalibrationResult],
    variance_result: VarianceResult,
    require_drift_breach: bool,
    require_performance_degradation: bool,
    psi_trigger_exempt: list[str] | None = None,
) -> AlertSummary:
    """Roll up every sub-metric's breach flag into a single alert record and
    decide whether the two-stage retrain trigger fires.

    `psi_trigger_exempt` lists features whose PSI is still computed and
    logged (visible in metrics_timeseries.parquet for Grafana / trend
    analysis) but excluded from deciding `drift_breached`. This exists for
    temperature_2m / windspeed_10m specifically: three rounds of backtest
    diagnostics (24h->168h->720h window, 0.25->1.5 threshold) confirmed
    their PSI stays persistently elevated regardless of window size or
    threshold, with no time trend -- a known limitation of PSI on
    temporally-autocorrelated continuous features compared against a
    full-period reference, not real drift. Rather than keep guessing at a
    5th arbitrary threshold, these features are excluded from triggering
    until a seasonally-stratified reference (documented future work) makes
    PSI meaningful for them again.
    """
    psi_trigger_exempt = set(psi_trigger_exempt or [])
    psi_breaches_all = [r.feature for r in psi_results if r.breached]
    psi_breaches = [f for f in psi_breaches_all if f not in psi_trigger_exempt]
    null_breaches = [r.feature for r in null_rate_results if r.breached]
    correlation_breaches = [r.feature for r in correlation_results if r.breached]
    label_kl_breached = label_kl > label_kl_threshold

    drift_breached = bool(
        psi_breaches or null_breaches or correlation_breaches or label_kl_breached
    )

    bias_breaches = [r.zone_id for r in zone_bias_results if r.alert_fired]
    calibration_breaches = [r.zone_id for r in calibration_results if r.breached]
    variance_breached = variance_result.breached

    performance_degraded = bool(
        bias_breaches or calibration_breaches or variance_breached
    )

    if require_drift_breach and require_performance_degradation:
        retrain_trigger_fired = drift_breached and performance_degraded
    elif require_drift_breach:
        retrain_trigger_fired = drift_breached
    elif require_performance_degradation:
        retrain_trigger_fired = performance_degraded
    else:
        retrain_trigger_fired = False

    breach_details = {
        "psi_breaches": psi_breaches,               # feeds drift_breached
        "psi_breaches_exempt_from_trigger": [f for f in psi_breaches_all if f in psi_trigger_exempt],
        "null_rate_breaches": null_breaches,
        "correlation_breaches": correlation_breaches,
        "label_kl_breached": label_kl_breached,
        "label_kl_value": label_kl,
        "zone_bias_breaches": bias_breaches,
        "zone_calibration_breaches": calibration_breaches,
        "variance_breached": variance_breached,
    }

    return AlertSummary(
        as_of=as_of_iso,
        drift_breached=drift_breached,
        performance_degraded=performance_degraded,
        retrain_trigger_fired=retrain_trigger_fired,
        breach_details=breach_details,
    )