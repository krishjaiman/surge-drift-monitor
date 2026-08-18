"""Unit tests for the core drift math. These use synthetic data (not the real
reference snapshot or prediction log) so they run standalone and catch
regressions in the math itself, independent of pipeline data availability.

Add to your existing test suite alongside Phase 1/2 tests; run with pytest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_metrics import (
    _bucket_categorical_series,
    _bucket_numeric_series,
    compute_null_rate_drift,
    compute_psi,
)
from src.monitoring.reference_loader import FeatureProfile, ReferenceSnapshot


def test_psi_zero_for_identical_distributions():
    ref = np.array([0.25, 0.25, 0.25, 0.25])
    cur = np.array([0.25, 0.25, 0.25, 0.25])
    assert compute_psi(ref, cur) == pytest.approx(0.0, abs=1e-9)


def test_psi_positive_for_shifted_distribution():
    ref = np.array([0.4, 0.3, 0.2, 0.1])
    cur = np.array([0.1, 0.2, 0.3, 0.4])  # fully inverted -> should be a large PSI
    psi = compute_psi(ref, cur)
    assert psi > 0.25  # should clear the alert threshold


def test_psi_handles_zero_bins_without_error():
    ref = np.array([0.5, 0.5, 0.0])
    cur = np.array([0.0, 0.5, 0.5])
    psi = compute_psi(ref, cur)
    assert np.isfinite(psi)
    assert psi > 0


def test_psi_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_psi(np.array([0.5, 0.5]), np.array([0.3, 0.3, 0.4]))


def test_bucket_numeric_series_matches_reference_bin_edges():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    values = pd.Series([0.5, 0.5, 1.5, 2.5, 2.9])
    probs = _bucket_numeric_series(values, edges)
    assert probs.sum() == pytest.approx(1.0)
    assert len(probs) == 3
    # two values in [0,1), one in [1,2), two in [2,3)
    assert probs[0] == pytest.approx(2 / 5)
    assert probs[2] == pytest.approx(2 / 5)


def test_bucket_numeric_series_clips_out_of_range_into_edge_bins():
    edges = np.array([0.0, 1.0, 2.0])
    values = pd.Series([-5.0, 0.5, 100.0])
    probs = _bucket_numeric_series(values, edges)
    # out-of-range values must land in an edge bin, not be dropped
    assert probs.sum() == pytest.approx(1.0)


def test_bucket_categorical_series_new_category_reduces_known_bin_share():
    order = ["A", "B"]
    values = pd.Series(["A", "A", "B", "C", "C"])  # "C" is unseen in reference
    probs = _bucket_categorical_series(values, order)
    # "C" isn't in `order` so it contributes zero mass to A/B bins,
    # correctly shrinking their apparent share vs a world without "C"
    assert probs.sum() < 1.0
    assert probs[0] == pytest.approx(2 / 5)
    assert probs[1] == pytest.approx(1 / 5)


def _make_numeric_profile(name: str, bin_edges, bin_counts, null_rate=0.0) -> FeatureProfile:
    return FeatureProfile(
        name=name, dtype="numeric", null_rate=null_rate,
        bin_edges=np.array(bin_edges), bin_counts=np.array(bin_counts),
    )


def _make_snapshot_with_features(features: dict[str, FeatureProfile]) -> ReferenceSnapshot:
    target = _make_numeric_profile("trip_count", [0, 10, 20], [50, 50])
    return ReferenceSnapshot(
        path="dummy.json", n_rows=1000, training_months=["2023-01"], n_zones=10,
        features=features, target=target,
    )


def test_null_rate_drift_flags_increase_above_multiplier():
    profile = _make_numeric_profile("lag_1h", [0, 5, 10], [50, 50], null_rate=0.01)
    snapshot = _make_snapshot_with_features({"lag_1h": profile})

    current = pd.DataFrame({"lag_1h": [1.0, None, None, None, 5.0]})  # 60% null vs 1% ref
    results = compute_null_rate_drift(current, snapshot, ["lag_1h"], multiplier_threshold=2.0)

    assert len(results) == 1
    assert bool(results[0].breached) is True
    assert results[0].ratio > 2.0


def test_null_rate_drift_does_not_flag_stable_rate():
    profile = _make_numeric_profile("lag_1h", [0, 5, 10], [50, 50], null_rate=0.5)
    snapshot = _make_snapshot_with_features({"lag_1h": profile})

    current = pd.DataFrame({"lag_1h": [1.0, None, 3.0, None, 5.0]})  # 40% null vs 50% ref
    results = compute_null_rate_drift(current, snapshot, ["lag_1h"], multiplier_threshold=2.0)

    assert bool(results[0].breached) is False


def test_psi_threshold_override_applies_per_feature():
    from src.monitoring.drift_metrics import compute_feature_psi

    profile = _make_numeric_profile("temperature_2m", [0, 1, 2, 3], [25, 25, 25])
    snapshot = _make_snapshot_with_features({"temperature_2m": profile})

    # A window concentrated in one bin vs. a uniform reference -> real PSI here
    # should land somewhere clearly between the two thresholds we're testing.
    current = pd.DataFrame({"temperature_2m": [0.1] * 10})

    default_results = compute_feature_psi(
        current, snapshot, ["temperature_2m"], psi_threshold=0.25
    )
    assert default_results[0].breached is True  # breaches the tight default threshold

    override_results = compute_feature_psi(
        current, snapshot, ["temperature_2m"], psi_threshold=0.25,
        threshold_overrides={"temperature_2m": 100.0},  # deliberately unreachable
    )
    assert override_results[0].breached is False  # same PSI value, but now under its own threshold
    assert override_results[0].psi == default_results[0].psi  # override changes the verdict, not the math


def test_psi_trigger_exempt_excludes_feature_from_drift_breached():
    from src.monitoring.alerting import evaluate_alerts
    from src.monitoring.drift_metrics import FeaturePSIResult

    breaching_psi = [FeaturePSIResult(feature="temperature_2m", psi=6.0, breached=True)]

    # Without exemption: this alone should breach drift.
    alert_no_exempt = evaluate_alerts(
        as_of_iso="2024-07-01T00:00:00", psi_results=breaching_psi,
        label_kl=0.0, label_kl_threshold=0.15, null_rate_results=[],
        correlation_results=[], zone_bias_results=[], calibration_results=[],
        variance_result=_no_variance_breach(), require_drift_breach=True,
        require_performance_degradation=False, psi_trigger_exempt=[],
    )
    assert alert_no_exempt.drift_breached is True

    # With exemption: same breaching PSI value, but excluded from the decision.
    alert_exempt = evaluate_alerts(
        as_of_iso="2024-07-01T00:00:00", psi_results=breaching_psi,
        label_kl=0.0, label_kl_threshold=0.15, null_rate_results=[],
        correlation_results=[], zone_bias_results=[], calibration_results=[],
        variance_result=_no_variance_breach(), require_drift_breach=True,
        require_performance_degradation=False, psi_trigger_exempt=["temperature_2m"],
    )
    assert alert_exempt.drift_breached is False
    # But it's still visible for Grafana/trend analysis, just tagged as exempt.
    assert alert_exempt.breach_details["psi_breaches_exempt_from_trigger"] == ["temperature_2m"]
    assert alert_exempt.breach_details["psi_breaches"] == []


def _no_variance_breach():
    from src.monitoring.performance_metrics import VarianceResult
    return VarianceResult(breached=False, current_variance=1.0,
                           historical_lower_bound=0.5, historical_upper_bound=2.0)