"""End-to-end smoke test on fully synthetic data: exercises reference
loading, drift metrics, performance metrics, correlation stability, and
alert rollup together through run_monitoring_for_hour. This is what catches
integration bugs (column-name mismatches across modules, window slicing
errors) that isolated unit tests can't see. Not a substitute for running the
backtest against real project data -- run that too before trusting output.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.monitoring.monitoring_job import run_monitoring_for_hour
from src.monitoring.reference_loader import ReferenceSnapshot

FEATURES = ["lag_1h", "lag_24h", "temperature_2m", "hour_of_day"]


def _build_synthetic_reference_json(path: Path) -> None:
    # Matches the REAL schema from src/training/reference_snapshot.py:
    # target lives inside "features" keyed by target_col ("trip_count"),
    # not as a separate top-level key; histogram counts key is "counts".
    def numeric_profile(mean, std, edges, counts):
        return {
            "dtype": "numeric", "mean": mean, "std": std,
            "percentiles": {"50": mean}, "null_rate": 0.01,
            "histogram": {"bin_edges": edges, "counts": counts},
        }

    snapshot = {
        "metadata": {"n_rows": 10000, "training_months": ["2022-01"], "n_zones": 5},
        "features": {
            "lag_1h": numeric_profile(10, 5, [0, 5, 10, 15, 20], [100, 200, 400, 300]),
            "lag_24h": numeric_profile(10, 5, [0, 5, 10, 15, 20], [100, 200, 400, 300]),
            "temperature_2m": numeric_profile(15, 8, [-10, 0, 10, 20, 30], [50, 200, 500, 250]),
            "hour_of_day": numeric_profile(12, 6, [0, 6, 12, 18, 24], [250, 250, 250, 250]),
            "trip_count": numeric_profile(12, 6, [0, 5, 10, 15, 20, 25], [100, 200, 300, 250, 150]),
        },
    }
    path.write_text(json.dumps(snapshot))


def _build_synthetic_joined_df(n_hours: int, n_zones: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-07-01")
    rows = []
    for h in range(n_hours):
        ts = start + pd.Timedelta(hours=h)
        for zone in range(n_zones):
            actual = max(0.0, rng.normal(10, 3))
            predicted = max(0.0, actual + rng.normal(0, 1.5))
            rows.append({
                "timestamp": ts, "zone_id": zone, "predicted_demand": predicted,
                "trip_count": actual, "model_version": "v1",
                "lag_1h": max(0.0, rng.normal(10, 5)),
                "lag_24h": max(0.0, rng.normal(10, 5)),
                "temperature_2m": rng.normal(15, 8),
                "hour_of_day": ts.hour,
            })
    return pd.DataFrame(rows)


def test_run_monitoring_for_hour_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ref_path = tmp_path / "reference_snapshot_current.json"
        _build_synthetic_reference_json(ref_path)
        reference = ReferenceSnapshot.from_json(ref_path)

        joined_df = _build_synthetic_joined_df(n_hours=48, n_zones=5)
        as_of = joined_df["timestamp"].max()

        params = {
            "monitoring": {
                "windows": {
                    "drift_window_hours": 24,
                    "rmse_windows_hours": [24],
                    "correlation_baseline_window_hours": 24,
                },
                "drift_thresholds": {
                    "psi_features": FEATURES,
                    "psi_alert_threshold": 0.25,
                    "psi_alert_threshold_overrides": {},
                    "psi_trigger_exempt": [],
                    "label_kl_alert_threshold": 0.15,
                    "spearman_drop_pct_threshold": 0.30,
                    "null_rate_multiplier_threshold": 2.0,
                },
                "performance_thresholds": {
                    "zone_bias_pct_threshold": 0.25,
                    "zone_bias_consecutive_hours": 3,
                    "calibration_error_threshold": 0.10,
                    "prediction_variance_multiplier": 2.0,
                },
                "retrain_trigger": {
                    "require_drift_breach": True,
                    "require_performance_degradation": True,
                },
            }
        }

        correlation_baseline = {f: 0.5 for f in FEATURES}  # fixed baseline for determinism

        alert, metrics_row, updated_counts = run_monitoring_for_hour(
            as_of, reference, joined_df, correlation_baseline, params, {}
        )

        # Structural assertions -- the point of this test is that the pipeline
        # runs end-to-end and produces well-formed output, not specific values.
        assert metrics_row["as_of"] == as_of.isoformat()
        assert "rmse_24h" in metrics_row
        assert np.isfinite(metrics_row["rmse_24h"])
        assert any(k.startswith("psi__") for k in metrics_row)
        assert isinstance(alert.drift_breached, bool)
        assert isinstance(alert.performance_degraded, bool)
        assert isinstance(alert.retrain_trigger_fired, bool)
        assert isinstance(updated_counts, dict)

        # Must be JSON-serializable -- this is what append_alert() does for real.
        json.dumps(alert.to_dict())


if __name__ == "__main__":
    test_run_monitoring_for_hour_end_to_end()
    print("Smoke test passed.")