"""Unit tests for src/training/reference_snapshot.py"""
import numpy as np
import pandas as pd
import pytest

from src.training.reference_snapshot import (
    _categorical_feature_profile,
    _numeric_feature_profile,
    build_reference_snapshot,
)


@pytest.fixture
def params():
    return {"reference_snapshot": {"percentiles": [5, 50, 95], "histogram_bins": 10}}


class TestNumericFeatureProfile:
    def test_captures_basic_stats(self, params):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        profile = _numeric_feature_profile(series, params)
        assert profile["mean"] == 3.0
        assert profile["min"] == 1.0
        assert profile["max"] == 5.0
        assert profile["null_rate"] == 0.0

    def test_handles_nulls_without_crashing(self, params):
        series = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        profile = _numeric_feature_profile(series, params)
        assert profile["null_count"] == 2
        assert profile["null_rate"] == 0.4
        assert profile["mean"] == 3.0  # computed on non-null values only

    def test_histogram_bin_count_matches_config(self, params):
        series = pd.Series(np.random.uniform(0, 100, 500))
        profile = _numeric_feature_profile(series, params)
        assert len(profile["histogram"]["counts"]) == 10


class TestCategoricalFeatureProfile:
    def test_frequencies_sum_to_one(self):
        series = pd.Series(["a", "a", "b", "c", "a"])
        profile = _categorical_feature_profile(series)
        assert abs(sum(profile["category_frequencies"].values()) - 1.0) < 1e-9

    def test_n_unique_correct(self):
        series = pd.Series(["a", "a", "b", "c"])
        profile = _categorical_feature_profile(series)
        assert profile["n_unique"] == 3


class TestBuildReferenceSnapshot:
    def test_includes_target_and_all_features(self, params):
        params["features"] = {"target_col": "trip_count"}
        params["data"] = {"training_months": ["2022-01"]}
        df = pd.DataFrame({
            "zone_id": ["1", "2", "1", "2"],
            "lag_1h": [1.0, 2.0, 3.0, 4.0],
            "trip_count": [5, 6, 7, 8],
            "hour_ts": pd.date_range("2022-01-01", periods=4, freq="h"),
        })
        snapshot = build_reference_snapshot(
            df, feature_cols=["zone_id", "lag_1h"], categorical_cols=["zone_id"], params=params
        )
        assert "zone_id" in snapshot["features"]
        assert "lag_1h" in snapshot["features"]
        assert "trip_count" in snapshot["features"]  # target always included
        assert snapshot["features"]["zone_id"]["dtype"] == "categorical"
        assert snapshot["features"]["lag_1h"]["dtype"] == "numeric"
        assert snapshot["metadata"]["n_rows"] == 4
        assert snapshot["metadata"]["n_zones"] == 2
