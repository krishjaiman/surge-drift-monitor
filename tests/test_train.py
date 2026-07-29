"""Unit tests for src/training/train.py"""
import pandas as pd
import pytest

from src.training.train import get_feature_columns, time_based_split


@pytest.fixture
def params():
    return {
        "features": {
            "time_features": ["hour_of_day", "day_of_week", "month", "is_weekend"],
            "lag_features": {"lag_hours": [1, 24, 168]},
            "rolling_features": {"windows_hours": [3, 24]},
            "categorical_features": ["zone_id"],
        },
        "data": {"weather": {"hourly_vars": ["temperature_2m", "precipitation", "windspeed_10m"]}},
        "split": {
            "train_end_date": "2022-01-20",
            "val_start_date": "2022-01-21",
            "val_end_date": "2022-01-31",
            "val_fraction_fallback": 0.2,
        },
    }


class TestGetFeatureColumns:
    def test_includes_all_feature_groups(self, params):
        cols = get_feature_columns(params)
        assert "zone_id" in cols
        assert "hour_of_day" in cols
        assert "lag_1h" in cols
        assert "rolling_mean_3h" in cols
        assert "temperature_2m" in cols

    def test_no_duplicate_columns(self, params):
        cols = get_feature_columns(params)
        assert len(cols) == len(set(cols))


class TestTimeBasedSplit:
    def test_no_temporal_overlap(self, params):
        """Train and val sets must not share any timestamps — a random
        split would leak adjacent-hour autocorrelation."""
        df = pd.DataFrame({
            "hour_ts": pd.date_range("2022-01-01", "2022-01-31", freq="h"),
        })
        df["trip_count"] = range(len(df))
        train_df, val_df = time_based_split(df, params)

        assert train_df["hour_ts"].max() <= pd.Timestamp(params["split"]["train_end_date"])
        assert val_df["hour_ts"].min() >= pd.Timestamp(params["split"]["val_start_date"])
        assert len(set(train_df["hour_ts"]) & set(val_df["hour_ts"])) == 0

    def test_fallback_split_when_val_too_small(self, params):
        """If the configured val window is nearly empty, falls back to a
        random-fraction split rather than training on almost no validation
        data silently."""
        params["split"]["val_start_date"] = "2022-01-30"
        params["split"]["val_end_date"] = "2022-01-30"  # ~1 day of val data
        df = pd.DataFrame({
            "hour_ts": pd.date_range("2022-01-01", "2022-01-31", freq="h"),
        })
        df["trip_count"] = range(len(df))
        train_df, val_df = time_based_split(df, params)

        expected_val_size = int(len(df) * params["split"]["val_fraction_fallback"])
        assert abs(len(val_df) - expected_val_size) <= 1
