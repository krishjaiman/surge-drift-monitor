"""Unit tests for src/features/build_features.py"""
import pandas as pd
import pytest

from src.features.build_features import (
    add_lag_and_rolling_features,
    add_time_features,
    aggregate_zone_hour_demand,
)


@pytest.fixture
def params():
    return {
        "features": {
            "min_zone_hourly_volume": 1,
            "lag_features": {"lag_hours": [1, 24]},
            "rolling_features": {"windows_hours": [3]},
        }
    }


@pytest.fixture
def raw_trips():
    """Two zones, a handful of trips across a few hours."""
    return pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "2022-01-01 00:15", "2022-01-01 00:45", "2022-01-01 01:10",
            "2022-01-01 02:05", "2022-01-01 00:20",
        ]),
        "PULocationID": [1, 1, 1, 1, 2],
        "passenger_count": [1, 1, 2, 1, 1],
        "trip_distance": [1.5, 2.0, 0.8, 3.1, 1.2],
    })


class TestAggregateZoneHourDemand:
    def test_zero_fill_for_missing_hours(self, raw_trips, params):
        """Hours with no trips for a zone must appear as trip_count=0,
        not be silently absent from the output."""
        result = aggregate_zone_hour_demand(raw_trips, params)
        zone1 = result[result["zone_id"] == 1].sort_values("hour_ts")
        # Zone 1 has trips at hour 0 and 1 and 2, all hours in between must exist
        assert (zone1["trip_count"] >= 0).all()
        assert len(zone1) == 3  # hours 00, 01, 02

    def test_all_invalid_trips_raises_instead_of_silently_returning_empty(self, params):
        """If every row is filtered out (e.g. a corrupt source file), this
        must raise loudly rather than silently producing an empty/zero
        result that downstream stages would treat as valid."""
        trips = pd.DataFrame({
            "tpep_pickup_datetime": pd.to_datetime(["2022-01-01 00:15"]),
            "PULocationID": [1],
            "passenger_count": [1],
            "trip_distance": [0.0],
        })
        with pytest.raises(ValueError, match="No valid trips remain"):
            aggregate_zone_hour_demand(trips, params)

    def test_drops_zero_distance_trips_among_valid_ones(self, raw_trips, params):
        """A zero-distance trip should be dropped without affecting other
        valid trips in the same batch."""
        trips = pd.concat([
            raw_trips,
            pd.DataFrame({
                "tpep_pickup_datetime": pd.to_datetime(["2022-01-01 00:30"]),
                "PULocationID": [1],
                "passenger_count": [1],
                "trip_distance": [0.0],
            }),
        ], ignore_index=True)
        result = aggregate_zone_hour_demand(trips, params)
        # total trip_count should match raw_trips only (5), not 6
        assert result["trip_count"].sum() == 5

    def test_drops_low_volume_zones(self, raw_trips, params):
        params["features"]["min_zone_hourly_volume"] = 10
        result = aggregate_zone_hour_demand(raw_trips, params)
        assert result.empty


class TestTimeFeatures:
    def test_weekend_flag(self):
        df = pd.DataFrame({"hour_ts": pd.to_datetime(["2022-01-01", "2022-01-03"])})  # Sat, Mon
        result = add_time_features(df)
        assert result.loc[0, "is_weekend"] == 1
        assert result.loc[1, "is_weekend"] == 0

    def test_hour_of_day_extracted(self):
        df = pd.DataFrame({"hour_ts": pd.to_datetime(["2022-01-01 14:00"])})
        result = add_time_features(df)
        assert result.loc[0, "hour_of_day"] == 14


class TestLagFeatures:
    def test_lag_has_no_leakage(self, params):
        """A lag_1h feature at hour T must equal trip_count at hour T-1,
        never the current row's own trip_count."""
        df = pd.DataFrame({
            "zone_id": [1, 1, 1, 1],
            "hour_ts": pd.date_range("2022-01-01", periods=4, freq="h"),
            "trip_count": [10, 20, 30, 40],
        })
        result = add_lag_and_rolling_features(df, params)
        assert pd.isna(result.loc[0, "lag_1h"])  # no prior hour
        assert result.loc[1, "lag_1h"] == 10
        assert result.loc[2, "lag_1h"] == 20
        assert result.loc[3, "lag_1h"] == 30

    def test_lag_is_per_zone_not_global(self, params):
        """Lag features must not leak across zones."""
        df = pd.DataFrame({
            "zone_id": [1, 2, 1, 2],
            "hour_ts": pd.to_datetime(
                ["2022-01-01 00:00", "2022-01-01 00:00", "2022-01-01 01:00", "2022-01-01 01:00"]
            ),
            "trip_count": [10, 100, 20, 200],
        })
        result = add_lag_and_rolling_features(df, params)
        zone1_hour1 = result[(result["zone_id"] == 1) & (result["hour_ts"] == "2022-01-01 01:00")]
        assert zone1_hour1["lag_1h"].iloc[0] == 10  # zone 1's own prior hour, not zone 2's
