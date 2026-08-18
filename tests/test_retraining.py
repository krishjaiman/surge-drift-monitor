"""Unit tests for Phase 4 retraining logic. Deliberately avoid anything
needing MLflow or real data files -- those are exercised by actually
running the backtest, not by these tests. These cover the pure logic:
cooldown gating, promotion arithmetic, split construction, and repredict
row-selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.retraining.promotion import decide_promotion
from src.retraining.repredict import repredict_future_hours
from src.retraining.retrain_data import build_retrain_split
from src.retraining.retrain_orchestrator import should_attempt_retrain


# --- should_attempt_retrain ---

def test_should_attempt_retrain_false_when_trigger_not_fired():
    should, reason = should_attempt_retrain(
        retrain_trigger_fired=False, as_of=pd.Timestamp("2024-11-01"),
        last_promotion_as_of=None, last_attempt_as_of=None,
        promotion_cooldown_days=7, failed_attempt_cooldown_days=1,
    )
    assert should is False
    assert reason == "trigger_not_fired"


def test_should_attempt_retrain_true_on_first_ever_trigger():
    should, reason = should_attempt_retrain(
        retrain_trigger_fired=True, as_of=pd.Timestamp("2024-11-01"),
        last_promotion_as_of=None, last_attempt_as_of=None,
        promotion_cooldown_days=7, failed_attempt_cooldown_days=1,
    )
    assert should is True


def test_should_attempt_retrain_blocked_during_promotion_cooldown():
    should, reason = should_attempt_retrain(
        retrain_trigger_fired=True, as_of=pd.Timestamp("2024-11-03"),
        last_promotion_as_of=pd.Timestamp("2024-11-01"), last_attempt_as_of=None,
        promotion_cooldown_days=7, failed_attempt_cooldown_days=1,
    )
    assert should is False
    assert "promotion_cooldown_active" in reason


def test_should_attempt_retrain_true_after_promotion_cooldown_elapses():
    should, reason = should_attempt_retrain(
        retrain_trigger_fired=True, as_of=pd.Timestamp("2024-11-10"),
        last_promotion_as_of=pd.Timestamp("2024-11-01"), last_attempt_as_of=None,
        promotion_cooldown_days=7, failed_attempt_cooldown_days=1,
    )
    assert should is True


def test_should_attempt_retrain_blocked_during_failed_attempt_cooldown():
    # This is the exact real-world case that motivated adding this cooldown:
    # a failed attempt an hour ago must not immediately retry.
    should, reason = should_attempt_retrain(
        retrain_trigger_fired=True, as_of=pd.Timestamp("2024-11-01T01:00:00"),
        last_promotion_as_of=None, last_attempt_as_of=pd.Timestamp("2024-11-01T00:00:00"),
        promotion_cooldown_days=7, failed_attempt_cooldown_days=1,
    )
    assert should is False
    assert "failed_attempt_cooldown_active" in reason


def test_should_attempt_retrain_true_after_failed_attempt_cooldown_elapses():
    should, reason = should_attempt_retrain(
        retrain_trigger_fired=True, as_of=pd.Timestamp("2024-11-03"),
        last_promotion_as_of=None, last_attempt_as_of=pd.Timestamp("2024-11-01"),
        promotion_cooldown_days=7, failed_attempt_cooldown_days=1,
    )
    assert should is True


def test_should_attempt_retrain_promotion_cooldown_takes_priority_when_both_active():
    # Both cooldowns technically active -- promotion cooldown (the longer,
    # more important one) should be the reported reason.
    should, reason = should_attempt_retrain(
        retrain_trigger_fired=True, as_of=pd.Timestamp("2024-11-02"),
        last_promotion_as_of=pd.Timestamp("2024-11-01"),
        last_attempt_as_of=pd.Timestamp("2024-11-01T12:00:00"),
        promotion_cooldown_days=7, failed_attempt_cooldown_days=1,
    )
    assert should is False
    assert "promotion_cooldown_active" in reason


# --- decide_promotion ---

def test_decide_promotion_promotes_on_sufficient_improvement():
    should_promote, pct = decide_promotion(champion_rmse=10.0, challenger_rmse=9.0, min_improvement_pct=0.05)
    assert should_promote is True
    assert pct == pytest.approx(0.10)


def test_decide_promotion_rejects_marginal_improvement():
    should_promote, pct = decide_promotion(champion_rmse=10.0, challenger_rmse=9.7, min_improvement_pct=0.05)
    assert should_promote is False
    assert pct == pytest.approx(0.03)


def test_decide_promotion_rejects_when_challenger_is_worse():
    should_promote, pct = decide_promotion(champion_rmse=10.0, challenger_rmse=11.0, min_improvement_pct=0.05)
    assert should_promote is False
    assert pct < 0


# --- build_retrain_split ---

def _synthetic_joined_df(n_hours: int, n_zones: int, start: str = "2024-09-01") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    start_ts = pd.Timestamp(start)
    rows = []
    for h in range(n_hours):
        ts = start_ts + pd.Timedelta(hours=h)
        for zone in range(n_zones):
            rows.append({
                "timestamp": ts, "zone_id": zone,
                "trip_count": max(0.0, rng.normal(10, 3)),
                "predicted_demand": max(0.0, rng.normal(10, 3)),
                "lag_1h": rng.normal(10, 5), "hour_of_day": ts.hour,
            })
    return pd.DataFrame(rows)


def test_build_retrain_split_raises_on_empty_window():
    df = _synthetic_joined_df(n_hours=10, n_zones=5)
    as_of = pd.Timestamp("2020-01-01")  # far before any data
    with pytest.raises(ValueError, match="No data in retrain window"):
        build_retrain_split(df, as_of, ["lag_1h", "hour_of_day"], categorical_cols=[],
                             lookback_days=30, holdout_days=5)


def test_build_retrain_split_raises_on_missing_feature_column():
    df = _synthetic_joined_df(n_hours=24 * 40, n_zones=20)
    as_of = df["timestamp"].max()
    with pytest.raises(ValueError, match="missing expected feature columns"):
        build_retrain_split(df, as_of, ["lag_1h", "this_feature_does_not_exist"],
                             categorical_cols=[], lookback_days=30, holdout_days=5)


def test_build_retrain_split_raises_below_min_rows_floor():
    df = _synthetic_joined_df(n_hours=24 * 40, n_zones=2)  # too few rows even over 30 days
    as_of = df["timestamp"].max()
    with pytest.raises(ValueError, match="too small to trust"):
        build_retrain_split(df, as_of, ["lag_1h", "hour_of_day"], categorical_cols=[],
                             lookback_days=30, holdout_days=5)


def test_build_retrain_split_holdout_is_strictly_after_train():
    df = _synthetic_joined_df(n_hours=24 * 40, n_zones=50)
    as_of = df["timestamp"].max()
    split = build_retrain_split(df, as_of, ["lag_1h", "hour_of_day"], categorical_cols=[],
                                 lookback_days=30, holdout_days=5)

    assert split.train_end <= split.val_start
    assert split.val_end == as_of
    assert len(split.X_train) == len(split.y_train)
    assert len(split.X_val) == len(split.y_val)
    assert list(split.X_train.columns) == ["lag_1h", "hour_of_day"]
    # train_df_raw must carry hour_ts for the reference-snapshot builder
    assert "hour_ts" in split.train_df_raw.columns


def test_build_retrain_split_casts_categorical_columns():
    df = _synthetic_joined_df(n_hours=24 * 40, n_zones=50)
    as_of = df["timestamp"].max()
    split = build_retrain_split(df, as_of, ["lag_1h", "hour_of_day"], categorical_cols=["hour_of_day"],
                                 lookback_days=30, holdout_days=5)

    # The exact bug that crashed the real backtest: predicting against a
    # champion trained with category-dtype columns using plain int64 columns.
    # Confirms build_retrain_split casts the named columns, matching train.py.
    assert str(split.X_train["hour_of_day"].dtype) == "category"
    assert str(split.X_val["hour_of_day"].dtype) == "category"
    # Untouched columns must NOT be cast.
    assert str(split.X_train["lag_1h"].dtype) != "category"


# --- repredict_future_hours ---

def test_repredict_only_updates_rows_after_as_of():
    df = _synthetic_joined_df(n_hours=10, n_zones=3)
    as_of = df["timestamp"].unique()[4]  # 5th hour

    class FakeModel:
        def predict(self, X):
            return np.full(len(X), 999.0)

    updated = repredict_future_hours(df, pd.Timestamp(as_of), FakeModel(), ["lag_1h", "hour_of_day"])

    before_mask = updated["timestamp"] <= as_of
    after_mask = updated["timestamp"] > as_of

    # rows at/before as_of must be untouched
    assert (updated.loc[before_mask, "predicted_demand"] == df.loc[before_mask, "predicted_demand"]).all()
    # rows after as_of must all be the fake model's constant output
    assert (updated.loc[after_mask, "predicted_demand"] == 999.0).all()
    # original df must not be mutated (repredict returns a copy)
    assert not (df["predicted_demand"] == 999.0).any()


# --- Real end-to-end LightGBM categorical parity test ---
# This is the actual bug from the real backtest run, reproduced with a real
# Booster (not a fake), to make sure it can never silently regress: a champion
# trained with a category-dtype column must not crash when evaluated against
# a holdout built through build_retrain_split's cast.

def test_categorical_cast_prevents_champion_predict_crash():
    import lightgbm as lgb

    rng = np.random.default_rng(1)
    n = 3000
    zone_values = rng.integers(1, 20, n)
    df = pd.DataFrame({
        "zone_id_cat": pd.Series(zone_values).astype("category"),
        "lag_1h": rng.normal(10, 3, n),
        "y": rng.normal(10, 3, n) + zone_values * 0.1,
    })

    # Train a tiny real champion with zone_id_cat as category dtype,
    # exactly mirroring train.py's categorical_feature handling.
    train_set = lgb.Dataset(
        df[["zone_id_cat", "lag_1h"]], label=df["y"], categorical_feature=["zone_id_cat"]
    )
    champion = lgb.train({"objective": "regression", "verbosity": -1}, train_set, num_boost_round=5)

    # Build a holdout the WRONG way (plain int, not category) -- this is
    # what the real backtest did before the fix, and it must still crash,
    # proving the test actually exercises the failure mode.
    bad_holdout = pd.DataFrame({
        "zone_id_cat": zone_values[:100],  # plain int64, NOT category
        "lag_1h": rng.normal(10, 3, 100),
    })
    with pytest.raises(Exception):
        champion.predict(bad_holdout)

    # Build a holdout the RIGHT way -- cast to category, matching build_retrain_split.
    good_holdout = pd.DataFrame({
        "zone_id_cat": pd.Series(zone_values[:100]).astype("category"),
        "lag_1h": rng.normal(10, 3, 100),
    })
    preds = champion.predict(good_holdout)  # must not raise
    assert len(preds) == 100