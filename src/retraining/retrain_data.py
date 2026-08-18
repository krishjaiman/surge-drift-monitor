"""Reconstruct a retraining dataset from data already on disk.

Design decision (confirmed): reuse the joined predictions+actuals dataframe
that Phase 3 already builds, rather than re-ingesting/re-engineering raw NYC
TLC data. This works cleanly because the `feat_*` columns logged by Phase 2
at prediction time ARE the exact model inputs -- no feature recomputation
needed, just a time-window slice.

This does mean the retrain dataset's feature values were computed by
Phase 2's serving-time feature logic, not Phase 1's offline batch pipeline.
If those two pipelines have ever silently diverged, this reuses whichever
one actually ran in production -- which is arguably more correct for a
retrain (train on what production actually saw), but flagging it as a
design tradeoff rather than a hidden assumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrainSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    train_df_raw: pd.DataFrame  # full training rows incl. hour_ts/zone_id/target --
                                 # needed by reference_snapshot.py's real builder,
                                 # which expects that shape, not just X/y
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def build_retrain_split(
    joined_df: pd.DataFrame,
    as_of: pd.Timestamp,
    feature_names: list[str],
    categorical_cols: list[str],
    target_col: str = "trip_count",
    lookback_days: int = 30,
    holdout_days: int = 5,
) -> RetrainSplit:
    """Slice the most recent `lookback_days` before `as_of` into a
    time-based train/holdout split. The holdout is the LAST `holdout_days`
    of that window -- consistent with Phase 1's time-based split philosophy
    (validate on the most recent slice, never a random split, to avoid
    leaking future information into the champion-challenger decision).

    `categorical_cols` MUST match `params["features"]["categorical_features"]`
    exactly -- confirmed against the real train.py, categorical columns are
    cast to pandas "category" dtype before being handed to LightGBM
    (`X_train[col] = X_train[col].astype("category")`). Skipping this cast,
    or casting different columns than training did, produces the exact
    "train and valid dataset categorical_feature do not match" crash seen
    when evaluating the champion on a differently-prepared X_val.

    Raises ValueError rather than silently training on too little data --
    a retrain decision based on a starved window is worse than no retrain.
    """
    window_start = as_of - pd.Timedelta(days=lookback_days)
    window_df = joined_df[
        (joined_df["timestamp"] > window_start) & (joined_df["timestamp"] <= as_of)
    ].copy()

    if window_df.empty:
        raise ValueError(
            f"No data in retrain window ({window_start} -> {as_of}). Cannot retrain."
        )

    val_start = as_of - pd.Timedelta(days=holdout_days)
    train_df = window_df[window_df["timestamp"] <= val_start]
    val_df = window_df[window_df["timestamp"] > val_start]

    missing = [f for f in feature_names if f not in window_df.columns]
    if missing:
        raise ValueError(
            f"Retrain window is missing expected feature columns: {missing}. "
            f"Available: {sorted(window_df.columns)}"
        )

    MIN_ROWS = 1000  # arbitrary but deliberate floor -- a few hundred rows
    # across 261 zones isn't enough signal to trust a retrained model over
    # the existing champion; fail loudly rather than promote on noise.
    if len(train_df) < MIN_ROWS or len(val_df) < MIN_ROWS:
        raise ValueError(
            f"Retrain split too small to trust: train={len(train_df)} rows, "
            f"val={len(val_df)} rows (floor={MIN_ROWS} each). Skipping this "
            f"retrain attempt rather than training on insufficient data."
        )

    logger.info(
        "Retrain split built: train=%d rows [%s -> %s], val=%d rows [%s -> %s]",
        len(train_df), train_df["timestamp"].min(), train_df["timestamp"].max(),
        len(val_df), val_df["timestamp"].min(), val_df["timestamp"].max(),
    )

    X_train = train_df[feature_names].copy()
    X_val = val_df[feature_names].copy()
    for col in categorical_cols:
        # Cast independently on each split, exactly matching train.py --
        # LightGBM's Dataset(..., reference=train_set) mechanism aligns
        # category codes across train/val internally; it does not require
        # both splits to see the identical set of categories locally.
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")

    # reference_snapshot.py's real builder expects "hour_ts", not "timestamp" --
    # confirmed against its actual source. Alias rather than rename in place,
    # so callers relying on "timestamp" elsewhere in train_df_raw aren't broken.
    train_df_raw = train_df.copy()
    train_df_raw["hour_ts"] = train_df_raw["timestamp"]

    return RetrainSplit(
        X_train=X_train.reset_index(drop=True),
        y_train=train_df[target_col].reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_val=val_df[target_col].reset_index(drop=True),
        train_df_raw=train_df_raw.reset_index(drop=True),
        train_start=train_df["timestamp"].min(),
        train_end=train_df["timestamp"].max(),
        val_start=val_df["timestamp"].min(),
        val_end=val_df["timestamp"].max(),
    )