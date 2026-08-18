"""Loads the Phase 2 prediction log and joins it to ground-truth actuals.

Two important constraints carried over from the Phase 2 handoff, both
non-obvious enough to be worth restating here rather than trusting memory:

1. `data/predictions/date=*/*.parquet` is many small flush-batch files per
   day, not one file per day. Must be read via a dataset API that handles a
   partitioned directory (pyarrow.dataset), never a single-file assumption.
2. Actuals live ONLY in `demand_history` (SQLite), never written back into
   the Parquet log. Predictions and actuals must be joined at read time on
   (zone_id, timestamp). There is no pre-joined table anywhere.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)


def load_predictions(
    predictions_dir: str | Path,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Load logged predictions in [start_ts, end_ts) from the partitioned
    Parquet directory. Filtering is pushed down via pyarrow dataset filters
    rather than loading everything into memory and slicing in pandas --
    six months of hourly x ~261-zone predictions is not huge, but there's no
    reason to pay the full scan cost on every hourly monitoring run.
    """
    predictions_dir = Path(predictions_dir)
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    dataset = ds.dataset(str(predictions_dir), format="parquet", partitioning="hive")
    filter_expr = (ds.field("timestamp") >= start_ts) & (ds.field("timestamp") < end_ts)
    table = dataset.to_table(filter=filter_expr)
    df = table.to_pandas()

    if df.empty:
        logger.warning(
            "No predictions found in [%s, %s). Returning empty frame.", start_ts, end_ts
        )
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Confirmed via real column dump: logged input features carry a "feat_"
    # prefix (feat_lag_168h, feat_hour_of_day, ...) that does NOT match the
    # unprefixed names used in reference_snapshot_current.json (lag_168h,
    # hour_of_day, ...). Strip the prefix here, once, at the read boundary,
    # so every downstream module (drift_metrics, correlation_stability) can
    # assume prediction-log feature names already match reference-snapshot
    # feature names -- rather than every caller needing to know about this
    # prefix separately.
    #
    # feat_zone_id is dropped rather than renamed: it's redundant with the
    # top-level zone_id column already used for the join, and renaming it
    # to "zone_id" would collide with that existing column.
    feature_prefix = "feat_"
    rename_map = {}
    for col in df.columns:
        if col.startswith(feature_prefix):
            stripped = col[len(feature_prefix):]
            if stripped != "zone_id":
                rename_map[col] = stripped
    df = df.rename(columns=rename_map)
    if "feat_zone_id" in df.columns:
        df = df.drop(columns=["feat_zone_id"])

    logger.info(
        "Loaded %d predictions across %d zones in [%s, %s)",
        len(df), df["zone_id"].nunique(), start_ts, end_ts,
    )
    return df


def load_actuals(
    sqlite_path: str | Path,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Load ground-truth (zone_id, ts, trip_count) from demand_history for
    the given window. Schema confirmed via PRAGMA table_info against the
    real Phase 2 database: columns are (zone_id, ts, trip_count, is_actual).

    Filters on is_actual = 1 explicitly -- the table has a flag distinguishing
    actual rows from (presumably) some other row type, and pulling ground
    truth for RMSE/bias must never silently include non-actual rows.
    """
    sqlite_path = Path(sqlite_path)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"Historical store not found: {sqlite_path}")

    query = """
        SELECT zone_id, ts, trip_count
        FROM demand_history
        WHERE ts >= ? AND ts < ? AND is_actual = 1
    """
    with sqlite3.connect(str(sqlite_path)) as conn:
        df = pd.read_sql_query(
            query, conn, params=[start_ts.isoformat(), end_ts.isoformat()]
        )
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def load_predictions_with_actuals(
    predictions_dir: str | Path,
    sqlite_path: str | Path,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Predictions joined to actuals on (zone_id, timestamp). Inner join --
    a prediction with no recorded actual yet cannot contribute to RMSE/bias
    and is correctly dropped, not imputed.
    """
    predictions = load_predictions(predictions_dir, start_ts, end_ts)
    if predictions.empty:
        return predictions.assign(trip_count=pd.Series(dtype="int64"))

    actuals = load_actuals(sqlite_path, start_ts, end_ts)

    # zone_id dtype has already surprised us once this phase (schema
    # mismatches on ts/hour_ts) -- normalize explicitly rather than trust
    # both sources agree, since a silent dtype mismatch here doesn't error,
    # it just drops every row in the merge.
    if predictions["zone_id"].dtype != actuals["zone_id"].dtype:
        logger.warning(
            "zone_id dtype mismatch: predictions=%s, actuals=%s. Casting both to int64.",
            predictions["zone_id"].dtype, actuals["zone_id"].dtype,
        )
    predictions = predictions.assign(zone_id=predictions["zone_id"].astype("int64"))
    actuals = actuals.assign(zone_id=actuals["zone_id"].astype("int64"))

    merged = predictions.merge(
        actuals,
        left_on=["zone_id", "timestamp"],
        right_on=["zone_id", "ts"],
        how="inner",
    )
    dropped = len(predictions) - len(merged)
    if dropped:
        logger.info(
            "%d/%d predictions had no matching actual yet and were excluded "
            "from performance metrics.", dropped, len(predictions),
        )
    return merged.drop(columns=["ts"])