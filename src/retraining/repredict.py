"""Re-score predictions after promotion.

Design decision (flagged earlier, now implemented): without this step, the
backtest's `predicted_demand` column stays frozen from the ORIGINAL champion
for the rest of the run, and no metric could ever visibly recover after a
promotion -- defeating the point of demonstrating TTR. Since the feature
columns (`feat_*`, already unprefixed by prediction_log_reader.py) don't
depend on which model produced the prediction, this is a plain batch
`.predict()` call over already-existing feature rows, not a re-run of
Phase 2's serving stack.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def repredict_future_hours(
    joined_df: pd.DataFrame,
    as_of: pd.Timestamp,
    new_model,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return a COPY of joined_df with predicted_demand replaced for every
    row with timestamp > as_of, using new_model's predictions on the
    existing feature columns. Rows at or before as_of are untouched --
    history should reflect what was actually predicted at the time, not be
    rewritten retroactively.
    """
    future_mask = joined_df["timestamp"] > as_of
    n_future = int(future_mask.sum())
    if n_future == 0:
        logger.warning("No future rows to re-predict after %s -- nothing to do.", as_of)
        return joined_df

    updated = joined_df.copy()
    X_future = updated.loc[future_mask, feature_names]
    updated.loc[future_mask, "predicted_demand"] = new_model.predict(X_future)

    logger.info(
        "Re-predicted %d rows after %s with the newly promoted model.", n_future, as_of
    )
    return updated