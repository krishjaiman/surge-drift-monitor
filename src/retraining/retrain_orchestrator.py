"""Ties together retrain_data, challenger_trainer, promotion, and repredict
into the single entrypoint the backtest loop calls each hour.

This module owns the cooldown policy: even if the trigger keeps firing
every hour (as observed in the Phase 3 backtest -- once real staleness
sets in, it stays triggered continuously), a real system should not
retrain every single hour. A cooldown after each promotion lets the new
baseline and metrics stabilize before considering another retrain.

Config sourcing (confirmed against real train.py / reference_snapshot.py,
no placeholders): mlflow_tracking_uri/registered_model_name/champion_alias
come from params["mlflow"]; LightGBM hyperparameters come from
params["model"]["params"] and params["model"]["early_stopping_rounds"];
categorical columns and target_col come from params["features"]. The
`retraining:` params block only holds genuinely NEW Phase 4 settings
(lookback_days, holdout_days, min_improvement_pct, cooldown_days) that
don't exist anywhere else in the project.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd

from src.retraining.challenger_trainer import evaluate_champion_on_holdout, train_challenger
from src.retraining.promotion import (
    decide_promotion,
    promote_challenger,
    rebuild_reference_snapshot,
    reset_monitoring_state,
)
from src.retraining.repredict import repredict_future_hours
from src.retraining.retrain_data import build_retrain_split

logger = logging.getLogger(__name__)


@dataclass
class RetrainOutcome:
    attempted: bool
    promoted: bool
    reason: str
    champion_rmse: float | None = None
    challenger_rmse: float | None = None
    improvement_pct: float | None = None
    new_model_version: str | None = None
    new_reference_snapshot_path: str | None = None


def should_attempt_retrain(
    retrain_trigger_fired: bool,
    as_of: pd.Timestamp,
    last_promotion_as_of: pd.Timestamp | None,
    last_attempt_as_of: pd.Timestamp | None,
    promotion_cooldown_days: int,
    failed_attempt_cooldown_days: int,
) -> tuple[bool, str]:
    """Gate: only attempt a retrain if the trigger fired AND neither
    cooldown is active. Two distinct cooldowns, tracked separately:

    - `promotion_cooldown_days` (longer, e.g. 7 days): after an actual
      promotion, lets the new baseline stabilize before trying again.
    - `failed_attempt_cooldown_days` (shorter, e.g. 1 day): after ANY
      attempt that did NOT promote, avoids retraining on every single
      triggered hour while the trigger stays hot but nothing about the
      situation has changed. Added after a real backtest showed 683
      retrain attempts (one per triggered hour) with 0 promotions --
      683 wasted LightGBM training runs with no cooldown between them,
      since `last_promotion_as_of` only updates on an actual promotion.

    Returns (should_attempt, reason) so callers can log WHY a retrain was
    skipped, not just that it was.
    """
    if not retrain_trigger_fired:
        return False, "trigger_not_fired"

    if last_promotion_as_of is not None:
        elapsed_days = (as_of - last_promotion_as_of).total_seconds() / 86400
        if elapsed_days < promotion_cooldown_days:
            return False, (
                f"promotion_cooldown_active "
                f"({elapsed_days:.1f}/{promotion_cooldown_days} days elapsed)"
            )

    if last_attempt_as_of is not None:
        elapsed_days = (as_of - last_attempt_as_of).total_seconds() / 86400
        if elapsed_days < failed_attempt_cooldown_days:
            return False, (
                f"failed_attempt_cooldown_active "
                f"({elapsed_days:.2f}/{failed_attempt_cooldown_days} days elapsed)"
            )

    return True, "trigger_fired_and_eligible"


def run_retrain_cycle(
    as_of: pd.Timestamp,
    joined_df: pd.DataFrame,
    feature_names: list[str],
    params: dict,
) -> tuple[RetrainOutcome, pd.DataFrame | None]:
    """Execute the full retrain -> evaluate -> promote (maybe) -> reset ->
    repredict pipeline for a single trigger event.

    Returns (outcome, updated_joined_df). updated_joined_df is None unless
    a promotion actually happened -- callers should only swap their working
    joined_df when it's not None.
    """
    retrain_cfg = params["retraining"]
    mon = params["monitoring"]
    mlflow_cfg = params["mlflow"]
    categorical_cols = params["features"]["categorical_features"]
    target_col = params["features"]["target_col"]
    model_params = params["model"]["params"]
    early_stopping_rounds = params["model"]["early_stopping_rounds"]

    try:
        split = build_retrain_split(
            joined_df, as_of, feature_names, categorical_cols, target_col=target_col,
            lookback_days=retrain_cfg["lookback_days"],
            holdout_days=retrain_cfg["holdout_days"],
        )
    except ValueError as exc:
        logger.warning("Retrain attempt at %s aborted: %s", as_of, exc)
        return RetrainOutcome(attempted=True, promoted=False, reason=f"data_error: {exc}"), None

    challenger_result = train_challenger(
        split.X_train, split.y_train, split.X_val, split.y_val,
        categorical_cols=categorical_cols,
        model_params=model_params, early_stopping_rounds=early_stopping_rounds,
    )

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    champion_model = mlflow.lightgbm.load_model(
        f"models:/{mlflow_cfg['registered_model_name']}@{mlflow_cfg['champion_alias']}"
    )
    champion_rmse = evaluate_champion_on_holdout(champion_model, split.X_val, split.y_val)

    should_promote, improvement_pct = decide_promotion(
        champion_rmse, challenger_result.val_rmse,
        min_improvement_pct=retrain_cfg["min_improvement_pct"],
    )

    outcome = RetrainOutcome(
        attempted=True, promoted=False, reason="challenger_did_not_beat_champion",
        champion_rmse=champion_rmse, challenger_rmse=challenger_result.val_rmse,
        improvement_pct=improvement_pct,
    )

    if not should_promote:
        logger.info(
            "Retrain at %s: challenger did NOT beat champion by %.1f%% -- keeping current champion.",
            as_of, retrain_cfg["min_improvement_pct"] * 100,
        )
        return outcome, None

    new_version = promote_challenger(challenger_result.model, mlflow_cfg)

    snapshot_path = rebuild_reference_snapshot(
        split.train_df_raw, feature_names, categorical_cols, params,
        version_tag=f"retrain_{as_of.strftime('%Y%m%d_%H%M')}",
    )

    reset_monitoring_state(mon["prediction_correlation_baseline_path"])

    updated_joined_df = repredict_future_hours(
        joined_df, as_of, challenger_result.model, feature_names
    )

    outcome.promoted = True
    outcome.reason = "promoted"
    outcome.new_model_version = new_version
    outcome.new_reference_snapshot_path = str(snapshot_path)

    logger.warning(
        "PROMOTED new champion at %s: version=%s, improvement=%.2f%%, "
        "snapshot=%s. Monitoring state reset.",
        as_of, new_version, improvement_pct * 100, snapshot_path,
    )

    return outcome, updated_joined_df


def append_retrain_event(outcome: RetrainOutcome, as_of: pd.Timestamp, output_path: str | Path) -> None:
    """Append one record per retrain attempt (promoted or not) -- an audit
    trail for TTR analysis: how many hours between trigger-fired and
    promoted, how often retraining actually clears the bar.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"as_of": as_of.isoformat(), **outcome.__dict__}
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")