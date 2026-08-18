"""Promotion pipeline: decide whether the challenger beats the champion,
and if so, execute every side effect a promotion requires -- MLflow
registry update, reference snapshot rebuild, and Phase 3 monitoring state
reset (so drift accumulation restarts against the new baseline, per the
project's core principle that this is a hard requirement of retraining,
not optional cleanup).

Config note: this module reads mlflow_tracking_uri/model_name/champion_alias
from params["mlflow"] (tracking_uri, registered_model_name, champion_alias)
-- the SAME keys train.py itself reads. Earlier drafts of this module had
their own duplicate `retraining.mlflow_tracking_uri` / `retraining.model_name`
keys with "champion" hardcoded as the alias string; that's two sources of
truth for the same config and a real risk (exactly the class of bug this
project already hit once with the reference snapshot). Fixed to read the
one real config section instead.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import lightgbm as lgb
import mlflow
import pandas as pd

logger = logging.getLogger(__name__)


def decide_promotion(
    champion_rmse: float, challenger_rmse: float, min_improvement_pct: float = 0.05
) -> tuple[bool, float]:
    """Promote only if the challenger's RMSE is at least `min_improvement_pct`
    better than the champion's, both measured on the SAME holdout. Returns
    (should_promote, actual_improvement_pct) so the decision is always
    logged with the number that drove it, not just a boolean.
    """
    improvement_pct = (champion_rmse - challenger_rmse) / champion_rmse
    should_promote = improvement_pct > min_improvement_pct
    logger.info(
        "Promotion decision: champion_rmse=%.4f, challenger_rmse=%.4f, "
        "improvement=%.2f%% (threshold=%.2f%%) -> %s",
        champion_rmse, challenger_rmse, improvement_pct * 100,
        min_improvement_pct * 100, "PROMOTE" if should_promote else "KEEP CHAMPION",
    )
    return should_promote, improvement_pct


def promote_challenger(
    challenger_model: lgb.Booster,
    mlflow_cfg: dict,
) -> str:
    """Log the challenger and move the champion alias onto it.

    `mlflow_cfg` is `params["mlflow"]` verbatim -- must contain
    tracking_uri, registered_model_name, champion_alias, and (for
    mlflow.set_experiment) experiment_name, matching train.py's own usage
    of this config section exactly.

    Mirrors train.py's single-call registration pattern
    (mlflow.lightgbm.log_model(booster, ..., registered_model_name=...))
    rather than a separate log-then-register step, for consistency with
    the rest of this codebase.

    The previous champion version keeps its version number and is tagged
    'previous_champion' rather than deleted -- demoted, not destroyed, so
    you can roll back or audit history.

    Returns the new version number as a string.
    """
    tracking_uri = mlflow_cfg["tracking_uri"]
    model_name = mlflow_cfg["registered_model_name"]
    champion_alias = mlflow_cfg["champion_alias"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    client = mlflow.MlflowClient()

    # Tag whatever currently holds the champion alias as previous, BEFORE
    # reassigning -- alias reassignment is atomic per-name in MLflow, so we
    # must capture the outgoing version first or lose track of it.
    try:
        current_champion = client.get_model_version_by_alias(model_name, champion_alias)
        client.set_model_version_tag(
            model_name, current_champion.version, "role", "previous_champion"
        )
        logger.info("Tagged outgoing champion (version %s) as previous_champion.",
                    current_champion.version)
    except mlflow.exceptions.RestException:
        logger.info("No existing @%s alias found -- this is the first promotion.", champion_alias)

    with mlflow.start_run(run_name="phase4_retrain_promotion"):
        model_info = mlflow.lightgbm.log_model(
            challenger_model, artifact_path="model", registered_model_name=model_name,
        )
        new_version = model_info.registered_model_version

    client.set_registered_model_alias(model_name, champion_alias, new_version)
    client.set_model_version_tag(model_name, new_version, "role", "champion")
    logger.info("Promoted new challenger to @%s (version %s).", champion_alias, new_version)

    return new_version


def rebuild_reference_snapshot(
    training_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    params: dict,
    version_tag: str,
) -> Path:
    """Rebuild the reference snapshot from the challenger's actual training
    data, using the REAL Phase 1 builder (not a reimplementation) so the
    schema is guaranteed identical to what reference_loader.py already
    parses correctly. This is the "monitoring resets against the new
    baseline" step -- per the project's stated principle, this must happen
    on every promotion, never be skipped.
    """
    from src.training.reference_snapshot import build_reference_snapshot, save_reference_snapshot

    snapshot = build_reference_snapshot(training_df, feature_cols, categorical_cols, params)
    out_path = save_reference_snapshot(snapshot, params, version_tag=version_tag)
    logger.info("Reference snapshot rebuilt and saved: %s", out_path)
    return out_path


def reset_monitoring_state(
    correlation_baseline_path: str | Path,
) -> dict:
    """Reset everything Phase 3 accumulates between evaluations, so drift
    detection starts clean against the new baseline:
      - deletes the cached feature-prediction correlation baseline (it will
        reseed itself from the next `correlation_baseline_window_hours` of
        post-promotion predictions, same as a fresh deployment)
      - returns an empty consecutive-bias-breach counter dict for the
        caller to use going forward

    Does NOT touch metrics_timeseries.parquet or alerts_log.jsonl -- those
    are historical records and should keep the pre-promotion history for
    TTD/TTR analysis, not be wiped.
    """
    correlation_baseline_path = Path(correlation_baseline_path)
    if correlation_baseline_path.exists():
        # Archive rather than silently delete -- useful if you want to
        # compare pre/post-promotion correlation baselines later.
        archive_path = correlation_baseline_path.with_suffix(".pre_promotion.json")
        shutil.move(str(correlation_baseline_path), str(archive_path))
        logger.info(
            "Archived pre-promotion correlation baseline to %s; will reseed fresh.",
            archive_path,
        )
    return {}