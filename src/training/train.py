"""
Training stage.

Trains a LightGBM demand forecasting model on the engineered feature table,
logs params/metrics/model to MLflow, registers the model, and assigns the
`@champion` alias. This alias — not a hardcoded version number — is what
the serving app (Phase 2) and the retraining loop (Phase 4) will reference,
so promotion later is a metadata swap, not a redeploy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def get_feature_columns(params: dict) -> list[str]:
    """Single source of truth for which columns are model inputs."""
    time_feats = params["features"]["time_features"]
    lag_feats = [f"lag_{h}h" for h in params["features"]["lag_features"]["lag_hours"]]
    rolling_feats = [f"rolling_mean_{h}h" for h in params["features"]["rolling_features"]["windows_hours"]]
    weather_feats = params["data"]["weather"]["hourly_vars"]
    categorical_feats = params["features"]["categorical_features"]
    return categorical_feats + time_feats + lag_feats + rolling_feats + weather_feats


def time_based_split(df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by date, not randomly — a demand forecaster must be validated on
    a future time window, since random splits leak adjacent-hour
    autocorrelation between train and val and give an overly optimistic
    RMSE that will not hold up in production.
    """
    train_end = pd.Timestamp(params["split"]["train_end_date"])
    val_start = pd.Timestamp(params["split"]["val_start_date"])
    val_end = pd.Timestamp(params["split"]["val_end_date"])

    train_df = df[df["hour_ts"] <= train_end]
    val_df = df[(df["hour_ts"] >= val_start) & (df["hour_ts"] <= val_end)]

    if len(val_df) < 0.05 * len(df):
        logger.warning(
            "Date-based validation split is very small (%d rows). "
            "Falling back to a random split fraction of %.2f.",
            len(val_df),
            params["split"]["val_fraction_fallback"],
        )
        frac = params["split"]["val_fraction_fallback"]
        df_sorted = df.sort_values("hour_ts")
        cutoff = int(len(df_sorted) * (1 - frac))
        train_df, val_df = df_sorted.iloc[:cutoff], df_sorted.iloc[cutoff:]

    logger.info("Train rows: %d | Val rows: %d", len(train_df), len(val_df))
    return train_df, val_df


def train_model(params_path: str = "params.yaml") -> str:
    """
    Full training entry point.

    Returns:
        The MLflow run_id of the training run.
    """
    params = load_params(params_path)

    processed_path = Path(params["data"]["processed_dir"]) / "features.parquet"
    df = pd.read_parquet(processed_path)

    feature_cols = get_feature_columns(params)
    target_col = params["features"]["target_col"]
    categorical_cols = params["features"]["categorical_features"]

    train_df, val_df = time_based_split(df, params)

    X_train, y_train = train_df[feature_cols].copy(), train_df[target_col]
    X_val, y_val = val_df[feature_cols].copy(), val_df[target_col]

    for col in categorical_cols:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run() as run:
        model_params = params["model"]["params"]
        mlflow.log_params(model_params)
        mlflow.log_param("feature_cols", feature_cols)
        mlflow.log_param("training_months", params["data"]["training_months"])

        train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
        val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_cols, reference=train_set)

        booster = lgb.train(
            model_params,
            train_set,
            num_boost_round=model_params["n_estimators"],
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(params["model"]["early_stopping_rounds"]),
                lgb.log_evaluation(period=50),
            ],
        )

        val_preds = booster.predict(X_val, num_iteration=booster.best_iteration)
        rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
        mae = float(mean_absolute_error(y_val, val_preds))

        # Zone-level RMSE, not just global — per the project's non-negotiable
        # that global aggregates must never hide zone-specific failures.
        val_df = val_df.copy()
        val_df["pred"] = val_preds
        zone_rmse = (
            val_df.groupby("zone_id")[[target_col, "pred"]]
            .apply(lambda g: np.sqrt(mean_squared_error(g[target_col], g["pred"])), include_groups=False)
            .rename("zone_rmse")
        )

        mlflow.log_metric("val_rmse", rmse)
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_zone_rmse_worst", float(zone_rmse.max()))
        mlflow.log_metric("val_zone_rmse_median", float(zone_rmse.median()))
        mlflow.log_metric("best_iteration", booster.best_iteration)

        logger.info("Validation RMSE=%.3f MAE=%.3f (worst zone RMSE=%.3f)", rmse, mae, zone_rmse.max())

        model_info = mlflow.lightgbm.log_model(
            booster,
            artifact_path ="model",
            registered_model_name=params["mlflow"]["registered_model_name"],
        )

        run_id = run.info.run_id
        best_iteration = booster.best_iteration
        registered_version = model_info.registered_model_version

    _write_dvc_metrics(rmse, mae, float(zone_rmse.max()), float(zone_rmse.median()), best_iteration)
    _assign_champion_alias(params, registered_version)
    return run_id


def _write_dvc_metrics(
    rmse: float, mae: float, worst_zone_rmse: float, median_zone_rmse: float, best_iteration: int
) -> None:
    """
    Write training metrics to a DVC-tracked JSON file (separate from the
    MLflow run). MLflow is the system of record for experiment history and
    model artifacts; this file is what `dvc metrics diff` reads to compare
    metrics across pipeline runs/commits without needing an MLflow query.
    """
    import json

    metrics_path = Path("metrics/train_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "val_rmse": rmse,
                "val_mae": mae,
                "val_zone_rmse_worst": worst_zone_rmse,
                "val_zone_rmse_median": median_zone_rmse,
                "best_iteration": best_iteration,
            },
            f,
            indent=2,
        )
    logger.info("Metrics written to %s", metrics_path)


def _assign_champion_alias(params: dict, version: str) -> None:
    """
    Assign the `@champion` alias to the newly registered model version.

    Phase 1 has no existing champion to compare against, so the first
    trained model is promoted unconditionally. From Phase 4 onward, this
    function is replaced by the champion-challenger comparison gate —
    do not reuse this unconditional-promotion logic once that exists.
    """
    client = MlflowClient(tracking_uri=params["mlflow"]["tracking_uri"])
    model_name = params["mlflow"]["registered_model_name"]
    alias = params["mlflow"]["champion_alias"]

    client.set_registered_model_alias(model_name, alias, version)
    logger.info("Assigned alias '@%s' to %s version %s", alias, model_name, version)


if __name__ == "__main__":
    train_model()
