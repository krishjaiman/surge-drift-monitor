"""Train a challenger model on the reconstructed retrain window.

Mirrors src/training/train.py's actual training approach exactly -- raw
lgb.Dataset + lgb.train (not the sklearn wrapper), same categorical_feature
handling, and REAL hyperparameters read from params["model"]["params"] /
params["model"]["early_stopping_rounds"]. No placeholder hyperparameter
dict: this module has zero opinions of its own about model configuration,
it only reads what train.py already reads.

One deliberate deviation from train.py, flagged rather than silent: the
`lgb.log_evaluation(period=50)` callback is dropped here. train.py runs
once per invocation, so per-round logging is useful; this runs potentially
dozens of times during a single backtest, so that callback would flood the
log with per-boosting-round noise. `lgb.early_stopping` is kept (it's not
just logging, it controls training). Everything that affects the actual
model -- objective, learning rate, num_estimators, categorical handling --
is unchanged from train.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChallengerResult:
    model: lgb.Booster
    val_rmse: float
    val_mae: float
    best_iteration: int
    n_train_rows: int
    n_val_rows: int


def train_challenger(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_cols: list[str],
    model_params: dict,
    early_stopping_rounds: int,
) -> ChallengerResult:
    """Train a challenger via the same lgb.Dataset/lgb.train path train.py
    uses, with the same real hyperparameters (`model_params` should be
    `params["model"]["params"]` verbatim, `early_stopping_rounds` should be
    `params["model"]["early_stopping_rounds"]` verbatim -- callers pass
    these through, this function does not default or override them).

    X_train/X_val must already have `categorical_cols` cast to "category"
    dtype (build_retrain_split.py does this) -- `categorical_feature` is
    passed here purely to tell lgb.Dataset which columns to treat as
    categorical, matching train.py's call signature exactly.
    """
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_cols, reference=train_set)

    booster = lgb.train(
        model_params,
        train_set,
        num_boost_round=model_params["n_estimators"],
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )

    preds = booster.predict(X_val, num_iteration=booster.best_iteration)
    rmse = float(np.sqrt(np.mean((preds - y_val.to_numpy()) ** 2)))
    mae = float(np.mean(np.abs(preds - y_val.to_numpy())))

    logger.info(
        "Challenger trained: val_rmse=%.4f, val_mae=%.4f, best_iteration=%d "
        "(train=%d rows, val=%d rows)",
        rmse, mae, booster.best_iteration, len(X_train), len(X_val),
    )

    return ChallengerResult(
        model=booster, val_rmse=rmse, val_mae=mae, best_iteration=booster.best_iteration,
        n_train_rows=len(X_train), n_val_rows=len(X_val),
    )


def evaluate_champion_on_holdout(champion_model: lgb.Booster, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """RMSE of the CURRENT champion on the SAME holdout the challenger was
    evaluated on -- apples-to-apples, required for a fair promotion decision.

    `champion_model` is a raw lgb.Booster (train.py logs the Booster
    directly via mlflow.lightgbm.log_model, not the sklearn wrapper, so
    mlflow.lightgbm.load_model returns a Booster here too). X_val must
    already have categorical_cols cast to "category" dtype, matching what
    the champion was trained with -- passing plain int64/object columns
    here is exactly what caused the "train and valid dataset
    categorical_feature do not match" crash.
    """
    preds = champion_model.predict(X_val, num_iteration=champion_model.best_iteration)
    return float(np.sqrt(np.mean((np.asarray(preds) - y_val.to_numpy()) ** 2)))