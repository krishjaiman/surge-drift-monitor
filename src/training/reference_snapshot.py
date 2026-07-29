"""
Reference distribution snapshot stage.

This is the most critical artifact in the project. It captures the
statistical profile of every model input feature (and the target) at
training time: mean, std, percentiles, and histogram bucket edges/counts.

All drift calculations in Phase 3 — PSI, KL divergence, null-rate checks —
diff live production distributions against this exact snapshot. It is
computed once here, versioned with DVC, and is NOT recomputed until Phase 4
promotes a new champion model (at which point a new snapshot replaces it
and monitoring resets against the new baseline, per the project spec).

Do not call this function outside of the training pipeline — a snapshot
computed from the wrong dataset silently invalidates every downstream
drift alert.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def _numeric_feature_profile(series: pd.Series, params: dict) -> dict[str, Any]:
    """Compute the statistical profile for a single numeric feature."""
    clean = series.dropna()
    percentiles = params["reference_snapshot"]["percentiles"]
    n_bins = params["reference_snapshot"]["histogram_bins"]

    counts, bin_edges = np.histogram(clean, bins=n_bins)

    return {
        "dtype": "numeric",
        "count": int(series.shape[0]),
        "null_count": int(series.isna().sum()),
        "null_rate": float(series.isna().mean()),
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "percentiles": {
            str(p): float(np.percentile(clean, p)) for p in percentiles
        },
        "histogram": {
            "bin_edges": bin_edges.tolist(),
            "counts": counts.tolist(),
        },
    }


def _categorical_feature_profile(series: pd.Series) -> dict[str, Any]:
    """Compute the statistical profile for a single categorical feature."""
    value_counts = series.value_counts(normalize=True, dropna=False)
    return {
        "dtype": "categorical",
        "count": int(series.shape[0]),
        "null_count": int(series.isna().sum()),
        "null_rate": float(series.isna().mean()),
        "category_frequencies": {str(k): float(v) for k, v in value_counts.items()},
        "n_unique": int(series.nunique()),
    }


def build_reference_snapshot(
    training_df: pd.DataFrame, feature_cols: list[str], categorical_cols: list[str], params: dict
) -> dict[str, Any]:
    """
    Build the full reference snapshot for the given training dataframe.

    Args:
        training_df: the exact dataframe rows used for model training
            (must match what train.py used — same filter, same split).
        feature_cols: all model input feature column names.
        categorical_cols: subset of feature_cols that are categorical.
        params: loaded params.yaml.

    Returns:
        A JSON-serializable dict: one profile entry per feature, plus
        the target column, plus snapshot-level metadata.
    """
    snapshot: dict[str, Any] = {"features": {}, "metadata": {}}

    for col in feature_cols:
        if col in categorical_cols:
            snapshot["features"][col] = _categorical_feature_profile(training_df[col])
        else:
            snapshot["features"][col] = _numeric_feature_profile(training_df[col], params)

    target_col = params["features"]["target_col"]
    snapshot["features"][target_col] = _numeric_feature_profile(training_df[target_col], params)

    snapshot["metadata"] = {
        "n_rows": int(len(training_df)),
        "training_months": params["data"]["training_months"],
        "date_range": [
            str(training_df["hour_ts"].min()),
            str(training_df["hour_ts"].max()),
        ],
        "n_zones": int(training_df["zone_id"].nunique()),
    }

    return snapshot


def save_reference_snapshot(snapshot: dict, params: dict, version_tag: str = "v1") -> Path:
    """Persist the snapshot as versioned JSON under data/reference/."""
    out_dir = Path(params["data"]["reference_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"reference_snapshot_{version_tag}.json"
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    # Also write/overwrite a stable "current" pointer file that Phase 3
    # monitoring always reads from, so promotion in Phase 4 is just
    # overwriting this one file (plus a new versioned copy for history).
    current_path = out_dir / "reference_snapshot_current.json"
    with open(current_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    logger.info("Reference snapshot saved to %s and %s", out_path, current_path)
    return out_path


if __name__ == "__main__":
    from src.training.train import get_feature_columns, time_based_split

    params = load_params()
    processed_path = Path(params["data"]["processed_dir"]) / "features.parquet"
    df = pd.read_parquet(processed_path)

    feature_cols = get_feature_columns(params)
    categorical_cols = params["features"]["categorical_features"]

    train_df, _ = time_based_split(df, params)

    snapshot = build_reference_snapshot(train_df, feature_cols, categorical_cols, params)
    save_reference_snapshot(snapshot, params, version_tag="v1")
