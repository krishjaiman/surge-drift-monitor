"""Feature-prediction Spearman correlation stability.

See the design note in the handoff: this baseline does NOT live in
`reference_snapshot_current.json` because it's a prediction-time quantity
that didn't exist when that snapshot was frozen. It is seeded once, from an
early window of production predictions, into its own cached artifact
(`prediction_correlation_baseline.json`), and every later evaluation compares
against that cache -- never against a freshly recomputed "current" baseline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorrelationStabilityResult:
    feature: str
    baseline_correlation: float
    current_correlation: float
    relative_drop_pct: float
    breached: bool


def seed_or_load_baseline(
    baseline_path: str | Path,
    joined_df_for_seeding: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, float]:
    """Load the cached baseline if it exists; otherwise compute it once from
    `joined_df_for_seeding` (expected to be the first `correlation_baseline_window_hours`
    of production predictions) and persist it. Idempotent by design -- safe
    to call on every monitoring run.
    """
    baseline_path = Path(baseline_path)
    if baseline_path.exists():
        with baseline_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    logger.info(
        "No cached correlation baseline found at %s -- seeding from %d rows.",
        baseline_path, len(joined_df_for_seeding),
    )
    baseline: dict[str, float] = {}
    for feature in feature_names:
        if feature not in joined_df_for_seeding.columns:
            continue
        corr = _safe_spearman(
            joined_df_for_seeding[feature], joined_df_for_seeding["predicted_demand"]
        )
        if corr is not None:
            baseline[feature] = corr

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with baseline_path.open("w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    logger.info("Cached correlation baseline written to %s", baseline_path)
    return baseline


def _safe_spearman(feature_series: pd.Series, prediction_series: pd.Series) -> float | None:
    """Spearman correlation, returning None for degenerate inputs (e.g. a
    constant column, which has undefined correlation) rather than raising or
    silently emitting NaN into downstream comparisons.
    """
    paired = pd.concat([feature_series, prediction_series], axis=1).dropna()
    if len(paired) < 10:
        return None
    if paired.iloc[:, 0].nunique() <= 1:
        return None
    corr, _ = spearmanr(paired.iloc[:, 0], paired.iloc[:, 1])
    if pd.isna(corr):
        return None
    return float(corr)


def compute_correlation_stability(
    joined_df: pd.DataFrame,
    baseline: dict[str, float],
    feature_names: list[str],
    drop_pct_threshold: float,
) -> list[CorrelationStabilityResult]:
    """Compare current-window feature-prediction correlation to the cached
    baseline. `relative_drop_pct` is signed: positive means the correlation
    weakened (moved toward zero) relative to baseline magnitude; a
    correlation that strengthened is not an alert condition per the spec.
    """
    results: list[CorrelationStabilityResult] = []
    for feature in feature_names:
        if feature not in baseline:
            continue
        current_corr = _safe_spearman(joined_df.get(feature, pd.Series(dtype=float)),
                                       joined_df.get("predicted_demand", pd.Series(dtype=float)))
        if current_corr is None:
            continue
        baseline_corr = baseline[feature]
        if abs(baseline_corr) < 1e-6:
            continue  # no meaningful baseline signal to measure decay against
        relative_drop = (abs(baseline_corr) - abs(current_corr)) / abs(baseline_corr)
        results.append(
            CorrelationStabilityResult(
                feature=feature,
                baseline_correlation=baseline_corr,
                current_correlation=current_corr,
                relative_drop_pct=float(relative_drop),
                breached=relative_drop > drop_pct_threshold,
            )
        )
    return results