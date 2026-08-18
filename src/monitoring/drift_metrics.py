"""Distribution drift metrics: PSI, label KL divergence, null-rate drift.

All functions compare a *current* window of data against the frozen Phase 1
`ReferenceSnapshot`. Nothing in this module recomputes or mutates the
reference artifact -- it is read-only input.

Reading on the "why" behind PSI as the standard covariate-drift metric,
and its 0.1/0.25 conventional thresholds:
https://www.blog.trainindata.com/population-stability-index/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.monitoring.reference_loader import FeatureProfile, ReferenceSnapshot

logger = logging.getLogger(__name__)

_EPSILON = 1e-6  # avoids log(0) / div-by-zero on empty bins


@dataclass(frozen=True)
class FeaturePSIResult:
    feature: str
    psi: float
    breached: bool


@dataclass(frozen=True)
class NullRateResult:
    feature: str
    reference_null_rate: float
    current_null_rate: float
    ratio: float
    breached: bool


def compute_psi(reference_probs: np.ndarray, current_probs: np.ndarray) -> float:
    """Population Stability Index between two aligned probability vectors.

    Both vectors must sum to ~1 and share the same bin order. Bins with zero
    probability on either side are clipped to `_EPSILON` rather than dropped,
    so a bin that vanished entirely (a real drift signal) still contributes
    to the score instead of being silently ignored.
    """
    if reference_probs.shape != current_probs.shape:
        raise ValueError(
            f"Shape mismatch: reference={reference_probs.shape}, "
            f"current={current_probs.shape}. Bins must align exactly."
        )
    ref = np.clip(reference_probs, _EPSILON, None)
    cur = np.clip(current_probs, _EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _bucket_numeric_series(values: pd.Series, bin_edges: np.ndarray) -> np.ndarray:
    """Bucket a current numeric series into the reference's fixed bin edges.

    Using the reference's own edges (not freshly computed quantile bins) is
    the point: bins must be identical on both sides for PSI to mean anything.
    Values outside the reference range fall into the first/last bin rather
    than being dropped -- an out-of-range value is itself a drift signal.
    """
    clean = values.dropna().to_numpy()
    if clean.size == 0:
        raise ValueError("No non-null values available to bucket.")
    # np.histogram with explicit edges clips out-of-range into edge bins
    # only if we pre-clip; otherwise out-of-range values are excluded, which
    # would understate drift. Clip explicitly to preserve them.
    clipped = np.clip(clean, bin_edges[0], bin_edges[-1])
    counts, _ = np.histogram(clipped, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        raise ValueError("Bucketed histogram is empty.")
    return counts.astype(float) / total


def _bucket_categorical_series(values: pd.Series, category_order: list[str]) -> np.ndarray:
    """Bucket a current categorical series into the reference's category set.

    Categories present in current data but absent from the reference (e.g. a
    genuinely new zone_id) are folded into an implicit 'unseen' mass that
    inflates every existing bin's apparent shrinkage -- which is correct
    behavior: a new category is drift, and PSI should reflect it rather than
    silently discard those rows.
    """
    clean = values.dropna().astype(str)
    if clean.empty:
        raise ValueError("No non-null values available to bucket.")
    counts = clean.value_counts()
    total = len(clean)
    return np.array([counts.get(cat, 0) / total for cat in category_order], dtype=float)


def compute_feature_psi(
    current_df: pd.DataFrame,
    reference: ReferenceSnapshot,
    feature_names: list[str],
    psi_threshold: float,
    threshold_overrides: dict[str, float] | None = None,
) -> list[FeaturePSIResult]:
    """PSI for each named feature, current window vs frozen reference.

    `threshold_overrides` lets specific features use a different alert
    threshold than `psi_threshold` -- needed for temporally-autocorrelated
    continuous features (temperature, windspeed) whose PSI carries a
    structurally higher baseline under any practical window length, as
    confirmed via backtest diagnostics (PSI stayed elevated across every
    calendar month, not just around a seasonal mismatch). This is a
    documented limitation of vanilla PSI, not something a window-size
    change alone fully resolves -- see params.yaml for the note on the
    real fix (seasonal reference) being out of scope for this phase.
    """
    threshold_overrides = threshold_overrides or {}
    results: list[FeaturePSIResult] = []
    for name in feature_names:
        profile = reference.get_feature(name)
        if name not in current_df.columns:
            logger.warning("Feature '%s' missing from current window; skipping PSI.", name)
            continue
        try:
            if profile.dtype == "numeric":
                current_probs = _bucket_numeric_series(current_df[name], profile.bin_edges)
            else:
                current_probs = _bucket_categorical_series(
                    current_df[name], profile.category_order()
                )
            ref_probs = profile.reference_probabilities()
            psi = compute_psi(ref_probs, current_probs)
        except ValueError as exc:
            logger.warning("Skipping PSI for '%s': %s", name, exc)
            continue
        effective_threshold = threshold_overrides.get(name, psi_threshold)
        results.append(
            FeaturePSIResult(feature=name, psi=psi, breached=psi > effective_threshold)
        )
    return results


def compute_label_kl_divergence(
    current_target: pd.Series,
    reference: ReferenceSnapshot,
) -> float:
    """KL(current || reference) over the target (trip_count) distribution."""
    profile = reference.target
    current_probs = _bucket_numeric_series(current_target, profile.bin_edges)
    ref_probs = profile.reference_probabilities()
    ref = np.clip(ref_probs, _EPSILON, None)
    cur = np.clip(current_probs, _EPSILON, None)
    return float(np.sum(cur * np.log(cur / ref)))


def compute_null_rate_drift(
    current_df: pd.DataFrame,
    reference: ReferenceSnapshot,
    feature_names: list[str],
    multiplier_threshold: float,
) -> list[NullRateResult]:
    """Flag features whose current null rate exceeds `multiplier_threshold`x
    the training-time null rate. Uses a small floor on the reference rate so
    a feature that was never null in training doesn't produce an infinite
    or undefined ratio the moment a single null appears.
    """
    floor = 1e-4
    results: list[NullRateResult] = []
    for name in feature_names:
        profile = reference.get_feature(name)
        if name not in current_df.columns:
            continue
        current_rate = float(current_df[name].isna().mean())
        ref_rate = max(profile.null_rate, floor)
        ratio = current_rate / ref_rate
        results.append(
            NullRateResult(
                feature=name,
                reference_null_rate=profile.null_rate,
                current_null_rate=current_rate,
                ratio=ratio,
                # cast explicitly: numpy bool_ (which `ratio > threshold` can silently
                # produce if any upstream value is still an ndarray scalar) breaks
                # json.dumps() in output_writer.append_alert -- always store plain bool.
                breached=bool(ratio > multiplier_threshold),
            )
        )
    return results