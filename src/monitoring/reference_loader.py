"""Read-only access to the Phase 1 reference distribution snapshot.

SCHEMA CONFIRMED against the real src/training/reference_snapshot.py
(build_reference_snapshot / save_reference_snapshot). Actual shape of
reference_snapshot_current.json:

    {
      "metadata": {
        "n_rows": int,
        "training_months": [str, ...],
        "date_range": [str, str],      # [min(hour_ts), max(hour_ts)] -- a
                                        # 2-element list, NOT a {start,end} dict.
                                        # Unused by this loader today.
        "n_zones": int
      },
      "features": {
        # Every model input feature AND the target column live in this same
        # dict, keyed by column name. The target is NOT a separate top-level
        # key -- there is no raw["target"]. It is
        # raw["features"][params["features"]["target_col"]], which is
        # "trip_count" throughout this project (confirmed via the SQLite
        # demand_history schema and Phase 2 prediction log).
        "<feature_name_or_trip_count>": {
          "dtype": "numeric" | "categorical",
          "count": int, "null_count": int, "null_rate": float,
          # numeric only:
          "mean": float, "std": float, "min": float, "max": float,
          "percentiles": {"<p>": float, ...},   # keys are str(p), e.g. "50", not "p50"
          "histogram": {
            "bin_edges": [float, ...],
            "counts": [int, ...]                # NOT "bin_counts" -- key is "counts"
          },
          # categorical only:
          "category_frequencies": {"<category>": float, ...},
          "n_unique": int
        }, ...
      }
    }

Do not recompute or mutate this artifact from monitoring code. Per the Phase 1
principle: a snapshot computed from the wrong dataset silently invalidates
every downstream drift alert. This module only reads.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureProfile:
    """Frozen training-time profile for a single feature."""

    name: str
    dtype: str  # "numeric" or "categorical"
    null_rate: float
    mean: Optional[float] = None
    std: Optional[float] = None
    percentiles: Optional[dict[str, float]] = None
    bin_edges: Optional[np.ndarray] = None
    bin_counts: Optional[np.ndarray] = None
    category_frequencies: Optional[dict[str, float]] = None

    def reference_probabilities(self) -> np.ndarray:
        """Return the reference histogram as a normalized probability vector.

        For numeric features this is bin_counts / sum(bin_counts). For
        categorical features it's the stored category frequencies, in a
        stable sorted-key order (callers must bucket current data with the
        same category set/order via `category_order`).
        """
        if self.dtype == "numeric":
            if self.bin_counts is None:
                raise ValueError(f"Feature '{self.name}' has no reference histogram.")
            counts = self.bin_counts.astype(float)
            total = counts.sum()
            if total == 0:
                raise ValueError(f"Feature '{self.name}' reference histogram is empty.")
            return counts / total
        if self.category_frequencies is None:
            raise ValueError(f"Feature '{self.name}' has no reference category frequencies.")
        ordered = [self.category_frequencies[k] for k in self.category_order()]
        return np.array(ordered, dtype=float)

    def category_order(self) -> list[str]:
        if self.category_frequencies is None:
            raise ValueError(f"Feature '{self.name}' is not categorical.")
        return sorted(self.category_frequencies.keys())


@dataclass(frozen=True)
class ReferenceSnapshot:
    """Typed, read-only view over the frozen Phase 1 reference artifact."""

    path: Path
    n_rows: int
    training_months: list[str]
    n_zones: int
    features: dict[str, FeatureProfile]
    target: FeatureProfile

    @classmethod
    def from_json(cls, path: str | Path, target_col: str = "trip_count") -> "ReferenceSnapshot":
        """Load and parse the snapshot. `target_col` must match
        params["features"]["target_col"] used when the snapshot was built --
        defaults to "trip_count", the name used everywhere else in this
        project (demand_history, prediction log), but is a parameter rather
        than a silent hardcode so a mismatch is something the caller can
        override instead of something this loader guesses wrong about.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Reference snapshot not found at {path}. Phase 3 must not proceed "
                f"without the frozen Phase 1 baseline -- do not recompute it here."
            )
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        raw_features = dict(raw["features"])  # copy: we pop the target out below
        if target_col not in raw_features:
            raise KeyError(
                f"Target column '{target_col}' not found among snapshot features "
                f"({sorted(raw_features.keys())}). The target profile is stored "
                f"inside 'features', not as a separate top-level key -- pass the "
                f"correct target_col if it isn't 'trip_count'."
            )
        target_raw = raw_features.pop(target_col)
        target_profile = cls._parse_profile(target_col, target_raw)

        features: dict[str, FeatureProfile] = {
            name: cls._parse_profile(name, profile) for name, profile in raw_features.items()
        }

        snapshot = cls(
            path=path,
            n_rows=raw["metadata"]["n_rows"],
            training_months=raw["metadata"]["training_months"],
            n_zones=raw["metadata"]["n_zones"],
            features=features,
            target=target_profile,
        )
        logger.info(
            "Loaded reference snapshot from %s (n_rows=%d, n_zones=%d, %d features, target='%s')",
            path, snapshot.n_rows, snapshot.n_zones, len(features), target_col,
        )
        return snapshot

    @staticmethod
    def _parse_profile(name: str, profile: dict) -> FeatureProfile:
        dtype = profile.get("dtype", "numeric")
        if dtype == "categorical":
            return FeatureProfile(
                name=name,
                dtype="categorical",
                null_rate=profile.get("null_rate", 0.0),
                category_frequencies=profile["category_frequencies"],
            )
        histogram = profile.get("histogram")
        return FeatureProfile(
            name=name,
            dtype="numeric",
            null_rate=profile.get("null_rate", 0.0),
            mean=profile.get("mean"),
            std=profile.get("std"),
            percentiles=profile.get("percentiles"),
            bin_edges=np.array(histogram["bin_edges"]) if histogram else None,
            # real schema key is "counts", not "bin_counts" -- confirmed against
            # src/training/reference_snapshot.py's _numeric_feature_profile
            bin_counts=np.array(histogram["counts"]) if histogram else None,
        )

    def get_feature(self, name: str) -> FeatureProfile:
        if name not in self.features:
            raise KeyError(
                f"Feature '{name}' not found in reference snapshot. Available: "
                f"{sorted(self.features.keys())}"
            )
        return self.features[name]