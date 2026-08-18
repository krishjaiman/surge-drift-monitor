"""Converts the existing Phase 3/4 staging layer (metrics_timeseries.parquet,
retrain_events.jsonl) into an OpenMetrics text file suitable for
`promtool tsdb create-blocks-from openmetrics` -- the documented, correct
way to load historical time-series data into Prometheus (not a live scrape).

Why this exists: Prometheus is designed to scrape a live target, not import
a backfilled history. The backfill tool sidesteps that by writing TSDB
blocks directly from a file, so Grafana can show the full Jul-Dec drift
ramp -> trigger -> retrain story on first launch, not just data collected
after you started the containers.

OpenMetrics format requires each metric FAMILY (name) to be grouped
together, not interleaved by timestamp -- `# TYPE <name> gauge` declared
once, followed by every sample for that name across all timestamps/labels,
then the next family. This module builds that structure explicitly rather
than writing row-by-row, which would produce invalid interleaved families.

Run: python -m src.monitoring.backfill_prometheus
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Column-name prefixes in metrics_timeseries.parquet that carry a per-feature
# label, mapped to the Prometheus metric family name they become.
_LABELED_PSI_PREFIXES = {
    "psi__": "surge_psi",
    "null_rate_ratio__": "surge_null_rate_ratio",
    "corr_drop_pct__": "surge_corr_drop_pct",
}

# Flat (non-prefixed, non-windowed) columns -> metric family name.
_FLAT_COLUMNS = {
    "label_kl": "surge_label_kl",
    "zones_with_bias_alert": "surge_zones_with_bias_alert",
    "zones_with_calibration_alert": "surge_zones_with_calibration_alert",
    "prediction_variance_current": "surge_prediction_variance_current",
    "prediction_variance_breached": "surge_prediction_variance_breached",
    "drift_breached": "surge_drift_breached",
    "performance_degraded": "surge_performance_degraded",
    "retrain_trigger_fired": "surge_retrain_trigger_fired",
}

# Windowed columns (rmse_24h, mae_168h, n_preds_720h, ...) -> family name,
# with the window (e.g. "24h") captured as a label.
_WINDOWED_PREFIXES = {
    "rmse_": "surge_rmse",
    "mae_": "surge_mae",
    "n_preds_": "surge_n_preds",
}


def _to_epoch_seconds(iso_ts: str) -> int:
    return int(pd.Timestamp(iso_ts).timestamp())


def _add_sample(families: dict, name: str, labels: dict, value: float, ts_epoch: int) -> None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return  # OpenMetrics can represent NaN, but skipping is simpler and
                 # avoids ambiguity -- these are expected early-window gaps
                 # (not enough lookback yet), not real "unknown" readings.
    families[name].append((labels, float(value), ts_epoch))


def _bool_to_float(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return 1.0 if bool(value) else 0.0


def build_metric_families_from_timeseries(df: pd.DataFrame) -> dict:
    """Returns {metric_family_name: [(labels_dict, value, epoch_seconds), ...]}."""
    families: dict[str, list] = defaultdict(list)

    for _, row in df.iterrows():
        ts_epoch = _to_epoch_seconds(row["as_of"])

        for prefix, family in _LABELED_PSI_PREFIXES.items():
            for col in df.columns:
                if col.startswith(prefix):
                    feature = col[len(prefix):]
                    _add_sample(families, family, {"feature": feature}, row[col], ts_epoch)

        for col, family in _FLAT_COLUMNS.items():
            if col not in df.columns:
                continue
            value = row[col]
            if family.endswith("_breached") or family.endswith("_degraded") or family.endswith("_fired"):
                value = _bool_to_float(value)
            _add_sample(families, family, {}, value, ts_epoch)

        for prefix, family in _WINDOWED_PREFIXES.items():
            for col in df.columns:
                if col.startswith(prefix):
                    window = col[len(prefix):]  # e.g. "24h", "168h", "720h"
                    _add_sample(families, family, {"window": window}, row[col], ts_epoch)

    return families


def build_metric_families_from_retrain_events(events_path: str | Path) -> dict:
    """Reads retrain_events.jsonl (Phase 4 audit trail) into the same
    families structure. Skipped entirely (returns {}) if the file doesn't
    exist -- not every backtest run includes a Phase 4 pass.
    """
    events_path = Path(events_path)
    families: dict[str, list] = defaultdict(list)
    if not events_path.exists():
        logger.info("No retrain_events.jsonl found at %s -- skipping retrain metrics.", events_path)
        return families

    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ts_epoch = _to_epoch_seconds(record["as_of"])
            _add_sample(families, "surge_retrain_attempted", {}, 1.0, ts_epoch)
            _add_sample(families, "surge_retrain_promoted", {},
                        _bool_to_float(record.get("promoted")), ts_epoch)
            if record.get("champion_rmse") is not None:
                _add_sample(families, "surge_retrain_champion_rmse", {}, record["champion_rmse"], ts_epoch)
            if record.get("challenger_rmse") is not None:
                _add_sample(families, "surge_retrain_challenger_rmse", {}, record["challenger_rmse"], ts_epoch)
            if record.get("improvement_pct") is not None:
                _add_sample(families, "surge_retrain_improvement_pct", {}, record["improvement_pct"], ts_epoch)

    return families


def write_openmetrics_file(families: dict, output_path: str | Path) -> None:
    """Serialize the families dict to a valid OpenMetrics text file.

    OpenMetrics requires each unique SERIES (metric name + exact label set)
    to be fully contiguous -- not just grouped by metric name. Interleaving
    e.g. psi{feature="temperature_2m"} and psi{feature="lag_1h"} samples by
    timestamp within the same family is invalid and rejected by strict
    parsers (confirmed against prometheus_client's real OpenMetrics parser,
    which raised "Invalid metric grouping" on an earlier version of this
    function that sorted only by timestamp). Each family is grouped by
    label-set first, each label-set's samples then sorted by time.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # newline="\n" is required, not optional, on Windows: Python's default
    # text-mode write silently converts \n to \r\n, and promtool (running
    # inside a Linux container) rejects CRLF-terminated OpenMetrics lines
    # with a cryptic "invalid metric type" error pointing at the trailing
    # \r rather than the real cause. Force LF explicitly regardless of
    # platform.
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for name in sorted(families.keys()):
            samples = families[name]
            if not samples:
                continue
            f.write(f"# TYPE {name} gauge\n")

            # Group by label-set (as a sorted tuple, so identical label
            # dicts always hash/group the same way regardless of insertion
            # order), preserving each series as one contiguous block.
            by_labels: dict[tuple, list] = defaultdict(list)
            for labels, value, ts_epoch in samples:
                key = tuple(sorted(labels.items()))
                by_labels[key].append((value, ts_epoch))

            for label_key in sorted(by_labels.keys()):
                label_str = ",".join(f'{k}="{v}"' for k, v in label_key)
                label_part = f"{{{label_str}}}" if label_str else ""
                for value, ts_epoch in sorted(by_labels[label_key], key=lambda s: s[1]):
                    f.write(f"{name}{label_part} {value} {ts_epoch}\n")
        f.write("# EOF\n")

    logger.info("Wrote OpenMetrics backfill file to %s (%d metric families).",
                output_path, len(families))


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate OpenMetrics backfill file.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--output", default="docker/prometheus/backfill.openmetrics")
    args = parser.parse_args()

    with open(args.params, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    mon = params["monitoring"]
    retrain_cfg = params["retraining"]

    df = pd.read_parquet(mon["outputs"]["metrics_timeseries_path"])
    families = build_metric_families_from_timeseries(df)

    retrain_families = build_metric_families_from_retrain_events(
        retrain_cfg["outputs"]["retrain_events_path"]
    )
    for name, samples in retrain_families.items():
        families[name].extend(samples)

    write_openmetrics_file(families, args.output)