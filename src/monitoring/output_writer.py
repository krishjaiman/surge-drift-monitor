"""Writes monitoring output to the local staging layer.

Chosen format (per the earlier design decision): Parquet time series for
metrics (queryable, Prometheus-exporter-friendly to backfill from later) and
JSONL for alerts (append-only, one record per line, trivial to tail during
Docker install / Prometheus wiring later without needing a Parquet reader).

Both writers are append-safe across repeated runs: metrics dedupe on
`as_of`, alerts simply accumulate (an alert record is an event log, not a
current-state table).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.monitoring.alerting import AlertSummary

logger = logging.getLogger(__name__)


def write_metrics_row(metrics_row: dict, output_path: str | Path) -> None:
    """Append one as-of evaluation's flattened metrics to the Parquet time
    series. Reads-modifies-writes the full file -- fine at this data volume
    (hourly rows over a 6-month backtest is ~4,400 rows); revisit with a
    partitioned write if this becomes a bottleneck at true production scale.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([metrics_row])
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        existing = existing[existing["as_of"] != metrics_row["as_of"]]  # dedupe on rerun
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined = combined.sort_values("as_of").reset_index(drop=True)
    combined.to_parquet(output_path, index=False)


def append_alert(alert: AlertSummary, output_path: str | Path) -> None:
    """Append one alert record as a line of JSON. Every evaluation writes a
    record here (not just breaches) so the log is a complete audit trail for
    computing TTD later, not just a list of fires.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(alert.to_dict()) + "\n")
    if alert.retrain_trigger_fired:
        logger.warning("RETRAIN TRIGGER FIRED at %s: %s", alert.as_of, alert.breach_details)