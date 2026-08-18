"""Reads the LATEST row of metrics_timeseries.parquet and latest
alerts_log.jsonl record, exposes them as Prometheus gauges on /metrics.

This is the LIVE path -- it always reflects whatever scheduled_run.py (or
backtest_runner.py, if still running) most recently wrote. Historical data
is loaded separately via the one-off backfill procedure BEFORE Prometheus
first starts (see docker/README.md); this exporter has no memory of
history, only "what does the staging layer say right now."

Metric names deliberately match backfill_prometheus.py's naming exactly
(surge_psi, surge_rmse, etc.) so a Grafana panel querying `surge_psi` works
identically whether the data point came from the historical backfill or a
live scrape -- one continuous query surface, not two.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
from prometheus_client import Gauge, start_http_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METRICS_PARQUET_PATH = Path(os.environ.get("METRICS_PARQUET_PATH", "/data/monitoring/metrics_timeseries.parquet"))
ALERTS_JSONL_PATH = Path(os.environ.get("ALERTS_JSONL_PATH", "/data/monitoring/alerts_log.jsonl"))
POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "30"))

_LABELED_PSI_PREFIXES = {
    "psi__": "surge_psi",
    "null_rate_ratio__": "surge_null_rate_ratio",
    "corr_drop_pct__": "surge_corr_drop_pct",
}
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
_WINDOWED_PREFIXES = {
    "rmse_": "surge_rmse",
    "mae_": "surge_mae",
    "n_preds_": "surge_n_preds",
}

# Gauge objects, created lazily on first use since we don't know feature
# names / window values until we see real data. Keyed by (family, label_names_tuple).
_gauges: dict[tuple, Gauge] = {}


def _get_gauge(family: str, label_names: tuple[str, ...]) -> Gauge:
    key = (family, label_names)
    if key not in _gauges:
        _gauges[key] = Gauge(family, f"{family} (surge-drift-monitor)", list(label_names))
        logger.info("Registered new gauge: %s%s", family, label_names)
    return _gauges[key]


def _set_flat(family: str, value: float) -> None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return
    _get_gauge(family, ()).set(float(value))


def _set_labeled(family: str, label_name: str, label_value: str, value: float) -> None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return
    _get_gauge(family, (label_name,)).labels(**{label_name: label_value}).set(float(value))


def _bool_to_float(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return 1.0 if bool(value) else 0.0


def update_from_latest_metrics_row() -> None:
    if not METRICS_PARQUET_PATH.exists():
        logger.warning("Metrics file not found yet: %s", METRICS_PARQUET_PATH)
        return
    try:
        df = pd.read_parquet(METRICS_PARQUET_PATH)
    except Exception as exc:  # noqa: BLE001 -- a mid-write read is expected occasionally
        logger.warning("Could not read metrics parquet (likely mid-write, will retry): %s", exc)
        return
    if df.empty:
        return

    latest = df.sort_values("as_of").iloc[-1]

    for prefix, family in _LABELED_PSI_PREFIXES.items():
        for col in df.columns:
            if col.startswith(prefix):
                feature = col[len(prefix):]
                _set_labeled(family, "feature", feature, latest[col])

    for col, family in _FLAT_COLUMNS.items():
        if col not in df.columns:
            continue
        value = latest[col]
        if family.endswith(("_breached", "_degraded", "_fired")):
            value = _bool_to_float(value)
        _set_flat(family, value)

    for prefix, family in _WINDOWED_PREFIXES.items():
        for col in df.columns:
            if col.startswith(prefix):
                window = col[len(prefix):]
                _set_labeled(family, "window", window, latest[col])

    logger.info("Updated gauges from latest row: as_of=%s", latest["as_of"])


def update_from_latest_alert() -> None:
    if not ALERTS_JSONL_PATH.exists():
        return
    try:
        with ALERTS_JSONL_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read alerts log (likely mid-write, will retry): %s", exc)
        return
    if not lines:
        return

    latest = json.loads(lines[-1])
    _set_flat("surge_drift_breached", _bool_to_float(latest.get("drift_breached")))
    _set_flat("surge_performance_degraded", _bool_to_float(latest.get("performance_degraded")))
    _set_flat("surge_retrain_trigger_fired", _bool_to_float(latest.get("retrain_trigger_fired")))


def main() -> None:
    start_http_server(8000)
    logger.info("Exporter listening on :8000/metrics, polling every %ds", POLL_INTERVAL_SECONDS)
    while True:
        update_from_latest_metrics_row()
        update_from_latest_alert()
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()