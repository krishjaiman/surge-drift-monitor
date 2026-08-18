"""Tests for backfill_prometheus.py. Every generated file is validated
against prometheus_client's real OpenMetrics parser, not just checked by
eye -- this is what caught a real bug during development (samples for
different label sets interleaved by timestamp within a family, which is
invalid per the OpenMetrics spec even though it looks reasonable at a
glance).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from prometheus_client.openmetrics.parser import text_string_to_metric_families

from src.monitoring.backfill_prometheus import (
    build_metric_families_from_retrain_events,
    build_metric_families_from_timeseries,
    write_openmetrics_file,
)


def _parse(path) -> dict:
    """Parse an OpenMetrics file and return {family_name: [samples]}."""
    with open(path) as f:
        text = f.read()
    families = list(text_string_to_metric_families(text))
    return {fam.name: fam.samples for fam in families}


def test_multi_feature_psi_produces_valid_openmetrics(tmp_path):
    """The exact bug found during development: interleaving different
    labeled series within one family by timestamp is invalid OpenMetrics.
    """
    rows = []
    start = pd.Timestamp("2024-07-01")
    for h in range(5):
        rows.append({
            "as_of": (start + pd.Timedelta(hours=h)).isoformat(),
            "psi__temperature_2m": 0.1 + h * 0.01,
            "psi__lag_1h": 0.05,
        })
    df = pd.DataFrame(rows)
    families = build_metric_families_from_timeseries(df)
    out_path = tmp_path / "backfill.openmetrics"
    write_openmetrics_file(families, out_path)

    parsed = _parse(out_path)  # raises if invalid -- the actual assertion
    assert "surge_psi" in parsed
    assert len(parsed["surge_psi"]) == 10  # 2 features x 5 timestamps


def test_nan_values_are_skipped_not_written_as_invalid_samples(tmp_path):
    rows = []
    start = pd.Timestamp("2024-07-01")
    for h in range(5):
        rows.append({
            "as_of": (start + pd.Timedelta(hours=h)).isoformat(),
            "rmse_24h": 7.5 if h > 1 else np.nan,
        })
    df = pd.DataFrame(rows)
    families = build_metric_families_from_timeseries(df)
    out_path = tmp_path / "backfill.openmetrics"
    write_openmetrics_file(families, out_path)

    parsed = _parse(out_path)
    assert len(parsed["surge_rmse"]) == 3  # only the 3 non-NaN rows


def test_boolean_columns_convert_to_zero_one(tmp_path):
    rows = [
        {"as_of": "2024-07-01T00:00:00", "drift_breached": True, "retrain_trigger_fired": False},
        {"as_of": "2024-07-01T01:00:00", "drift_breached": False, "retrain_trigger_fired": True},
    ]
    df = pd.DataFrame(rows)
    families = build_metric_families_from_timeseries(df)
    out_path = tmp_path / "backfill.openmetrics"
    write_openmetrics_file(families, out_path)

    parsed = _parse(out_path)
    drift_values = sorted(s.value for s in parsed["surge_drift_breached"])
    assert drift_values == [0.0, 1.0]


def test_windowed_columns_get_window_label(tmp_path):
    rows = [{
        "as_of": "2024-07-01T00:00:00",
        "rmse_24h": 7.0, "rmse_168h": 7.5, "rmse_720h": 8.0,
    }]
    df = pd.DataFrame(rows)
    families = build_metric_families_from_timeseries(df)
    out_path = tmp_path / "backfill.openmetrics"
    write_openmetrics_file(families, out_path)

    parsed = _parse(out_path)
    windows = {s.labels["window"] for s in parsed["surge_rmse"]}
    assert windows == {"24h", "168h", "720h"}


def test_retrain_events_produce_valid_openmetrics(tmp_path):
    events = [
        {"as_of": "2024-10-06T03:00:00", "attempted": True, "promoted": False,
         "champion_rmse": 10.44, "challenger_rmse": 11.6, "improvement_pct": -0.11},
        {"as_of": "2024-10-21T14:00:00", "attempted": True, "promoted": True,
         "champion_rmse": 12.1, "challenger_rmse": 9.8, "improvement_pct": 0.19},
    ]
    events_path = tmp_path / "retrain_events.jsonl"
    with events_path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    families = build_metric_families_from_retrain_events(events_path)
    out_path = tmp_path / "backfill.openmetrics"
    write_openmetrics_file(families, out_path)

    parsed = _parse(out_path)  # raises if invalid
    assert len(parsed["surge_retrain_attempted"]) == 2
    promoted_values = sorted(s.value for s in parsed["surge_retrain_promoted"])
    assert promoted_values == [0.0, 1.0]


def test_missing_retrain_events_file_returns_empty(tmp_path):
    families = build_metric_families_from_retrain_events(tmp_path / "does_not_exist.jsonl")
    assert families == {}


def test_empty_families_produces_eof_only_file(tmp_path):
    out_path = tmp_path / "backfill.openmetrics"
    write_openmetrics_file({}, out_path)
    content = out_path.read_text()
    assert content.strip() == "# EOF"


def test_output_file_uses_unix_line_endings_not_crlf(tmp_path):
    # Real bug found on Windows: Python's default text-mode write silently
    # converts \n to \r\n there, and promtool (Linux container) rejects
    # CRLF-terminated lines with a misleading "invalid metric type" error.
    # Read the file in BINARY mode to check for literal \r\n -- reading in
    # text mode would itself normalize line endings and hide the bug.
    rows = [{"as_of": "2024-07-01T00:00:00", "label_kl": 0.02}]
    df = pd.DataFrame(rows)
    families = build_metric_families_from_timeseries(df)
    out_path = tmp_path / "backfill.openmetrics"
    write_openmetrics_file(families, out_path)

    raw_bytes = out_path.read_bytes()
    assert b"\r\n" not in raw_bytes
    assert b"\n" in raw_bytes