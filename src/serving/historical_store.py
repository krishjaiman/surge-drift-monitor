"""
SQLite-backed store of per-(zone, hour) demand history.

This is the feature-serving store: it answers point/range lookups needed to
build lag_1h/24h/168h and rolling_mean_3h/24h at request time. It is distinct
from the Parquet prediction log — this store is mutated as new ground-truth
actuals arrive; the prediction log never is.

Causality note: rolling_mean_Nh(t) is computed over the window (t-N, t), i.e.
it EXCLUDES the current hour t. This mirrors the Phase 1 training-time
definition and avoids the leakage bug class you already fixed once
(features must never see the value they're trying to predict, or its future).
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS demand_history (
    zone_id INTEGER NOT NULL,
    ts TEXT NOT NULL,              -- ISO8601, floored to the hour
    trip_count REAL NOT NULL,
    is_actual INTEGER NOT NULL DEFAULT 1,  -- 1 = observed ground truth, 0 = reserved for future use
    PRIMARY KEY (zone_id, ts)
);
CREATE INDEX IF NOT EXISTS idx_zone_ts ON demand_history (zone_id, ts);
"""


def _floor_hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)


class HistoricalDemandStore:
    """Thread-safe-enough for a single-process FastAPI dev server (WAL mode)."""

    def __init__(self, sqlite_path: str) -> None:
        self.path = Path(sqlite_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def seed_from_dataframe(self, df) -> int:
        """
        Bulk-load historical actuals, e.g. from Phase 1's processed training
        data, so lag/rolling features are available from the first replay hour.

        Expects columns: zone_id, timestamp, trip_count.
        """
        rows = [
            (int(r.zone_id), _floor_hour(r.timestamp).isoformat(), float(r.trip_count), 1)
            for r in df.itertuples(index=False)
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO demand_history (zone_id, ts, trip_count, is_actual) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        logger.info("Seeded historical store with %d rows", len(rows))
        return len(rows)

    def record_actual(self, zone_id: int, timestamp: datetime, trip_count: float) -> None:
        """Backfill ground truth once it's observed (simulated T+1h arrival)."""
        ts = _floor_hour(timestamp).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO demand_history (zone_id, ts, trip_count, is_actual) VALUES (?, ?, ?, 1)",
            (zone_id, ts, trip_count),
        )
        self._conn.commit()

    def _lookup(self, zone_id: int, timestamp: datetime) -> float | None:
        ts = _floor_hour(timestamp).isoformat()
        row = self._conn.execute(
            "SELECT trip_count FROM demand_history WHERE zone_id = ? AND ts = ?",
            (zone_id, ts),
        ).fetchone()
        return row[0] if row else None

    def _rolling_mean(self, zone_id: int, timestamp: datetime, window_hours: int) -> float | None:
        end = _floor_hour(timestamp)  # exclusive
        start = end - timedelta(hours=window_hours)
        row = self._conn.execute(
            """
            SELECT AVG(trip_count), COUNT(*) FROM demand_history
            WHERE zone_id = ? AND ts >= ? AND ts < ?
            """,
            (zone_id, start.isoformat(), end.isoformat()),
        ).fetchone()
        avg, count = row
        if count < window_hours:
            # Incomplete window (e.g. near start of history) -> caller decides
            # whether to skip this zone/hour rather than serve a biased mean.
            return None
        return avg

    def get_lag_and_rolling_features(
        self,
        zone_id: int,
        timestamp: datetime,
        lag_hours: list[int],
        rolling_windows: list[int],
    ) -> dict[str, float] | None:
        """
        Returns None if any required lag is missing -> caller should skip
        this (zone, hour) rather than impute, since a fabricated lag value
        is exactly the kind of silent data-fabrication bug you already hit
        once in Phase 1 grid-fill.
        """
        features: dict[str, float] = {}
        for h in lag_hours:
            val = self._lookup(zone_id, timestamp - timedelta(hours=h))
            if val is None:
                return None
            features[f"lag_{h}h"] = val

        for w in rolling_windows:
            val = self._rolling_mean(zone_id, timestamp, w)
            if val is None:
                return None
            features[f"rolling_mean_{w}h"] = val

        return features

    def known_zone_ids(self) -> list[int]:
        rows = self._conn.execute("SELECT DISTINCT zone_id FROM demand_history").fetchall()
        return sorted(r[0] for r in rows)

    def close(self) -> None:
        self._conn.close()