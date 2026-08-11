"""
Buffered, partitioned Parquet writer for the prediction log -- the raw
material for all of Phase 3. Logs are append-only and immutable; ground
truth backfill goes into HistoricalDemandStore instead, and is joined at
analysis time on (zone_id, timestamp).

Partitioning by date keeps Phase 3's daily PSI job cheap (reads only
today's/yesterday's partition, not the whole log).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class PredictionLogger:
    def __init__(self, output_dir: str, flush_every_n: int, flush_interval_s: int) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every_n = flush_every_n
        self.flush_interval_s = flush_interval_s

        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def log(
        self,
        zone_id: int,
        timestamp: datetime,
        predicted_demand: float,
        model_name: str,
        model_version: str,
        features: dict,
    ) -> None:
        record = {
            "zone_id": zone_id,
            "timestamp": timestamp,
            "predicted_demand": predicted_demand,
            "model_name": model_name,
            "model_version": model_version,
            "logged_at": datetime.utcnow(),
            **{f"feat_{k}": v for k, v in features.items()},
        }
        with self._lock:
            self._buffer.append(record)
            should_flush = len(self._buffer) >= self.flush_every_n
        if should_flush:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            batch = self._buffer
            self._buffer = []

        df = pd.DataFrame(batch)
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
        try:
            df.to_parquet(
                self.output_dir,
                partition_cols=["date"],
                engine="pyarrow",
                existing_data_behavior="overwrite_or_ignore",
            )
            logger.info("Flushed %d prediction records to %s", len(df), self.output_dir)
        except Exception:
            # Don't lose data on a write failure - put it back in the buffer.
            with self._lock:
                self._buffer = batch + self._buffer
            logger.exception("Failed to flush prediction log; %d records requeued", len(batch))

    def _flush_loop(self) -> None:
        while not self._stop.wait(self.flush_interval_s):
            self.flush()

    def shutdown(self) -> None:
        self._stop.set()
        self.flush()