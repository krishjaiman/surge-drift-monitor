"""
MLflow @champion loader with background version-refresh.

Serving must pick up a new champion the moment Phase 4 swaps the alias,
without a service restart -- this is directly on your TTR clock. We keep
the loaded pyfunc model in memory and periodically compare the registered
version behind the alias against what's cached.

Reading: MLflow model aliases (replaces the old stage-based API):
https://mlflow.org/docs/latest/model-registry.html#model-registry-workflows
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import mlflow
from mlflow import MlflowClient
from mlflow.pyfunc import PyFuncModel

logger = logging.getLogger(__name__)


class ChampionModelLoader:
    def __init__(self, tracking_uri: str, model_name: str, alias: str, refresh_interval_s: int) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._model_name = model_name
        self._alias = alias
        self._refresh_interval_s = refresh_interval_s

        self._model: PyFuncModel | None = None
        self._version: str | None = None
        self._loaded_at: datetime | None = None
        self._refresh_task: asyncio.Task | None = None

    def load_sync(self) -> None:
        version_info = self._client.get_model_version_by_alias(self._model_name, self._alias)
        model_uri = f"models:/{self._model_name}@{self._alias}"
        logger.info("Loading model %s (version %s)", model_uri, version_info.version)
        self._model = mlflow.pyfunc.load_model(model_uri)
        self._version = version_info.version
        self._loaded_at = datetime.now(timezone.utc)

    def check_and_reload_if_stale(self) -> bool:
        """Returns True if a reload happened."""
        version_info = self._client.get_model_version_by_alias(self._model_name, self._alias)
        if version_info.version != self._version:
            logger.info(
                "Champion version changed (%s -> %s) - reloading",
                self._version, version_info.version,
            )
            self.load_sync()
            return True
        return False

    async def start_background_refresh(self) -> None:
        async def _loop():
            while True:
                await asyncio.sleep(self._refresh_interval_s)
                try:
                    await asyncio.to_thread(self.check_and_reload_if_stale)
                except Exception:
                    logger.exception("Champion refresh check failed; keeping current model in serving")

        self._refresh_task = asyncio.create_task(_loop())

    def stop_background_refresh(self) -> None:
        if self._refresh_task:
            self._refresh_task.cancel()

    @property
    def model(self) -> PyFuncModel:
        if self._model is None:
            raise RuntimeError("Model not loaded yet - call load_sync() first")
        return self._model

    @property
    def version(self) -> str:
        return self._version or "unknown"

    @property
    def loaded_at(self) -> datetime:
        return self._loaded_at or datetime.now(timezone.utc)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def alias(self) -> str:
        return self._alias