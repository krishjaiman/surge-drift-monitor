"""FastAPI serving app for the surge-demand forecaster (Phase 2)."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import yaml
from fastapi import FastAPI, HTTPException

from src.serving.feature_builder import FeatureBuilder
from src.serving.historical_store import HistoricalDemandStore
from src.serving.model_loader import ChampionModelLoader
from src.serving.prediction_logger import PredictionLogger
from src.serving.schemas import (
    ActualDemandRecord, BatchPredictionRequest, BatchPredictionResponse,
    ModelInfoResponse, PredictionResponse, SinglePredictionRequest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("params.yaml") as f:
    CONFIG = yaml.safe_load(f)          # full config
    PARAMS = CONFIG["serving"]           # serving-specific subset, for convenience

store: HistoricalDemandStore
loader: ChampionModelLoader
feature_builder: FeatureBuilder
pred_logger: PredictionLogger


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, loader, feature_builder, pred_logger

    store = HistoricalDemandStore(PARAMS["historical_store"]["sqlite_path"])
    pred_logger = PredictionLogger(
        output_dir=PARAMS["prediction_log"]["output_dir"],
        flush_every_n=PARAMS["prediction_log"]["flush_every_n_records"],
        flush_interval_s=PARAMS["prediction_log"]["flush_interval_seconds"],
    )
    loader = ChampionModelLoader(
    tracking_uri=CONFIG["mlflow"]["tracking_uri"],
    model_name=CONFIG["mlflow"]["registered_model_name"],
    alias=CONFIG["mlflow"]["champion_alias"],
    refresh_interval_s=CONFIG["serving"]["model_refresh_interval_seconds"],
    )
    loader.load_sync()
    await loader.start_background_refresh()

    feature_builder = FeatureBuilder(
    store=store,
    weather_cache_path=CONFIG["serving"]["weather_cache_path"],
    lag_hours=CONFIG["features"]["lag_features"]["lag_hours"],
    rolling_windows=CONFIG["features"]["rolling_features"]["windows_hours"],
    categorical_features=CONFIG["features"]["categorical_features"],
    )

    yield

    loader.stop_background_refresh()
    pred_logger.shutdown()
    store.close()


app = FastAPI(title="Surge Demand Forecaster", lifespan=lifespan)


def _predict_one(zone_id: int, timestamp: datetime) -> PredictionResponse | None:
    features_df = feature_builder.build(zone_id, timestamp)
    if features_df is None:
        return None
    pred = float(loader.model.predict(features_df)[0])
    pred_logger.log(
        zone_id=zone_id,
        timestamp=timestamp,
        predicted_demand=pred,
        model_name=loader.model_name,
        model_version=loader.version,
        features=features_df.iloc[0].to_dict(),
    )
    return PredictionResponse(
        zone_id=zone_id, timestamp=timestamp, predicted_demand=pred,
        model_name=loader.model_name, model_version=loader.version, logged=True,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_version": loader.version}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        model_name=loader.model_name, alias=loader.alias,
        version=loader.version, loaded_at=loader.loaded_at,
    )


@app.post("/admin/reload-model")
def reload_model():
    reloaded = loader.check_and_reload_if_stale()
    return {"reloaded": reloaded, "version": loader.version}


@app.post("/predict", response_model=PredictionResponse)
def predict_single(req: SinglePredictionRequest):
    result = _predict_one(req.zone_id, req.timestamp)
    if result is None:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient history/weather to featurize zone {req.zone_id} at {req.timestamp}",
        )
    return result


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchPredictionRequest):
    zone_ids = req.zone_ids or store.known_zone_ids()
    predictions: list[PredictionResponse] = []
    skipped = 0
    for zone_id in zone_ids:
        result = _predict_one(zone_id, req.timestamp)
        if result is None:
            skipped += 1
            continue
        predictions.append(result)

    return BatchPredictionResponse(
        timestamp=req.timestamp,
        model_name=loader.model_name,
        model_version=loader.version,
        predictions=predictions,
        n_zones_scored=len(predictions),
        n_zones_skipped=skipped,
    )


@app.post("/actuals")
def record_actual(record: ActualDemandRecord):
    """Backfills observed ground truth into the historical store so
    subsequent lag features see it. In production this would be triggered
    by an hourly job once actual trip counts settle; in your replay driver
    it's called right after you 'observe' the next hour of historical data."""
    store.record_actual(record.zone_id, record.timestamp, record.actual_trip_count)
    return {"status": "recorded"}