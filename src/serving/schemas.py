"""Pydantic request/response schemas for the demand-forecast serving API."""
from datetime import datetime
from pydantic import BaseModel, Field


class SinglePredictionRequest(BaseModel):
    zone_id: int = Field(..., ge=1, description="NYC TLC taxi zone ID")
    timestamp: datetime = Field(..., description="Hour to predict demand for (will be floored to the hour)")


class PredictionResponse(BaseModel):
    zone_id: int
    timestamp: datetime
    predicted_demand: float
    model_name: str
    model_version: str
    logged: bool


class BatchPredictionRequest(BaseModel):
    timestamp: datetime = Field(..., description="Hour to score all zones for")
    zone_ids: list[int] | None = Field(
        default=None,
        description="Optional subset of zones. If omitted, scores all zones known to the historical store.",
    )


class BatchPredictionResponse(BaseModel):
    timestamp: datetime
    model_name: str
    model_version: str
    predictions: list[PredictionResponse]
    n_zones_scored: int
    n_zones_skipped: int  # e.g. insufficient history for lag_168h


class ActualDemandRecord(BaseModel):
    """Used to backfill ground truth into the historical store (T+1h arrival)."""
    zone_id: int
    timestamp: datetime
    actual_trip_count: float


class ModelInfoResponse(BaseModel):
    model_name: str
    alias: str
    version: str
    loaded_at: datetime