import datetime as dt
from typing import Optional, Any

from pydantic import BaseModel


# ---------- Projects ----------
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    created_at: dt.datetime

    class Config:
        from_attributes = True


# ---------- Datasets ----------
class DatasetOut(BaseModel):
    id: int
    filename: str
    n_rows: Optional[int]
    n_cols: Optional[int]
    target_column: Optional[str]
    date_column: Optional[str]
    version: int
    uploaded_at: dt.datetime

    class Config:
        from_attributes = True


class DatasetProfile(BaseModel):
    n_rows: int
    n_cols: int
    columns: list
    dtypes: dict
    null_counts: dict
    duplicate_rows: int
    numeric_summary: dict
    detected_date_columns: list
    suggested_target: Optional[str]


class SetTargetRequest(BaseModel):
    target_column: Optional[str] = None
    date_column: Optional[str] = None


# ---------- Recommendation ----------
class RecommendationOut(BaseModel):
    workflow_type: str
    reason: str
    confidence: float
    candidate_models: list[str]
    suggested_metric: str


# ---------- Experiments ----------
class ExperimentCreate(BaseModel):
    dataset_id: int
    workflow_type: Optional[str] = None  # if None, uses recommendation
    hyperparameter_tuning: Optional[bool] = False
    custom_hyperparameters: Optional[Any] = None
    business_objective: Optional[str] = None


class ExperimentOut(BaseModel):
    id: int
    workflow_type: str
    status: str
    best_model_name: Optional[str]
    metrics_json: Optional[Any]
    recommendation_reason: Optional[str]
    confidence: Optional[float]
    error: Optional[str]
    model_version: Optional[int]
    model_status: Optional[str]
    pipeline_logs: Optional[str]
    pipeline_progress: Optional[str]
    leaderboard_json: Optional[Any]
    created_at: dt.datetime
    completed_at: Optional[dt.datetime]
    hyperparameter_tuning: Optional[bool] = False
    custom_hyperparameters: Optional[Any] = None
    business_objective: Optional[str] = None
    model_path: Optional[str] = None

    class Config:
        from_attributes = True


class DriftRequest(BaseModel):
    target_dataset_id: int


class ForecastMonitoringRequest(BaseModel):
    target_dataset_id: int


class PredictRequest(BaseModel):
    records: list[dict]
