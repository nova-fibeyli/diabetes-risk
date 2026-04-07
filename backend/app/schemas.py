from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.feature_schema import DISCLAIMER


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    model_ready: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    trained_at: str | None
    target_column: str
    feature_columns: list[str]
    metrics: dict[str, float] = Field(default_factory=dict)
    all_model_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    available_models: list[str] = Field(default_factory=list)
    dataset_rows: int | None = None
    disclaimer: str = DISCLAIMER


class PredictionInput(BaseModel):
    age: int | None = Field(default=None, ge=18, le=120)
    sex: Literal["female", "male", "other"] | None = None
    height_cm: float | None = Field(default=None, ge=100, le=250)
    weight_kg: float | None = Field(default=None, ge=20, le=350)
    bmi: float | None = Field(default=None, ge=10, le=80)
    systolic_bp: int | None = Field(default=None, ge=70, le=250)
    diastolic_bp: int | None = Field(default=None, ge=40, le=150)
    heart_rate: int | None = Field(default=None, ge=30, le=220)
    fasting_glucose_mg_dl: float | None = Field(default=None, ge=40, le=500)
    fasting_glucose_mmol_l: float | None = Field(default=None, ge=2, le=30)
    hba1c: float | None = Field(default=None, ge=3, le=20)
    insulin: float | None = Field(default=None, ge=0, le=1000)
    homa_ir: float | None = Field(default=None, ge=0, le=30)
    cholesterol_total: float | None = Field(default=None, ge=50, le=500)
    physical_activity_days_per_week: int | None = Field(default=None, ge=0, le=7)
    smoking: bool | None = None
    family_history_diabetes: bool | None = None
    hypertension: bool | None = None
    cardiovascular_disease: bool | None = None
    stroke_history: bool | None = None
    pregnancy_count: int | None = Field(default=None, ge=0, le=20)
    notes: str | None = None

    model_config = ConfigDict(extra="ignore")


class PredictionRequest(BaseModel):
    input: PredictionInput
    model_name: str | None = None


class BatchPredictionRequest(BaseModel):
    inputs: list[PredictionInput]
    model_name: str | None = None


class NormalizedFeatures(BaseModel):
    age: int | None = None
    sex: str | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    bmi: float | None = None
    systolic_bp: int | None = None
    diastolic_bp: int | None = None
    heart_rate: int | None = None
    fasting_glucose_mg_dl: float | None = None
    fasting_glucose_mmol_l: float | None = None
    hba1c: float | None = None
    insulin: float | None = None
    homa_ir: float | None = None
    cholesterol_total: float | None = None
    physical_activity_days_per_week: int | None = None
    smoking: bool | None = None
    family_history_diabetes: bool | None = None
    hypertension: bool | None = None
    cardiovascular_disease: bool | None = None
    stroke_history: bool | None = None
    pregnancy_count: int | None = None
    derived_model_features: dict[str, float | int] = Field(default_factory=dict)


class RecommendationItem(BaseModel):
    title: str
    rationale: str
    priority: Literal["high", "medium", "low"]
    category: Literal["lifestyle", "monitoring", "laboratory", "medical_follow_up"]


class PredictionResponse(BaseModel):
    model_name: str
    risk_probability: float
    risk_percent: float
    risk_band: str
    prediction_confidence: float
    explanation: str
    key_factors: list[str]
    missing_required_fields: list[str]
    normalized_input: NormalizedFeatures
    model_metrics: dict[str, float] = Field(default_factory=dict)
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    saved_prediction_id: int | None = None
    disclaimer: str = DISCLAIMER


class BatchPredictionItem(BaseModel):
    index: int
    result: PredictionResponse


class BatchPredictionResponse(BaseModel):
    items: list[BatchPredictionItem]
    disclaimer: str = DISCLAIMER


class ParsedField(BaseModel):
    key: str
    label: str
    value: float | str | None
    confidence: float = 0.0
    source_text: str | None = None
    required_for_prediction: bool = False


class ParsePdfResponse(BaseModel):
    filename: str
    extracted_text_preview: str
    extracted_fields: list[ParsedField]
    missing_required_fields: list[str]
    uploaded_report_id: int | None = None
    disclaimer: str = DISCLAIMER


class FeatureSchemaResponse(BaseModel):
    fields: list[dict[str, Any]]
    disclaimer: str = DISCLAIMER


class MetricsResponse(BaseModel):
    prediction_requests: int
    batch_requests: int
    pdf_parse_requests: int
    model_metrics: dict[str, float] = Field(default_factory=dict)
    model_name: str


class AuthConfigResponse(BaseModel):
    configured: bool
    login_url: str


class UserProfileResponse(BaseModel):
    id: int
    email: str
    full_name: str | None = None
    avatar_url: str | None = None
    joined_at: datetime
    last_login_at: datetime
    is_admin: bool = False


class HistoryItemResponse(BaseModel):
    id: int
    created_at: datetime
    risk_percent: float
    risk_band: str
    prediction_confidence: float
    explanation: str
    key_metrics: dict[str, Any]


class HistoryResponse(BaseModel):
    items: list[HistoryItemResponse]


class ProfileResponse(BaseModel):
    user: UserProfileResponse
    recent_predictions: list[HistoryItemResponse]
    disclaimer: str = DISCLAIMER


class AdminUserSummary(BaseModel):
    id: int
    email: str
    full_name: str | None = None
    avatar_url: str | None = None
    joined_at: datetime
    last_login_at: datetime
    prediction_count: int
    report_count: int
    average_risk_percent: float | None = None
    average_confidence: float | None = None
    recent_predictions: list[HistoryItemResponse] = Field(default_factory=list)


class AdminOverviewResponse(BaseModel):
    users: list[AdminUserSummary]
    generated_at: datetime
