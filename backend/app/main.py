import logging
from datetime import UTC, datetime
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse

from app.auth import (
    create_access_token,
    get_current_admin,
    get_current_user,
    google_auth_configured,
    oauth,
    settings,
    user_is_admin,
    upsert_google_user,
)
from app.db import get_db, init_db
from app.feature_schema import FEATURE_SCHEMA
from app.ml.inference import ModelService
from app.models import Prediction, UploadedReport, User
from app.schemas import (
    AuthConfigResponse,
    AdminOverviewResponse,
    AdminUserSummary,
    BatchPredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeatureSchemaResponse,
    HealthResponse,
    HistoryItemResponse,
    HistoryResponse,
    MetricsResponse,
    ModelInfoResponse,
    ParsePdfResponse,
    PredictionRequest,
    PredictionResponse,
    ProfileResponse,
    UserProfileResponse,
)
from app.services.metrics import runtime_metrics
from app.services.pdf_parser import parse_lab_report_text, read_pdf_text


model_service = ModelService(settings=settings)
logger = logging.getLogger(__name__)


def _history_item(record: Prediction) -> HistoryItemResponse:
    normalized = record.normalized_payload or {}
    return HistoryItemResponse(
        id=record.id,
        created_at=record.created_at,
        risk_percent=record.risk_percent,
        risk_band=record.risk_band,
        prediction_confidence=record.prediction_confidence,
        explanation=record.explanation,
        key_metrics={
            "model_name": normalized.get("_model_name"),
            "age": normalized.get("age"),
            "bmi": normalized.get("bmi"),
            "fasting_glucose_mg_dl": normalized.get("fasting_glucose_mg_dl"),
            "hba1c": normalized.get("hba1c"),
        },
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    try:
        model_service.load()
    except Exception as exc:
        logger.exception("Model failed to load during startup: %s", exc)
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret_key,
    same_site="lax",
    https_only=settings.auth_cookie_secure,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        model_ready=model_service.ready,
    )


@app.get("/auth/google", response_model=AuthConfigResponse)
def google_auth_info() -> AuthConfigResponse:
    return AuthConfigResponse(
        configured=google_auth_configured(),
        login_url=f"{settings.backend_base_url}/auth/google/login",
    )


@app.get("/auth/google/login")
async def google_login(request: Request):
    if not google_auth_configured():
        raise HTTPException(status_code=503, detail="Google OAuth is not configured.")
    return await oauth.google.authorize_redirect(request, settings.google_redirect_uri)


@app.get("/auth/google/callback")
async def google_callback(request: Request, db: AsyncSession = Depends(get_db)):
    if not google_auth_configured():
        raise HTTPException(status_code=503, detail="Google OAuth is not configured.")

    token = await oauth.google.authorize_access_token(request)
    userinfo = token.get("userinfo")
    if userinfo is None:
        userinfo = await oauth.google.parse_id_token(request, token)
    user = await upsert_google_user(db, userinfo)
    access_token = create_access_token(user)

    response = RedirectResponse(url=f"{settings.frontend_url.rstrip('/')}/")
    response.set_cookie(
        settings.auth_cookie_name,
        access_token,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="lax",
        max_age=settings.auth_token_expiry_days * 24 * 60 * 60,
    )
    return response


@app.post("/auth/logout")
async def logout() -> Response:
    response = Response(status_code=204)
    response.delete_cookie(settings.auth_cookie_name, samesite="lax")
    return response


@app.get("/profile", response_model=ProfileResponse)
async def profile(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ProfileResponse:
    result = await db.execute(
        select(Prediction)
        .where(Prediction.user_id == current_user.id)
        .order_by(desc(Prediction.created_at))
        .limit(5)
    )
    recent = result.scalars().all()
    return ProfileResponse(
        user=UserProfileResponse(
            id=current_user.id,
            email=current_user.email,
            full_name=current_user.full_name,
            avatar_url=current_user.avatar_url,
            joined_at=current_user.created_at,
            last_login_at=current_user.last_login_at,
            is_admin=user_is_admin(current_user),
        ),
        recent_predictions=[_history_item(item) for item in recent],
    )


@app.get("/history", response_model=HistoryResponse)
async def history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> HistoryResponse:
    result = await db.execute(
        select(Prediction)
        .where(Prediction.user_id == current_user.id)
        .order_by(desc(Prediction.created_at))
        .limit(25)
    )
    items = result.scalars().all()
    return HistoryResponse(items=[_history_item(item) for item in items])


@app.get("/admin/overview", response_model=AdminOverviewResponse)
async def admin_overview(
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> AdminOverviewResponse:
    users_result = await db.execute(select(User).order_by(desc(User.last_login_at)))
    users = users_result.scalars().all()

    predictions_result = await db.execute(select(Prediction).order_by(desc(Prediction.created_at)))
    predictions = predictions_result.scalars().all()

    reports_result = await db.execute(select(UploadedReport))
    reports = reports_result.scalars().all()

    predictions_by_user: dict[int, list[Prediction]] = {}
    for item in predictions:
        predictions_by_user.setdefault(item.user_id, []).append(item)

    report_count_by_user: dict[int, int] = {}
    for report in reports:
        report_count_by_user[report.user_id] = report_count_by_user.get(report.user_id, 0) + 1

    summaries: list[AdminUserSummary] = []
    for user in users:
        user_predictions = predictions_by_user.get(user.id, [])
        prediction_count = len(user_predictions)
        average_risk = (
            round(sum(item.risk_percent for item in user_predictions) / prediction_count, 2)
            if prediction_count
            else None
        )
        average_confidence = (
            round(sum(item.prediction_confidence for item in user_predictions) / prediction_count, 2)
            if prediction_count
            else None
        )
        summaries.append(
            AdminUserSummary(
                id=user.id,
                email=user.email,
                full_name=user.full_name,
                avatar_url=user.avatar_url,
                joined_at=user.created_at,
                last_login_at=user.last_login_at,
                prediction_count=prediction_count,
                report_count=report_count_by_user.get(user.id, 0),
                average_risk_percent=average_risk,
                average_confidence=average_confidence,
                recent_predictions=[_history_item(item) for item in user_predictions[:5]],
            )
        )

    return AdminOverviewResponse(users=summaries, generated_at=datetime.now(UTC))


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if not model_service.ready:
        try:
            model_service.load()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Model is not ready yet. Startup or training failed: {exc}",
            ) from exc
    metadata = model_service.metadata or {}
    return ModelInfoResponse(**metadata)


@app.get("/feature-schema", response_model=FeatureSchemaResponse)
def feature_schema() -> FeatureSchemaResponse:
    return FeatureSchemaResponse(fields=FEATURE_SCHEMA)


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PredictionResponse:
    runtime_metrics.increment_predictions()
    result = model_service.predict(request.input, request.model_name)
    prediction = Prediction(
        user_id=current_user.id,
        risk_probability=result.risk_probability,
        risk_percent=result.risk_percent,
        risk_band=result.risk_band,
        prediction_confidence=result.prediction_confidence,
        explanation=result.explanation,
        input_payload={**request.input.model_dump(), "_model_name": result.model_name},
        normalized_payload={**result.normalized_input.model_dump(), "_model_name": result.model_name},
    )
    db.add(prediction)
    await db.commit()
    await db.refresh(prediction)
    result.saved_prediction_id = prediction.id
    return result


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    _: User = Depends(get_current_user),
) -> BatchPredictionResponse:
    runtime_metrics.increment_batch()
    items = [
        BatchPredictionItem(index=index, result=model_service.predict(item, request.model_name))
        for index, item in enumerate(request.inputs)
    ]
    return BatchPredictionResponse(items=items)


@app.post("/parse-pdf", response_model=ParsePdfResponse)
async def parse_pdf(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ParsePdfResponse:
    runtime_metrics.increment_pdf()
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    data = await file.read()
    if len(data) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Uploaded PDF is too large.")

    target_path = settings.upload_dir / f"{current_user.id}_{file.filename}"
    target_path.write_bytes(data)

    try:
        text = read_pdf_text(Path(target_path))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read this PDF as text. It may be scanned, encrypted, or unsupported. {exc}",
        ) from exc

    extracted_fields = parse_lab_report_text(text) if text.strip() else []
    report = UploadedReport(
        user_id=current_user.id,
        filename=file.filename,
        storage_path=str(target_path),
        extracted_text_preview=text[:3000],
        extracted_fields=[field.model_dump() for field in extracted_fields],
    )
    db.add(report)
    await db.commit()
    await db.refresh(report)

    return ParsePdfResponse(
        filename=file.filename,
        extracted_text_preview=text[:3000],
        extracted_fields=extracted_fields,
        missing_required_fields=[
            field["key"]
            for field in FEATURE_SCHEMA
            if field.get("section") == "labs"
            and field["key"] not in {item.key for item in extracted_fields}
        ],
        uploaded_report_id=report.id,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    metadata = model_service.metadata or {}
    return MetricsResponse(
        prediction_requests=runtime_metrics.prediction_requests,
        batch_requests=runtime_metrics.batch_requests,
        pdf_parse_requests=runtime_metrics.pdf_parse_requests,
        model_metrics=metadata.get("metrics", {}),
        model_name=metadata.get("model_name", "unknown"),
    )
