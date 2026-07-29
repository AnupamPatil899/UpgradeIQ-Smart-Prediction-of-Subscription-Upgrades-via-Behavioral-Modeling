"""
api.py

Thin FastAPI wrapper around predictor.py. This is the file that gets
containerized and deployed to Cloud Run.

Endpoints:
    GET  /health   — liveness/readiness check (also confirms artifacts loaded OK)
    POST /predict  — run a churn prediction for one customer

Artifacts are loaded once, at startup, not per request — see the
startup event below. If ARTIFACT_DIR is misconfigured or an artifact
is missing, the app fails fast at startup instead of failing on the
first real request.
"""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("upgradeiq-api")

app = FastAPI(title="UpgradeIQ Churn Prediction API", version="1.0.0")


class CustomerProfile(BaseModel):
    """
    Raw customer fields — same schema as train.csv minus CustomerID/Churn.
    SubscriptionType stays a raw string here ("Basic"/"Standard"/"Premium");
    predictor.py handles the numeric encoding internally.
    """

    AccountAge: int = Field(..., ge=0, description="Months subscribed")
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    SubscriptionType: str = Field(..., description="Basic, Standard, or Premium")
    PaymentMethod: str
    PaperlessBilling: str
    ContentType: str
    MultiDeviceAccess: str
    DeviceRegistered: str
    ViewingHoursPerWeek: float = Field(..., ge=0)
    AverageViewingDuration: float = Field(..., ge=0)
    ContentDownloadsPerMonth: int = Field(..., ge=0)
    GenrePreference: str
    UserRating: float = Field(..., ge=1, le=5)
    SupportTicketsPerMonth: int = Field(..., ge=0)
    Gender: str
    WatchlistSize: int = Field(..., ge=0)
    ParentalControl: str
    SubtitlesEnabled: str

    class Config:
        json_schema_extra = {
            "example": {
                "AccountAge": 12,
                "MonthlyCharges": 50.0,
                "TotalCharges": 600.0,
                "SubscriptionType": "Basic",
                "PaymentMethod": "Credit Card",
                "PaperlessBilling": "Yes",
                "ContentType": "Both",
                "MultiDeviceAccess": "Yes",
                "DeviceRegistered": "Mobile",
                "ViewingHoursPerWeek": 10.0,
                "AverageViewingDuration": 60.0,
                "ContentDownloadsPerMonth": 5,
                "GenrePreference": "Drama",
                "UserRating": 3.5,
                "SupportTicketsPerMonth": 1,
                "Gender": "Male",
                "WatchlistSize": 10,
                "ParentalControl": "No",
                "SubtitlesEnabled": "Yes",
            }
        }


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    churn_probability: float
    churn_prediction: int
    model_version: str


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_version: str


@app.on_event("startup")
def load_artifacts_on_startup() -> None:
    """
    Force artifacts to load once at container startup, not on the
    first request. If ARTIFACT_DIR is wrong or a file is missing,
    the container fails to become ready, which Cloud Run will surface
    immediately in deploy logs — much easier to debug than a 500 on
    someone's first real prediction request.
    """
    version = predictor.get_model_version()
    logger.info(f"Artifacts loaded successfully. Model version: {version}")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_version=predictor.get_model_version())


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerProfile) -> PredictionResponse:
    try:
        result = predictor.predict_single(customer.model_dump())
        return PredictionResponse(**result)
    except ValueError as e:
        # Bad/missing input data — client error, not a server error.
        logger.warning(f"Bad request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Anything else is unexpected — log with full context, return 500.
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Internal prediction error") from e
