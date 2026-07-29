"""
api.py

Thin FastAPI wrapper around predictor.py. Containerized and deployed to Cloud Run.

Endpoints:
    GET  /health   — liveness/readiness check (also confirms artifacts loaded OK)
    POST /predict  — run a churn prediction for one customer
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure directory containing this file is in sys.path for local uvicorn execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("upgradeiq-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Force artifacts to load once at container startup.
    If ARTIFACT_DIR is wrong or an artifact is missing, the container fails fast.
    """
    version = predictor.get_model_version()
    logger.info(f"Artifacts loaded successfully. Model version: {version}")
    yield


app = FastAPI(
    title="UpgradeIQ Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan
)


class CustomerProfile(BaseModel):
    """
    Raw customer fields — same schema as train.csv minus CustomerID/Churn.
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


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_version=predictor.get_model_version())


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerProfile) -> PredictionResponse:
    try:
        result = predictor.predict_single(customer.model_dump())
        return PredictionResponse(**result)
    except ValueError as e:
        logger.warning(f"Bad request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Internal prediction error") from e

