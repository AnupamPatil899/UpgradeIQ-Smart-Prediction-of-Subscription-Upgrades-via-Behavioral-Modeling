# Implementation Plan — UpgradeIQ Backend & Streamlit Frontend Restructuring

Restructure the UpgradeIQ project codebase into clean, modular Backend (`src/`) and Frontend (`Frontend/`) services, ensuring seamless local execution and production readiness for GCP (Cloud Run / Artifact Registry).

## Architecture Overview

```
July_2026/
├── Dockerfile                  # GCP Container spec for FastAPI Backend
├── models/
│   └── v1/                     # Serialized artifacts (best_model.pkl, one_hot_encoder.pkl, metadata.json)
├── src/                        # Backend (FastAPI Service)
│   ├── api.py                  # REST API endpoints (/health, /predict)
│   ├── predictor.py            # Model loader & inference pipeline (Local or GCS paths)
│   ├── engineering.py          # Shared feature engineering logic & quantiles
│   ├── generate_metadata.py    # Artifact metadata generator
│   └── requirements.txt        # Backend dependencies (FastAPI, XGBoost, etc.)
└── Frontend/                   # Streamlit Web App Service
    ├── app.py                  # Main Streamlit UI (Restructured from Upgrade_app_old.py)
    ├── api_client.py           # Requests client for backend communication
    ├── config.py               # Config with fallback environment variable handling
    ├── styles.py               # Custom dark-theme CSS and modern styling
    ├── utils.py                # Metric calculations, risk badges & recommendation engine
    ├── requirements.txt        # Frontend dependencies (streamlit, requests, pandas)
    └── Dockerfile              # GCP Container spec for Streamlit Frontend
```

## User Review Required

> [!NOTE]
> The backend and frontend are decoupled into two standalone microservices:
> 1. **Backend Service (`src/`)**: Runs FastAPI with Uvicorn. Reads artifacts from local `models/v1` or GCS (`gs://...`).
> 2. **Frontend Service (`Frontend/`)**: Runs Streamlit UI. Communicates with Backend via HTTP API (`API_URL`).

> [!TIP]
> Both services are configured with default local fallbacks so you can test locally using `uvicorn src.api:app` and `streamlit run Frontend/app.py` without needing to set environment variables manually.

---

## Proposed Changes

### Backend (`src/`)

#### [MODIFY] [src/api.py](file:///c:/Users/anupa/OneDrive/Desktop/Anupam/Python_SURE/UpgradeIQ_project/July_2026/src/api.py)
- Remove hardcoded Windows path `ARTIFACT_DIR = r"C:\..."`.
- Use dynamic environment variable loading via `predictor.py` with fallback to `models/v1`.
- Add `model_config = {"protected_namespaces": ()}` to Pydantic models (`PredictionResponse`, `HealthResponse`) to suppress Pydantic v2 protected namespace warnings.
- Modernize `@app.on_event("startup")` lifecycle management to lifespan handler.

#### [MODIFY] [src/predictor.py](file:///c:/Users/anupa/OneDrive/Desktop/Anupam/Python_SURE/UpgradeIQ_project/July_2026/src/predictor.py)
- Add smart default path fallback for `ARTIFACT_DIR` (check `models/v1` relative to project root if `ARTIFACT_DIR` env var is missing).
- Ensure robust matrix concatenation and return model version metadata cleanly.

#### [MODIFY] [Dockerfile](file:///c:/Users/anupa/OneDrive/Desktop/Anupam/Python_SURE/UpgradeIQ_project/July_2026/Dockerfile)
- Add default environment variable `ENV ARTIFACT_DIR=/app/models/v1`.
- Expose port 8080 for GCP Cloud Run compatibility.

---

### Frontend (`Frontend/`)

#### [MODIFY] [Frontend/config.py](file:///c:/Users/anupa/OneDrive/Desktop/Anupam/Python_SURE/UpgradeIQ_project/July_2026/Frontend/config.py)
- Update `API_URL = os.environ.get("API_URL", "http://localhost:8080")` to avoid `KeyError` when running locally.

#### [MODIFY] [Frontend/app.py](file:///c:/Users/anupa/OneDrive/Desktop/Anupam/Python_SURE/UpgradeIQ_project/July_2026/Frontend/app.py)
- Rebuild Streamlit UI combining the rich visual layout of `Upgrade_app_old.py` with API-based architecture.
- Group sidebar controls into styled categories (Account, Financials, Viewing Behaviour, Content Preferences, Satisfaction & Support, Demographics).
- Display live metric card header (Backend connection status, Model version, Latency, API status).
- Render gradient churn probability bar, risk breakdown tags, engineered metrics table, and recommended customer action box.

#### [MODIFY] [Frontend/utils.py](file:///c:/Users/anupa/OneDrive/Desktop/Anupam/Python_SURE/UpgradeIQ_project/July_2026/Frontend/utils.py)
- Update risk factor analysis to use model quantiles (Low Monthly Viewing, Recent Activity Drop, Low Satisfaction, High Support Usage, High Watcher, High Satisfaction).
- Calculate engineered metrics matching the backend's `engineering.py` calculations.

#### [MODIFY] [Frontend/Dockerfile](file:///c:/Users/anupa/OneDrive/Desktop/Anupam/Python_SURE/UpgradeIQ_project/July_2026/Frontend/Dockerfile)
- Use standard `PORT` variable substitution for Streamlit Cloud Run container.

---

## Verification Plan

### Manual Verification
1. **Test FastAPI Backend locally**:
   - Run backend server: `python -m uvicorn src.api:app --host 127.0.0.1 --port 8080` (or `uvicorn api:app` inside `src`).
   - Check `/health` endpoint response.
   - Send sample JSON request to `/predict` endpoint and verify predictions & response structure.

2. **Test Streamlit Frontend locally**:
   - Run frontend app: `streamlit run Frontend/app.py`.
   - Verify UI rendering, metric cards, sidebar inputs, and click **⚡ Predict Churn**.
   - Verify connection to backend, probability gauge, risk breakdown badges, and recommendations.

3. **Validate Docker readiness**:
   - Check Dockerfile syntax and paths for both Backend and Frontend containers.
