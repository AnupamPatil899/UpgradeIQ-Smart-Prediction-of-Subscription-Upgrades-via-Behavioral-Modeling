# Walkthrough — UpgradeIQ Backend & Frontend Restructuring

We have restructured the UpgradeIQ project into a decoupled microservices architecture with a **FastAPI Backend** and a **Streamlit Frontend**, configured for local testing and containerized GCP Cloud Run deployment.

## Architecture

```
July_2026/
├── Dockerfile                  # Container definition for FastAPI Backend
├── models/
│   └── v1/                     # Trained ML artifacts (best_model.pkl, one_hot_encoder.pkl, metadata.json)
├── src/                        # Backend microservice
│   ├── api.py                  # FastAPI REST endpoints (/health, /predict)
│   ├── predictor.py            # Singleton artifact loader & inference engine
│   ├── engineering.py          # Feature engineering pipeline & quantile definitions
│   └── requirements.txt        # Backend dependencies
└── Frontend/                   # Streamlit web app
    ├── app.py                  # Streamlit UI powered by API backend
    ├── api_client.py           # Requests client wrapper
    ├── config.py               # Config with dynamic API_URL fallback
    ├── styles.py               # Dark mode CSS design system
    ├── utils.py                # Quantile risk breakdown & metric formatters
    ├── requirements.txt        # Frontend dependencies
    └── Dockerfile              # Container definition for Streamlit Frontend
```

---

## Key Changes Made

### Backend (`src/`)
- [api.py](file:///c:/Users/anupa/OneDrive/Desktop/Anupam/Python_SURE/UpgradeIQ_project/July_2026/src/api.py): Replaced hardcoded local Windows path with dynamic environment variable artifact loading. Migrated to modern FastAPI `lifespan` handler and added Pydantic v2 `model_config`.
- [predictor.py](file:///c:/Users/anupa/OneDrive/Desktop/Anupam/Python_SURE/UpgradeIQ_project/July_2026/src/predictor.py): Added automatic local fallback to `models/v1/` when `ARTIFACT_DIR` environment variable is omitted locally.
- [Dockerfile](file:///c:/Users/anupa/OneDrive/Desktop/Anupam/Python_SURE/UpgradeIQ_project/July_2026/Dockerfile): Added default `ENV ARTIFACT_DIR=/app/models/v1` and exposed port 8080.

### Frontend (`Frontend/`)
- [config.py](file:///c:/Users/anupa/OneDrive/Desktop/Anupam/Python_SURE/UpgradeIQ_project/July_2026/Frontend/config.py): Set default `API_URL = os.environ.get("API_URL", "http://localhost:8080")` to prevent KeyError during local execution.
- [app.py](file:///c:/Users/anupa/OneDrive/Desktop/Anupam/Python_SURE/UpgradeIQ_project/July_2026/Frontend/app.py): Rebuilt Streamlit application combining rich UI aesthetics from legacy monolith (`Upgrade_app_old.py`) with backend REST API integration. Features include:
  - Top status bar (Backend status, Model version, Latency, API status)
  - Grouped sidebar profile inputs
  - Real-time prediction results with gradient probability bar
  - Risk factor breakdown badges (🔴/🟢/🟡)
  - Engineered metrics table & actionable recommendations
- [utils.py](file:///c:/Users/anupa/OneDrive/Desktop/Anupam/Python_SURE/UpgradeIQ_project/July_2026/Frontend/utils.py): Aligned risk flag thresholds with model metadata quantiles (`VHPW_Q25`, `VHPW_Q75`, `SUPPORT_Q75`).
- [Frontend/Dockerfile](file:///c:/Users/anupa/OneDrive/Desktop/Anupam/Python_SURE/UpgradeIQ_project/July_2026/Frontend/Dockerfile): Enabled dynamic `PORT` expansion for Cloud Run compatibility.

---

## Verification Results

- **Python Syntax Check**: Validated all Python files using `py_compile` with zero syntax errors.
- **Backend Artifact Loading**: Verified artifact loading in `predictor.py` against local `models/v1/`:
  - Artifact model version: `log_model_2`
  - Quantiles & column definitions successfully parsed.

---

## How to Run Locally

### 1. Start FastAPI Backend:
```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8080 --reload
```
*Health Check*: Open `http://localhost:8080/health` in your browser.

### 2. Start Streamlit Frontend:
```bash
streamlit run Frontend/app.py
```
*App UI*: Opens automatically at `http://localhost:8501`.
