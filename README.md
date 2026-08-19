# ⚡ UpgradeIQ: Customer Subscription Churn & Upgrade Intelligence Platform

[![Live App](https://img.shields.io/badge/Live_Dashboard-Google_Cloud_Run-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://upgradeiq-frontend-149522512282.us-central1.run.app)
[![API Docs](https://img.shields.io/badge/API_Docs-FastAPI_Swagger-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://upgradeiq-backend-149522512282.us-central1.run.app/docs)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-v1.2%2B-FFCC00.svg?logo=catboost&logoColor=black)](https://catboost.ai/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Google Cloud Run](https://img.shields.io/badge/GCP-Cloud_Run-4285F4.svg?logo=googlecloud&logoColor=white)](https://cloud.google.com/run)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg?logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

> [!NOTE]
> **Repository Branch Navigation**:
> - **`Deployment` (Default / Active Branch)**: Production microservices architecture (FastAPI Backend + Streamlit Dashboard), Docker container definitions, automated GCP Cloud Run CI/CD, and serialized model artifacts (`models/v3/`).
> - **[`Training` Branch](https://github.com/AnupamPatil899/UpgradeIQ-Smart-Prediction-of-Subscription-Upgrades-via-Behavioral-Modeling/tree/Training)**: Complete offline training pipeline, leak-free preprocessing, 5-Fold Stratified Cross-Validation, 1,500+ trial Bayesian HPO sweep (Optuna TPE), and checkpointing engine.

---

## 📌 Overview

**UpgradeIQ** is an enterprise machine learning platform engineered to identify subscription churn risk and pinpoint upsell/tier upgrade candidates from customer behavioral metrics. Built on **243,000+ subscription records**, the system features a decoupled microservices architecture with a **FastAPI** REST prediction engine and a **Streamlit** user dashboard deployed to **Google Cloud Run** with automated **GitHub Actions CI/CD**.

---

## 🌐 Live Microservice Endpoints

| Service | Technology | Live URL / Endpoint |
| :--- | :--- | :--- |
| **Interactive Dashboard** | Streamlit (Python 3.11) | [Streamlit Web UI](https://upgradeiq-frontend-149522512282.us-central1.run.app) |
| **REST Prediction API** | FastAPI / Uvicorn | [Swagger API Documentation](https://upgradeiq-backend-149522512282.us-central1.run.app/docs) |

---

## 📊 Model Performance & Benchmarks

The production classifier (**CatBoost v3**) was selected following an extensive **1,500+ trial Bayesian hyperparameter sweep (Optuna TPE)** with **5-Fold Stratified Cross-Validation**, leak-free preprocessing, and decision threshold calibration.

| Metric | Baseline | Production CatBoost (v3) | Business Impact |
| :--- | :--- | :--- | :--- |
| **Recall (Churn Detection Rate)** | 12.04% | **60.88%** | **🔥 5× Gain** (Catches ~61% of all churners) |
| **F1-Score** | 0.1973 | **0.4463** | **+126% Balanced Accuracy** |
| **PR-AUC (Average Precision)** | 0.2400 | **0.4072** | **2.25× Over Baseline** |
| **ROC-AUC** | 0.7462 | **0.7533** | **Strict Out-of-Fold Evaluation** |
| **Precision** | 54.60% (uncalibrated) | **35.23%** | **High-ROI Retention Campaign Targeting** |
| **Decision Threshold** | 0.50 | **0.22** | **Calibrated for 4.5:1 Class Imbalance** |

---

## 🏗️ System Architecture & Execution Flow

```
┌─────────────────────────┐          HTTP / JSON          ┌─────────────────────────┐
│   Streamlit Frontend    │  ───────────────────────────► │     FastAPI Backend     │
│ (Interactive Dashboard) │  ◄─────────────────────────── │  (Inference Microservice)│
└─────────────────────────┘      Churn Probability &      └────────────┬────────────┘
                                   Risk Factor Flags                   │
                                                          ┌────────────┴────────────┐
                                                          │   CatBoost v3 Engine    │
                                                          │  (Calibrated Threshold) │
                                                          └─────────────────────────┘
```

```
Upgradeiq/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions CI/CD for automated Cloud Run deployments
├── Dockerfile                  # Container definition for FastAPI backend
├── models/
│   └── v3/                     # Active production artifacts
│       ├── best_model.pkl      # Tuned CatBoost classifier
│       ├── one_hot_encoder.pkl # Categorical encoder (zero data leakage)
│       └── metadata.json       # Feature schema, distribution quantiles, & threshold
├── src/                        # ⚙️ BACKEND MICROSERVICE (FastAPI)
│   ├── api.py                  # Endpoints (/health, /predict)
│   ├── predictor.py            # Singleton artifact loader & inference pipeline
│   ├── engineering.py          # Pure feature engineering functions
│   └── requirements.txt        # Backend dependencies
└── Frontend/                   # 🎨 FRONTEND MICROSERVICE (Streamlit)
    ├── app.py                  # Interactive customer intelligence dashboard
    ├── api_client.py           # Resilient backend client wrapper
    ├── config.py               # Runtime environment configuration
    ├── styles.py               # Dark-mode dashboard design system
    ├── utils.py                # Behavioral risk badge and scoring utilities
    ├── requirements.txt        # Frontend dependencies
    └── Dockerfile              # Container definition for Streamlit frontend
```

---

## ⚡ Quick Test with `curl`

You can test the live prediction API directly from your terminal:

```bash
curl -X POST "https://upgradeiq-backend-149522512282.us-central1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "AccountAge": 12,
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
    "SubscriptionType": "Standard",
    "PaymentMethod": "Credit card",
    "PaperlessBilling": "Yes",
    "ContentType": "Both",
    "MultiDeviceAccess": "Yes",
    "DeviceRegistered": "TV",
    "ViewingHoursPerWeek": 14.5,
    "AverageViewingDuration": 85.0,
    "ContentDownloadsPerMonth": 4,
    "GenrePreference": "Action",
    "UserRating": 4.2,
    "SupportTicketsPerMonth": 1,
    "Gender": "Female",
    "WatchlistSize": 8,
    "ParentalControl": "No",
    "SubtitlesEnabled": "Yes"
  }'
```

**Sample Response**:
```json
{
  "churn_probability": 0.4626,
  "churn_prediction": 1,
  "model_version": "v3"
}
```

---

## 🚀 Local Development Setup

### 1. Prerequisites
- Python 3.11+
- Virtual environment (`venv`)

```bash
git clone -b Deployment https://github.com/AnupamPatil899/UpgradeIQ-Smart-Prediction-of-Subscription-Upgrades-via-Behavioral-Modeling.git
cd UpgradeIQ-Smart-Prediction-of-Subscription-Upgrades-via-Behavioral-Modeling

python3 -m venv .venv
source .venv/bin/activate
```

### 2. Start Backend REST API (FastAPI)
```bash
pip install -r src/requirements.txt
python -m uvicorn src.api:app --host 127.0.0.1 --port 8080 --reload
```
- API Docs: `http://localhost:8080/docs`
- Health Check: `http://localhost:8080/health`

### 3. Start Frontend Dashboard (Streamlit)
In a separate terminal:
```bash
source .venv/bin/activate
pip install -r Frontend/requirements.txt
API_URL=http://127.0.0.1:8080 streamlit run Frontend/app.py --server.port 8501
```
- Web Dashboard: `http://localhost:8501`

---

## 🚢 Continuous Integration & Deployment (GCP Cloud Run)

Whenever changes are pushed to the `Deployment` branch:
1. **CI Testing**: Validates Python syntax and dependency integrity.
2. **Docker Buildx**: Builds optimized multi-layer Docker container images.
3. **Artifact Registry**: Pushes container images to Google Artifact Registry.
4. **Cloud Run Deployment**: Deploys both services to **GCP Cloud Run**, dynamically linking the backend URL into the frontend environment.

---

## 🔬 Model Training & Offline Experimentation

The offline model training, feature extraction, and hyperparameter optimization suite is maintained on the dedicated **[`Training` branch](https://github.com/AnupamPatil899/UpgradeIQ-Smart-Prediction-of-Subscription-Upgrades-via-Behavioral-Modeling/tree/Training)**.

Key training highlights in the `Training` branch:
- **Leak-Free Data Ingestion**: Quantiles and categorical encoders are computed strictly on training partitions during 5-Fold Stratified Cross-Validation.
- **Automated Bayesian HPO**: Optuna Tree-structured Parzen Estimator (TPE) search running **1,500+ trials** across CatBoost, XGBoost, and LightGBM.
- **Decision Threshold Calibration**: Mathematical threshold search maximizing $F_1$ and churn recall on imbalanced distributions.
- **Persistent State & Checkpoints**: SQLite trial tracking (`optuna_study.db`) and automated champion checkpoint saving.

To explore training scripts, run local cross-validation, or launch hyperparameter tuning sweeps:
```bash
git checkout Training
```
For detailed training documentation, refer to the [Training Branch README](https://github.com/AnupamPatil899/UpgradeIQ-Smart-Prediction-of-Subscription-Upgrades-via-Behavioral-Modeling/tree/Training).

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).

