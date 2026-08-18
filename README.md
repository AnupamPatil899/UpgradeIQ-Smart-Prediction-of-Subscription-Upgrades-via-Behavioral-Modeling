# ⚡ UpgradeIQ: Customer Subscription Churn & Upgrade Intelligence Platform

UpgradeIQ is a production-grade machine learning system designed to predict subscription churn and identify expansion/upsell opportunities from customer behavioral data. The system features a decoupled architecture with a **FastAPI** REST prediction engine and a **Streamlit** retention dashboard deployed to **Google Cloud Run** with automated **GitHub Actions CI/CD**.

---

## 🌐 Live Services

| Service | Stack | URL / Endpoint |
| :--- | :--- | :--- |
| **Interactive Dashboard** | Streamlit (Python 3.11) | [Live Frontend App](https://upgradeiq-frontend-149522512282.us-central1.run.app) |
| **REST Prediction API** | FastAPI / Uvicorn | [Swagger API Documentation](https://upgradeiq-backend-149522512282.us-central1.run.app/docs) |

---

## 📊 Model Performance & Benchmarks

The production model (**CatBoost v3**) was trained on **243,000+ subscriber records** using **5-Fold Stratified Cross-Validation** with Bayesian hyperparameter optimization (Optuna TPE) and decision threshold calibration.

| Metric | Baseline | Production CatBoost (v3) | Improvement |
| :--- | :--- | :--- | :--- |
| **Recall (Detection Rate)** | 12.04% | **60.88%** | **5× Churn Detection Gain** |
| **F1-Score** | 0.1973 | **0.4463** | **+126% Balanced Accuracy** |
| **PR-AUC (Avg Precision)** | 0.2400 | **0.4072** | **2.25× Over Baseline** |
| **ROC-AUC** | 0.7462 | **0.7533** | **Strict Out-of-Fold Evaluation** |
| **Precision** | 54.60% (uncalibrated) | **35.23%** | **High-ROI Campaign Targeting** |
| **Decision Threshold** | 0.50 | **0.22** | **Calibrated for 4.5:1 Class Imbalance** |

---

## 🏗️ System Architecture

```
July_2026/
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

## 🔧 Feature Engineering Highlights

The feature engineering layer transforms raw profile and usage data into high-signal behavioral indicators:
- **Tenure-Spend Consistency**: Evaluates cumulative billing integrity relative to account age and monthly tier.
- **Frustration Index**: Support ticket volume normalized by weekly viewing hours.
- **Download Intensity**: Content download rate per weekly viewing hour.
- **Viewing Session Ratio**: Average viewing session duration relative to weekly watch time.
- **Engagement Score**: Weighted composite of downloads, watchlist size, and weekly active hours.
- **Quantile Risk Flags**: Outlier detection on activity drops and support ticket spikes.

---

## 🚀 Local Development Setup

### 1. Prerequisites
- Python 3.11+
- Virtual environment (`venv`)

```bash
# Clone the deployment branch
git clone -b Deployment https://github.com/AnupamPatil899/UpgradeIQ-Smart-Prediction-of-Subscription-Upgrades-via-Behavioral-Modeling.git
cd UpgradeIQ-Smart-Prediction-of-Subscription-Upgrades-via-Behavioral-Modeling

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Start Backend REST API
```bash
pip install -r src/requirements.txt
python -m uvicorn src.api:app --host 127.0.0.1 --port 8080 --reload
```
- API Docs: `http://localhost:8080/docs`
- Health Check: `http://localhost:8080/health`

### 3. Start Frontend Dashboard
In a separate terminal:
```bash
source .venv/bin/activate
pip install -r Frontend/requirements.txt
API_URL=http://127.0.0.1:8080 streamlit run Frontend/app.py --server.port 8501
```
- Web Dashboard: `http://localhost:8501`

---

## 🚢 Continuous Integration & Deployment (GCP Cloud Run)

The repository includes a GitHub Actions workflow ([`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)) that automates deployment:
1. **CI Testing**: Validates code syntax and verifies dependencies.
2. **Buildx Caching**: Builds optimized multi-layer Docker images for both services.
3. **Artifact Registry**: Pushes container images to Google Artifact Registry.
4. **Cloud Run Deployment**: Automatically deploys the backend and frontend to **GCP Cloud Run**, injecting dynamic URLs and artifact configurations.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
