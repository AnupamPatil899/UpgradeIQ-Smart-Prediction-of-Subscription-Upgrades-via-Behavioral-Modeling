# ⚡ UpgradeIQ: Smart Prediction of Subscription Upgrades & Churn via Behavioral Modeling

> **Note on Repository Branch**: This is the **`Deployment`** branch, which contains the production-ready, microservices-based architecture (FastAPI Backend + Streamlit Frontend) deployed to **Google Cloud Run** with automated GitHub Actions CI/CD. For model training code, exploratory data analysis, and notebook experiments, please refer to the model training branches.

---

## 📌 Overview

**UpgradeIQ** is an enterprise-grade machine learning application designed to analyze customer behavioral data and predict subscription changes (such as churn or potential tier upgrades). By leveraging key metrics—including viewing habits, engagement scores, support ticket frequency, and account age—UpgradeIQ provides real-time, actionable insights to help customer success teams maximize retention and target upsell opportunities.

---

## 🌐 Live Microservice Deployment

| Component | Technology | Live URL / Endpoint |
| :--- | :--- | :--- |
| **Frontend Web App** | Streamlit (Python 3.11) | [Streamlit Interactive UI](http://8.231.82.156:8501/) *(or GCP Cloud Run Frontend)* |
| **Backend REST API** | FastAPI / Uvicorn | Swagger Docs: `http://localhost:8080/docs` *(or GCP Cloud Run Backend)* |

---

## 🌟 Key Features

- **Decoupled Microservice Architecture**: Clean separation between the ML prediction engine (FastAPI) and the user-facing web dashboard (Streamlit).
- **Extensive Feature Engineering**: Calculates 28+ composite metrics, including:
  - **Engagement Score**: Weighted combination of downloads, watchlist size, and weekly viewing.
  - **Support Intensity**: Ticket frequency relative to account tenure.
  - **Recent Activity Drop**: Flags sudden engagement drop-offs for established accounts.
  - **Total Risk Score**: Quantile-based composite risk flag sum.
- **Handling Imbalanced Data**: Utilizes **SMOTE** (Synthetic Minority Over-sampling Technique) to balance class distributions during model training.
- **High-Performance ML Model**: Powered by **XGBoost Classifier** (ROC-AUC ~ **0.75**) tuned on **243,000+ subscription records**.
- **Automated CI/CD**: Seamless GitHub Actions workflow deploying containerized Docker images to **GCP Cloud Run** via Artifact Registry.

---

## 🏗️ Repository Architecture

```
July_2026/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions CI/CD workflow for GCP Cloud Run
├── Dockerfile                  # Container spec for FastAPI Backend microservice
├── models/
│   └── v1/
│       ├── best_model.pkl      # Trained XGBoost classifier
│       ├── one_hot_encoder.pkl # Categorical OneHotEncoder
│       └── metadata.json       # Column definitions, quantiles, & version metadata
├── src/                        # ⚙️ BACKEND MICROSERVICE (FastAPI)
│   ├── api.py                  # REST API endpoints (/health, /predict)
│   ├── predictor.py            # Singleton artifact loader & inference pipeline
│   ├── engineering.py          # Shared feature engineering logic
│   ├── generate_metadata.py    # Artifact metadata generator tool
│   └── requirements.txt        # Backend dependencies
└── Frontend/                   # 🎨 FRONTEND MICROSERVICE (Streamlit)
    ├── app.py                  # Interactive user dashboard UI
    ├── api_client.py           # REST client wrapper for backend communication
    ├── config.py               # Config with dynamic environment variable fallback
    ├── styles.py               # Modern dark-mode CSS theme
    ├── utils.py                # Quantile risk breakdown badges & metric formatting
    ├── requirements.txt        # Frontend dependencies
    └── Dockerfile              # Container spec for Streamlit Frontend
```

---

## 📊 Dataset Schema

The underlying model is trained on subscription dataset attributes including:

- **Account Profile**: `AccountAge`, `SubscriptionType` (Basic / Standard / Premium), `PaymentMethod`, `PaperlessBilling`, `Gender`.
- **Viewing Behavior**: `ViewingHoursPerWeek`, `AverageViewingDuration`, `ContentDownloadsPerMonth`, `WatchlistSize`, `ContentType`, `GenrePreference`.
- **User Satisfaction & Support**: `UserRating` (1–5★), `SupportTicketsPerMonth`.
- **Target Variable**: `Churn` (0 = Retained/Upgrade candidate, 1 = Churn risk).

---

## 🚀 Local Development Setup

### Prerequisites
- Python 3.11+
- Virtual environment (`venv` or `uv`)

### 1. Clone & Set Up Virtual Environment

```bash
git clone -b Deployment https://github.com/AnupamPatil899/UpgradeIQ-Smart-Prediction-of-Subscription-Upgrades-via-Behavioral-Modeling.git
cd UpgradeIQ-Smart-Prediction-of-Subscription-Upgrades-via-Behavioral-Modeling

# Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. Start Backend REST API (FastAPI)

```bash
pip install -r src/requirements.txt
python -m uvicorn src.api:app --host 127.0.0.1 --port 8080 --reload
```
*Health Check*: Open `http://localhost:8080/health` or `http://localhost:8080/docs`.

### 3. Start Frontend Dashboard (Streamlit)

In a new terminal window:

```bash
pip install -r Frontend/requirements.txt
streamlit run Frontend/app.py --server.port 8501
```
*App UI*: Opens automatically at `http://localhost:8501`.

---

## ⚙️ CI/CD & GCP Deployment

This repository includes a continuous integration and deployment pipeline ([deploy.yml](.github/workflows/deploy.yml)).

Whenever changes are pushed to the `Deployment` branch:
1. **CI Check**: Verifies dependency compilation and validates code syntax.
2. **Docker Buildx**: Builds container images for both Backend and Frontend using GitHub Actions caching.
3. **GCP Artifact Registry**: Pushes container images to Google Artifact Registry.
4. **Cloud Run Deploy**: Deploys both services to **GCP Cloud Run**, automatically injecting the backend URL into the frontend environment.

---

## 🤝 Contributing & Branching Guide

- **`Deployment` Branch (Current)**: Primary branch for production API & UI deployment.
- **Model Training Branches**: Contain exploratory Jupyter Notebooks (`.ipynb`), model comparison experiments (Logistic Regression vs. XGBoost), and raw training scripts.

Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
