# ⚡ UpgradeIQ Enterprise Retraining & Optimization Suite

The `Retraining/` module is an enterprise-grade offline retraining engine designed for multi-day parameter searches and model optimization. It combines **5-Fold Stratified Cross-Validation**, **Leak-Free Preprocessing Pipelines**, a **Hybrid Random Search + Bayesian Optimization Engine (Optuna TPESampler)**, and **Optimal Decision Threshold Calibration**.

---

## 📁 Architecture Overview

```
Retraining/
├── config.py                 # Central config dataclass & environment variable overrides
├── dataset.py                # Zero-leakage data loader, train/test split, and fold pipelines
├── engineering.py            # Pure feature engineering functions (single source of truth)
├── models.py                 # Parameter spaces & builders (XGBoost, LightGBM, CatBoost)
├── evaluator.py              # Stratified K-Fold CV, OOF scoring, & F1 threshold optimizer
├── tune_optuna.py            # Hybrid Random Search + Bayesian TPE optimization engine
├── export_artifacts.py       # Production artifact generator (model.pkl, ohe.pkl, metadata.json)
├── train_master.py           # Master CLI orchestrator (tune / quick / export modes)
├── run_tmux.sh               # One-click tmux session launcher with live logging
├── requirements.txt          # Training dependencies
└── README.md                 # Documentation and command guide
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
cd /home/anupa/Upgradeiq/Retraining

# Create and activate a dedicated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Run Long Multi-Day Training in `tmux`

To start an extensive search that explores across XGBoost, LightGBM, and CatBoost over several days:

```bash
./run_tmux.sh --mode tune --model all --n-trials 500 --startup-trials 50 --timeout-hours 72 --cv-folds 5
```

#### What this does:
1. Creates a detached `tmux` session named `upgradeiq-retrain`.
2. Runs **50 initial trials of pure Random Exploration** across hyperparameters to broadly sample the parameter space.
3. Automatically transitions to **Bayesian Optimization (TPE Sampler)** to exploit the most promising parameter regions.
4. Uses **5-Fold Stratified Cross Validation** per trial to prevent overfitting and split variance.
5. Optimizes the **decision threshold** to maximize churn detection rate (Recall) and ROI (Precision).
6. Persists every trial state to `optuna_study.db` (SQLite) so runs can be paused, resumed, or parallelized across workers.
7. Automatically saves a checkpoint whenever a new best model is discovered!

---

### 3. Monitoring & Managing Background Training

| Task | Command |
| :--- | :--- |
| **Attach to Live Session** | `tmux attach -t upgradeiq-retrain` |
| **Detach from Session** | Press `Ctrl + b`, then release and press `d` |
| **View Live Logs** | `tail -f retraining.log` |
| **Launch Optuna Dashboard** | `optuna-dashboard sqlite:///optuna_study.db` (opens web UI) |
| **Stop / Terminate Run** | `tmux kill-session -t upgradeiq-retrain` |

---

### 4. Direct CLI Options

You can also run `train_master.py` directly:

```bash
# 1. Quick baseline check
python train_master.py --mode quick --model xgb

# 2. Deep XGBoost sweep with 200 trials
python train_master.py --mode tune --model xgb --n-trials 200 --cv-folds 5

# 3. Export production artifacts from current best checkpoint
python train_master.py --mode export --model xgb --model-version v3
```

---

## 📦 Production Artifact Compatibility

When optimization finishes, the system exports:
- `best_model.pkl` (and `model.pkl`)
- `one_hot_encoder.pkl` (and `ohe.pkl`)
- `metadata.json` (Includes feature schema, quantiles, best hyperparameters, test metrics, and calibrated `decision_threshold`)

These artifacts are **100% plug-and-play compatible** with the FastAPI backend (`July_2026/src/predictor.py`) and Streamlit frontend. Simply copy them to `July_2026/models/v3` or upload to GCS:

```bash
# Redeploy to GCP Cloud Run pointing to new model (Zero Container Rebuild!):
gcloud run deploy upgradeiq-api \
  --image gcr.io/YOUR_PROJECT_ID/upgradeiq-api:latest \
  --set-env-vars ARTIFACT_DIR=gs://upgradeiq-ml-artifacts/models/v3 \
  --region us-central1
```
