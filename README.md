# ⚡ UpgradeIQ: Model Training & Hyperparameter Optimization Engine

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian_HPO-3B82F6.svg?logo=optuna&logoColor=white)](https://optuna.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-v1.2%2B-FFCC00.svg?logo=catboost&logoColor=black)](https://catboost.ai/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1%2B-11754C.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-008080.svg)](https://lightgbm.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Stratified CV](https://img.shields.io/badge/Validation-5--Fold_Stratified_CV-success.svg)](https://scikit-learn.org/stable/modules/cross_validation.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

The `Training` repository provides a leak-free offline machine learning and optimization framework designed for multi-day parameter sweeps across gradient-boosted decision tree architectures (**CatBoost**, **XGBoost**, and **LightGBM**). 

The engine combines **5-Fold Stratified Cross-Validation**, a **Hybrid Random Search + Bayesian Optimization Engine (Optuna TPESampler)**, **continuous best-model checkpointing**, and **decision threshold calibration**.

---

## 🏆 Multi-Model Benchmark Comparison

Results evaluated across **243,000+ subscriber records** with 5-Fold Stratified Cross-Validation:

| Model Architecture | Total Trials | PR-AUC (Avg Precision) | ROC-AUC | Recall | Precision | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CatBoost (v3)** 🥇 | 500 | **0.4072** | **0.7533** | **60.88%** | **35.23%** | **0.4463** | **Production Champion** |
| **XGBoost** 🥈 | 500 | 0.3989 | 0.7472 | 60.08% | 34.98% | 0.4422 | High Generalization |
| **LightGBM** 🥉 | 500 | 0.3979 | 0.7465 | 59.67% | 34.99% | 0.4412 | Fast Training Speed |
| **Baseline (v2)** | - | 0.2400 | 0.7462 | 12.04% | 54.60% | 0.1973 | Uncalibrated (Default 0.5) |

---

## 🏗️ Repository Architecture

```
Retraining/
├── config.py                 # Central config dataclass & environment variable overrides
├── dataset.py                # Zero-leakage data loader, train/test split, and fold transformers
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
cd Retraining

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Multi-Day Background Execution via `tmux`

To launch an automated optimization sweep exploring across CatBoost, XGBoost, and LightGBM:

```bash
./run_tmux.sh --mode tune --model all --n-trials 500 --startup-trials 50 --timeout-hours 72 --cv-folds 5
```

#### Workflow Breakdown:
1. **Tmux Isolation**: Spawns a background session named `upgradeiq-retrain`.
2. **Random Exploration Phase**: Runs 50 random trials to uniformly sample parameter boundaries.
3. **Bayesian Exploitation Phase**: Uses Tree-structured Parzen Estimators (TPE) to model $P(x|y)$ and focus on high-performing parameter regions.
4. **Leak-Free 5-Fold Stratified CV**: Quantiles and categorical encoders are fitted strictly on training partitions per fold.
5. **Threshold Calibration**: Evaluates validation probability distributions to optimize decision thresholds for maximum F1 / churn recall.
6. **Continuous Checkpointing**: Saves trial states into `optuna_study.db` (SQLite) and records candidate models into `checkpoints/`.

---

### 3. Monitoring & Managing Background Runs

| Goal | Command |
| :--- | :--- |
| **Attach to Live Session** | `tmux attach -t upgradeiq-retrain` |
| **Detach from Session** | Press `Ctrl + b`, then release and press `d` |
| **View Live Logs** | `tail -f retraining.log` |
| **Launch Optuna Dashboard** | `optuna-dashboard sqlite:///optuna_study.db` |
| **Stop / Terminate Run** | `tmux kill-session -t upgradeiq-retrain` |

---

### 4. Direct CLI Execution Options

```bash
# 1. Quick baseline check
python train_master.py --mode quick --model cat

# 2. Deep CatBoost sweep with 200 trials
python train_master.py --mode tune --model cat --n-trials 200 --cv-folds 5

# 3. Export production artifacts from current best checkpoint
python train_master.py --mode export --model cat --model-version v3
```

---

## 📦 Production Artifact Compatibility

When optimization completes, the pipeline automatically exports:
- `best_model.pkl` (and `model.pkl`)
- `one_hot_encoder.pkl` (and `ohe.pkl`)
- `metadata.json` (Feature schema, distribution quantiles, best parameters, and calibrated threshold)
- `predictions_test.csv` (Batch predictions for 104,480 customers in `test.csv`)

These artifacts are **100% plug-and-play compatible** with the FastAPI backend in the `Deployment` branch.

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).
