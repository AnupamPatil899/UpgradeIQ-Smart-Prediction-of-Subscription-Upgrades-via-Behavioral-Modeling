"""
config.py — Central Configuration & Path Resolution for Retraining Suite
Provides structured dataclasses, default paths, and environment variable overrides.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RetrainingConfig:
    # Data Paths
    data_path: str = os.getenv("DATA_PATH", "../Dataset/train.csv")
    test_data_path: str = os.getenv("TEST_DATA_PATH", "../Dataset/test.csv")
    model_version: str = os.getenv("MODEL_VERSION", "v3")
    
    # Artifact Output Directories
    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", f"./artifacts/{os.getenv('MODEL_VERSION', 'v3')}")
    checkpoints_dir: str = os.getenv("CHECKPOINTS_DIR", "./checkpoints")
    artifact_upload_dir: str = os.getenv("ARTIFACT_UPLOAD_DIR", f"gs://upgradeiq-ml-artifacts/models/{os.getenv('MODEL_VERSION', 'v3')}")
    
    # Database and Tracking
    optuna_db_url: str = os.getenv("OPTUNA_DB_URL", "sqlite:///optuna_study.db")
    study_name: str = os.getenv("STUDY_NAME", "upgradeiq_churn_retrain")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "UpgradeIQ_Retraining_Suite")
    
    # Cross Validation & Training
    n_splits: int = int(os.getenv("N_SPLITS", "5"))
    test_size: float = float(os.getenv("TEST_SIZE", "0.20"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    
    # Sampling & Imbalance
    use_smote: bool = os.getenv("USE_SMOTE", "false").lower() in ("true", "1", "yes")
    smote_sampling_strategy: float = float(os.getenv("SMOTE_SAMPLING_STRATEGY", "0.5"))
    
    # Evaluation & Gate Check
    primary_metric: str = os.getenv("PRIMARY_METRIC", "pr_auc")  # 'pr_auc' or 'roc_auc'
    min_acceptable_auc: float = float(os.getenv("MIN_ACCEPTABLE_AUC", "0.75"))
    target_recall: float = float(os.getenv("TARGET_RECALL", "0.70"))
    
    # Tuning Budget
    n_trials: int = int(os.getenv("N_TRIALS", "100"))
    n_startup_trials: int = int(os.getenv("N_STARTUP_TRIALS", "25"))  # Random search phase before Bayesian exploitation
    timeout_seconds: Optional[int] = int(os.getenv("TIMEOUT_SECONDS", "86400"))  # Default 24h per run (can be increased)
    n_jobs: int = int(os.getenv("N_JOBS", "-1"))
    
    # Schema Definition
    target_column: str = "Churn"
    id_column: str = "CustomerID"
    categorical_columns: List[str] = field(
        default_factory=lambda: [
            "PaymentMethod",
            "PaperlessBilling",
            "ContentType",
            "MultiDeviceAccess",
            "DeviceRegistered",
            "GenrePreference",
            "Gender",
            "ParentalControl",
            "SubtitlesEnabled",
        ]
    )


def get_default_config() -> RetrainingConfig:
    """Return default configuration instance."""
    return RetrainingConfig()
