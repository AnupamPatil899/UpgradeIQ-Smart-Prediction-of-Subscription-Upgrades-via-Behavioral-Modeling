"""
models.py — Model Factories & Search Space Definitions
Supports XGBoost, LightGBM, and CatBoost with parameter generators for Random & Bayesian search.
"""

from typing import Dict, Any, Optional
import numpy as np


def get_xgb_model(params: Dict[str, Any], scale_pos_weight: float = 1.0):
    """Build XGBoost Classifier instance."""
    from xgboost import XGBClassifier

    default_params = {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "scale_pos_weight": scale_pos_weight,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "gamma": 0.2,
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    merged_params = {**default_params, **params}
    return XGBClassifier(**merged_params)


def get_lgb_model(params: Dict[str, Any], scale_pos_weight: float = 1.0):
    """Build LightGBM Classifier instance."""
    from lightgbm import LGBMClassifier

    default_params = {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 25,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "scale_pos_weight": scale_pos_weight,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "objective": "binary",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    merged_params = {**default_params, **params}
    return LGBMClassifier(**merged_params)


def get_cat_model(params: Dict[str, Any], scale_pos_weight: float = 1.0):
    """Build CatBoost Classifier instance."""
    from catboost import CatBoostClassifier

    default_params = {
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "auto_class_weights": "Balanced" if scale_pos_weight > 1.5 else None,
        "eval_metric": "PRAUC",
        "random_seed": 42,
        "verbose": 0,
        "thread_count": -1,
    }
    merged_params = {**default_params, **params}
    return CatBoostClassifier(**merged_params)


def sample_xgb_params(trial, is_random: bool = False) -> Dict[str, Any]:
    """Sample hyperparameters for XGBoost via Optuna."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.05),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.5, 6.0, step=0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0, step=0.1),
    }


def sample_lgb_params(trial, is_random: bool = False) -> Dict[str, Any]:
    """Sample hyperparameters for LightGBM via Optuna."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.05),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.5, 6.0, step=0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
    }


def sample_cat_params(trial, is_random: bool = False) -> Dict[str, Any]:
    """Sample hyperparameters for CatBoost via Optuna."""
    return {
        "iterations": trial.suggest_int("iterations", 300, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 9),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }
