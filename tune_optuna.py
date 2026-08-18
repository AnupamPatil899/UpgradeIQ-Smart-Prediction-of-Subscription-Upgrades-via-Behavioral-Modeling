"""
tune_optuna.py — Hybrid Random Search + Bayesian Optimization Engine
Uses Optuna with TPESampler (random startup trials + Bayesian exploitation)
and SQLite storage for persistent, pauseable multi-day training runs.
"""

import os
import json
import time
from typing import Dict, Any, Optional
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import mlflow

from config import RetrainingConfig
from models import (
    get_xgb_model,
    get_lgb_model,
    get_cat_model,
    sample_xgb_params,
    sample_lgb_params,
    sample_cat_params,
)
from evaluator import evaluate_cv


class OptunaTuningEngine:
    """
    Orchestrates continuous multi-model tuning with checkpointing.
    """

    def __init__(self, config: RetrainingConfig, X_train: np.ndarray, y_train: np.ndarray):
        self.config = config
        self.X_train = X_train
        self.y_train = y_train
        os.makedirs(self.config.checkpoints_dir, exist_ok=True)

        # Imbalance ratio calculation
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        self.base_imbalance_ratio = float(neg_count / max(pos_count, 1))
        print(f"[IMBALANCE] Class Ratio: {neg_count:,} negative / {pos_count:,} positive (Ratio: {self.base_imbalance_ratio:.2f})")

    def run_study(self, model_type: str = "xgb") -> optuna.Study:
        """
        Run or resume an Optuna study with TPE (Random Exploration -> Bayesian Exploitation).
        """
        study_name = f"{self.config.study_name}_{model_type}"
        storage_url = self.config.optuna_db_url

        print(f"\n[OPTUNA] Initializing study '{study_name}' with storage: {storage_url}")
        print(f"[OPTUNA] Strategy: {self.config.n_startup_trials} Random Search trials -> Bayesian TPE Exploitation")

        sampler = TPESampler(
            n_startup_trials=self.config.n_startup_trials,
            multivariate=True,
            seed=self.config.random_state,
        )
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"[OPTUNA] Study state: {completed_trials} existing completed trials.")

        def objective(trial: optuna.Trial) -> float:
            start_time = time.time()

            # 1. Sample hyperparameters based on model type
            if model_type == "xgb":
                params = sample_xgb_params(trial)
                model_fn = lambda: get_xgb_model(params, scale_pos_weight=params.get("scale_pos_weight", self.base_imbalance_ratio))
            elif model_type == "lgb":
                params = sample_lgb_params(trial)
                model_fn = lambda: get_lgb_model(params, scale_pos_weight=params.get("scale_pos_weight", self.base_imbalance_ratio))
            elif model_type == "cat":
                params = sample_cat_params(trial)
                model_fn = lambda: get_cat_model(params, scale_pos_weight=self.base_imbalance_ratio)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # 2. Stratified CV Evaluation
            cv_results = evaluate_cv(
                model_factory=model_fn,
                X=self.X_train,
                y=self.y_train,
                n_splits=self.config.n_splits,
                random_state=self.config.random_state,
                use_smote=self.config.use_smote,
                smote_ratio=self.config.smote_sampling_strategy,
            )

            primary_score = cv_results["oof_pr_auc"] if self.config.primary_metric == "pr_auc" else cv_results["oof_roc_auc"]
            elapsed = time.time() - start_time

            # Log attributes to trial
            trial.set_user_attr("oof_pr_auc", cv_results["oof_pr_auc"])
            trial.set_user_attr("oof_roc_auc", cv_results["oof_roc_auc"])
            trial.set_user_attr("precision", cv_results["precision"])
            trial.set_user_attr("recall", cv_results["recall"])
            trial.set_user_attr("f1_score", cv_results["f1_score"])
            trial.set_user_attr("optimal_threshold", cv_results["optimal_threshold"])
            trial.set_user_attr("elapsed_sec", elapsed)

            # Check if this trial beat previous best
            try:
                best_value = study.best_value
            except ValueError:
                best_value = -1.0

            if primary_score > best_value:
                print(f"  [★ NEW BEST TRIAL #{trial.number}] {self.config.primary_metric.upper()} = {primary_score:.4f} "
                      f"(ROC-AUC: {cv_results['oof_roc_auc']:.4f}, F1: {cv_results['f1_score']:.4f}, Recall: {cv_results['recall']:.4f}, Thresh: {cv_results['optimal_threshold']})")
                self._save_checkpoint(model_type, trial.number, params, cv_results, primary_score)

            return primary_score

        # Start Optuna optimization loop
        try:
            study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout_seconds,
                n_jobs=1,  # Sequential trials with internal multi-threading
                show_progress_bar=True,
            )
        except KeyboardInterrupt:
            print("\n[OPTUNA] Optimization paused by user. Progress is safely preserved in SQLite!")

        print(f"\n[OPTUNA] Completed study '{study_name}'!")
        print(f"[OPTUNA] Best Trial #{study.best_trial.number}: Score = {study.best_value:.4f}")
        print(f"[OPTUNA] Best Hyperparameters: {json.dumps(study.best_params, indent=2)}")

        return study

    def _save_checkpoint(self, model_type: str, trial_num: int, params: dict, cv_results: dict, score: float):
        """Save continuous checkpoint to disk so progress is never lost."""
        ckpt_path = os.path.join(self.config.checkpoints_dir, f"best_{model_type}_checkpoint.json")
        data = {
            "model_type": model_type,
            "best_trial_number": trial_num,
            "score": score,
            "primary_metric": self.config.primary_metric,
            "params": params,
            "cv_metrics": {
                "oof_pr_auc": cv_results["oof_pr_auc"],
                "oof_roc_auc": cv_results["oof_roc_auc"],
                "precision": cv_results["precision"],
                "recall": cv_results["recall"],
                "f1_score": cv_results["f1_score"],
                "optimal_threshold": cv_results["optimal_threshold"],
            },
        }
        with open(ckpt_path, "w") as f:
            json.dump(data, f, indent=2)
