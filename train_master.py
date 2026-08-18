"""
train_master.py — Master Retraining CLI Orchestrator
Entrypoint for quick runs, hyperparameter sweeps (Random + Bayesian), and production export.
"""

import argparse
import json
import os
import sys
from sklearn.model_selection import train_test_split

from config import get_default_config, RetrainingConfig
from dataset import find_and_load_data, DataPipeline
from tune_optuna import OptunaTuningEngine
from export_artifacts import export_final_model


def parse_args():
    parser = argparse.ArgumentParser(description="UpgradeIQ Model Retraining & Hyperparameter Optimization Master")
    parser.add_argument("--mode", type=str, choices=["tune", "quick", "export"], default="tune",
                        help="Execution mode: 'tune' (deep Optuna sweep), 'quick' (single baseline), 'export' (export best model)")
    parser.add_argument("--model", type=str, choices=["xgb", "lgb", "cat", "all"], default="xgb",
                        help="Model architecture to train/tune: xgb, lgb, cat, or all")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Total number of Optuna trials to run")
    parser.add_argument("--startup-trials", type=int, default=25,
                        help="Number of initial pure Random Search trials before Bayesian TPE exploitation")
    parser.add_argument("--timeout-hours", type=float, default=24.0,
                        help="Maximum training runtime in hours before graceful stopping")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of Stratified K-Fold CV splits")
    parser.add_argument("--use-smote", action="store_true", default=False,
                        help="Enable SMOTE oversampling during cross-validation")
    parser.add_argument("--model-version", type=str, default="v3",
                        help="Model version string (e.g. v3, v4)")
    parser.add_argument("--data-path", type=str, default="",
                        help="Explicit path to train.csv or Dataset.zip")
    parser.add_argument("--gcs-upload", type=str, default="",
                        help="Optional GCS target URI (e.g. gs://bucket/models/v3)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Initialize Configuration
    config = get_default_config()
    config.model_version = args.model_version
    config.n_trials = args.n_trials
    config.n_startup_trials = args.startup_trials
    config.timeout_seconds = int(args.timeout_hours * 3600)
    config.n_splits = args.cv_folds
    config.use_smote = args.use_smote
    if args.data_path:
        config.data_path = args.data_path
    if args.gcs_upload:
        config.artifact_upload_dir = args.gcs_upload

    print("=" * 70)
    print(f"🚀 UPGRADEIQ ENTERPRISE RETRAINING ENGINE — Version {config.model_version}")
    print("=" * 70)
    print(f"Mode:             {args.mode.upper()}")
    print(f"Model(s):         {args.model.upper()}")
    print(f"Trials Budget:    {config.n_trials} ({config.n_startup_trials} Random Search + Bayesian Exploitation)")
    print(f"Time Budget:      {args.timeout_hours} hours")
    print(f"CV Folds:         {config.n_splits}-Fold Stratified")
    print(f"Imbalance SMOTE:  {'Enabled' if config.use_smote else 'Disabled (Tree ScalePosWeight)'}")
    print(f"Artifacts Dir:    {config.artifacts_dir}")
    print("=" * 70)

    # 2. Ingest Data
    df_raw = find_and_load_data(config.data_path)

    # 3. Train / Test Split (Strict Zero Leakage)
    print(f"\n[SPLIT] Performing 80/20 Stratified Train/Test Split...")
    df_train, df_test = train_test_split(
        df_raw, test_size=config.test_size, random_state=config.random_state, stratify=df_raw["Churn"]
    )
    print(f"[SPLIT] Training Set: {len(df_train):,} rows | Test Set: {len(df_test):,} rows")

    # 4. Fit Preprocessing Pipeline on Train Only
    pipeline = DataPipeline(cat_cols=config.categorical_columns)
    X_train, y_train = pipeline.fit_transform_train(df_train)
    X_test, y_test = pipeline.transform_eval(df_test)

    print(f"[FEATURES] Extracted {X_train.shape[1]} total features ({len(pipeline.num_cols)} numerical, {len(pipeline.cat_cols)} categorical).")

    # 5. Execution Modes
    models_to_run = ["xgb", "lgb", "cat"] if args.model == "all" else [args.model]

    if args.mode == "tune":
        tuning_engine = OptunaTuningEngine(config, X_train, y_train.values)
        best_overall_score = -1.0
        best_overall_model = None
        best_overall_study = None

        for m_type in models_to_run:
            print(f"\n>>> Starting Retraining Sweep for [{m_type.upper()}] <<<")
            study = tuning_engine.run_study(m_type)
            if study.best_value > best_overall_score:
                best_overall_score = study.best_value
                best_overall_model = m_type
                best_overall_study = study

        # Export the best overall model
        print("\n" + "=" * 70)
        print(f"🏆 Best Architecture Discovered: {best_overall_model.upper()} (OOF PR-AUC: {best_overall_score:.4f})")
        print("=" * 70)

        best_params = best_overall_study.best_params
        best_threshold = best_overall_study.best_trial.user_attrs.get("optimal_threshold", 0.35)

        export_final_model(
            config=config,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train.values,
            X_test=X_test,
            y_test=y_test,
            best_params=best_params,
            model_type=best_overall_model,
            decision_threshold=best_threshold,
        )

    elif args.mode == "quick":
        # Quick baseline run
        print("\n[QUICK] Running fast single-pass model...")
        best_params = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05}
        export_final_model(
            config=config,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train.values,
            X_test=X_test,
            y_test=y_test,
            best_params=best_params,
            model_type=models_to_run[0],
            decision_threshold=0.35,
        )

    elif args.mode == "export":
        ckpt_file = os.path.join(config.checkpoints_dir, f"best_{models_to_run[0]}_checkpoint.json")
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f"Checkpoint {ckpt_file} not found. Run with --mode tune first.")
        with open(ckpt_file) as f:
            ckpt_data = json.load(f)

        print(f"\n[EXPORT] Loaded checkpoint for {ckpt_data['model_type']} (Trial #{ckpt_data['best_trial_number']})")
        export_final_model(
            config=config,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train.values,
            X_test=X_test,
            y_test=y_test,
            best_params=ckpt_data["params"],
            model_type=ckpt_data["model_type"],
            decision_threshold=ckpt_data["cv_metrics"].get("optimal_threshold", 0.35),
        )

    print("\n[COMPLETE] Retraining execution finished successfully!")


if __name__ == "__main__":
    main()
