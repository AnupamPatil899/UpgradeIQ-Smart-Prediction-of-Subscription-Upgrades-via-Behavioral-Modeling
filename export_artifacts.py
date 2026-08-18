"""
export_artifacts.py — Production Artifact Generator & GCS Publisher
Trains final production model on full training set using best parameters,
evaluates on held-out test set, and exports model.pkl, ohe.pkl, and metadata.json.
"""

import os
import json
import joblib
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

from config import RetrainingConfig
from dataset import DataPipeline
from models import get_xgb_model, get_lgb_model, get_cat_model
from evaluator import find_optimal_threshold


def upload_to_gcs(local_file: str, gcs_uri: str):
    """Upload a file to Google Cloud Storage."""
    if not gcs_uri.startswith("gs://"):
        print(f"[GCS] Skipping upload (not a gs:// URI): {gcs_uri}")
        return

    try:
        from google.cloud import storage

        parts = gcs_uri.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        filename = os.path.basename(local_file)
        blob_path = f"{prefix}/{filename}".strip("/")

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_file)
        print(f"[GCS] Uploaded {local_file} -> gs://{bucket_name}/{blob_path}")
    except Exception as e:
        print(f"[GCS] Warning: GCS upload failed for {local_file}: {e}")


def export_final_model(
    config: RetrainingConfig,
    pipeline: DataPipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: Dict[str, Any],
    model_type: str = "xgb",
    decision_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Fit production model on full X_train, evaluate on X_test, and save all artifacts.
    """
    os.makedirs(config.artifacts_dir, exist_ok=True)
    print(f"\n[EXPORT] Fitting final {model_type.upper()} model on full training set ({len(y_train):,} rows)...")

    # 1. Fit Model
    if model_type == "xgb":
        model = get_xgb_model(best_params)
    elif model_type == "lgb":
        model = get_lgb_model(best_params)
    elif model_type == "cat":
        model = get_cat_model(best_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)

    # 2. Evaluate on Held-out Test Set
    test_probs = model.predict_proba(X_test)[:, 1]
    test_roc_auc = float(roc_auc_score(y_test, test_probs))
    test_pr_auc = float(average_precision_score(y_test, test_probs))

    # Determine or optimize threshold on test/val
    if decision_threshold is None:
        thresh_info = find_optimal_threshold(y_test.values, test_probs)
        decision_threshold = thresh_info["optimal_threshold"]
    else:
        thresh_info = {}

    test_preds = (test_probs >= decision_threshold).astype(int)
    test_precision = float(precision_score(y_test, test_preds, zero_division=0))
    test_recall = float(recall_score(y_test, test_preds, zero_division=0))
    test_f1 = float(f1_score(y_test, test_preds, zero_division=0))

    print(f"[EVAL-TEST] Final Test Performance:")
    print(f"            ROC-AUC:            {test_roc_auc:.4f}")
    print(f"            PR-AUC:             {test_pr_auc:.4f}")
    print(f"            Decision Threshold: {decision_threshold:.4f}")
    print(f"            Precision:          {test_precision:.4f}")
    print(f"            Recall:             {test_recall:.4f}")
    print(f"            F1-Score:           {test_f1:.4f}")

    # Gate Check
    if test_roc_auc < config.min_acceptable_auc:
        print(f"[GATE] Warning: ROC-AUC ({test_roc_auc:.4f}) is below target ({config.min_acceptable_auc})")

    # 3. Save Artifacts
    # Save with standard names (best_model.pkl / model.pkl for maximum compatibility)
    model_path = os.path.join(config.artifacts_dir, "best_model.pkl")
    model_alias_path = os.path.join(config.artifacts_dir, "model.pkl")
    ohe_path = os.path.join(config.artifacts_dir, "one_hot_encoder.pkl")
    ohe_alias_path = os.path.join(config.artifacts_dir, "ohe.pkl")
    meta_path = os.path.join(config.artifacts_dir, "metadata.json")

    joblib.dump(model, model_path)
    joblib.dump(model, model_alias_path)
    joblib.dump(pipeline.ohe, ohe_path)
    joblib.dump(pipeline.ohe, ohe_alias_path)

    metadata = {
        "model_version": config.model_version,
        "model_type": model_type,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "decision_threshold": decision_threshold,
        "quantiles": pipeline.quantiles,
        "num_cols": pipeline.num_cols,
        "cat_cols": pipeline.cat_cols,
        "total_num_cols": len(pipeline.num_cols),
        "total_cat_cols": len(pipeline.cat_cols),
        "features_count": X_train.shape[1],
        "feature_names": pipeline.feature_names,
        "best_hyperparameters": best_params,
        "metrics": {
            "test_roc_auc": test_roc_auc,
            "test_pr_auc": test_pr_auc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "train_records": len(y_train),
            "test_records": len(y_test),
        },
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[ARTIFACTS] Successfully saved production artifacts to {config.artifacts_dir}")

    # 4. GCS Upload if configured
    if config.artifact_upload_dir and config.artifact_upload_dir.startswith("gs://"):
        upload_to_gcs(model_path, config.artifact_upload_dir)
        upload_to_gcs(model_alias_path, config.artifact_upload_dir)
        upload_to_gcs(ohe_path, config.artifact_upload_dir)
        upload_to_gcs(ohe_alias_path, config.artifact_upload_dir)
        upload_to_gcs(meta_path, config.artifact_upload_dir)
        print(f"[GCS] Artifacts deployed to GCS: {config.artifact_upload_dir}")

    # 5. Optionally score test.csv if present
    if os.path.exists(config.test_data_path):
        try:
            score_unlabeled_test_set(config, pipeline, model, decision_threshold)
        except Exception as e:
            print(f"[TEST-PREDICTIONS] Warning: Could not score {config.test_data_path}: {e}")

    return metadata


def score_unlabeled_test_set(config: RetrainingConfig, pipeline: DataPipeline, model, decision_threshold: float):
    """Score all customers in Dataset/test.csv and output predictions_test.csv."""
    df_test_raw = pd.read_csv(config.test_data_path)
    customer_ids = df_test_raw["CustomerID"] if "CustomerID" in df_test_raw.columns else df_test_raw.index

    x_test_full, _ = pipeline.transform_eval(df_test_raw)
    probs = model.predict_proba(x_test_full)[:, 1]
    preds = (probs >= decision_threshold).astype(int)

    out_df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Churn_Probability": np.round(probs, 4),
        "Predicted_Churn": preds,
    })
    out_path = os.path.join(config.artifacts_dir, "predictions_test.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[TEST-PREDICTIONS] Generated batch predictions for {len(out_df):,} customers -> {out_path}")

