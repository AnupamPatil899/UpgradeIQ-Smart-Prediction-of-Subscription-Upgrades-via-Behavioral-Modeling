"""
predictor.py

Loads the three versioned model artifacts (model, one-hot encoder,
metadata) and exposes a single predict_single() function used by the
serving API.

Artifacts are loaded from EITHER a local folder OR a GCS path
(gs://bucket/path/), depending on the ARTIFACT_DIR value passed in.
This lets you test the full predict path locally against files copied
down from the bucket, then point at gs://... for the real deployment
without touching any other code.

Artifacts are loaded once and cached in memory (module-level
singleton) — not reloaded per request. On Cloud Run this means one
load per container instance start, which is what you want.
"""

import io
import json
import os
import sys

import joblib
import pandas as pd
from scipy.sparse import hstack

# Ensure directory containing this file is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engineering import add_engineered_features, encode_subscription_type

# Populated lazily by _load_artifacts(). Do not import these directly
# from outside this module — call predict_single() / get_model_version()
# instead, so loading always goes through the same path.
_model = None
_ohe = None
_metadata = None


def predict_single(user_input: dict) -> dict:
    """
    Run a churn prediction for a single raw customer record.

    Args:
        user_input: dict matching the raw schema (same fields your
            Streamlit sidebar collects) — CustomerID and Churn should
            NOT be present, SubscriptionType should be the raw string
            ("Basic"/"Standard"/"Premium"), not pre-encoded.

    Returns:
        {
            "churn_probability": float,
            "churn_prediction": int (0 or 1),
            "model_version": str,
        }
    """
    _ensure_loaded()

    df = pd.DataFrame([user_input])
    df = encode_subscription_type(df)
    df = add_engineered_features(df, _metadata["quantiles"])

    num_cols = _metadata["num_cols"]
    cat_cols = _metadata["cat_cols"]

    missing = [c for c in num_cols + cat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required column(s) after engineering: {missing}")

    x_num = df[num_cols].values.astype(float)
    x_cat = _ohe.transform(df[cat_cols])
    x_vec = _hstack(x_num, x_cat)

    probability = float(_model.predict_proba(x_vec)[0, 1])
    threshold = _metadata.get("decision_threshold", 0.5)

    return {
        "churn_probability": probability,
        "churn_prediction": int(probability >= threshold),
        "model_version": _metadata.get("model_version", "unknown"),
    }


def get_model_version() -> str:
    """Small helper for the API's /health endpoint."""
    _ensure_loaded()
    return _metadata.get("model_version", "unknown")


def _ensure_loaded() -> None:
    global _model, _ohe, _metadata
    if _model is not None and _ohe is not None and _metadata is not None:
        return

    artifact_dir = os.environ.get("ARTIFACT_DIR")
    if not artifact_dir:
        # Fallback to local models/v1 directory relative to workspace root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_dir = os.path.join(base_dir, "models", "v1")
        if os.path.exists(default_dir):
            artifact_dir = default_dir
        else:
            raise RuntimeError(
                "ARTIFACT_DIR environment variable is not set and fallback path "
                f"({default_dir}) was not found. Please set ARTIFACT_DIR."
            )

    _model = _load_pickle(artifact_dir, "best_model.pkl")
    _ohe = _load_pickle(artifact_dir, "one_hot_encoder.pkl")
    _metadata = _load_json(artifact_dir, "metadata.json")


def _load_pickle(artifact_dir: str, filename: str):
    raw_bytes = _read_bytes(artifact_dir, filename)
    return joblib.load(io.BytesIO(raw_bytes))


def _load_json(artifact_dir: str, filename: str) -> dict:
    raw_bytes = _read_bytes(artifact_dir, filename)
    return json.loads(raw_bytes.decode("utf-8"))


def _read_bytes(artifact_dir: str, filename: str) -> bytes:
    if artifact_dir.startswith("gs://"):
        return _read_bytes_from_gcs(artifact_dir, filename)
    return _read_bytes_from_local(artifact_dir, filename)


def _read_bytes_from_local(artifact_dir: str, filename: str) -> bytes:
    path = os.path.join(artifact_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found at {path}")
    with open(path, "rb") as f:
        return f.read()


def _read_bytes_from_gcs(artifact_dir: str, filename: str) -> bytes:
    # Imported here (not at module top) so this module still works in
    # environments without google-cloud-storage installed, e.g. pure
    # local testing against a local ARTIFACT_DIR.
    from google.cloud import storage

    without_scheme = artifact_dir[len("gs://"):]
    bucket_name, _, prefix = without_scheme.partition("/")
    blob_path = f"{prefix.rstrip('/')}/{filename}"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        raise FileNotFoundError(f"Artifact not found at gs://{bucket_name}/{blob_path}")

    return blob.download_as_bytes()


def _hstack(x_num, x_cat):
    return hstack([x_num, x_cat])
