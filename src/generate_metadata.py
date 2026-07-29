"""
generate_metadata.py

Run this ONCE, right after training finishes (in the same
notebook/script session, so quantiles/num_cols/cat_cols are already
in memory — this does not retrain anything).

Produces metadata.json, the third artifact predictor.py needs
alongside model.pkl and ohe.pkl.

Usage (paste into your training notebook, after fitting log_model/
xgb_model and OHEncoder):

    from generate_metadata import generate_metadata
    from engineering import compute_quantiles

    quantiles = compute_quantiles(train_df)   # call BEFORE feature engineering columns are added

    generate_metadata(
        quantiles=quantiles,
        num_cols=num_cols,      # the exact list used to build X_train's numeric block
        cat_cols=cat_cols,      # the exact list used to build X_train's categorical block
        model_version="v1",
        output_path="metadata.json",
    )
"""

import json


def generate_metadata(
    quantiles: dict,
    num_cols: list,
    cat_cols: list,
    model_version: str,
    decision_threshold: float = 0.5,
    output_path: str = "metadata.json",
) -> dict:
    """
    Build and write metadata.json.

    Args:
        quantiles: dict with vhpw_q25, vhpw_q75, support_q75 — from
            engineering.compute_quantiles(train_df).
        num_cols: exact ordered list of numeric column names used when
            building the training feature matrix (must match the order
            predictor.py will slice at serving time).
        cat_cols: exact ordered list of categorical column names passed
            into OHEncoder.fit(...).
        model_version: a short version tag, e.g. "v1", "log_model_2".
        decision_threshold: probability cutoff for churn_prediction=1.
        output_path: where to write the file locally before uploading
            to gs://your-bucket/models/<version>/metadata.json.

    Returns:
        The metadata dict that was written (useful if you want to
        print/inspect it before uploading).
    """
    _validate(quantiles, num_cols, cat_cols, model_version)

    metadata = {
        "quantiles": quantiles,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "model_version": model_version,
        "decision_threshold": decision_threshold,
    }

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"metadata.json written to {output_path}")
    print(json.dumps(metadata, indent=2))

    return metadata


def _validate(quantiles: dict, num_cols: list, cat_cols: list, model_version: str) -> None:
    required_quantile_keys = ["vhpw_q25", "vhpw_q75", "support_q75"]
    missing = [k for k in required_quantile_keys if k not in quantiles]
    if missing:
        raise ValueError(f"quantiles is missing required key(s): {missing}")

    if not num_cols:
        raise ValueError("num_cols is empty — predictor.py needs this to slice the feature matrix")
    if not cat_cols:
        raise ValueError("cat_cols is empty — predictor.py needs this to slice the feature matrix")
    if not model_version:
        raise ValueError("model_version must be a non-empty string")
