"""
dataset.py — Data Ingestion & Leak-Free Preprocessing Pipeline
Handles data loading, fallback searches, train/test splitting, and fold transformers.
"""

import os
import zipfile
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from engineering import encode_subscription_type, compute_quantiles, add_engineered_features


def find_and_load_data(path: str) -> pd.DataFrame:
    """Load dataset with multi-location fallback search."""
    resolved_path = os.path.abspath(path)
    print(f"[DATA] Attempting to load from: {resolved_path}")

    if not os.path.exists(resolved_path):
        fallbacks = [
            "../Dataset/train.csv",
            "/home/anupa/Upgradeiq/Dataset/train.csv",
            "./Dataset/train.csv",
            "../Dataset/Dataset.zip",
            "../Datasets/Train_test_all/train.csv",
            "../Datasets/Train_test_all/Dataset.zip",
            "../../Datasets/Train_test_all/train.csv",
            "./Dataset.zip",
            "../July_2026_docs/Dataset.zip",
            "./train.csv",
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                resolved_path = os.path.abspath(fb)
                print(f"[DATA] Found dataset at fallback: {resolved_path}")
                break

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}' or any standard fallback paths.\n"
            f"Please ensure train.csv or Dataset.zip is accessible."
        )

    if resolved_path.endswith(".zip"):
        with zipfile.ZipFile(resolved_path) as z:
            csv_names = [f for f in z.namelist() if f.endswith(".csv")]
            if not csv_names:
                raise FileNotFoundError(f"No CSV file found in archive {resolved_path}")
            with z.open(csv_names[0]) as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(resolved_path)

    print(f"[DATA] Successfully loaded {len(df):,} rows, {len(df.columns)} columns.")
    return df


class DataPipeline:
    """
    Leak-Free Data Pipeline.
    Guarantees quantiles and categorical encoders are fitted ONLY on training partitions.
    """

    def __init__(self, cat_cols: Optional[List[str]] = None):
        self.cat_cols = cat_cols or [
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
        self.num_cols: List[str] = []
        self.quantiles: Dict[str, float] = {}
        self.ohe: Optional[OneHotEncoder] = None
        self.feature_names: List[str] = []

    def fit_transform_train(self, df_train: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        """
        Fit transformers and compute quantiles on training split, returning (X_train_matrix, y_train).
        """
        df = df_train.copy()
        if "CustomerID" in df.columns:
            df = df.drop(columns=["CustomerID"])

        if "Churn" not in df.columns:
            raise KeyError("Target column 'Churn' not found in training data.")

        y = df["Churn"].astype(int)
        df_raw = df.drop(columns=["Churn"])

        # 1. Encode subscription type
        df_raw = encode_subscription_type(df_raw)

        # 2. Compute Quantiles strictly on train
        self.quantiles = compute_quantiles(df_raw)

        # 3. Add engineered features
        df_engineered = add_engineered_features(df_raw, self.quantiles)

        # 4. Resolve column types
        self.num_cols = [c for c in df_engineered.columns if c not in self.cat_cols]

        # 5. Fit OneHotEncoder strictly on train
        try:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        x_cat = self.ohe.fit_transform(df_engineered[self.cat_cols])
        x_num = df_engineered[self.num_cols].values.astype(float)
        x_full = np.hstack([x_num, x_cat])

        # Track feature names
        try:
            ohe_feature_names = list(self.ohe.get_feature_names_out(self.cat_cols))
        except AttributeError:
            ohe_feature_names = [f"cat_{i}" for i in range(x_cat.shape[1])]
        self.feature_names = self.num_cols + ohe_feature_names

        return x_full, y

    def transform_eval(self, df_eval: pd.DataFrame) -> Tuple[np.ndarray, Optional[pd.Series]]:
        """
        Transform validation/test split using pre-fitted encoders and quantiles.
        """
        if self.ohe is None or not self.quantiles:
            raise RuntimeError("Pipeline must be fitted with fit_transform_train first.")

        df = df_eval.copy()
        if "CustomerID" in df.columns:
            df = df.drop(columns=["CustomerID"])

        y = df["Churn"].astype(int) if "Churn" in df.columns else None
        df_raw = df.drop(columns=["Churn"]) if "Churn" in df.columns else df

        df_raw = encode_subscription_type(df_raw)
        df_engineered = add_engineered_features(df_raw, self.quantiles)

        x_cat = self.ohe.transform(df_engineered[self.cat_cols])
        x_num = df_engineered[self.num_cols].values.astype(float)
        x_full = np.hstack([x_num, x_cat])

        return x_full, y
