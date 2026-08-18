"""
engineering.py — Shared Feature Engineering Module
Single source of truth for all feature transformations.
Pure functions, zero I/O, zero side effects. Only depends on pandas and numpy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List

REQUIRED_RAW_COLUMNS: List[str] = [
    "AccountAge",
    "MonthlyCharges",
    "TotalCharges",
    "SubscriptionType",
    "ViewingHoursPerWeek",
    "AverageViewingDuration",
    "ContentDownloadsPerMonth",
    "UserRating",
    "SupportTicketsPerMonth",
    "WatchlistSize",
]

QUANTILE_KEYS: List[str] = ["vhpw_q25", "vhpw_q75", "support_q75"]


def encode_subscription_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode SubscriptionType categorical column to numeric values (Basic: 0, Standard: 1, Premium: 2).
    Safe for both raw string columns and already numeric columns.
    """
    df = df.copy()
    if "SubscriptionType" in df.columns:
        mapping = {"Basic": 0, "Standard": 1, "Premium": 2}
        if not pd.api.types.is_numeric_dtype(df["SubscriptionType"]):
            df["SubscriptionType"] = df["SubscriptionType"].map(mapping).fillna(0).astype(int)
    return df


def compute_quantiles(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute distribution quantiles from training dataset.
    Must be called ONLY on training partitions (X_train) to prevent data leakage.
    """
    for col in ["ViewingHoursPerWeek", "SupportTicketsPerMonth"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' required to compute quantiles.")

    return {
        "vhpw_q25": float(df["ViewingHoursPerWeek"].quantile(0.25)),
        "vhpw_q75": float(df["ViewingHoursPerWeek"].quantile(0.75)),
        "support_q75": float(df["SupportTicketsPerMonth"].quantile(0.75)),
    }


def add_engineered_features(df: pd.DataFrame, quantiles: Dict[str, float]) -> pd.DataFrame:
    """
    Apply feature engineering transformations to a dataframe using given quantile thresholds.
    Accepts both batch training data and single-row inference payloads.
    """
    _validate_columns(df, REQUIRED_RAW_COLUMNS)
    _validate_quantiles(quantiles)

    vhpw_q25 = quantiles["vhpw_q25"]
    vhpw_q75 = quantiles["vhpw_q75"]
    support_q75 = quantiles["support_q75"]

    df = df.copy()
    eps = 1e-9

    # 1. Economic & Tenure Metrics
    df["valueperhourmonthly"] = df["MonthlyCharges"] / (df["ViewingHoursPerWeek"] * 4 + eps)
    df["avgmonthlyusage"] = (df["ViewingHoursPerWeek"] * 4) / (df["AccountAge"] + eps)
    df["ChargesToAge_Ratio"] = df["MonthlyCharges"] / (df["AccountAge"] + 1)
    df["TenureSpendConsistency"] = df["TotalCharges"] / (df["AccountAge"] * df["MonthlyCharges"] + eps)

    # 2. Engagement & Interaction Metrics
    df["EngagementScore"] = (
        df["ContentDownloadsPerMonth"] + df["WatchlistSize"] + (df["ViewingHoursPerWeek"] * 4)
    ) / 3
    df["EngagementSatisfaction"] = df["ViewingHoursPerWeek"] * df["UserRating"]
    df["DownloadIntensity"] = df["ContentDownloadsPerMonth"] / (df["ViewingHoursPerWeek"] + 1.0)
    df["ViewingSessionRatio"] = df["AverageViewingDuration"] / ((df["ViewingHoursPerWeek"] * 60 / 7) + eps)

    # 3. Frustration & Support Intensity
    df["SupportIntensity"] = df["SupportTicketsPerMonth"] / (df["AccountAge"] + 1)
    df["FrustrationIndex"] = df["SupportTicketsPerMonth"] / (df["ViewingHoursPerWeek"] + 1.0)

    # 4. Behavioral Flags & Threshold Indicators
    df["HighSatisfaction"] = (df["UserRating"] >= 4.0).astype(int)
    df["LowSatisfaction"] = (df["UserRating"] <= 2.0).astype(int)
    df["Highwatching"] = (df["ViewingHoursPerWeek"] > vhpw_q75).astype(int)
    df["Low_view_monthly"] = ((df["ViewingHoursPerWeek"] * 4) < (vhpw_q25 * 4)).astype(int)
    df["HighSupport"] = (df["SupportTicketsPerMonth"] > support_q75).astype(int)
    df["RecentActivityDrop"] = (
        (df["ViewingHoursPerWeek"] < vhpw_q25) & (df["AccountAge"] > 6)
    ).astype(int)

    # 5. Composite Risk Score (Cleaned without duplicate flags)
    df["Total_risk_score"] = (
        df["Low_view_monthly"]
        + df["LowSatisfaction"]
        + df["HighSupport"]
        + df["RecentActivityDrop"]
    )

    return df


def _validate_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")


def _validate_quantiles(quantiles: Dict[str, float]) -> None:
    missing = [k for k in QUANTILE_KEYS if k not in quantiles]
    if missing:
        raise ValueError(f"Missing required quantile keys: {missing}")
