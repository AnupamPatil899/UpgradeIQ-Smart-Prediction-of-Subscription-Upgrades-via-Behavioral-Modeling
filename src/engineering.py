"""
engineering.py

Single source of truth for feature engineering.

This module is imported by BOTH the training script and the serving
predictor. There is no second copy of this logic anywhere else in the
repo. That is intentional: it is what prevents train/serve skew, which
is the class of bug (see the old train_df/test_df leakage) this file
exists to eliminate.

Quantile thresholds (vhpw_q25, vhpw_q75, support_q75) must be computed
ONCE at training time, from the training split only, and then reused
as-is at serving time. They are never recomputed from live/incoming
data. This module does not compute them — see `compute_quantiles`
below for that, which training calls once; serving only ever calls
`add_engineered_features` with quantiles loaded from metadata.json.
"""

import pandas as pd


# Columns required to be present before engineering can run.
REQUIRED_RAW_COLUMNS = [
    "AccountAge",
    "MonthlyCharges",
    "ViewingHoursPerWeek",
    "ContentDownloadsPerMonth",
    "WatchlistSize",
    "SupportTicketsPerMonth",
    "UserRating",
]

# Keys expected inside the `quantiles` dict passed to add_engineered_features.
QUANTILE_KEYS = ["vhpw_q25", "vhpw_q75", "support_q75"]


def compute_quantiles(df: pd.DataFrame) -> dict:
    """
    Compute the quantile thresholds used by feature engineering.

    Call this ONCE during training, on the training split BEFORE any
    train/test split leakage can occur (i.e. compute on X_train, not
    on the full dataset). Persist the returned dict into metadata.json
    so serving can reuse the exact same thresholds.
    """
    _validate_columns(df, ["ViewingHoursPerWeek", "SupportTicketsPerMonth"])

    return {
        "vhpw_q25": float(df["ViewingHoursPerWeek"].quantile(0.25)),
        "vhpw_q75": float(df["ViewingHoursPerWeek"].quantile(0.75)),
        "support_q75": float(df["SupportTicketsPerMonth"].quantile(0.75)),
    }


def add_engineered_features(df: pd.DataFrame, quantiles: dict) -> pd.DataFrame:
    """
    Apply the full set of engineered features to a dataframe.

    Works identically whether `df` has one row (a single serving
    request) or the full training set — that consistency is the
    entire point of this function existing as shared code.

    Args:
        df: raw dataframe with the original schema columns
            (CustomerID and Churn should already be dropped before
            this is called; SubscriptionType should already be
            numerically encoded before this is called).
        quantiles: dict with keys vhpw_q25, vhpw_q75, support_q75 —
            must come from `compute_quantiles` at training time, or
            from a loaded metadata.json at serving time. Never
            recomputed here.

    Returns:
        A new dataframe (input is not mutated) with engineered
        columns appended.
    """
    _validate_columns(df, REQUIRED_RAW_COLUMNS)
    _validate_quantiles(quantiles)

    vhpw_q25 = quantiles["vhpw_q25"]
    vhpw_q75 = quantiles["vhpw_q75"]
    support_q75 = quantiles["support_q75"]

    df = df.copy()

    # small epsilon guards against division by zero on edge-case inputs
    # (e.g. AccountAge == 0, ViewingHoursPerWeek == 0)
    eps = 1e-9

    df["valueperhourmonthly"] = df["MonthlyCharges"] / (df["ViewingHoursPerWeek"] * 4 + eps)
    df["avgmonthlyusage"] = (df["ViewingHoursPerWeek"] * 4) / (df["AccountAge"] + eps)
    df["EngagementScore"] = (
        df["ContentDownloadsPerMonth"] + df["WatchlistSize"] + (df["ViewingHoursPerWeek"] * 4)
    ) / 3
    df["SupportIntensity"] = df["SupportTicketsPerMonth"] / (df["AccountAge"] + 1)
    df["HighSatisfaction"] = (df["UserRating"] >= 4.0).astype(int)
    df["ChargesToAge_Ratio"] = df["MonthlyCharges"] / (df["AccountAge"] + 1)
    df["EngagementSatisfaction"] = df["ViewingHoursPerWeek"] * df["UserRating"]
    df["Highwatching"] = (df["ViewingHoursPerWeek"] > vhpw_q75).astype(int)
    df["RecentActivityDrop"] = (
        (df["ViewingHoursPerWeek"] < vhpw_q25) & (df["AccountAge"] > 6)
    ).astype(int)
    df["Low_view_monthly"] = ((df["ViewingHoursPerWeek"] * 4) < (vhpw_q25 * 4)).astype(int)
    df["LowSatisfaction"] = (df["UserRating"] <= 2.0).astype(int)
    df["HighSupport"] = (df["SupportTicketsPerMonth"] > support_q75).astype(int)
    # Total_risk_score is a 4-term sum. Low_view_session was dropped as a
    # duplicate of Low_view_monthly during training experimentation — if
    # this changes again, this is the one place to update it.
    df["Total_risk_score"] = (
        df["Low_view_monthly"]
        + df["LowSatisfaction"]
        + df["HighSupport"]
        + df["RecentActivityDrop"]
    )

    return df


def encode_subscription_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map SubscriptionType from string to the numeric encoding used
    consistently across training and serving.

    Kept as its own small function (rather than a free-floating
    .replace() call) so both training and serving call the exact
    same mapping, and so the mapping is only defined in one place.
    """
    mapping = {"Basic": 0, "Standard": 1, "Premium": 2}
    df = df.copy()

    # Checking `dtype == object` is not reliable across pandas versions —
    # newer pandas can default string columns to a dedicated string dtype
    # rather than `object`. Checking "is this NOT numeric" instead is
    # correct regardless of which string dtype pandas happens to use.
    if not pd.api.types.is_numeric_dtype(df["SubscriptionType"]):
        unknown = set(df["SubscriptionType"].unique()) - set(mapping.keys())
        if unknown:
            raise ValueError(f"Unknown SubscriptionType value(s): {unknown}")
        df["SubscriptionType"] = df["SubscriptionType"].map(mapping)

    return df


def _validate_columns(df: pd.DataFrame, required_columns: list) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) for feature engineering: {missing}")


def _validate_quantiles(quantiles: dict) -> None:
    missing = [k for k in QUANTILE_KEYS if k not in quantiles]
    if missing:
        raise ValueError(f"Missing required quantile key(s): {missing}")
