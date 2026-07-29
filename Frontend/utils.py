"""
Utility functions used by Streamlit UI.
"""

import pandas as pd

# Model quantiles matching metadata.json (v1)
VHPW_Q25 = 10.76
VHPW_Q75 = 30.22
SUPPORT_Q75 = 7.0


def calculate_engineered_metrics(data: dict) -> dict:
    account_age = data["AccountAge"]
    viewing = data["ViewingHoursPerWeek"]
    downloads = data["ContentDownloadsPerMonth"]
    watchlist = data["WatchlistSize"]
    monthly = data["MonthlyCharges"]
    support = data["SupportTicketsPerMonth"]
    rating = data["UserRating"]

    engagement = (downloads + watchlist + viewing * 4) / 3
    support_intensity = support / (account_age + 1)
    charge_age = monthly / (account_age + 1)
    engagement_rating = viewing * rating

    low_view = int((viewing * 4) < (VHPW_Q25 * 4))
    low_sat = int(rating <= 2.0)
    high_sup = int(support > SUPPORT_Q75)
    recent_drop = int((viewing < VHPW_Q25) and (account_age > 6))

    total_risk = low_view + low_sat + high_sup + recent_drop

    return {
        "Engagement Score": round(engagement, 2),
        "Support Intensity": round(support_intensity, 4),
        "Charges:Age Ratio": round(charge_age, 2),
        "Engagement × Satisfaction": round(engagement_rating, 2),
        "Total Risk Score": total_risk,
    }


def risk_badges(data: dict) -> list:
    viewing = data["ViewingHoursPerWeek"]
    account_age = data["AccountAge"]
    rating = data["UserRating"]
    support = data["SupportTicketsPerMonth"]

    badges = []

    # Risk factors (high risk)
    if (viewing * 4) < (VHPW_Q25 * 4):
        badges.append(("🔴 Low Monthly Viewing", "high"))

    if (viewing < VHPW_Q25) and (account_age > 6):
        badges.append(("🔴 Recent Activity Drop", "high"))

    if rating <= 2.0:
        badges.append(("🔴 Low Satisfaction", "high"))

    if support > SUPPORT_Q75:
        badges.append(("🔴 High Support Usage", "high"))

    # Positive factors (good)
    if viewing > VHPW_Q75:
        badges.append(("🟢 High Watcher", "good"))

    if rating >= 4.0:
        badges.append(("🟢 High Satisfaction", "good"))

    if not badges:
        badges.append(("🟡 Standard Risk Profile", "neutral"))

    return badges


def recommendation(probability: float) -> tuple:
    if probability >= 0.70:
        return (
            "🚨 Immediate intervention needed. Offer a discount or upgrade incentive. Assign a customer success rep.",
            "error",
        )
    elif probability >= 0.50:
        return (
            "⚠️ Monitor closely. Send a targeted win-back email or personalized content recommendations.",
            "warning",
        )
    elif probability >= 0.30:
        return (
            "ℹ️ Moderate risk. Keep up regular communication and engagement campaigns.",
            "info",
        )
    return (
        "✅ Loyal customer. Great candidate for upselling to a higher tier or referral programs.",
        "success",
    )


def metrics_dataframe(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value": [str(v) for v in metrics.values()]
    })