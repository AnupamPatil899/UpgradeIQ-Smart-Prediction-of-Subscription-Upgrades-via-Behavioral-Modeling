"""
UpgradeIQ Streamlit Frontend
Uses FastAPI backend deployed on Cloud Run / Local Uvicorn.
"""

import streamlit as st
import pandas as pd

from styles import apply_styles
from api_client import APIClient
from utils import (
    calculate_engineered_metrics,
    metrics_dataframe,
    recommendation,
    risk_badges,
)

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UpgradeIQ — Churn Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()

api = APIClient()
health = api.health()

# ─── HERO SECTION ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>⚡ UpgradeIQ</h1>
  <p>Smart Prediction of Subscription Upgrades &amp; Churn &nbsp;|&nbsp; Enterprise AI Suite</p>
</div>
""", unsafe_allow_html=True)

# ─── STATUS HEADER ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
cards = [
    ("Backend", "🟢 Connected" if health["connected"] else "🔴 Offline"),
    ("Model Version", health.get("model_version", "-")),
    ("Latency", f"{health['latency']} ms" if health.get("latency") is not None else "-"),
    ("API Status", health.get("status", "offline")),
]
for col, (l, v) in zip((c1, c2, c3, c4), cards):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">{l}</div>
            <div class="value">{v}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── SIDEBAR FORM ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧑‍💼 Customer Profile")

    st.markdown("<div class='section-title'>Account</div>", unsafe_allow_html=True)
    account_age = st.slider("Account Age (months)", 1, 119, 30)
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    payment_method = st.selectbox("Payment Method",
        ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.markdown("<div class='section-title'>Financials</div>", unsafe_allow_html=True)
    monthly_charges = st.slider("Monthly Charges ($)", 5.0, 150.0, 50.0, step=0.5)
    total_charges = st.slider("Total Charges ($)", 5.0, 15000.0,
                                float(round(monthly_charges * account_age, 1)), step=10.0)

    st.markdown("<div class='section-title'>Viewing Behaviour</div>", unsafe_allow_html=True)
    viewing_hours = st.slider("Viewing Hours / Week (hrs)", 0.0, 40.0, 10.0, step=0.5)
    avg_view_duration = st.slider("Avg Session Duration (mins)", 5.0, 200.0, 60.0, step=5.0)
    downloads_per_month = st.slider("Content Downloads / Month", 0, 50, 5)
    watchlist_size = st.slider("Watchlist Size", 0, 24, 10)

    st.markdown("<div class='section-title'>Content Preferences</div>", unsafe_allow_html=True)
    content_type = st.selectbox("Content Type", ["Movies", "TV Shows", "Both"])
    genre_preference = st.selectbox("Genre Preference",
        ["Action", "Comedy", "Drama", "Horror", "Documentary", "Fantasy", "Romance", "Sci-Fi", "Thriller"])
    multi_device = st.selectbox("Multi-Device Access", ["Yes", "No"])
    device_registered = st.selectbox("Device Registered",
        ["Mobile", "Tablet", "Computer", "TV"])
    parental_control = st.selectbox("Parental Control", ["Yes", "No"])
    subtitles_enabled = st.selectbox("Subtitles Enabled", ["Yes", "No"])

    st.markdown("<div class='section-title'>Satisfaction & Support</div>", unsafe_allow_html=True)
    user_rating = st.slider("User Rating (1–5)", 1.0, 5.0, 3.5, step=0.1)
    support_tickets = st.slider("Support Tickets / Month", 0, 10, 1)

    st.markdown("<div class='section-title'>Demographics</div>", unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Predict Churn", use_container_width=True)

# ─── MAIN PANEL ──────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.4, 1], gap="large")

if not predict_btn:
    with left_col:
        st.markdown("""
        <div class="info-box">
        👈 &nbsp; <strong>Fill in the customer profile</strong> in the left sidebar,
        then click <strong>⚡ Predict Churn</strong> for an instant prediction.
        </div>
        <div class="info-box">
        🔬 <strong>How it works:</strong> Predictions are processed in real-time by an XGBoost model
        served via FastAPI microservice. The model uses <strong>28+ engineered features</strong> including
        engagement scores, support intensity, activity-drop signals, and charge-to-age ratios.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 Feature Guide")
        feat_df = pd.DataFrame({
            "Feature": ["AccountAge", "SubscriptionType", "MonthlyCharges",
                        "ViewingHoursPerWeek", "UserRating", "SupportTicketsPerMonth",
                        "ContentDownloadsPerMonth", "WatchlistSize"],
            "Description": [
                "How long the customer has been subscribed (months)",
                "Plan tier: Basic / Standard / Premium",
                "Current monthly billing in USD",
                "Average weekly streaming hours",
                "Customer satisfaction rating (1–5★)",
                "Recent complaint / support frequency",
                "Offline content download activity",
                "Items saved for later but not yet watched",
            ]
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    with right_col:
        st.markdown("### ℹ️ Microservice Architecture")
        st.markdown("""
        - **Frontend**: Streamlit (Python 3.11)
        - **Backend**: FastAPI / Uvicorn REST API
        - **Model Engine**: XGBoost Classifier with SMOTE
        - **Deployment**: GCP Cloud Run Ready (Stateless Container)
        """)

else:
    payload = {
        "AccountAge": account_age,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SubscriptionType": subscription_type,
        "PaymentMethod": payment_method,
        "PaperlessBilling": paperless_billing,
        "ContentType": content_type,
        "MultiDeviceAccess": multi_device,
        "DeviceRegistered": device_registered,
        "ViewingHoursPerWeek": viewing_hours,
        "AverageViewingDuration": avg_view_duration,
        "ContentDownloadsPerMonth": downloads_per_month,
        "GenrePreference": genre_preference,
        "UserRating": user_rating,
        "SupportTicketsPerMonth": support_tickets,
        "Gender": gender,
        "WatchlistSize": watchlist_size,
        "ParentalControl": parental_control,
        "SubtitlesEnabled": subtitles_enabled,
    }

    try:
        result = api.predict(payload)
        prob = result["churn_probability"]
        pred = result["churn_prediction"]

        with left_col:
            # Result Card
            if pred == 1:
                st.markdown(f"""
                <div class="result-churn">
                    <div class="result-icon">⚠️</div>
                    <div class="result-title">High Churn Risk</div>
                    <div class="result-prob">This customer is likely to cancel their subscription.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <div class="result-icon">✅</div>
                    <div class="result-title">Low Churn Risk</div>
                    <div class="result-prob">This customer is likely to stay or upgrade.</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability Gauge Bar
            pct = f"{prob * 100:.1f}"
            st.markdown(f"<div class='prob-label'>Churn Probability: <strong>{pct}%</strong></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill" style="width:{pct}%"></div>
            </div>
            <div style='display:flex; justify-content:space-between; margin-top:4px;'>
                <span style='color:#7effc4; font-size:0.8rem'>0% — Low Risk</span>
                <span style='color:#ff8fa3; font-size:0.8rem'>100% — Certain Churn</span>
            </div>""", unsafe_allow_html=True)

        with right_col:
            st.markdown("### 🔍 Risk Factor Breakdown")
            for label, badge_type in risk_badges(payload):
                if badge_type == "high":
                    cls = "risk-high"
                elif badge_type == "good":
                    cls = "risk-low"
                else:
                    cls = "risk-neutral"
                st.markdown(f"<span class='risk-badge {cls}'>{label}</span>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 Engineered Metrics")
            metrics = calculate_engineered_metrics(payload)
            st.dataframe(metrics_dataframe(metrics), use_container_width=True, hide_index=True)

            st.markdown("### 💡 Recommended Action")
            msg, level = recommendation(prob)
            getattr(st, level)(msg)

            st.caption(f"Model Version: `{result.get('model_version', 'v1')}` | Inference Latency: `{result.get('latency', 0)} ms`")

    except Exception as e:
        with left_col:
            st.error(f"❌ Prediction request failed: {e}")
            st.info("Ensure the FastAPI backend is running (`uvicorn src.api:app --port 8080`) or check `API_URL` environment variable.")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("<br><hr style='border-color:rgba(255,255,255,0.1)'>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#555577; font-size:0.8rem;'>
UpgradeIQ &nbsp;·&nbsp; Streamlit &amp; FastAPI Architecture &nbsp;·&nbsp; GCP Cloud Run Ready
</p>""", unsafe_allow_html=True)

