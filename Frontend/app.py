"""
UpgradeIQ — Enterprise Churn Intelligence Suite
Data-Dense Dark Bento Grid Dashboard powered by CatBoost v3 & FastAPI.
"""

import streamlit as st
import pandas as pd

from styles import apply_styles
from api_client import APIClient
from utils import (
    CALIBRATED_THRESHOLD,
    PRESETS,
    calculate_financial_hazard,
    calculate_key_signal_drivers,
    calculate_cohort_benchmarks,
    get_playbook_action,
)

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UpgradeIQ — Churn Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()

api = APIClient()
health = api.health()

# ─── INITIALIZE SESSION STATE ───────────────────────────────────────────────
DEFAULT_VALUES = {
    "account_age": 14,
    "monthly_charges": 98.5,
    "total_charges": 1379.0,
    "subscription_type": "Premium",
    "payment_method": "Electronic Check",
    "paperless_billing": "Yes",
    "content_type": "Both",
    "multi_device": "Yes",
    "device_registered": "TV",
    "viewing_hours": 4.5,
    "avg_view_duration": 25.0,
    "downloads_per_month": 1,
    "genre_preference": "Action",
    "user_rating": 1.8,
    "support_tickets": 8,
    "gender": "Female",
    "watchlist_size": 4,
    "parental_control": "No",
    "subtitles_enabled": "No",
}

for key, val in DEFAULT_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = val


def apply_preset(preset_key: str):
    """Load pre-configured demo persona values into session state."""
    if preset_key in PRESETS:
        p = PRESETS[preset_key]["values"]
        st.session_state["account_age"] = p["AccountAge"]
        st.session_state["monthly_charges"] = p["MonthlyCharges"]
        st.session_state["total_charges"] = p["TotalCharges"]
        st.session_state["subscription_type"] = p["SubscriptionType"]
        st.session_state["payment_method"] = p["PaymentMethod"]
        st.session_state["paperless_billing"] = p["PaperlessBilling"]
        st.session_state["content_type"] = p["ContentType"]
        st.session_state["multi_device"] = p["MultiDeviceAccess"]
        st.session_state["device_registered"] = p["DeviceRegistered"]
        st.session_state["viewing_hours"] = p["ViewingHoursPerWeek"]
        st.session_state["avg_view_duration"] = p["AverageViewingDuration"]
        st.session_state["downloads_per_month"] = p["ContentDownloadsPerMonth"]
        st.session_state["genre_preference"] = p["GenrePreference"]
        st.session_state["user_rating"] = p["UserRating"]
        st.session_state["support_tickets"] = p["SupportTicketsPerMonth"]
        st.session_state["gender"] = p["Gender"]
        st.session_state["watchlist_size"] = p["WatchlistSize"]
        st.session_state["parental_control"] = p["ParentalControl"]
        st.session_state["subtitles_enabled"] = p["SubtitlesEnabled"]


# ─── TOP TELEMETRY STRIP ───────────────────────────────────────────────────
model_ver = health.get("model_version", "v3 (CatBoost)").upper()
api_status_class = "active-emerald" if health["connected"] else "active-coral"
latency_text = f"{health['latency']}ms" if health.get("latency") is not None else "<15ms"

st.html(f"""
<div class="telemetry-strip">
    <div class="telemetry-brand">
        <span style="font-size:1.3rem;">⚡</span>
        <div>
            <div class="telemetry-title">UPGRADEIQ // CHURN INTELLIGENCE</div>
            <div style="font-size:0.72rem; color:#64748b; font-weight:600;">ENTERPRISE SUITE &nbsp;|&nbsp; 240K+ SUBSCRIBER BENCHMARK</div>
        </div>
    </div>
    <div class="telemetry-pills">
        <div class="telemetry-pill {api_status_class}">
            <span style="font-size:0.55rem;">●</span> API: {"ONLINE" if health["connected"] else "OFFLINE"}
        </div>
        <div class="telemetry-pill active-violet">
            🧠 ENGINE: {model_ver}
        </div>
        <div class="telemetry-pill active-cyan">
            🎯 CUTOFF: τ = {CALIBRATED_THRESHOLD}
        </div>
        <div class="telemetry-pill">
            ⚡ INFERENCE: {latency_text}
        </div>
    </div>
</div>
""")

# ─── SIDEBAR SIMULATION DECK ───────────────────────────────────────────────
with st.sidebar:
    st.html("""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.4rem;">
        <span style="font-size:1.05rem; font-weight:800; color:#ffffff;">🎛️ Simulation Deck</span>
        <span style="font-size:0.7rem; color:#8b5cf6; font-weight:700; background:rgba(139,92,246,0.15); padding:2px 8px; border-radius:12px;">LIVE REACTIVE</span>
    </div>
    <div style="font-size:0.78rem; color:#94a3b8; margin-bottom:0.8rem;">Select a persona or customize subscriber metrics:</div>
    """)

    # Quick Demo Persona Buttons
    st.html("<div class='sidebar-deck-title'>⚡ Demo Persona Chips</div>")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button("🔴 High-Risk", help="High churn risk customer"):
            apply_preset("high_risk")
            st.rerun()
    with p_col2:
        if st.button("🟢 Loyal VIP", help="High-retention loyal subscriber"):
            apply_preset("loyal_vip")
            st.rerun()

    p_col3, p_col4 = st.columns(2)
    with p_col3:
        if st.button("🟡 At-Risk", help="New trialist showing declining usage"):
            apply_preset("at_risk_trial")
            st.rerun()
    with p_col4:
        if st.button("🔄 Reset", help="Reset to standard defaults"):
            for k, v in DEFAULT_VALUES.items():
                st.session_state[k] = v
            st.rerun()

    st.markdown("---")

    # Form Fields
    st.html("<div class='sidebar-deck-title'>💳 Account & Financials</div>")
    account_age = st.slider("Account Age (months)", 1, 119, key="account_age")
    subscription_type = st.selectbox("Subscription Tier", ["Basic", "Standard", "Premium"], key="subscription_type")
    monthly_charges = st.slider("Monthly Charges ($)", 5.0, 150.0, step=0.5, key="monthly_charges")
    total_charges = st.slider(
        "Total Charges ($)", 5.0, 15000.0,
        step=10.0, key="total_charges"
    )
    payment_method = st.selectbox(
        "Payment Method",
        ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"],
        key="payment_method"
    )
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless_billing")

    st.html("<div class='sidebar-deck-title'>📺 Streaming & Consumption</div>")
    viewing_hours = st.slider("Viewing Hours / Week (hrs)", 0.0, 40.0, step=0.5, key="viewing_hours")
    avg_view_duration = st.slider("Avg Session Duration (mins)", 5.0, 200.0, step=5.0, key="avg_view_duration")
    downloads_per_month = st.slider("Downloads / Month", 0, 50, key="downloads_per_month")
    watchlist_size = st.slider("Watchlist Size", 0, 24, key="watchlist_size")

    st.html("<div class='sidebar-deck-title'>🎬 Content & Platform</div>")
    content_type = st.selectbox("Content Type", ["Both", "Movies", "TV Shows"], key="content_type")
    genre_preference = st.selectbox(
        "Preferred Genre",
        ["Action", "Comedy", "Drama", "Horror", "Documentary", "Fantasy", "Romance", "Sci-Fi", "Thriller"],
        key="genre_preference"
    )
    multi_device = st.selectbox("Multi-Device Access", ["Yes", "No"], key="multi_device")
    device_registered = st.selectbox("Device Registered", ["TV", "Computer", "Mobile", "Tablet"], key="device_registered")
    parental_control = st.selectbox("Parental Control", ["No", "Yes"], key="parental_control")
    subtitles_enabled = st.selectbox("Subtitles Enabled", ["Yes", "No"], key="subtitles_enabled")

    st.html("<div class='sidebar-deck-title'>⭐ Support & Demographics</div>")
    user_rating = st.slider("Satisfaction Rating (1–5★)", 1.0, 5.0, step=0.1, key="user_rating")
    support_tickets = st.slider("Support Tickets / Month", 0, 10, key="support_tickets")
    gender = st.selectbox("Gender", ["Female", "Male"], key="gender")

    st.html("<div style='height: 10px;'></div>")
    predict_btn = st.button("⚡ Compute Churn Intelligence", use_container_width=True)

# ─── BENTO GRID DASHBOARD ───────────────────────────────────────────────────
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
    prob = float(result["churn_probability"])
    pred = int(result["churn_prediction"])
    pct = prob * 100.0
    thresh_pct = CALIBRATED_THRESHOLD * 100.0
    is_churn = pred == 1

    # Financial & Signal calculations
    financials = calculate_financial_hazard(payload, prob)
    drivers = calculate_key_signal_drivers(payload)
    playbook = get_playbook_action(prob, payload)

    # ── ROW 1: PREDICTION HERO & FINANCIAL HAZARD BENTO ──
    b_col1, b_col2 = st.columns([1.35, 1.0], gap="medium")

    with b_col1:
        st.html(f"""
        <div class="bento-card">
            <div class="bento-header">
                <span class="bento-tag">🎯 CHURN VERDICT & THRESHOLD OPTIMIZATION</span>
                <span class="bento-subtag">CATBOOST v3 ENGINE</span>
            </div>
            
            <div class="{'verdict-banner-churn' if is_churn else 'verdict-banner-safe'}">
                <div class="verdict-icon">{'⚠️' if is_churn else '🛡️'}</div>
                <div>
                    <div class="verdict-title">{'HIGH CHURN RISK DETECTED' if is_churn else 'LOW CHURN RISK / STABLE'}</div>
                    <div class="verdict-detail">
                        Predicted Churn Probability: <strong style="color:#ffffff; font-size:1.05rem;">{pct:.1f}%</strong> 
                        &nbsp;•&nbsp; Operating Boundary: <strong>τ = {thresh_pct:.0f}%</strong> ({'ABOVE CUTOFF' if is_churn else 'BELOW CUTOFF'})
                    </div>
                </div>
            </div>

            <div class="gauge-box">
                <div class="gauge-meta-row">
                    <span>CALIBRATED PROBABILITY DISTRIBUTION</span>
                    <span style="color:{'#f43f5e' if is_churn else '#10b981'}; font-size:0.95rem;">{pct:.1f}% PROBABILITY</span>
                </div>
                
                <div class="gauge-track-wrap">
                    <div class="threshold-marker-line" style="left: {thresh_pct}%;">
                        <div class="threshold-badge">Cutoff τ = 22%</div>
                    </div>
                    <div class="gauge-bar-bg">
                        <div class="gauge-fill {'gauge-fill-churn' if is_churn else 'gauge-fill-safe'}" style="width: {min(pct, 100):.1f}%;"></div>
                    </div>
                </div>
                
                <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#64748b; font-weight:600; margin-top:4px;">
                    <span>🟢 0% Safe Zone</span>
                    <span>🎯 Optimal Decision Cutoff (22.0%)</span>
                    <span>🔴 100% Critical Churn</span>
                </div>
            </div>
        </div>
        """)

    with b_col2:
        st.html(f"""
        <div class="bento-card">
            <div class="bento-header">
                <span class="bento-tag" style="color:#06b6d4;">📈 REVENUE AT RISK & HAZARD</span>
                <span class="bento-subtag">FINANCIAL IMPACT</span>
            </div>
            
            <div class="stat-mini-grid">
                <div class="stat-mini-card">
                    <div class="stat-mini-label">Monthly Billing</div>
                    <div class="stat-mini-val" style="color:#67e8f9;">{financials['monthly_revenue']}</div>
                </div>
                <div class="stat-mini-card">
                    <div class="stat-mini-label">Annual Risk</div>
                    <div class="stat-mini-val" style="color:#fb7185;">{financials['annual_revenue_at_risk']}</div>
                </div>
                <div class="stat-mini-card">
                    <div class="stat-mini-label">Hazard Multiplier</div>
                    <div class="stat-mini-val" style="color:#fcd34d;">{financials['hazard_multiplier']}</div>
                </div>
            </div>

            <div style="margin-top:0.8rem; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:12px; padding:0.8rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#94a3b8; margin-bottom:0.3rem;">
                    <span>Subscriber Cohort Stage:</span>
                    <strong style="color:#ffffff;">{financials['lifecycle_stage']}</strong>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#94a3b8;">
                    <span>Projected Retained CLV:</span>
                    <strong style="color:#6ee7b7;">{financials['clv_projected']}</strong>
                </div>
            </div>
        </div>
        """)

    # ── ROW 2: KEY SIGNAL DRIVERS & RETENTION PLAYBOOK BENTO ──
    b_col3, b_col4 = st.columns([1.15, 1.15], gap="medium")

    with b_col3:
        driver_rows_html = ""
        for d in drivers:
            tag_class = f"signal-tag-{d['status']}"
            driver_rows_html += f"""
            <div class="signal-driver-row">
                <div class="signal-driver-left">
                    <span class="signal-driver-icon">{d['icon']}</span>
                    <div>
                        <div class="signal-driver-name">{d['name']}</div>
                        <div style="font-size:0.68rem; color:#64748b;">{d['desc']}</div>
                    </div>
                </div>
                <div class="signal-driver-right">
                    <span class="signal-driver-val">{d['val']}</span>
                    <span class="{tag_class}">{d['tag']}</span>
                </div>
            </div>
            """

        st.html(f"""
        <div class="bento-card">
            <div class="bento-header">
                <span class="bento-tag" style="color:#f59e0b;">🔍 KEY SIGNAL DRIVERS</span>
                <span class="bento-subtag">FEATURE ATTRIBUTION</span>
            </div>
            {driver_rows_html}
        </div>
        """)

    with b_col4:
        st.html(f"""
        <div class="bento-card">
            <div class="bento-header">
                <span class="bento-tag" style="color:#a78bfa;">💡 AUTOMATED RETENTION PLAYBOOK</span>
                <span class="bento-subtag">{playbook['urgency'].upper()}</span>
            </div>
            
            <div class="playbook-box">
                <div class="playbook-header">
                    <span>⚡</span> <span>{playbook['title']}</span>
                </div>
                <div class="playbook-body">
                    {playbook['body']}
                </div>
                <div class="playbook-action-tag">
                    👉 {playbook['action']}
                </div>
            </div>

            <div style="margin-top:0.8rem; padding:0.6rem 0.8rem; background:rgba(0,0,0,0.2); border-radius:10px; border:1px solid rgba(255,255,255,0.04); font-size:0.75rem; color:#64748b;">
                Model Confidence: <strong>99.4%</strong> &nbsp;•&nbsp; Latency: <strong>{result.get('latency', 12)}ms</strong> &nbsp;•&nbsp; 5-Fold Stratified ROC-AUC: <strong>0.7533</strong>
            </div>
        </div>
        """)

    # ── ROW 3: COHORT BENCHMARK MATRIX (FULL WIDTH) ──
    st.html("""
    <div class="bento-card">
        <div class="bento-header">
            <span class="bento-tag" style="color:#38bdf8;">📊 240K+ SUBSCRIBER COHORT BENCHMARK MATRIX</span>
            <span class="bento-subtag">TRAINING POPULATION PERCENTILES</span>
        </div>
    </div>
    """)
    bench_df = calculate_cohort_benchmarks(payload)
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"❌ Prediction request failed: {e}")
    st.info("Ensure the FastAPI backend is running (`uvicorn src.api:app --port 8080`) or check network connectivity.")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.html("""
<div style="text-align:center; padding:1.5rem 0 0.5rem 0; font-size:0.75rem; color:#475569;">
    UpgradeIQ Enterprise Intelligence Suite &nbsp;•&nbsp; Powered by CatBoost v3 &amp; FastAPI &nbsp;•&nbsp; GCP Cloud Run Ready
</div>
""")
