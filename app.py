"""
UpgradeIQ — Smart Prediction of Subscription Upgrades & Churn
Streamlit App — locally runnable, reads Dataset.zip automatically
"""

import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UpgradeIQ — Churn Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e0e0ff;
}
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.1);
}
section[data-testid="stSidebar"] * { color: #d4d4f7 !important; }

.hero {
    background: linear-gradient(90deg, #6c63ff 0%, #48b1bf 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; color: white;
}
.hero h1 { font-size: 2.6rem; font-weight: 900; margin: 0; letter-spacing:-1px; }
.hero p  { font-size: 1.05rem; opacity: 0.85; margin-top: 0.4rem; }

.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
}
.metric-card .label { font-size: 0.8rem; color: #a0a0cc; text-transform: uppercase; letter-spacing: 1px; }
.metric-card .value { font-size: 2rem; font-weight: 700; color: #ffffff; }

.result-churn {
    background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    border-radius: 16px; padding: 2rem; text-align: center; color: white;
    box-shadow: 0 8px 32px rgba(255,65,108,0.4);
}
.result-safe {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    border-radius: 16px; padding: 2rem; text-align: center; color: white;
    box-shadow: 0 8px 32px rgba(56,239,125,0.3);
}
.result-icon  { font-size: 3.5rem; }
.result-title { font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0; }
.result-prob  { font-size: 1rem; opacity: 0.9; }

.prob-bar-wrap {
    background: rgba(255,255,255,0.1);
    border-radius: 50px; height: 22px; margin: 0.6rem 0; overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 50px; background: linear-gradient(90deg, #38ef7d, #ff416c); }
.prob-label { font-size: 0.78rem; color: #a0a0cc; margin-bottom: 4px; }

.risk-badge {
    display: inline-block; padding: 0.25rem 0.8rem; border-radius: 20px;
    font-size: 0.82rem; font-weight: 600; margin: 0.2rem;
}
.risk-high   { background: rgba(255,65,108,0.25); color: #ff8fa3; border: 1px solid #ff416c; }
.risk-low    { background: rgba(56,239,125,0.2);  color: #7effc4; border: 1px solid #38ef7d; }
.risk-neutral{ background: rgba(255,165,0,0.2);   color: #ffd580; border: 1px solid #ffa500; }

.stButton > button {
    background: linear-gradient(90deg, #6c63ff, #48b1bf);
    color: white; border: none; border-radius: 10px;
    padding: 0.75rem 2.5rem; font-size: 1.05rem; font-weight: 700; width: 100%;
}
.stButton > button:hover { box-shadow: 0 6px 20px rgba(108,99,255,0.5); }

.info-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(108,99,255,0.4);
    border-radius: 12px; padding: 1rem 1.4rem; margin: 0.8rem 0;
    font-size: 0.9rem; color: #c0c0e0;
}
.info-box strong { color: #c8b6ff; }
.section-title {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #7c6fff; margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────
def add_engineered_features(df: pd.DataFrame, quantiles: dict) -> pd.DataFrame:
    """Apply identical feature engineering to train df or a single-row inference df."""
    vhpw_q25    = quantiles["vhpw_q25"]
    vhpw_q75    = quantiles["vhpw_q75"]
    support_q75 = quantiles["support_q75"]

    df = df.copy()

    df["valueperhourmonthly"]    = df["MonthlyCharges"] / (df["ViewingHoursPerWeek"] * 4 + 1e-9)
    df["avgmonthlyusage"]        = (df["ViewingHoursPerWeek"] * 4) / (df["AccountAge"] + 1e-9)
    df["EngagementScore"]        = (df["ContentDownloadsPerMonth"] + df["WatchlistSize"] + (df["ViewingHoursPerWeek"] * 4)) / 3
    df["SupportIntensity"]       = df["SupportTicketsPerMonth"] / (df["AccountAge"] + 1)
    df["HighSatisfaction"]       = (df["UserRating"] >= 4.0).astype(int)
    df["ChargesToAge_Ratio"]     = df["MonthlyCharges"] / (df["AccountAge"] + 1)
    df["EngagementSatisfaction"] = df["ViewingHoursPerWeek"] * df["UserRating"]
    df["Highwatching"]           = (df["ViewingHoursPerWeek"] > vhpw_q75).astype(int)
    df["RecentActivityDrop"]     = ((df["ViewingHoursPerWeek"] < vhpw_q25) & (df["AccountAge"] > 6)).astype(int)
    df["Low_view_monthly"]       = ((df["ViewingHoursPerWeek"] * 4) < (vhpw_q25 * 4)).astype(int)
    df["Low_view_session"]       = df["Low_view_monthly"]
    df["LowSatisfaction"]        = (df["UserRating"] <= 2.0).astype(int)
    df["HighSupport"]            = (df["SupportTicketsPerMonth"] > support_q75).astype(int)
    df["Total_risk_score"]       = (df["Low_view_monthly"] + df["Low_view_session"] +
                                    df["LowSatisfaction"] + df["HighSupport"] + df["RecentActivityDrop"])
    return df


# ─── TRAINING PIPELINE (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model():
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score

    # ── Load data from zip
    zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset.zip")
    with zipfile.ZipFile(zip_path) as z:
        with z.open("Dataset/train.csv") as f:
            df = pd.read_csv(f)

    # ── Drop ID, encode subscription type as numeric
    df.drop(columns=["CustomerID"], inplace=True)
    df.replace({"SubscriptionType": {"Basic": 0, "Standard": 1, "Premium": 2}}, inplace=True)

    # ── Compute quantiles BEFORE feature engineering
    quantiles = {
        "vhpw_q25":    df["ViewingHoursPerWeek"].quantile(0.25),
        "vhpw_q75":    df["ViewingHoursPerWeek"].quantile(0.75),
        "support_q75": df["SupportTicketsPerMonth"].quantile(0.75),
    }

    # ── Feature engineering
    df = add_engineered_features(df, quantiles)

    # ── Separate X / y
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # Identify cat and num columns (in stable order)
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if X[c].dtype != "object"]

    # ── OHE
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    X_cat = ohe.fit_transform(X[cat_cols])
    X_num = X[num_cols].values.astype(float)
    X_full = np.hstack([X_num, X_cat])

    # ── Split + SMOTE
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_tr, y_tr)

    # ── XGBoost
    model = XGBClassifier(
        random_state=42, eval_metric="logloss", n_jobs=-1,
        n_estimators=200, max_depth=6, learning_rate=0.1
    )
    model.fit(X_res, y_res)

    # ── Test AUC
    y_prob = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_prob)

    return model, {
        "ohe":         ohe,
        "cat_cols":    cat_cols,
        "num_cols":    num_cols,
        "quantiles":   quantiles,
        "auc":         auc,
        "train_size":  len(df),
        "n_features":  X_full.shape[1],
    }


def predict_single(user_input: dict, model, meta: dict) -> tuple:
    """
    Build feature vector from a user input dict (same schema as train.csv minus CustomerID/Churn),
    run prediction, return (churn_probability, feature_array).
    """
    ohe      = meta["ohe"]
    cat_cols = meta["cat_cols"]
    num_cols = meta["num_cols"]
    quantiles= meta["quantiles"]

    # Build a 1-row DataFrame with the RAW columns (before eng features)
    row = dict(user_input)
    # Encode SubscriptionType to int, same as during training
    row["SubscriptionType"] = {"Basic": 0, "Standard": 1, "Premium": 2}[row["SubscriptionType"]]

    df_single = pd.DataFrame([row])

    # Apply identical feature engineering using stored quantiles
    df_single = add_engineered_features(df_single, quantiles)

    # Slice in the exact column order used during training
    X_num = df_single[num_cols].values.astype(float)      # shape (1, len(num_cols))
    X_cat = ohe.transform(df_single[cat_cols])              # shape (1, n_ohe)
    X_vec = np.hstack([X_num, X_cat])                       # shape (1, n_features)

    prob = model.predict_proba(X_vec)[0, 1]
    return prob, df_single


# ─── HERO ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>⚡ UpgradeIQ</h1>
  <p>Smart Prediction of Subscription Upgrades &amp; Churn &nbsp;|&nbsp; Powered by XGBoost</p>
</div>
""", unsafe_allow_html=True)

# ─── TRAIN ───────────────────────────────────────────────────────────────────
with st.spinner("🔄 Training XGBoost model on 240K+ records… (first launch only, cached thereafter)"):
    model, meta = train_model()

# Stat bar
c1, c2, c3, c4 = st.columns(4)
for col, label, val in [
    (c1, "Training Records", f"{meta['train_size']:,}"),
    (c2, "Model",            "XGBoost"),
    (c3, "ROC-AUC Score",    f"{meta['auc']:.4f}"),
    (c4, "Total Features",   str(meta['n_features'])),
]:
    with col:
        st.markdown(f"""<div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧑‍💼 Customer Profile")

    st.markdown("<div class='section-title'>Account</div>", unsafe_allow_html=True)
    account_age       = st.slider("Account Age (months)", 1, 119, 30)
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    payment_method    = st.selectbox("Payment Method",
        ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.markdown("<div class='section-title'>Financials</div>", unsafe_allow_html=True)
    monthly_charges = st.slider("Monthly Charges ($)", 5.0, 150.0, 50.0, step=0.5)
    total_charges   = st.slider("Total Charges ($)", 5.0, 15000.0,
                                float(round(monthly_charges * account_age, 1)), step=10.0)

    st.markdown("<div class='section-title'>Viewing Behaviour</div>", unsafe_allow_html=True)
    viewing_hours       = st.slider("Viewing Hours / Week (hrs)", 0.0, 40.0, 10.0, step=0.5)
    avg_view_duration   = st.slider("Avg Session Duration (mins)", 5.0, 200.0, 60.0, step=5.0)
    downloads_per_month = st.slider("Content Downloads / Month", 0, 50, 5)
    watchlist_size      = st.slider("Watchlist Size", 0, 24, 10)

    st.markdown("<div class='section-title'>Content Preferences</div>", unsafe_allow_html=True)
    content_type      = st.selectbox("Content Type", ["Movies", "TV Shows", "Both"])
    genre_preference  = st.selectbox("Genre Preference",
        ["Action", "Comedy", "Drama", "Horror", "Documentary", "Fantasy", "Romance", "Sci-Fi", "Thriller"])
    multi_device      = st.selectbox("Multi-Device Access", ["Yes", "No"])
    device_registered = st.selectbox("Device Registered",
        ["Mobile", "Tablet", "Computer", "TV"])
    parental_control  = st.selectbox("Parental Control", ["Yes", "No"])
    subtitles_enabled = st.selectbox("Subtitles Enabled", ["Yes", "No"])

    st.markdown("<div class='section-title'>Satisfaction & Support</div>", unsafe_allow_html=True)
    user_rating     = st.slider("User Rating (1–5)", 1.0, 5.0, 3.5, step=0.1)
    support_tickets = st.slider("Support Tickets / Month", 0, 10, 1)

    st.markdown("<div class='section-title'>Demographics</div>", unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Predict Churn", use_container_width=True)

# ─── MAIN PANEL ──────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.4, 1], gap="large")

with left_col:
    if not predict_btn:
        st.markdown("""
        <div class="info-box">
        👈 &nbsp; <strong>Fill in the customer profile</strong> in the left sidebar,
        then click <strong>⚡ Predict Churn</strong> for an instant prediction.
        </div>
        <div class="info-box">
        🔬 <strong>How it works:</strong> XGBoost trained on <strong>243,787</strong> subscription
        records with SMOTE oversampling (18% churn rate). The model uses
        <strong>28+ engineered features</strong> including engagement scores, support intensity,
        activity-drop signals, and charge-to-age ratios.
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

    else:
        # ── Build raw input dict matching train.csv schema (minus CustomerID and Churn)
        user_input = {
            "AccountAge":               account_age,
            "MonthlyCharges":           monthly_charges,
            "TotalCharges":             total_charges,
            "SubscriptionType":         subscription_type,   # string — encoded inside predict_single
            "PaymentMethod":            payment_method,
            "PaperlessBilling":         paperless_billing,
            "ContentType":              content_type,
            "MultiDeviceAccess":        multi_device,
            "DeviceRegistered":         device_registered,
            "ViewingHoursPerWeek":      viewing_hours,
            "AverageViewingDuration":   avg_view_duration,
            "ContentDownloadsPerMonth": downloads_per_month,
            "GenrePreference":          genre_preference,
            "UserRating":               user_rating,
            "SupportTicketsPerMonth":   support_tickets,
            "Gender":                   gender,
            "WatchlistSize":            watchlist_size,
            "ParentalControl":          parental_control,
            "SubtitlesEnabled":         subtitles_enabled,
        }

        try:
            prob, df_eng = predict_single(user_input, model, meta)
            pred = int(prob >= 0.5)

            # Result card
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

            # Probability bar
            st.markdown(f"<div class='prob-label'>Churn Probability: <strong>{prob*100:.1f}%</strong></div>",
                        unsafe_allow_html=True)
            pct = f"{prob*100:.1f}"
            st.markdown(f"""
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill" style="width:{pct}%"></div>
            </div>
            <div style='display:flex; justify-content:space-between; margin-top:4px;'>
                <span style='color:#7effc4; font-size:0.8rem'>0% — No Risk</span>
                <span style='color:#ff8fa3; font-size:0.8rem'>100% — Certain Churn</span>
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)

with right_col:
    if predict_btn:
        try:
            st.markdown("### 🔍 Risk Factor Breakdown")
            q = meta["quantiles"]
            vhpw_q25    = q["vhpw_q25"]
            vhpw_q75    = q["vhpw_q75"]
            support_q75 = q["support_q75"]

            risk_flags = {
                "Low Monthly Viewing":  ((viewing_hours * 4) < (vhpw_q25 * 4),   True),
                "Recent Activity Drop": ((viewing_hours < vhpw_q25) and (account_age > 6), True),
                "Low Satisfaction":     (user_rating <= 2.0,                      True),
                "High Support Usage":   (support_tickets > support_q75,           True),
                "High Watcher":         (viewing_hours > vhpw_q75,                False),
                "High Satisfaction":    (user_rating >= 4.0,                      False),
            }

            for label, (active, is_risk) in risk_flags.items():
                if active and is_risk:
                    cls, icon = "risk-high",    "🔴"
                elif not active and is_risk:
                    cls, icon = "risk-low",     "🟢"
                elif active and not is_risk:
                    cls, icon = "risk-low",     "🟢"
                else:
                    cls, icon = "risk-neutral", "🟡"
                st.markdown(f"<span class='risk-badge {cls}'>{icon} {label}</span>",
                            unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 Engineered Metrics")
            eng_metrics = pd.DataFrame({
                "Metric": [
                    "Engagement Score", "Support Intensity",
                    "Charges:Age Ratio", "Engagement × Satisfaction", "Total Risk Score"
                ],
                "Value": [
                    f"{(downloads_per_month + watchlist_size + viewing_hours*4)/3:.2f}",
                    f"{support_tickets / (account_age + 1):.4f}",
                    f"${monthly_charges / (account_age + 1):.2f}",
                    f"{viewing_hours * user_rating:.2f}",
                    str(int((viewing_hours*4 < vhpw_q25*4)*2 +
                            (user_rating <= 2.0) +
                            (support_tickets > support_q75) +
                            ((viewing_hours < vhpw_q25) and (account_age > 6))))
                ],
            })
            st.dataframe(eng_metrics, use_container_width=True, hide_index=True)

            # Recommendation
            st.markdown("### 💡 Recommended Action")
            if prob >= 0.7:
                st.error("🚨 **Immediate intervention needed.** Offer a discount or upgrade incentive. Assign a customer success rep.")
            elif prob >= 0.5:
                st.warning("⚠️ **Monitor closely.** Send a targeted win-back email or personalized content recommendations.")
            elif prob >= 0.3:
                st.info("ℹ️ **Moderate risk.** Keep up regular communication and engagement campaigns.")
            else:
                st.success("✅ **Loyal customer.** Great candidate for upselling to a higher tier or referral programs.")
        except Exception:
            pass

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("<br><hr style='border-color:rgba(255,255,255,0.1)'>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#555577; font-size:0.8rem;'>
UpgradeIQ &nbsp;·&nbsp; Built with Streamlit &amp; XGBoost &nbsp;·&nbsp;
Dataset: 243K+ subscription records
</p>""", unsafe_allow_html=True)
