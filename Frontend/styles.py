"""
styles.py

Centralized CSS styling for the UpgradeIQ Streamlit frontend.
Import in app.py and call:

    from styles import apply_styles
    apply_styles()

The visual appearance matches the original local application.
"""

import streamlit as st


CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e0e0ff;
}

section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.10);
}

section[data-testid="stSidebar"] * {
    color: #d4d4f7 !important;
}

.hero {
    background: linear-gradient(90deg,#6c63ff 0%,#48b1bf 100%);
    border-radius:16px;
    padding:2rem 2.5rem;
    margin-bottom:1.5rem;
    color:white;
}

.hero h1{
    font-size:2.6rem;
    font-weight:900;
    margin:0;
    letter-spacing:-1px;
}

.hero p{
    font-size:1.05rem;
    opacity:.85;
    margin-top:.4rem;
}

.metric-card{
    background:rgba(255,255,255,.06);
    border:1px solid rgba(255,255,255,.12);
    border-radius:12px;
    padding:1.2rem 1.5rem;
    text-align:center;
}

.metric-card .label{
    font-size:.8rem;
    color:#a0a0cc;
    text-transform:uppercase;
    letter-spacing:1px;
}

.metric-card .value{
    font-size:2rem;
    font-weight:700;
    color:white;
}

.result-churn{
    background:linear-gradient(135deg,#ff416c,#ff4b2b);
    border-radius:16px;
    padding:2rem;
    text-align:center;
    color:white;
    box-shadow:0 8px 32px rgba(255,65,108,.40);
}

.result-safe{
    background:linear-gradient(135deg,#11998e,#38ef7d);
    border-radius:16px;
    padding:2rem;
    text-align:center;
    color:white;
    box-shadow:0 8px 32px rgba(56,239,125,.30);
}

.result-icon{
    font-size:3.5rem;
}

.result-title{
    font-size:1.8rem;
    font-weight:800;
    margin:.3rem 0;
}

.result-prob{
    opacity:.9;
}

.prob-label{
    font-size:.8rem;
    color:#a0a0cc;
    margin-bottom:4px;
}

.prob-bar-wrap{
    background:rgba(255,255,255,.10);
    height:22px;
    border-radius:50px;
    overflow:hidden;
}

.prob-bar-fill{
    height:100%;
    border-radius:50px;
    background:linear-gradient(90deg,#38ef7d,#ff416c);
}

.risk-badge{
    display:inline-block;
    padding:.25rem .8rem;
    border-radius:20px;
    font-size:.82rem;
    font-weight:600;
    margin:.2rem;
}

.risk-high{
    background:rgba(255,65,108,.25);
    color:#ff8fa3;
    border:1px solid #ff416c;
}

.risk-low{
    background:rgba(56,239,125,.20);
    color:#7effc4;
    border:1px solid #38ef7d;
}

.risk-neutral{
    background:rgba(255,165,0,.20);
    color:#ffd580;
    border:1px solid orange;
}

.info-box{
    background:rgba(255,255,255,.05);
    border:1px solid rgba(108,99,255,.4);
    border-radius:12px;
    padding:1rem 1.4rem;
    margin:.8rem 0;
    color:#c0c0e0;
}

.section-title{
    font-size:.72rem;
    font-weight:700;
    letter-spacing:2px;
    text-transform:uppercase;
    color:#7c6fff;
    margin-bottom:.5rem;
}

.backend-ok{
    padding:.5rem;
    border-radius:8px;
    background:rgba(56,239,125,.15);
    border:1px solid #38ef7d;
}

.backend-down{
    padding:.5rem;
    border-radius:8px;
    background:rgba(255,65,108,.15);
    border:1px solid #ff416c;
}

.stButton > button{
    width:100%;
    border:none;
    border-radius:10px;
    color:white;
    font-weight:700;
    font-size:1.05rem;
    padding:.75rem 2.5rem;
    background:linear-gradient(90deg,#6c63ff,#48b1bf);
}

.stButton > button:hover{
    box-shadow:0 6px 20px rgba(108,99,255,.45);
}

footer{
    visibility:hidden;
}
</style>
"""


def apply_styles():
    """Inject the CSS into the Streamlit app."""
    st.markdown(CSS, unsafe_allow_html=True)
