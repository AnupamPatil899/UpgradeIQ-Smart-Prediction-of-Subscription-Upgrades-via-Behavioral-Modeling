"""
styles.py — Modern Obsidian Bento Grid Design System for UpgradeIQ
Features frosted glassmorphism, glowing borders, crisp typography, and responsive bento cards.
"""

import streamlit as st

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Global Reset & Typography */
html, body, [class*="css"], .stMarkdown, .stText, p, span, div {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, kbd, samp, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Deep Obsidian Canvas */
.stApp {
    background: radial-gradient(circle at 10% 10%, #111625 0%, #0a0c16 45%, #05060b 100%) !important;
    color: #e2e8f0;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.02);
}
::-webkit-scrollbar-thumb {
    background: rgba(139, 92, 246, 0.25);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(139, 92, 246, 0.5);
}

/* Sidebar Glass Deck */
section[data-testid="stSidebar"] {
    background: rgba(13, 17, 28, 0.75) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
}

section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

/* Top Telemetry Header */
.telemetry-strip {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 0.8rem 1.4rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(16px);
}

.telemetry-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.telemetry-title {
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #ffffff, #c4b5fd, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.telemetry-pills {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
}

.telemetry-pill {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #94a3b8;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}

.telemetry-pill.active-emerald {
    border-color: rgba(16, 185, 129, 0.4);
    color: #6ee7b7;
    background: rgba(16, 185, 129, 0.1);
}

.telemetry-pill.active-violet {
    border-color: rgba(139, 92, 246, 0.4);
    color: #c4b5fd;
    background: rgba(139, 92, 246, 0.1);
}

.telemetry-pill.active-cyan {
    border-color: rgba(6, 182, 212, 0.4);
    color: #67e8f9;
    background: rgba(6, 182, 212, 0.1);
}

/* Bento Grid System */
.bento-card {
    background: rgba(18, 24, 38, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 18px;
    padding: 1.4rem;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
    overflow: hidden;
    margin-bottom: 1rem;
    box-shadow: 0 8px 30px -10px rgba(0, 0, 0, 0.5);
}

.bento-card:hover {
    border-color: rgba(139, 92, 246, 0.35);
    transform: translateY(-2px);
    box-shadow: 0 14px 40px -10px rgba(139, 92, 246, 0.15);
}

.bento-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    padding-bottom: 0.6rem;
}

.bento-tag {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8b5cf6;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.bento-subtag {
    font-size: 0.72rem;
    color: #64748b;
    font-weight: 500;
}

/* Prediction Verdict Bento Hero */
.verdict-banner-churn {
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.15) 0%, rgba(225, 29, 72, 0.25) 100%);
    border: 1px solid rgba(244, 63, 94, 0.5);
    border-radius: 14px;
    padding: 1.2rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 20px rgba(244, 63, 94, 0.2);
}

.verdict-banner-safe {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
    border: 1px solid rgba(16, 185, 129, 0.5);
    border-radius: 14px;
    padding: 1.2rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.2);
}

.verdict-icon {
    font-size: 2.4rem;
    line-height: 1;
}

.verdict-title {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.3px;
    color: #ffffff;
    margin: 0;
}

.verdict-detail {
    font-size: 0.84rem;
    color: #cbd5e1;
    margin-top: 0.2rem;
}

/* Calibrated Threshold Visual Gauge */
.gauge-box {
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.8rem 0;
}

.gauge-meta-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.78rem;
    font-weight: 700;
    color: #94a3b8;
    margin-bottom: 0.4rem;
}

.gauge-track-wrap {
    position: relative;
    margin: 2rem 0 0.8rem 0;
}

.gauge-bar-bg {
    position: relative;
    height: 22px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    overflow: hidden;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.6);
}

.gauge-fill {
    height: 100%;
    border-radius: 20px;
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

.gauge-fill-churn {
    background: linear-gradient(90deg, #f59e0b 0%, #f43f5e 100%);
}

.gauge-fill-safe {
    background: linear-gradient(90deg, #10b981 0%, #06b6d4 100%);
}

.threshold-marker-line {
    position: absolute;
    top: 0;
    height: 22px;
    width: 2px;
    background: #ffffff;
    box-shadow: 0 0 10px 2px rgba(255, 255, 255, 0.95);
    z-index: 10;
}

.threshold-badge {
    position: absolute;
    top: -24px;
    transform: translateX(-50%);
    background: #7c3aed;
    color: #ffffff;
    font-size: 0.68rem;
    font-weight: 800;
    padding: 2px 7px;
    border-radius: 6px;
    white-space: nowrap;
    border: 1px solid rgba(255, 255, 255, 0.4);
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}

.threshold-badge::after {
    content: '';
    position: absolute;
    bottom: -4px;
    left: 50%;
    transform: translateX(-50%);
    border-width: 4px 4px 0;
    border-style: solid;
    border-color: #7c3aed transparent;
    display: block;
    width: 0;
}

/* Revenue & Risk Hazard Stat Grid */
.stat-mini-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.6rem;
    margin-top: 0.6rem;
}

.stat-mini-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 0.8rem 0.9rem;
    text-align: center;
}

.stat-mini-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    color: #64748b;
    font-weight: 700;
    letter-spacing: 0.8px;
}

.stat-mini-val {
    font-size: 1.15rem;
    font-weight: 800;
    color: #ffffff;
    margin-top: 0.2rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Key Signal Drivers */
.signal-driver-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.65rem 0.8rem;
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    margin-bottom: 0.45rem;
}

.signal-driver-left {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.signal-driver-icon {
    font-size: 1rem;
}

.signal-driver-name {
    font-size: 0.8rem;
    font-weight: 600;
    color: #cbd5e1;
}

.signal-driver-right {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.signal-driver-val {
    font-size: 0.82rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #ffffff;
}

.signal-tag-high {
    font-size: 0.68rem;
    padding: 1px 6px;
    border-radius: 4px;
    background: rgba(244, 63, 94, 0.15);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.4);
    font-weight: 700;
}

.signal-tag-safe {
    font-size: 0.68rem;
    padding: 1px 6px;
    border-radius: 4px;
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.4);
    font-weight: 700;
}

.signal-tag-neutral {
    font-size: 0.68rem;
    padding: 1px 6px;
    border-radius: 4px;
    background: rgba(245, 158, 11, 0.15);
    color: #fcd34d;
    border: 1px solid rgba(245, 158, 11, 0.4);
    font-weight: 700;
}

/* Playbook Retention Box */
.playbook-box {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.12) 0%, rgba(6, 182, 212, 0.08) 100%);
    border: 1px solid rgba(139, 92, 246, 0.35);
    border-radius: 14px;
    padding: 1.1rem;
    margin-top: 0.5rem;
}

.playbook-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.88rem;
    font-weight: 700;
    color: #c4b5fd;
    margin-bottom: 0.4rem;
}

.playbook-body {
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.5;
}

.playbook-action-tag {
    display: inline-block;
    background: rgba(124, 58, 237, 0.3);
    color: #ffffff;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-top: 0.6rem;
    border: 1px solid rgba(139, 92, 246, 0.5);
}

/* Sidebar Section Titles */
.sidebar-deck-title {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #a78bfa;
    margin-top: 1.1rem;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 10px;
    color: white;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 0.7rem 1.8rem;
    background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%);
    box-shadow: 0 4px 16px rgba(124, 58, 237, 0.4);
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(124, 58, 237, 0.6);
    background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%);
}

footer {
    visibility: hidden;
}
</style>
"""


def apply_styles():
    """Inject Bento Grid CSS styles."""
    st.html(CSS)
