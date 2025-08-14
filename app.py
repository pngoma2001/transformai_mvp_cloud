# app.py
# TransformAI — Streamlit MVP (cloud-safe)
from __future__ import annotations

import io
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(
    page_title="TransformAI — Portfolio Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Safe header / logo ----------
def render_header():
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        st.image(str(logo_path), width=260)
    else:
        # Fallback title when logo file is not present in the repo
        st.markdown(
            "<div style='font-size:32px; font-weight:700; margin:0;'>TransformAI</div>"
            "<div style='color:#6b7280; margin-top:4px;'>AI playbooks for EBITDA growth & cost reduction</div>",
            unsafe_allow_html=True,
        )

render_header()
st.divider()

# ---------- Sample data ----------
def make_sample_df(kind: str) -> pd.DataFrame:
    # 8 quarters of mock data per company type
    periods = pd.period_range("2023Q1", "2024Q4", freq="Q")
    df = pd.DataFrame({"period": periods.astype(str)})

    if kind == "RetailCo":
        df["revenue"] = [4.1, 4.3, 4.0, 4.5, 4.9, 5.0, 5.2, 5.6]
        df["ebitda"] =  [0.35,0.38,0.30,0.42,0.48,0.51,0.55,0.62]
        df["customers"]=[120,125,118,130,136,138,142,150]
        df["cogs"] =     [2.7,2.8,2.7,2.9,3.0,3.1,3.2,3.3]
    elif kind == "SaaSCo":
        df["revenue"] = [1.8, 2.0, 2.2, 2.5, 2.8, 3.1, 3.3, 3.6]
        df["ebitda"] =  [-0.1,-0.05,0.0,0.1,0.18,0.25,0.32,0.40]
        df["customers"]=[220,240,260,290,320,350,380,420]
        df["cogs"] =     [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85]
    else: # MarketplaceCo
        df["revenue"] = [3.2, 3.4, 3.3, 3.7, 4.2, 4.6, 4.9, 5.3]
        df["ebitda"] =  [0.2, 0.22,0.18,0.25,0.3, 0.36,0.42,0.50]
        df["customers"]=[500,520,510,540,580,610,640,680]
        df["cogs"] =     [1.8,1.9,1.85,2.0,2.2,2.3,2.4,2.6]

    return df

SAMPLES = ["RetailCo", "SaaSCo", "MarketplaceCo"]

# ---------- Helpers ----------
def read_csv(file) -> pd.DataFrame:
    try:
        return pd.read_csv(file)
    except Exception:
        # Try bytes -> StringIO if user uploaded a BytesIO from Streamlit
        if hasattr(file, "getvalue"):
            return pd.read_csv(io.StringIO(file.getvalue().decode("utf-8", errors="ignore")))
        raise

def has_multi_periods(df: pd.DataFrame) -> bool:
    return "period" in df.columns and df["period"].nunique() > 1

def summarize(df: pd.DataFrame) -> dict:
    out = {}
    if "revenue" in df.columns:
        out["latest_revenue"] = float(df["revenue"].iloc[-1])
        if has_multi_periods(df):
            out["revenue_cagr"] = (
                (df["revenue"].iloc[-1] / max(df["revenue"].iloc[0], 1e-9)) ** (1 / (len(df) - 1)) - 1
            )
    if "ebitda" in df.columns:
        out["latest_ebitda"] = float(df["ebitda"].iloc[-1])
        if has_multi_periods(df):
            out["ebitda_margin_latest"] = float(df["ebitda"].iloc[-1] / max(df["revenue"].iloc[-1], 1e-9))
    return out

def simple_playbook(df: pd.DataFrame) -> list[str]:
    tips = []
    if "ebitda" in df.columns and "revenue" in df.columns:
        margin = df["ebitda"].iloc[-1] / max(df["revenue"].iloc[-1], 1e-9)
        if margin < 0.1:
            tips.append("Renegotiate top 5 vendor contracts; target 3–5% COGS reduction.")
        elif margin < 0.2:
            tips.append("Pilot pricing test on 10–20% of SKUs to lift blended take-rate by 50–100 bps.")
        else:
            tips.append("Double down on high-margin segments; shift 10% paid spend to proven channels.")
    if "customers" in df.columns and has_multi_periods(df):
        gr = (df["customers"].iloc[-1] / max(df["customers"].iloc[0], 1e-9)) ** (1/(len(df)-1)) - 1
        if gr < 0.05:
            tips.append("Stand up referral program and onboarding nudges to improve top-of-funnel velocity.")
    if "cogs" in df.columns and "revenue" in df.columns:
        cogs_ratio = df["cogs"].iloc[-1] / max(df["revenue"].iloc[-1], 1e-9)
        if cogs_ratio > 0.6:
            tips.append("Run supplier consolidation: 80/20 SKU coverage with volume discounts.")
    if not tips:
        tips.append("Maintain course; expand proven initiatives and monitor unit economics monthly.")
    return tips

# ---------- Sidebar (mock admin / templates) ----------
with st.sidebar:
    st.subheader("Templates (mock admin)")
    st.caption("Simulate learning loop by bumping the template version.")
    if "tpl_version" not in st.session_state:
        st.session_state.tpl_version = 1
    st.write(f"Template version: **v{st.session_state.tpl_version}**")
    if st.button("Bump Version"):
        st.session_state.tpl_version += 1
        st.success(f"Bumped to v{st.session_state.tpl_version}")

    st.markdown("---")
    st.subheader("Sample CSVs")
    chosen = st.selectbox("Download a sample", SAMPLES, index=0)
    sample_df = make_sample_df(chosen)
    csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download sample CSV",
        data=csv_bytes,
        file_name=f"{chosen}_sample.csv",
        mime="text/csv",
    )
    st.caption("Load this back into the app to try multi-period charts.")

# ---------- Main UI ----------
st.write("Upload a CSV or pick a sample company, then click **Analyze**.")

col1, col2 = st.columns([1, 1])
with col1:
    sample_choice = st.selectbox("Choose a sample company", SAMPLES, index=0)
with col2:
    uploaded = st.file_uploader("Or upload your CSV", type=["csv"])

with st.expander("Evidence Sources (optional)"):
    url_evidence = st.text_input("Paste a relevant URL (market data, news, etc.)", value="")
    pdfs = st.file_uploader("Attach PDFs (optional)", type=["pdf"], accept_multiple_files=True)

analyze = st.button("Analyze", type="primary")

# ---------- Analysis ----------
if analyze:
    if uploaded is not None:
        df = read_csv(uploaded)
        st.info("Using your uploaded CSV.")
    else:
        df = make_sample_df(sample_choice)
        st.info(f"Using sample dataset: **{sample_choice}**.")

    # Basic validations
    if "revenue" not in df.columns or "ebitda" not in df.columns:
        st.error("Your data should include at least 'period', 'revenue', and 'ebitda' columns.")
        st.stop()

    if "period" in df.columns:
        # Best effort ordering by period if values look sortable
        try:
            # Try to parse common period strings (YYYYQ#, YYYY-MM, etc.)
            # If parse fails, we’ll stick to the given order
            parsed = pd.PeriodIndex(df["period"].astype(str))
            df = df.iloc[pd.Series(parsed).argsort(kind="mergesort")].reset_index(drop=True)
        except Exception:
            pass

    # Summary
    st.subheader("Summary")
    meta = summarize(df)
    cols = st.columns(3)
    cols[0].metric("Latest Revenue", f"${meta.get('latest_revenue', float('nan')):,.2f}M")
    cols[1].metric("Latest EBITDA", f"${meta.get('latest_ebitda', float('nan')):,.2f}M")
    if "revenue_cagr" in meta:
        cols[2].metric("Revenue CAGR", f"{meta['revenue_cagr']*100:.1f}%")

    # Charts
    if has_multi_periods(df):
        st.subheader("Trends")
        try:
            fig = px.line(df, x="period", y=["revenue", "ebitda"], markers=True)
            fig.update_layout(yaxis_title="USD (Millions)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("Could not render chart. Check that 'period' is a consistent sequence.")

    # Playbook (rule-based stub; LLM can be added later)
    st.subheader("Recommended Playbook")
    for tip in simple_playbook(df):
        st.write(f"• {tip}")

    # Evidence echo (no parsing, just record)
    if url_evidence or pdfs:
        st.markdown("**Evidence provided:**")
        if url_evidence:
            st.write(f"- URL: {url_evidence}")
        for f in (pdfs or []):
            st.write(f"- PDF: {f.name}")

st.markdown("---")
st.caption("TransformAI MVP — front-end only. Logo optional; app runs without /assets.")

