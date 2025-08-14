# app.py — TransformAI (Streamlit cloud-safe, KPI fixes)
from __future__ import annotations

import io
from pathlib import Path
from datetime import datetime
import json
import time

import pandas as pd
import streamlit as st

# ───────────────────────────── Page setup ─────────────────────────────
st.set_page_config(
    page_title="TransformAI — Portfolio Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────── Header / Hero ──────────────────────────
def render_header() -> None:
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        st.image(str(logo_path), width=220)  # safe: only if file exists
    else:
        st.markdown(
            """
            <div style="font-size:30px;font-weight:700;margin:0;">TransformAI</div>
            <div style="opacity:.75;margin-top:4px;">AI playbooks for EBITDA growth & cost reduction</div>
            """,
            unsafe_allow_html=True,
        )

render_header()
st.divider()

# ───────────────────────────── Sidebar (optional AI) ─────────────────
st.sidebar.header("TransformAI")
use_ai = st.sidebar.toggle("Enable AI-generated plays", value=False,
                           help="Optional. App works without this.")
provider = st.sidebar.selectbox("Provider", ["OpenAI", "Anthropic"], index=0,
                                disabled=not use_ai)
model_name = st.sidebar.text_input("Model name", value="gpt-4o-mini",
                                   disabled=not use_ai)
creativity = st.sidebar.slider("Creativity", 0.0, 1.0, 0.2, 0.01,
                               disabled=not use_ai)
backend_url = st.sidebar.text_input("Backend URL (optional)")

# ───────────────────────────── Sample data helpers ────────────────────
def make_sample_df(kind: str) -> pd.DataFrame:
    """Return 8 quarters of made-up KPIs for a sample company."""
    periods = pd.period_range("2023Q1", "2024Q4", freq="Q")
    df = pd.DataFrame({"period": periods.astype(str)})
    if kind == "RetailCo":
        df["revenue"] = [41.4, 43.0, 44.5, 45.9, 49.5, 50.2, 52.0, 56.3]
        df["ebitda"]  = [ 9.5, 10.0, 10.2, 10.4, 10.8, 11.0, 11.5, 12.0]
        df["gross_margin"] = [34.0, 34.8, 35.2, 35.7, 36.1, 36.5, 36.9, 37.0]  # %
        df["churn_rate"]   = [21.0, 21.2, 21.0, 20.9, 20.8, 20.7, 20.6, 20.6]  # %
        df["turns_util"]   = [6.8,  6.9,  7.0,  7.1,  7.2,  7.2,  7.3,  7.45]
    else:  # HealthCo
        df["revenue"] = [28.2, 28.8, 29.1, 29.9, 31.2, 31.7, 32.4, 33.0]
        df["ebitda"]  = [ 5.1,  5.2,  5.3,  5.4,  5.6,  5.7,  5.8,  6.0]
        df["gross_margin"] = [38.0, 38.1, 38.2, 38.4, 38.6, 38.7, 38.8, 39.0]  # %
        df["churn_rate"]   = [12.0, 12.1, 12.2, 12.3, 12.3, 12.2, 12.1, 12.0]  # %
        df["turns_util"]   = [4.8,  4.9,  5.0,  5.1,  5.1,  5.2,  5.2,  5.3]
    return df

SAMPLES = {
    "RetailCo": make_sample_df("RetailCo"),
    "HealthCo": make_sample_df("HealthCo"),
}

def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def has_multi_periods(df: pd.DataFrame) -> bool:
    return "period" in df.columns and df["period"].nunique() > 1

# ───────────────────────────── UI — Data selection ────────────────────
st.markdown(
    "Upload a CSV or use a sample dataset to generate recommended plays with "
    "estimated EBITDA uplift."
)

sample = st.selectbox("Choose a sample company", list(SAMPLES.keys()))
uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

with st.expander("Evidence Sources (optional)"):
    st.write("Attach extra signals to make recommendations more contextual (optional).")
    survey_file = st.file_uploader("Customer survey (CSV)", type=["csv"], key="survey")
    market_file = st.file_uploader("Market data (CSV)", type=["csv"], key="market")
    use_sample_evidence = st.checkbox("Use sample evidence if not provided", value=True)

# Resolve primary dataframe
if uploaded is not None:
    df = read_csv(uploaded)
    st.session_state["company_name"] = Path(uploaded.name).stem.title()
else:
    df = SAMPLES[sample].copy()
    st.session_state["company_name"] = sample

# 🔧 DERIVED COLUMNS/FALLBACKS (fixes wrong % and 0.00 turns)
# - If gross_margin is missing, compute as ebitda/revenue (fraction).
# - If churn_rate provided as fraction, keep as fraction and multiply at display time.
# - If turns_util missing but inventory_turns exists, map it.
if "gross_margin" not in df.columns and {"revenue", "ebitda"} <= set(df.columns):
    df["gross_margin"] = (pd.to_numeric(df["ebitda"], errors="coerce") /
                          pd.to_numeric(df["revenue"], errors="coerce")).fillna(0.0) * 100.0  # percentage
# Normalize: if gross_margin looks like a fraction (e.g., 0.37), convert to %
if df["gross_margin"].max() <= 1.0:
    df["gross_margin"] = df["gross_margin"] * 100.0

if "turns_util" not in df.columns and "inventory_turns" in df.columns:
    df["turns_util"] = pd.to_numeric(df["inventory_turns"], errors="coerce").fillna(0.0)

# Optional evidence — safely read (kept minimal for now)
survey_df = None
market_df = None
if survey_file is not None:
    survey_df = read_csv(survey_file)
elif use_sample_evidence:
    p = Path("data/sample_customer_survey.csv")
    if p.exists():
        survey_df = pd.read_csv(p)

if market_file is not None:
    market_df = read_csv(market_file)
elif use_sample_evidence:
    p = Path("data/sample_market.csv")
    if p.exists():
        market_df = pd.read_csv(p)

# ───────────────────────────── Actions ────────────────────────────────
st.divider()
if st.button("Generate Playbook", type="primary"):
    # Compute KPIs snapshot and stash
    last = df.iloc[-1]
    kpis = {
        "company": st.session_state.get("company_name", "UnknownCo"),
        "period": last.get("period", "Latest"),
        "revenue": float(last.get("revenue", 0.0)),
        "ebitda": float(last.get("ebitda", 0.0)),
        # keep as PERCENT number (e.g., 37.0)
        "gross_margin": float(last.get("gross_margin", 0.0)),
        # keep churn as PERCENT if it looks like percent; if fraction, convert
        "churn_rate": float(last.get("churn_rate", 0.0)),
        "turns_util": float(last.get("turns_util", 0.0)),
    }
    # YoY %
    if has_multi_periods(df) and len(df) >= 5:
        kpis["revenue_yoy"] = float(
            (df["revenue"].iloc[-1] - df["revenue"].iloc[-5]) /
            max(df["revenue"].iloc[-5], 1e-9) * 100.0
        )
    else:
        kpis["revenue_yoy"] = None

    # Normalize churn to percent for display
    if kpis["churn_rate"] <= 1.0:
        kpis["churn_rate"] *= 100.0

    st.session_state["kpis"] = kpis
    st.session_state.setdefault("decisions", [])
    st.session_state.setdefault("activity", [])

# When we have KPIs, render the summary and plays
kpis = st.session_state.get("kpis")
if kpis:
    # Summary
    st.subheader(f"Summary — {kpis['company']} ({kpis.get('period', 'Latest')})")

    cols = st.columns(5)
    cols[0].metric(
        "Revenue",
        f"${kpis['revenue']:,.0f}",
        (f"{kpis['revenue_yoy']:.1f}% YoY" if kpis['revenue_yoy'] is not None else None),
    )
    cols[1].metric("EBITDA", f"${kpis['ebitda']:,.0f}")
    cols[2].metric("Gross Margin", f"{kpis['gross_margin']:.1f}%")
    cols[3].metric("Churn Rate", f"{kpis['churn_rate']:.1f}%")
    cols[4].metric("Turns/Utilization", f"{kpis['turns_util']:.2f}")

    # Trend chart (if multiple periods)
    if has_multi_periods(df):
        trend = df[["period", "revenue", "ebitda"]].copy().set_index("period")
        st.line_chart(trend)

    # Simple plays (non-LLM)
    def simple_playbook(df: pd.DataFrame, k: dict) -> list[dict]:
        plays = []
        plays.append({
            "id": "pricing",
            "title": "Pricing Optimization",
            "confidence": "High",
            "complexity": "Medium",
            "assumed_increase": 0.06,
            "rationale": "Top 30% SKUs show low elasticity; raise prices 5–7% with guardrails.",
            "uplift_estimate": round(k["revenue"] * 0.06 * 0.3 * 0.3, 2),
        })
        plays.append({
            "id": "retention",
            "title": "Customer Retention Program",
            "confidence": "Medium",
            "complexity": "Medium",
            "assumed_decrease": 0.02,
            "rationale": "Lifecycle messaging & loyalty offers to reduce churn by ~2%.",
            "uplift_estimate": round(k["revenue"] * 0.02 * 0.4, 2),
        })
        plays.append({
            "id": "utilization",
            "title": "Inventory & Utilization Tuning",
            "confidence": "Medium",
            "complexity": "Low",
            "rationale": "Tune reorder points and slotting; improve turns/utilization.",
            "uplift_estimate": round(k["revenue"] * 0.01, 2),
        })
        return plays

    st.subheader("Recommended Plays")
    plays = simple_playbook(df, kpis)

    for play in plays:
        with st.container(border=True):
            st.markdown(f"**{play['title']}**")
            st.caption(f"Confidence: {play['confidence']} • Complexity: {play['complexity']}")
            st.write(play["rationale"])

            if "assumed_increase" in play:
                st.slider(
                    "Assumed price increase",
                    min_value=0.00, max_value=0.12, value=play["assumed_increase"],
                    step=0.01, key=f"inc_{play['id']}",
                )

            st.write(f"Estimated revenue impact (very rough): **${play['uplift_estimate']:,.0f}**")

            # Decision radio
            decision = st.radio("Decision", ["Pending", "Approved", "Rejected"],
                                horizontal=True, key=f"dec_{play['id']}")
            rationale = st.text_input("Rationale (optional)", key=f"rat_{play['id']}")
            if st.button("Save decision", key=f"save_{play['id']}"):
                st.session_state["decisions"].append({
                    "id": play["id"],
                    "title": play["title"],
                    "decision": decision.lower(),
                    "rationale": rationale,
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("Saved.")

            st.divider()
            if st.button("Push to Salesforce (mock)", key=f"push_{play['id']}"):
                st.session_state["activity"].append({
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "action": "push",
                    "play": play["title"],
                    "target": "salesforce",
                    "status": "success",
                })
                st.success("Integration job queued → success")

    # Activity
    st.subheader("Activity Log")
    if len(st.session_state.get("activity", [])) == 0:
        st.info("No activity yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state["activity"]))

    # Export PDF (optional)
    if st.button("Export Summary PDF"):
        try:
            from pdf_utils import export_summary_pdf
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                export_summary_pdf(tmp.name, kpis.get("company"), kpis, st.session_state["decisions"])
                tmp.flush()
                st.download_button(
                    "Download PDF",
                    data=open(tmp.name, "rb").read(),
                    file_name="transformai_summary.pdf",
                    mime="application/pdf",
                )
        except Exception as ex:
            st.warning(f"PDF export unavailable: {ex}")

else:
    st.info("Choose a sample or upload a CSV, then click **Generate Playbook**.")
