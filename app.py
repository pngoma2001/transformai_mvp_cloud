# app.py — TransformAI (Streamlit cloud-safe, loads samples from /data)
from __future__ import annotations

from pathlib import Path
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

# ───────────────────────────── Data loading helpers ───────────────────
SAMPLE_MAP = {
    "RetailCo": ("data/sample_retailco.csv", "RetailCo"),
    "RetailCo (8q)": ("data/sample_retailco_long.csv", "RetailCo"),
    "HealthCo": ("data/sample_healthco.csv", "HealthCo"),
    "HealthCo (8q)": ("data/sample_healthco_long.csv", "HealthCo"),
}

def load_sample(label: str) -> pd.DataFrame:
    """Load a sample CSV from /data, with a safe fallback to an empty frame."""
    path, company = SAMPLE_MAP[label]
    p = Path(path)
    if p.exists():
        df = pd.read_csv(p)
        # ensure a 'period' column exists for charts (many samples already have it)
        if "period" not in df.columns:
            df["period"] = range(1, len(df) + 1)
        st.session_state["company_name"] = company
        return df
    # Fallback: empty frame (will trigger friendly error later)
    st.warning(f"Sample file not found: {p}. Using an empty dataset.")
    st.session_state["company_name"] = company
    return pd.DataFrame(columns=["period", "revenue", "ebitda"])

def normalize_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Fix missing/variant columns and units so tiles display correctly."""
    out = df.copy()
    # Derive gross_margin if missing (as %)
    if "gross_margin" not in out.columns and {"revenue", "ebitda"} <= set(out.columns):
        out["gross_margin"] = (
            pd.to_numeric(out["ebitda"], errors="coerce") /
            pd.to_numeric(out["revenue"], errors="coerce")
        ) * 100.0
    # If gross_margin looks like a fraction (<=1), convert to %
    if "gross_margin" in out.columns and pd.to_numeric(out["gross_margin"], errors="coerce").max() <= 1.0:
        out["gross_margin"] = pd.to_numeric(out["gross_margin"], errors="coerce") * 100.0
    # Map inventory_turns → turns_util
    if "turns_util" not in out.columns and "inventory_turns" in out.columns:
        out["turns_util"] = pd.to_numeric(out["inventory_turns"], errors="coerce")
    # Ensure churn stays as % (convert fraction → %)
    if "churn_rate" in out.columns:
        ch = pd.to_numeric(out["churn_rate"], errors="coerce")
        if ch.max() <= 1.0:
            out["churn_rate"] = ch * 100.0
        else:
            out["churn_rate"] = ch
    return out

def has_multi_periods(df: pd.DataFrame) -> bool:
    return "period" in df.columns and df["period"].nunique() > 1

# ───────────────────────────── UI — Data selection ────────────────────
st.markdown(
    "Upload a CSV or use a sample dataset to generate recommended plays with "
    "estimated EBITDA uplift."
)

sample = st.selectbox("Choose a sample company", list(SAMPLE_MAP.keys()), index=1)
uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

with st.expander("Evidence Sources (optional)"):
    st.write("Attach extra signals to make recommendations more contextual (optional).")
    survey_file = st.file_uploader("Customer survey (CSV)", type=["csv"], key="survey")
    market_file = st.file_uploader("Market data (CSV)", type=["csv"], key="market")
    use_sample_evidence = st.checkbox("Use sample evidence if not provided", value=True)

# Resolve primary dataframe
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.session_state["company_name"] = Path(uploaded.name).stem.title()
else:
    df = load_sample(sample)

df = normalize_kpis(df)

# ───────────────────────────── Action ────────────────────────────────
st.divider()
if st.button("Generate Playbook", type="primary"):
    if len(df) == 0 or {"revenue", "ebitda"} - set(df.columns):
        st.error("Your dataset should include at least 'revenue' and 'ebitda' columns.")
    else:
        last = df.iloc[-1]
        kpis = {
            "company": st.session_state.get("company_name", "UnknownCo"),
            "period": last.get("period", "Latest"),
            "revenue": float(last.get("revenue", 0.0)),
            "ebitda": float(last.get("ebitda", 0.0)),
            "gross_margin": float(last.get("gross_margin", 0.0)),   # already %
            "churn_rate": float(last.get("churn_rate", 0.0)),       # already %
            "turns_util": float(last.get("turns_util", 0.0)),
        }
        # YoY % if we have at least 5 periods
        if has_multi_periods(df) and len(df) >= 5:
            kpis["revenue_yoy"] = float(
                (df["revenue"].iloc[-1] - df["revenue"].iloc[-5]) /
                max(df["revenue"].iloc[-5], 1e-9) * 100.0
            )
        else:
            kpis["revenue_yoy"] = None

        st.session_state["kpis"] = kpis
        st.session_state.setdefault("decisions", [])
        st.session_state.setdefault("activity", [])

# ───────────────────────────── Render results ────────────────────────
kpis = st.session_state.get("kpis")
if kpis:
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

    # Simple demo plays (non-LLM)
    def simple_playbook(k: dict) -> list[dict]:
        return [
            {
                "id": "pricing",
                "title": "Pricing Optimization",
                "confidence": "High",
                "complexity": "Medium",
                "assumed_increase": 0.06,
                "rationale": "Top 30% SKUs show low elasticity; raise prices 5–7% with guardrails.",
                "uplift_estimate": round(k["revenue"] * 0.06 * 0.3 * 0.3, 2),
            },
            {
                "id": "retention",
                "title": "Customer Retention Program",
                "confidence": "Medium",
                "complexity": "Medium",
                "assumed_decrease": 0.02,
                "rationale": "Lifecycle messaging & loyalty offers to reduce churn by ~2%.",
                "uplift_estimate": round(k["revenue"] * 0.02 * 0.4, 2),
            },
            {
                "id": "utilization",
                "title": "Inventory & Utilization Tuning",
                "confidence": "Medium",
                "complexity": "Low",
                "rationale": "Tune reorder points and slotting; improve turns/utilization.",
                "uplift_estimate": round(k["revenue"] * 0.01, 2),
            },
        ]

    st.subheader("Recommended Plays")
    for play in simple_playbook(kpis):
        with st.container(border=True):
            st.markdown(f"**{play['title']}**")
            st.caption(f"Confidence: {play['confidence']} • Complexity: {play['complexity']}")
            st.write(play["rationale"])

            if "assumed_increase" in play:
                st.slider("Assumed price increase", 0.00, 0.12, play["assumed_increase"], 0.01, key=f"inc_{play['id']}")

            st.write(f"Estimated revenue impact (very rough): **${play['uplift_estimate']:,.0f}**")

            decision = st.radio("Decision", ["Pending", "Approved", "Rejected"], horizontal=True, key=f"dec_{play['id']}")
            rationale = st.text_input("Rationale (optional)", key=f"rat_{play['id']}")
            if st.button("Save decision", key=f"save_{play['id']}"):
                st.session_state["decisions"].append({
                    "id": play["id"],
                    "title": play["title"],
                    "decision": decision.lower(),
                    "rationale": rationale,
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
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

    st.subheader("Activity Log")
    if len(st.session_state.get("activity", [])) == 0:
        st.info("No activity yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state["activity"]))

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
