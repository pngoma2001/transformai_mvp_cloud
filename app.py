# app.py — TransformAI (Cloud) with evidence chips fix

import os, time, json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from engine import analyze, summarize_kpis
from evidence_utils import (
    load_sample_evidence, compute_evidence_signals,
    rank_plays_by_evidence, chips_from_signals
)

# Map Streamlit secrets to env vars (optional for SDKs)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

st.set_page_config(page_title="TransformAI", layout="wide")

# ───────────────── Sidebar ─────────────────
st.sidebar.title("TransformAI")
st.sidebar.caption("PE portfolio transformation demo — Streamlit Cloud safe build")

# AI mode (optional)
st.sidebar.subheader("AI Mode (optional)")
ai_mode = st.sidebar.toggle(
    "Enable AI-generated plays",
    value=False,
    help="When ON, uses OpenAI/Anthropic to generate plays. Falls back to rules if unavailable."
)
provider = st.sidebar.selectbox("Provider", ["OpenAI", "Anthropic"], disabled=not ai_mode)
default_model = "gpt-4o-mini" if provider == "OpenAI" else "claude-3-5-sonnet-20240620"
model_name = st.sidebar.text_input("Model name", value=default_model, disabled=not ai_mode)
temperature = st.sidebar.slider("Creativity", 0.0, 1.0, 0.2, 0.05, disabled=not ai_mode)

# Optional backend URL (if you run FastAPI)
backend_url = st.sidebar.text_input("Backend URL (optional)", value="")
use_backend = bool(backend_url.strip())

# ───────────────── Header / Hero ────────────────
logo_path = Path("assets/logo.png")
if logo_path.exists():
    st.image(str(logo_path), width=220)
else:
    st.markdown("### TransformAI")

st.title("Portfolio Company Analyzer (Cloud)")
st.caption("AI playbooks for EBITDA growth & cost reduction")

st.markdown(
    "Upload a CSV or use a sample dataset to generate recommended transformation plays "
    "with estimated EBITDA uplift."
)

# ───────────────── Data selection ───────────────
sample = st.selectbox(
    "Choose a sample company",
    ["RetailCo", "RetailCo (8q)", "HealthCo", "HealthCo (8q)"]
)
uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

# Evidence UI
with st.expander("Evidence Sources (optional)"):
    st.caption("Attach extra signals; we’ll set defaults, reorder plays, and show warnings.")
    use_sample_ev = st.checkbox("Use sample evidence if not provided", value=True)
    survey_file    = st.file_uploader("Customer survey (CSV)", type=["csv"], key="survey_csv")
    market_file    = st.file_uploader("Market prices (CSV)", type=["csv"], key="market_csv")
    macro_file     = st.file_uploader("Macro CPI/PPI (CSV)", type=["csv"], key="macro_csv")
    stockouts_file = st.file_uploader("Stockouts (CSV)", type=["csv"], key="stockouts_csv")
    util_file      = st.file_uploader("Utilization (CSV)", type=["csv"], key="util_csv")

# ───────────────── Analyze ──────────────────────
if st.button("Generate Playbook", type="primary"):
    # Load main dataset
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        if sample.startswith("RetailCo"):
            path = "data/sample_retailco_long.csv" if "8q" in sample else "data/sample_retailco.csv"
        else:
            path = "data/sample_healthco_long.csv" if "8q" in sample else "data/sample_healthco.csv"
        df = pd.read_csv(path)

    # Evidence load
    ev = {}
    if any([survey_file, market_file, macro_file, stockouts_file, util_file]) or use_sample_ev:
        ev = load_sample_evidence(
            use_sample_ev=use_sample_ev,
            survey_file=survey_file,
            market_file=market_file,
            macro_file=macro_file,
            stockouts_file=stockouts_file,
            util_file=util_file,
        )

    # Analyze (AI → backend → local rules)
    with st.spinner("Analyzing…"):
        if ai_mode:
            from llm_utils import call_openai, call_anthropic
            from jsonschema import validate
            kpis = summarize_kpis(df)
            plays = None
            try:
                plays = (call_openai if provider == "OpenAI" else call_anthropic)(model_name, kpis, temperature)
            except Exception:
                plays = None

            if not plays:
                result = analyze(df)
            else:
                try:
                    schema = json.load(open("schemas/analysis.schema.json"))
                    candidate = {"kpis": kpis, "plays": plays}
                    validate(instance=candidate, schema=schema)
                    result = candidate
                except Exception:
                    result = analyze(df)
        elif use_backend:
            import requests
            r = requests.post(
                f"{backend_url.rstrip('/')}/analyze",
                files={"file": ("data.csv", df.to_csv(index=False), "text/csv")},
            )
            resp = r.json()
            if resp.get("ok"):
                result = resp["result"]
            else:
                st.error(resp.get("error", "Backend error"))
                st.stop()
        else:
            result = analyze(df)

        # Apply evidence weighting/reordering
        signals = compute_evidence_signals(ev, df)
        result["plays"] = rank_plays_by_evidence(result["plays"], signals)

        # Save to session
        st.session_state.last_result = result
        st.session_state.last_signals = signals
        st.session_state.activity = st.session_state.get("activity", [])

        # Fetch backend activity (optional)
        if use_backend:
            try:
                import requests
                act = requests.get(f"{backend_url.rstrip('/')}/activity").json()
                if isinstance(act, list):
                    st.session_state.activity = act
            except Exception:
                pass

# ───────────────── Render ───────────────────────
if "last_result" in st.session_state:
    data = st.session_state.last_result
    kpis = data["kpis"]
    signals = st.session_state.get("last_signals", {})

    st.subheader(f"Summary — {kpis.get('company')} ({kpis.get('period')})")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💰 Revenue", f"${kpis['revenue']:,.0f}", f"{kpis['revenue_yoy']*100:.1f}% YoY")
    col2.metric("📈 EBITDA", f"${kpis['ebitda']:,.0f}", f"{(kpis['ebitda']/kpis['revenue'])*100:.1f}% margin")
    col3.metric("🧮 Gross Margin", f"{kpis['gross_margin']*100:.1f}%")
    if kpis.get("churn_rate") is not None:
        col4.metric("🔁 Churn Rate", f"{kpis['churn_rate']*100:.1f}%")
    if kpis.get("inventory_turns") is not None or kpis.get("utilization") is not None:
        col5.metric("🔄 Turns/Utilization", f"{kpis.get('inventory_turns') or kpis.get('utilization'):.2f}")

    # Optional trend chart when multiple periods exist
    try:
        if uploaded is not None:
            df_plot = pd.read_csv(uploaded)
        else:
            df_plot = (
                pd.read_csv("data/sample_healthco_long.csv")
                if "HealthCo" in sample and "8q" in sample
                else pd.read_csv("data/sample_retailco_long.csv")
                if "RetailCo" in sample and "8q" in sample
                else None
            )
        if df_plot is not None and "period" in df_plot.columns and "revenue" in df_plot.columns:
            fig = px.line(df_plot, x="period", y=["revenue", "ebitda"], markers=True, title="Trends")
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    # ───────── Evidence (bug fix: render chips as HTML)
    st.markdown("### Evidence")
    chips = chips_from_signals(signals)  # returns <span> badges from evidence_utils.py  :contentReference[oaicite:1]{index=1}
    if chips:
        st.markdown(" ".join(chips), unsafe_allow_html=True)  # <-- fix (was st.write)

    with st.expander("Evidence details"):
        for k, v in signals.get("details", {}).items():
            st.markdown(f"**{k}**")
            for line in v:
                st.markdown(f"- {line}")

    st.divider()
    st.subheader("Recommended Plays")

    # Type colors
    type_colors = {
        "pricing": "#10B981",
        "retention": "#3B82F6",
        "supply": "#F59E0B",
        "utilization": "#A855F7",
        "claims": "#EC4899",
        "referrals": "#22D3EE",
    }

    if "activity" not in st.session_state:
        st.session_state.activity = []
    if "decisions" not in st.session_state:
        st.session_state.decisions = {}

    defaults = signals.get("defaults", {})

    for play in data["plays"]:
        with st.container(border=True):
            bar_color = type_colors.get(play.get("type", ""), "#4F46E5")
            st.markdown(
                f"<div style='height:6px;background:{bar_color};"
                "margin:-0.5rem -0.5rem 0.75rem -0.5rem;border-radius:8px 8px 0 0'></div>",
                unsafe_allow_html=True,
            )

            left, right = st.columns([2, 1])
            with left:
                st.markdown(f"### {play['title']}")
                st.caption(play.get("hypothesis", ""))

                warn = signals.get("warnings", {}).get(play.get("type", ""))
                if warn:
                    st.warning(warn)

                # Evidence-informed sliders
                if play.get("type") == "pricing":
                    pct = st.slider(
                        "Assumed price increase", 0.0, 0.12, float(defaults.get("pricing_uplift", 0.06)), 0.01,
                        key=f"slider_{play['id']}",
                    )
                    gm = st.slider(
                        "Gross margin", 0.1, 0.7, float(kpis.get("gross_margin", 0.35)), 0.01,
                        key=f"gm_{play['id']}",
                    )
                    adoption = st.slider(
                        "Adoption", 0.1, 1.0, float(defaults.get("pricing_adoption", 0.6)), 0.05,
                        key=f"adopt_{play['id']}",
                    )
                    uplift = kpis["revenue"] * pct * gm * adoption
                elif play.get("type") == "retention":
                    reduce_pp = st.slider(
                        "Churn reduction (pp)", 0.00, 0.10, float(defaults.get("retention_reduce_pp", 0.03)), 0.005,
                        key=f"slider_{play['id']}",
                    )
                    gm = st.slider(
                        "Gross margin", 0.1, 0.7, float(kpis.get("gross_margin", 0.35)), 0.01,
                        key=f"gm_{play['id']}",
                    )
                    uplift = kpis["revenue"] * reduce_pp * gm * 0.5
                elif play.get("type") in ["supply", "utilization"]:
                    delta = st.slider(
                        "Impact factor", 0.00, 0.05, float(defaults.get("ops_delta", 0.02)), 0.005,
                        key=f"slider_{play['id']}",
                    )
                    gm = st.slider(
                        "Gross margin", 0.1, 0.7, float(kpis.get("gross_margin", 0.35)), 0.01,
                        key=f"gm_{play['id']}",
                    )
                    uplift = kpis["revenue"] * delta * gm
                else:
                    uplift = play.get("uplift_usd", 0.0)

                st.markdown(f"**EBITDA Uplift (est.):** ${uplift:,.0f}")

                with st.expander("Plan (4 weeks)"):
                    for step in play.get("plan", []):
                        st.markdown(f"- **Week {step['week']}** — {step['step']} (_Owner: {step['owner_role']}_)")

                with st.expander("Assumptions & Risks"):
                    if play.get("assumptions"):
                        st.markdown("**Assumptions**")
                        for a in play["assumptions"]:
                            st.markdown(f"- {a}")
                    if play.get("risks"):
                        st.markdown("**Risks**")
                        for r in play["risks"]:
                            st.markdown(f"- {r}")

            with right:
                st.write("**Approve for Execution**")
                decision = st.radio("Decision", ["Pending", "Approved", "Rejected"], index=0, key=f"decision_{play['id']}")
                rationale = st.text_area("Rationale (optional)", key=f"rationale_{play['id']}", height=80)

                if st.button("Save Decision", key=f"save_{play['id']}"):
                    st.session_state.decisions[play["id"]] = {"status": decision.lower(), "rationale": rationale}
                    st.session_state.activity.append({
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "action": "decision",
                        "play": play["title"],
                        "status": decision
                    })
                    if use_backend:
                        import requests
                        try:
                            requests.post(
                                f"{backend_url.rstrip('/')}/decision",
                                json={
                                    "play_id": play["id"],
                                    "play_title": play["title"],
                                    "status": decision.lower(),
                                    "rationale": rationale,
                                    "actor": "user",
                                },
                            )
                        except Exception:
                            pass
                    st.success("Saved.")

                st.divider()
                target = st.selectbox("Push target", ["Salesforce", "NetSuite", "HubSpot", "CSV"], key=f"target_{play['id']}")
                if st.button("Push (mock)", key=f"push_{play['id']}"):
                    st.session_state.activity.append({
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "action": "push",
                        "play": play["title"],
                        "target": target,
                        "status": "success",
                    })
                    if use_backend:
                        import requests
                        try:
                            requests.post(
                                f"{backend_url.rstrip('/')}/integrations/push",
                                json={"play_title": play["title"], "target": target},
                            )
                        except Exception:
                            pass
                    st.success(f"Integration job queued → {target}")

    # Activity
    st.subheader("Activity Log")
    if len(st.session_state.activity) == 0:
        st.info("No activity yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.activity))

    # Export PDF (with optional Evidence Appendix)
    if st.button("Export Summary PDF"):
        from pdf_utils import export_summary_pdf
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            export_summary_pdf(tmp.name, kpis.get("company"), kpis, st.session_state.decisions, evidence=signals)
            tmp.flush()
            st.download_button(
                "Download PDF",
                data=open(tmp.name, "rb").read(),
                file_name="transformai_summary.pdf",
                mime="application/pdf",
            )
else:
    st.info("Choose a sample or upload CSV, then click **Generate Playbook**.")
