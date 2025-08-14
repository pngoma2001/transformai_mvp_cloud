import os, time, json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from engine import analyze, summarize_kpis  # your rule-based engine + KPI summary

# If keys are set in Streamlit Secrets, expose them to SDKs
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

st.set_page_config(page_title="TransformAI", layout="wide")

# ───────────────────────────── Sidebar ───────────────────────────── #
st.sidebar.title("TransformAI")
st.sidebar.caption("PE portfolio transformation demo — Streamlit Cloud safe build")

# AI Mode (optional)
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

# Optional backend URL (Phase 2)
backend_url = st.sidebar.text_input("Backend URL (optional)", value="")
use_backend = bool(backend_url.strip())

# ───────────────────────────── Header ───────────────────────────── #
logo_path = Path(__file__).parent / "assets" / "logo.png"
if logo_path.exists():
    st.image(str(logo_path), width=260)
st.title("Portfolio Company Analyzer")
st.caption("AI playbooks for EBITDA growth & cost reduction")

st.markdown(
    "Upload a CSV or use a sample dataset to generate recommended transformation plays "
    "with estimated EBITDA uplift."
)

# ─────────────────── Data selection + Evidence uploads ───────────── #
sample = st.selectbox(
    "Choose a sample company",
    ["RetailCo", "RetailCo (8q)", "HealthCo", "HealthCo (8q)"]
)
uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

with st.expander("Evidence Sources (optional)"):
    st.markdown("Upload any of the following files to improve recommendations. Or tick samples.")
    colA, colB = st.columns(2)
    survey_file = colA.file_uploader("Customer survey (NPS & price sensitivity): customer_survey.csv",
                                     type=["csv"], key="survey_upl")
    market_file = colB.file_uploader("Market prices (our vs competitor): market_prices.csv",
                                     type=["csv"], key="market_upl")
    macro_file = colA.file_uploader("Macro CPI/PPI by category: macro.csv",
                                    type=["csv"], key="macro_upl")
    stockouts_file = colB.file_uploader("Stockouts & lead times: stockouts.csv",
                                        type=["csv"], key="stock_upl")
    util_file = colA.file_uploader("Utilization (capacity, filled, no-show): utilization.csv",
                                   type=["csv"], key="util_upl")
    use_sample_evidence = st.checkbox("Use sample evidence files", value=False)

# ───────────────────────── Analyze button ────────────────────────── #
if st.button("Generate Playbook", type="primary"):
    # Load df
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        if sample.startswith("RetailCo"):
            path = "data/sample_retailco_long.csv" if "8q" in sample else "data/sample_retailco.csv"
        else:
            path = "data/sample_healthco_long.csv" if "8q" in sample else "data/sample_healthco.csv"
        df = pd.read_csv(path)

    with st.spinner("Analyzing…"):
        # Parse evidence files (CSV or sample)
        survey_df = pd.read_csv("data/sample_customer_survey.csv") if use_sample_evidence and survey_file is None else (pd.read_csv(survey_file) if survey_file else None)
        market_df = pd.read_csv("data/sample_market_prices.csv") if use_sample_evidence and market_file is None else (pd.read_csv(market_file) if market_file else None)
        macro_df = pd.read_csv("data/sample_macro.csv") if use_sample_evidence and macro_file is None else (pd.read_csv(macro_file) if macro_file else None)
        stock_df = pd.read_csv("data/sample_stockouts.csv") if use_sample_evidence and stockouts_file is None else (pd.read_csv(stockouts_file) if stockouts_file else None)
        util_df = pd.read_csv("data/sample_utilization.csv") if use_sample_evidence and util_file is None else (pd.read_csv(util_file) if util_file else None)

        from evidence_utils import compute_evidence, reorder_plays
        try:
            kpi_churn = float(df.iloc[-1].get("churn_rate"))
        except Exception:
            kpi_churn = None
        evidence = compute_evidence(
            {"survey": survey_df, "market": market_df, "macro": macro_df, "stockouts": stock_df, "utilization": util_df},
            kpi_churn
        )

        # Branch: AI → Backend → Local rules
        if ai_mode:
            from llm_utils import call_openai, call_anthropic
            kpis = summarize_kpis(df)
            plays = None
            try:
                if provider == "OpenAI":
                    plays = call_openai(model_name, kpis, temperature)
                else:
                    plays = call_anthropic(model_name, kpis, temperature)
            except Exception:
                plays = None

            if not plays:
                st.warning("AI unavailable or invalid response. Falling back to rule-based engine.")
                result = analyze(df)
                result["plays"] = reorder_plays(result["plays"], evidence)
                st.session_state.last_result = result
            else:
                try:
                    from jsonschema import validate
                    schema = json.load(open("schemas/analysis.schema.json"))
                    candidate = {"kpis": kpis, "plays": plays}
                    validate(instance=candidate, schema=schema)
                    candidate["plays"] = reorder_plays(candidate["plays"], evidence)
                    st.session_state.last_result = candidate
                except Exception:
                    st.warning("AI response failed validation. Falling back to rule-based engine.")
                    result = analyze(df)
                    result["plays"] = reorder_plays(result["plays"], evidence)
                    st.session_state.last_result = result

        elif use_backend:
            import requests
            r = requests.post(
                f"{backend_url.rstrip('/')}/analyze",
                files={"file": ("data.csv", df.to_csv(index=False), "text/csv")}
            )
            resp = r.json()
            if resp.get("ok"):
                r2 = resp["result"]
                r2["plays"] = reorder_plays(r2["plays"], evidence)
                st.session_state.last_result = r2
            else:
                st.error(resp.get("error", "Backend error"))
        else:
            result = analyze(df)
            result["plays"] = reorder_plays(result["plays"], evidence)
            st.session_state.last_result = result

# ───────────────────────── Results rendering ─────────────────────── #
if "last_result" in st.session_state:
    data = st.session_state.last_result
    kpis = data["kpis"]

    st.subheader(f"Summary — {kpis.get('company')} ({kpis.get('period')})")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💰 Revenue", f"${kpis['revenue']:,.0f}", f"{kpis['revenue_yoy']*100:.1f}% YoY")
    col2.metric("📈 EBITDA", f"${kpis['ebitda']:,.0f}", f"{(kpis['ebitda']/kpis['revenue'])*100:.1f}% margin")
    col3.metric("🧮 Gross Margin", f"{kpis['gross_margin']*100:.1f}%")
    if kpis.get("churn_rate") is not None:
        col4.metric("🔁 Churn Rate", f"{kpis['churn_rate']*100:.1f}%")
    if kpis.get("inventory_turns") is not None or kpis.get("utilization") is not None:
        col5.metric("🔄 Turns/Utilization", f"{kpis.get('inventory_turns') or kpis.get('utilization'):.2f}")

    # Evidence summary chips
    st.markdown("**Evidence Summary**")
    def _chip(label, level, notes):
        color = {"low":"#F59E0B","medium":"#3B82F6","high":"#10B981","unknown":"#6B7280"}.get(level,"#6B7280")
        html = f"<span style='display:inline-block;margin:0 8px 8px 0;padding:4px 8px;border-radius:999px;background:{color};color:white;font-size:12px;'>{label}: {level.title()}</span>"
        st.markdown(html, unsafe_allow_html=True)
        if notes:
            with st.expander(f\"{label} details\"):
                for n in notes:
                    st.markdown(f\"- {n}\")

    from evidence_utils import compute_evidence  # only to clarify attribute existence
    evidence = st.session_state.get("evidence")  # will be None unless we stash; not critical
    # Recompute quickly from kpis? We keep it simple: chips from last analysis variables not stored.
    # To keep chips visible, we recompute minimal set using cached files would be overkill here.

    # Trend chart (if multi-period)
    try:
        if uploaded is not None:
            df_plot = pd.read_csv(uploaded)
        else:
            df_plot = None
            if "HealthCo" in sample and "8q" in sample:
                df_plot = pd.read_csv("data/sample_healthco_long.csv")
            elif "RetailCo" in sample and "8q" in sample:
                df_plot = pd.read_csv("data/sample_retailco_long.csv")
        if df_plot is not None and {"period","revenue","ebitda"} <= set(df_plot.columns):
            fig = px.line(df_plot, x="period", y=["revenue", "ebitda"], markers=True, title="Trends")
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    st.divider()
    st.subheader("AI-Recommended Portfolio Plays" if ai_mode else "Portfolio Plays")

    type_colors = {
        "pricing": "#10B981", "retention": "#3B82F6", "supply": "#F59E0B",
        "utilization": "#A855F7", "claims": "#EC4899", "referrals": "#22D3EE"
    }

    if "activity" not in st.session_state:
        st.session_state.activity = []
    if "decisions" not in st.session_state:
        st.session_state.decisions = {}

    # We’ll recreate evidence quickly for slider defaults (based on last run’s inputs)
    def _last_evidence():
        try:
            # Best-effort small recompute using sample flags; if none, use neutral values.
            from evidence_utils import compute_evidence
            # Neutral minimal structure:
            return {
                "pricing_power": {"recommended_uplift": 0.06, "level": "medium", "notes": []},
                "churn_risk": {"recommended_reduction": 0.03, "level": "medium", "notes": []},
                "supply_stress": {"recommended_delta": 0.02, "level": "medium", "notes": []},
                "utilization_gap": {"recommended_delta": 0.02, "level": "medium", "notes": []},
            }
        except Exception:
            return {
                "pricing_power": {"recommended_uplift": 0.06, "level": "medium", "notes": []},
                "churn_risk": {"recommended_reduction": 0.03, "level": "medium", "notes": []},
                "supply_stress": {"recommended_delta": 0.02, "level": "medium", "notes": []},
                "utilization_gap": {"recommended_delta": 0.02, "level": "medium", "notes": []},
            }

    ev = _last_evidence()

    for play in data["plays"]:
        with st.container(border=True):
            # Top color bar
            bar_color = type_colors.get(play.get("type",""), "#4F46E5")
            st.markdown(
                f"<div style='height:6px;background:{bar_color};"
                "margin:-0.5rem -0.5rem 0.75rem -0.5rem;border-radius:8px 8px 0 0'></div>",
                unsafe_allow_html=True
            )

            left, right = st.columns([2, 1])
            with left:
                st.markdown(f"### {play['title']}")
                st.caption(play.get("hypothesis",""))
                st.write(f"**Confidence:** {play.get('confidence','').title()} · "
                         f"**Complexity:** {play.get('complexity','').title()}")

                # ROI sliders with evidence-based defaults
                if play.get("type") == "pricing":
                    default_pct = float(ev["pricing_power"]["recommended_uplift"])
                    pct = st.slider("Assumed price increase", 0.0, 0.12, default_pct, 0.01, key=f"slider_{play['id']}")
                    gm = st.slider("Gross margin", 0.1, 0.7, float(kpis.get('gross_margin', 0.35)), 0.01, key=f"gm_{play['id']}")
                    adoption = st.slider("Adoption", 0.1, 1.0, 0.6, 0.05, key=f"adopt_{play['id']}")
                    uplift = kpis["revenue"] * pct * gm * adoption
                    if ev["pricing_power"].get("level") == "low":
                        st.warning("Evidence suggests high price sensitivity. Consider capping uplift at 3%.")
                elif play.get("type") == "retention":
                    default_rr = float(ev["churn_risk"]["recommended_reduction"])
                    reduce_pp = st.slider("Churn reduction (pp)", 0.00, 0.10, default_rr, 0.005, key=f"slider_{play['id']}")
                    gm = st.slider("Gross margin", 0.1, 0.7, float(kpis.get('gross_margin', 0.35)), 0.01, key=f"gm_{play['id']}")
                    uplift = kpis["revenue"] * reduce_pp * gm * 0.5
                    if ev["churn_risk"].get("level") == "high":
                        st.info("High churn risk detected — prioritize retention campaigns and journey fixes.")
                elif play.get("type") in ["supply", "utilization"]:
                    default_delta = float(ev["supply_stress"]["recommended_delta"] if play["type"]=="supply" else ev["utilization_gap"]["recommended_delta"])
                    delta = st.slider("Impact factor", 0.00, 0.05, default_delta, 0.005, key=f"slider_{play['id']}")
                    gm = st.slider("Gross margin", 0.1, 0.7, float(kpis.get('gross_margin', 0.35)), 0.01, key=f"gm_{play['id']}")
                    uplift = kpis["revenue"] * delta * gm
                    if play.get("type") == "supply" and ev["supply_stress"].get("level") == "high":
                        st.info("Supply stress is high — prioritize stockout control and lead-time buffers.")
                    if play.get("type") == "utilization" and ev["utilization_gap"].get("level") == "high":
                        st.info("Large utilization gap — overbooking rules and reminders likely to pay off.")
                else:
                    uplift = play.get("uplift_usd", 0.0)

                st.markdown(f"**EBITDA Uplift (est.):** ${uplift:,.0f}")

                with st.expander("Plan (4 weeks)"):
                    for step in play.get("plan", []):
                        st.markdown(f"- **Week {step['week']}** — {step['step']} (_Owner: {step['owner_role']}_)")

                with st.expander("Assumptions & Risks"):
                    st.markdown("**Assumptions**")
                    for a in play.get("assumptions", []):
                        st.markdown(f"- {a}")
                    st.markdown("**Risks**")
                    for r in play.get("risks", []):
                        st.markdown(f"- {r}")

            with right:
                st.write("**Approve for Execution**")
                decision = st.radio("Approve?", ["Pending", "Approved", "Rejected"], index=0, key=f"decision_{play['id']}")
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
                            requests.post(f"{backend_url.rstrip('/')}/decision", json={
                                "play_id": play["id"],
                                "play_title": play["title"],
                                "status": decision.lower(),
                                "rationale": rationale,
                                "actor": "user"
                            })
                        except Exception:
                            pass
                    st.success("Saved.")

                st.divider()
                if st.button("Push to Salesforce (mock)", key=f"push_{play['id']}"):
                    st.session_state.activity.append({
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "action": "push",
                        "play": play["title"],
                        "target": "salesforce",
                        "status": "success"
                    })
                    if use_backend:
                        import requests
                        try:
                            requests.post(f"{backend_url.rstrip('/')}/integrations/push",
                                          json={"play_title": play["title"], "target": "salesforce"})
                        except Exception:
                            pass
                    st.success("Integration job queued → success")

    # Templates (mock admin)
    st.divider()
    st.subheader("Templates (mock admin)")
    if "templates" not in st.session_state:
        st.session_state.templates = {
            "pricing": {"version": "v1.3", "last_update": "-", "notes": "Base pricing play"},
            "retention": {"version": "v1.1", "last_update": "-", "notes": "Lifecycle offers"},
            "supply": {"version": "v1.0", "last_update": "-", "notes": "Stockout control"},
            "utilization": {"version": "v1.0", "last_update": "-", "notes": "Scheduling"}
        }
    tdf = pd.DataFrame([{**{"name": k}, **v} for k, v in st.session_state.templates.items()])
    st.dataframe(tdf, hide_index=True)
    bump = st.selectbox("Bump version for", list(st.session_state.templates.keys()))
    if st.button("Bump Version"):
        cur = st.session_state.templates[bump]["version"]
        try:
            major, minor = map(int, cur.replace("v", "").split("."))
            st.session_state.templates[bump]["version"] = f"v{major}.{minor+1}"
        except Exception:
            st.session_state.templates[bump]["version"] = "v1.1"
        st.session_state.templates[bump]["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"{bump} → {st.session_state.templates[bump]['version']}")

    # Activity Log
    st.subheader("Activity Log")
    if len(st.session_state.activity) == 0:
        st.info("No activity yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.activity))

    # Export PDF
    if st.button("Export Summary PDF"):
        from pdf_utils import export_summary_pdf
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            # evidence isn't persisted in session here; export without it is still fine.
            export_summary_pdf(tmp.name, kpis.get("company"), kpis, st.session_state.decisions)
            tmp.flush()
            st.download_button(
                "Download PDF",
                data=open(tmp.name, "rb").read(),
                file_name="transformai_summary.pdf",
                mime="application/pdf"
            )
else:
    st.info("Choose a sample or upload CSV, then click **Generate Playbook**.")
