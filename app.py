
import streamlit as st
import pandas as pd
import json, time, io
from engine import analyze
import plotly.express as px

st.set_page_config(page_title="TransformAI MVP", layout="wide")

st.sidebar.title("TransformAI")
st.sidebar.caption("PE portfolio transformation demo — Streamlit Cloud safe build")

st.sidebar.subheader("AI Mode (optional)")
ai_mode = st.sidebar.toggle("Enable AI-generated plays", value=False, help="When ON, uses OpenAI/Anthropic to generate plays. Falls back to rules if unavailable.")
provider = st.sidebar.selectbox("Provider", ["OpenAI","Anthropic"], disabled=not ai_mode)
default_model = "gpt-4o-mini" if provider=="OpenAI" else "claude-3-5-sonnet-20240620"
model_name = st.sidebar.text_input("Model name", value=default_model, disabled=not ai_mode)
temperature = st.sidebar.slider("Creativity", 0.0, 1.0, 0.2, 0.05, disabled=not ai_mode)


# Optional backend URL (set in sidebar). If provided, the app will use REST endpoints instead of local engine.
backend_url = st.sidebar.text_input("Backend URL (optional)", value="")
use_backend = bool(backend_url.strip())


# Session state
if "activity" not in st.session_state:
    st.session_state.activity = []  # list of dicts
if "decisions" not in st.session_state:
    st.session_state.decisions = {}

st.title("TransformAI — Portfolio Company Analyzer (Cloud)")

st.markdown("Upload a CSV or use a sample dataset to generate recommended transformation plays with estimated EBITDA uplift.")

# Data selection
sample = st.selectbox("Choose a sample company", ["RetailCo", "RetailCo (8q)", "HealthCo", "HealthCo (8q)"])
uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

with st.expander("Evidence Sources (optional)"):
    st.markdown("Upload any of the following files to improve recommendations. Use our samples if you prefer.")
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


if st.button("Analyze Company", type="primary"):
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        if sample.startswith("RetailCo"):
            df = pd.read_csv("data/sample_retailco_long.csv") if "8q" in sample else pd.read_csv("data/sample_retailco.csv")
        else:
            df = pd.read_csv("data/sample_healthco_long.csv") if "8q" in sample else pd.read_csv("data/sample_healthco.csv")
    with st.spinner("Analyzing…"):

    # Parse evidence files (CSV or sample)
    survey_df = pd.read_csv("data/sample_customer_survey.csv") if use_sample_evidence and survey_file is None else (pd.read_csv(survey_file) if survey_file else None)
    market_df = pd.read_csv("data/sample_market_prices.csv") if use_sample_evidence and market_file is None else (pd.read_csv(market_file) if market_file else None)
    macro_df = pd.read_csv("data/sample_macro.csv") if use_sample_evidence and macro_file is None else (pd.read_csv(macro_file) if macro_file else None)
    stock_df = pd.read_csv("data/sample_stockouts.csv") if use_sample_evidence and stockouts_file is None else (pd.read_csv(stockouts_file) if stockouts_file else None)
    util_df = pd.read_csv("data/sample_utilization.csv") if use_sample_evidence and util_file is None else (pd.read_csv(util_file) if util_file else None)
    from evidence_utils import compute_evidence, reorder_plays
    kpi_churn = None
    try:
        kpi_churn = float(df.iloc[-1].get("churn_rate"))
    except Exception:
        kpi_churn = None
    evidence = compute_evidence({"survey": survey_df, "market": market_df, "macro": macro_df, "stockouts": stock_df, "utilization": util_df}, kpi_churn)

        time.sleep(0.6)
        import json
        if ai_mode:
            from jsonschema import validate, ValidationError
            from engine import summarize_kpis  # reuse KPI summary
            from llm_utils import call_openai, call_anthropic
            kpis = summarize_kpis(df)
            plays = None
            with st.spinner("Calling LLM…"):
                if provider == "OpenAI":
                    plays = call_openai(model_name, kpis, temperature)
                else:
                    plays = call_anthropic(model_name, kpis, temperature)
            if not plays:
                st.warning("AI unavailable or invalid response. Falling back to rule-based engine.")
                result = analyze(df)
                result["plays"] = reorder_plays(result["plays"], evidence)
                st.session_state.last_result = result
            else:
                # Build result and validate
                result = {"kpis": kpis, "plays": plays}
                try:
                    import json, os
                    from jsonschema import validate
                    schema = json.load(open("schemas/analysis.schema.json"))
                    validate(instance=result, schema=schema)
                    result["plays"] = reorder_plays(result["plays"], evidence)
                st.session_state.last_result = result
                except Exception:
                    st.warning("AI response failed validation. Falling back to rule-based engine.")
                    result = analyze(df)
                    result["plays"] = reorder_plays(result["plays"], evidence)
                st.session_state.last_result = result
        elif use_backend:
            import requests
            r = requests.post(f"{backend_url.rstrip('/')}/analyze", files={"file": ('data.csv', df.to_csv(index=False), 'text/csv')})
            resp = r.json()
            if resp.get("ok"):
                r2 = resp["result"]
                r2["plays"] = reorder_plays(r2["plays"], evidence)
                st.session_state.last_result = r2
            else:
                st.error(resp.get("error","Backend error"))
        else:
            result = analyze(df)
            result["plays"] = reorder_plays(result["plays"], evidence)
            st.session_state.last_result = result  # plain dict

if "last_result" in st.session_state:
    data = st.session_state.last_result
    kpis = data["kpis"]
    st.subheader(f"Summary — {kpis.get('company')} ({kpis.get('period')})")

    # KPI tiles
    col1, col2, col3, col4, col5 = st.columns(5)

    st.markdown("**Evidence Summary**")
    def _chip(label, level, notes):
        color = {"low":"#F59E0B","medium":"#3B82F6","high":"#10B981","unknown":"#6B7280"}.get(level,"#6B7280")
        html = f"<span style='display:inline-block;margin:0 8px 8px 0;padding:4px 8px;border-radius:999px;background:{color};color:white;font-size:12px;'>{label}: {level.title()}</span>"
        st.markdown(html, unsafe_allow_html=True)
        if notes:
            with st.expander(f"{label} details"):
                for n in notes:
                    st.markdown(f"- {n}")
    _chip("Pricing power", evidence['pricing_power']['level'], evidence['pricing_power']['notes'])
    _chip("Churn risk", evidence['churn_risk']['level'], evidence['churn_risk']['notes'])
    _chip("Supply stress", evidence['supply_stress']['level'], evidence['supply_stress']['notes'])
    _chip("Utilization gap", evidence['utilization_gap']['level'], evidence['utilization_gap']['notes'])

    col1.metric("Revenue", f"${kpis['revenue']:,.0f}", f"{kpis['revenue_yoy']*100:.1f}% YoY")
    col2.metric("EBITDA", f"${kpis['ebitda']:,.0f}", f"{(kpis['ebitda']/kpis['revenue'])*100:.1f}% margin")
    col3.metric("Gross Margin", f"{kpis['gross_margin']*100:.1f}%")
    if kpis.get("churn_rate") is not None:
        col4.metric("Churn Rate", f"{kpis['churn_rate']*100:.1f}%")
    if kpis.get("inventory_turns") is not None or kpis.get("utilization") is not None:
        col5.metric("Turns/Utilization", f"{kpis.get('inventory_turns') or kpis.get('utilization'):.2f}")

    st.divider()
    # Optional: KPI trend chart if CSV had multiple rows
    try:
        import plotly.express as px
        if uploaded is not None:
            df_plot = pd.read_csv(uploaded)
        else:
            df_plot = pd.read_csv("data/sample_healthco.csv") if sample=="HealthCo" else pd.read_csv("data/sample_retailco.csv")
        # Try plotting revenue and ebitda by period if period looks sortable
        fig = px.line(df_plot, x="period", y=["revenue","ebitda"], markers=True)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    st.subheader("Recommended Plays")
    for play in data["plays"]:
        with st.container(border=True):
            left, right = st.columns([2,1])
            with left:
                st.markdown(f"### {play['title']}")
                st.caption(play["hypothesis"])
                st.write(f"**Confidence:** {play['confidence'].title()} · **Complexity:** {play['complexity'].title()}")
                # ROI slider
                if play["type"] == "pricing":
                    default_pct = float(evidence["pricing_power"]["recommended_uplift"] or 0.06)
                    pct = st.slider("Assumed price increase", 0.0, 0.12, default_pct, 0.01, key=f"slider_{play['id']}")
                    gm = st.slider("Gross margin", 0.1, 0.7, float(kpis.get('gross_margin',0.35)), 0.01, key=f"gm_{play['id']}")
                    adoption = st.slider("Adoption", 0.1, 1.0, 0.6, 0.05, key=f"adopt_{play['id']}")
                    uplift = kpis["revenue"] * pct * gm * adoption
                elif play["type"] == "retention":
                    default_rr = float(evidence["churn_risk"]["recommended_reduction"] or 0.03)
                    reduce_pp = st.slider("Churn reduction (pp)", 0.00, 0.10, default_rr, 0.005, key=f"slider_{play['id']}")
                    gm = st.slider("Gross margin", 0.1, 0.7, float(kpis.get('gross_margin',0.35)), 0.01, key=f"gm_{play['id']}")
                    uplift = kpis["revenue"] * reduce_pp * gm * 0.5
                elif play["type"] in ["supply","utilization"]:
                    if play["type"]=="supply": default_delta = float(evidence["supply_stress"]["recommended_delta"] or 0.02)
                    else: default_delta = float(evidence["utilization_gap"]["recommended_delta"] or 0.02)
                    delta = st.slider("Impact factor", 0.00, 0.05, default_delta, 0.005, key=f"slider_{play['id']}")
                    gm = st.slider("Gross margin", 0.1, 0.7, float(kpis.get('gross_margin',0.35)), 0.01, key=f"gm_{play['id']}")
                    uplift = kpis["revenue"] * delta * gm
                else:
                    uplift = play["uplift_usd"]
                st.markdown(f"**EBITDA Uplift (est.):** ${uplift:,.0f}")

                # Evidence-based warnings
                if play["type"]=="pricing" and evidence["pricing_power"]["level"]=="low":
                    st.warning("Evidence suggests high price sensitivity. Consider capping uplift at 3%.")
                if play["type"]=="retention" and evidence["churn_risk"]["level"]=="high":
                    st.info("High churn risk detected — prioritize retention campaigns and journey fixes.")
                if play["type"]=="supply" and evidence["supply_stress"]["level"]=="high":
                    st.info("Supply stress is high — prioritize stockout control and lead-time buffers.")
                if play["type"]=="utilization" and evidence["utilization_gap"]["level"]=="high":
                    st.info("Large utilization gap — overbooking rules and reminders likely to pay off.")

                with st.expander("Plan (4 weeks)"):
                    for step in play["plan"]:
                        st.markdown(f"- **Week {step['week']}** — {step['step']} (_Owner: {step['owner_role']}_)")
                with st.expander("Assumptions & Risks"):
                    st.markdown("**Assumptions**")
                    for a in play["assumptions"]:
                        st.markdown(f"- {a}")
                    st.markdown("**Risks**")
                    for r in play["risks"]:
                        st.markdown(f"- {r}")
            with right:
                st.write("**Decision**")
                decision = st.radio("Approve?", ["Pending","Approved","Rejected"], index=0, key=f"decision_{play['id']}")
                rationale = st.text_area("Rationale (optional)", key=f"rationale_{play['id']}", height=80)
                if st.button("Save Decision", key=f"save_{play['id']}"):
                    st.session_state.decisions[play["id"]] = {"status": decision.lower(), "rationale": rationale}
                    st.session_state.activity.append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "action":"decision", "play": play["title"], "status": decision})
                    if use_backend:
                        import requests
                        requests.post(f"{backend_url.rstrip('/')}/decision", json={"play_id": play["id"], "play_title": play["title"], "status": decision.lower(), "rationale": rationale, "actor": "user"})
                    st.success("Saved.")
                st.divider()
                if st.button("Push to Salesforce (mock)", key=f"push_{play['id']}"):
                    st.session_state.activity.append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "action":"push", "play": play["title"], "target":"salesforce", "status":"success"})
                    if use_backend:
                        import requests
                        requests.post(f"{backend_url.rstrip('/')}/integrations/push", json={"play_title": play["title"], "target": "salesforce"})
                    st.success("Integration job queued → success")
    st.divider()

    st.divider()
    st.subheader("Templates (mock admin)")
    if "templates" not in st.session_state:
        st.session_state.templates = {
            "pricing": {"version":"v1.3","last_update":"-", "notes":"Base pricing play"},
            "retention": {"version":"v1.1","last_update":"-", "notes":"Lifecycle offers"},
            "supply": {"version":"v1.0","last_update":"-", "notes":"Stockout control"},
            "utilization": {"version":"v1.0","last_update":"-", "notes":"Scheduling"}
        }
    import pandas as _pd
    tdf = _pd.DataFrame([{**{"name":k}, **v} for k,v in st.session_state.templates.items()])
    st.dataframe(tdf, hide_index=True)
    bump = st.selectbox("Bump version for", list(st.session_state.templates.keys()))
    if st.button("Bump Version"):
        cur = st.session_state.templates[bump]["version"]
        try:
            base, num = cur.split("v")[0], cur.split("v")[1]
            major_minor = num.split(".")
            major = int(major_minor[0]); minor = int(major_minor[1])
            minor += 1
            st.session_state.templates[bump]["version"] = f"v{major}.{minor}"
        except Exception:
            st.session_state.templates[bump]["version"] = "v1.1"
        from time import strftime
        st.session_state.templates[bump]["last_update"] = strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"{bump} → {st.session_state.templates[bump]['version']}")


    st.subheader("Activity Log")
    if len(st.session_state.activity) == 0:
        st.info("No activity yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.activity))

    # Export
    if st.button("Export 1-page PDF (mock HTML)"):
        html = f"""
        <h2>TransformAI Summary — {kpis.get('company')}</h2>
        <p>Period: {kpis.get('period')}</p>
        <p>Revenue: ${kpis.get('revenue'):,.0f} · EBITDA: ${kpis.get('ebitda'):,.0f} · GM: {kpis.get('gross_margin')*100:.1f}%</p>
        <h3>Selected Decisions</h3>
        <ul>
        {''.join([f"<li>{pid}: {d['status']} — {d.get('rationale','')}</li>" for pid,d in st.session_state.decisions.items()])}
        </ul>
        """
        st.download_button("Download HTML", data=html, file_name="transformai_summary.html", mime="text/html")
else:
    st.info("Choose a sample or upload CSV, then click **Analyze Company**.")
