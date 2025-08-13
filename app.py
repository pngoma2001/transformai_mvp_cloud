
import streamlit as st
import pandas as pd
import json, time, io
from engine import analyze
import plotly.express as px

st.set_page_config(page_title="TransformAI MVP", layout="wide")

st.sidebar.title("TransformAI")
st.sidebar.caption("PE portfolio transformation demo — Streamlit Cloud safe build")

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
sample = st.selectbox("Choose a sample company", ["RetailCo", "HealthCo"])
uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

if st.button("Analyze Company", type="primary"):
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        if sample == "RetailCo":
            df = pd.read_csv("data/sample_retailco.csv")
        else:
            df = pd.read_csv("data/sample_healthco.csv")
    with st.spinner("Analyzing…"):
        time.sleep(0.6)
        if use_backend:
            import requests
            files = {"file": df.to_csv(index=False).encode("utf-8")}
            r = requests.post(f"{backend_url.rstrip('/')}/analyze", files={"file": ('data.csv', df.to_csv(index=False), 'text/csv')})
            resp = r.json()
            if resp.get("ok"):
                st.session_state.last_result = resp["result"]
            else:
                st.error(resp.get("error","Backend error"))
        else:
            result = analyze(df)
            st.session_state.last_result = result  # plain dict

if "last_result" in st.session_state:
    data = st.session_state.last_result
    kpis = data["kpis"]
    st.subheader(f"Summary — {kpis.get('company')} ({kpis.get('period')})")

    # KPI tiles
    col1, col2, col3, col4, col5 = st.columns(5)
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
                    pct = st.slider("Assumed price increase", 0.0, 0.12, 0.06, 0.01, key=f"slider_{play['id']}")
                    gm = st.slider("Gross margin", 0.1, 0.7, float(kpis.get('gross_margin',0.35)), 0.01, key=f"gm_{play['id']}")
                    adoption = st.slider("Adoption", 0.1, 1.0, 0.6, 0.05, key=f"adopt_{play['id']}")
                    uplift = kpis["revenue"] * pct * gm * adoption
                elif play["type"] == "retention":
                    reduce_pp = st.slider("Churn reduction (pp)", 0.00, 0.10, 0.03, 0.005, key=f"slider_{play['id']}")
                    gm = st.slider("Gross margin", 0.1, 0.7, float(kpis.get('gross_margin',0.35)), 0.01, key=f"gm_{play['id']}")
                    uplift = kpis["revenue"] * reduce_pp * gm * 0.5
                elif play["type"] in ["supply","utilization"]:
                    delta = st.slider("Impact factor", 0.00, 0.05, 0.02, 0.005, key=f"slider_{play['id']}")
                    gm = st.slider("Gross margin", 0.1, 0.7, float(kpis.get('gross_margin',0.35)), 0.01, key=f"gm_{play['id']}")
                    uplift = kpis["revenue"] * delta * gm
                else:
                    uplift = play["uplift_usd"]
                st.markdown(f"**EBITDA Uplift (est.):** ${uplift:,.0f}")
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
