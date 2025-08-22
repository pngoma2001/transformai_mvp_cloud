import os
import io
import math
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Page config & CSS (wider, modern, no cropped header)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Transform AI — Diligence Grid (Pro)", layout="wide")
st.markdown(
    """
    <style>
      .block-container {max-width: 1500px; padding-top: 1rem;}
      h1, h2, h3 { letter-spacing: 0.2px; }
      .metric-badge {font-size:12px; color:#a3a3a3;}
      .sticky-actions { position: sticky; top: 0; background: rgba(17,17,23,.9); z-index: 20; padding: 8px 0 4px 0; border-bottom: 1px solid rgba(255,255,255,.06); }
      .muted { color: #a1a1aa; font-size: 13px; }
      .good { color: #22c55e; }
      .warn { color: #f59e0b; }
      .bad  { color: #ef4444; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"

def _read_csv_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _month_floor(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.to_period("M").dt.to_timestamp()

# -----------------------------------------------------------------------------
# Sample & fallback data
# -----------------------------------------------------------------------------
def load_or_synthesize_data() -> Dict[str, pd.DataFrame]:
    _ensure_dir("data")
    data: Dict[str, pd.DataFrame] = {}

    # Transactions (core)
    tx_path = "data/sample_transactions.csv"
    if os.path.exists(tx_path):
        df_tx = pd.read_csv(tx_path)
    else:
        # Synthetic tiny dataset
        rng = pd.date_range("2024-01-01", periods=180, freq="D")
        customers = [f"C{i:03d}" for i in range(60)]
        np.random.seed(0)
        df_tx = pd.DataFrame({
            "customer_id": np.random.choice(customers, size=len(rng)),
            "order_date": np.random.choice(rng, size=len(rng)),
            "net_revenue": np.random.gamma(2.2, 60, size=len(rng)).round(2),
            "price": np.random.uniform(12, 36, size=len(rng)).round(2),
            "quantity": np.random.randint(1, 6, size=len(rng)),
        })
        df_tx.to_csv(tx_path, index=False)
    data["transactions"] = df_tx

    # QoE P&L monthly (optional)
    qoe_path = "data/sample_qoe_pnl_monthly.csv"
    if os.path.exists(qoe_path):
        data["qoe"] = _read_csv_safe(qoe_path)
    else:
        months = pd.period_range("2024-01", "2024-06", freq="M").astype(str)
        df_qoe = pd.DataFrame({
            "month": months,
            "revenue": [120000, 125000, 132000, 128000, 135000, 142000],
            "cogs":    [42000,  44000,  47000,  46000,  48000,  50000],
            "sga":     [26000,  26000,  27000,  26500,  27500,  28000],
            "marketing":[11000, 12500, 12000, 11500, 11800, 12000],
            "orders":  [2600,   2700,   2850,   2800,   2900,   3050],
            "customers":[1900,  1950,   2000,   2025,   2075,   2125]
        })
        df_qoe.to_csv(qoe_path, index=False)
        data["qoe"] = df_qoe

    # NRR/GRR monthly (optional)
    nrr_path = "data/sample_nrr_grr.csv"
    if os.path.exists(nrr_path):
        data["nrrgrr"] = _read_csv_safe(nrr_path)
    else:
        months = pd.period_range("2024-01", "2024-11", freq="M").astype(str)
        df_nrr = pd.DataFrame({
            "month": months,
            "GRR":  np.clip(0.95 + 0.03*np.sin(np.arange(len(months))/2), 0.90, 1.00).round(4),
            "NRR":  np.clip(1.00 + 0.02*np.cos(np.arange(len(months))/3), 0.95, 1.07).round(4),
        })
        data["nrrgrr"] = df_nrr

    return data

DATA = load_or_synthesize_data()

# -----------------------------------------------------------------------------
# Modules (analysis engines)
# -----------------------------------------------------------------------------
def mod_pdf_kpis_stub(_: Any) -> Dict[str, Any]:
    # Placeholder: surface investor-friendly summary.
    return {
        "value": None,
        "summary": "Revenue ≈ $12.5M; EBITDA ≈ $3.2M; Gross margin ≈ 64%; Churn ≈ 4%. (Stub)",
        "artifacts": {},
    }

def mod_cohort_retention(transactions: pd.DataFrame) -> Dict[str, Any]:
    df = transactions.copy()
    must = {"customer_id", "order_date", "net_revenue"}
    if not must.issubset(df.columns):
        return {"value": None, "summary": "Map schema: need customer_id, order_date, net_revenue.", "artifacts": {}}

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["month"] = _month_floor(df["order_date"])
    first = df.groupby("customer_id")["month"].min().rename("cohort")
    df = df.join(first, on="customer_id")

    # cohort sizes
    cohort_sizes = df.groupby("cohort")["customer_id"].nunique()
    # cohort periods
    df["period"] = ((df["month"] - df["cohort"]) / np.timedelta64(1, "M")).round().astype(int)

    # retention: purchasers per cohort-period / cohort size
    cohorts = df.groupby(["cohort", "period"])["customer_id"].nunique().unstack(fill_value=0)
    curves = (cohorts.T / cohort_sizes.values).T.fillna(0)

    # compute a simple aggregate curve by averaging cohorts
    agg_curve = curves.mean(axis=0).round(3).tolist()
    m3 = None
    if len(agg_curve) > 3:
        m3 = float(agg_curve[3])  # month 3
    summary = f"Retention stabilizes around M3 ≈ {m3:.0%}." if m3 is not None else "Retention curve computed."

    return {
        "value": m3,
        "summary": summary,
        "artifacts": {
            "agg_curve": agg_curve,
            "curves": curves.reset_index().to_dict(orient="list"),
        },
    }

def mod_nrr_grr(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    # Prefer explicit NRR/GRR file, else derive an approximation from transactions
    if "nrrgrr" in dataframes and not dataframes["nrrgrr"].empty:
        df = dataframes["nrrgrr"].copy()
        return {
            "value": float(df["NRR"].tail(1).values[0]),
            "summary": f"Latest (NRR, GRR) ≈ ({df['NRR'].tail(1).values[0]:.0%}, {df['GRR'].tail(1).values[0]:.0%}).",
            "artifacts": {"nrr": df["NRR"].tolist(), "grr": df["GRR"].tolist(), "month": df["month"].tolist()},
        }

    # Derive from transactions as revenue-retention proxy
    df_tx = DATA["transactions"].copy()
    df_tx["month"] = _month_floor(df_tx["order_date"])
    mrev = df_tx.groupby("month")["net_revenue"].sum().reset_index()
    mrev["NRR"] = (mrev["net_revenue"] / mrev["net_revenue"].shift(1)).fillna(1.0)
    mrev["GRR"] = np.clip(0.97 + 0.01*np.cos(np.arange(len(mrev))/2), 0.90, 1.00)
    return {
        "value": float(mrev["NRR"].tail(1).values[0]),
        "summary": f"Latest derived NRR ≈ {mrev['NRR'].tail(1).values[0]:.0%}.",
        "artifacts": {"nrr": mrev["NRR"].round(3).tolist(), "grr": mrev["GRR"].round(3).tolist(),
                      "month": mrev["month"].dt.strftime("%Y-%m").tolist()},
    }

def mod_pricing_power(transactions: pd.DataFrame) -> Dict[str, Any]:
    df = transactions.copy()
    must = {"price", "quantity"}
    if not must.issubset(df.columns):
        return {"value": None, "summary": "Map schema: need price, quantity.", "artifacts": {}}
    # guard
    df = df[(df["price"] > 0) & (df["quantity"] > 0)].copy()
    if len(df) < 6:
        return {"value": None, "summary": "Not enough points for elasticity.", "artifacts": {}}

    x = np.log(df["price"].values)
    y = np.log(df["quantity"].values)
    slope, intercept = np.polyfit(x, y, 1)

    klass = "inelastic" if abs(slope) < 1 else "elastic"
    summary = f"Own-price elasticity ≈ {slope:.2f} → {klass}."

    return {
        "value": float(slope),
        "summary": summary,
        "artifacts": {
            "points": {"log_price": x.tolist(), "log_qty": y.tolist()},
            "fit": {"slope": float(slope), "intercept": float(intercept)},
        },
    }

def mod_unit_econ(qoe: pd.DataFrame) -> Dict[str, Any]:
    df = qoe.copy()
    must = {"month", "revenue", "cogs", "sga", "marketing", "orders", "customers"}
    if not must.issubset(df.columns):
        return {"value": None, "summary": "QoE monthly P&L requires month, revenue, cogs, sga, marketing, orders, customers.", "artifacts": {}}

    df["GM"] = df["revenue"] - df["cogs"]
    df["GM_pct"] = (df["GM"] / df["revenue"]).fillna(0)
    df["AOV"] = (df["revenue"] / df["orders"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    # CAC approx monthly = marketing / new_customers (crude; here treat customers as active)
    df["CAC"] = (df["marketing"] / (df["customers"].diff().clip(lower=1).fillna(1))).clip(0)
    df["CM"] = df["GM"] - df["sga"] - df["marketing"]

    aov = float(df["AOV"].mean())
    gm_pct = float(df["GM_pct"].mean())
    cac = float(df["CAC"].mean())
    cm = float(df["CM"].mean())

    summary = f"AOV ≈ ${aov:,.0f}, GM ≈ {gm_pct:.0%}, CAC ≈ ${cac:,.0f}, CM ≈ ${cm:,.0f}."

    return {
        "value": aov,
        "summary": summary,
        "artifacts": df.to_dict(orient="list"),
    }

# -----------------------------------------------------------------------------
# Session state (rows, columns, mapping, cell results)
# -----------------------------------------------------------------------------
if "rows" not in st.session_state:
    st.session_state.rows = pd.DataFrame([
        {"id": _uid("row"), "alias": "sample_transactions (table)", "row_type": "table", "source": "sample_transactions.csv", "delete": False},
        {"id": _uid("row"), "alias": "sample_qoe_pnl_monthly (qoe)", "row_type": "table", "source": "sample_qoe_pnl_monthly.csv", "delete": False},
        {"id": _uid("row"), "alias": "sample_nrr_grr (table)", "row_type": "table", "source": "sample_nrr_grr.csv", "delete": False},
        {"id": _uid("row"), "alias": "Sample_KPI_Pack (pdf)", "row_type": "pdf",   "source": "Sample_KPI_Pack.pdf", "delete": False},
    ])

if "cols" not in st.session_state:
    st.session_state.cols = pd.DataFrame([
        {"id": _uid("col"), "label": "PDF KPIs",         "module": "PDF KPIs (PDF)",          "delete": False},
        {"id": _uid("col"), "label": "Cohort Retention", "module": "Cohort Retention (CSV)",  "delete": False},
        {"id": _uid("col"), "label": "NRR/GRR",          "module": "NRR/GRR (CSV)",           "delete": False},
        {"id": _uid("col"), "label": "Pricing Power",    "module": "Pricing Power (CSV)",     "delete": False},
        {"id": _uid("col"), "label": "Unit Economics",   "module": "Unit Economics (CSV)",    "delete": False},
    ])

if "matrix" not in st.session_state:
    # all False initially
    rows = st.session_state.rows["id"].tolist()
    cols = st.session_state.cols["id"].tolist()
    st.session_state.matrix = pd.DataFrame(
        {"row_id": np.repeat(rows, len(cols)),
         "col_id": cols * len(rows),
         "run": False}
    )

if "cells" not in st.session_state:
    # results per (row_id, col_id)
    st.session_state.cells: Dict[str, Dict[str, Any]] = {}

# -----------------------------------------------------------------------------
# Mapping helpers
# -----------------------------------------------------------------------------
MODULES = [
    "PDF KPIs (PDF)",
    "Cohort Retention (CSV)",
    "NRR/GRR (CSV)",
    "Pricing Power (CSV)",
    "Unit Economics (CSV)",
]

def _matrix_key(row_id: str, col_id: str) -> str:
    return f"{row_id}::{col_id}"

def _cell_status(row_id: str, col_id: str) -> str:
    key = _matrix_key(row_id, col_id)
    if key in st.session_state.cells:
        return st.session_state.cells[key].get("status", "cached")
    return "—"

# -----------------------------------------------------------------------------
# Title
# -----------------------------------------------------------------------------
st.markdown("<div class='sticky-actions'></div>", unsafe_allow_html=True)
st.title("Transform AI — Diligence Grid (Pro)")
st.caption("Rows = evidence (CSV/PDF). Columns = modules (analyses). Cells = runs. This is your *agentic spreadsheet* for diligence.")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_grid, tab_run, tab_sheet, tab_visuals = st.tabs(["Grid", "Run", "Sheet", "Visuals"])

# -----------------------------------------------------------------------------
# GRID TAB — rows/columns + matrix mapping
# -----------------------------------------------------------------------------
with tab_grid:
    st.subheader("1) Build the grid")
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 1.0])

    with c1:
        st.markdown("**Rows (evidence)**")
        rows_edit = st.data_editor(
            st.session_state.rows,
            hide_index=True,
            use_container_width=True,
            column_config={
                "id": st.column_config.TextColumn("id", disabled=True),
                "alias": st.column_config.TextColumn("Alias"),
                "row_type": st.column_config.SelectboxColumn("Type", options=["table", "pdf"]),
                "source": st.column_config.TextColumn("Source"),
                "delete": st.column_config.CheckboxColumn("Delete"),
            },
            num_rows="dynamic",
            key="rows_editor",
        )
        if st.button("Apply row edits", type="secondary"):
            st.session_state.rows = rows_edit[~rows_edit["delete"]].drop(columns=["delete"], errors="ignore")
            # refresh matrix for new rows
            rows = st.session_state.rows["id"].tolist()
            cols = st.session_state.cols["id"].tolist()
            st.session_state.matrix = pd.DataFrame(
                {"row_id": np.repeat(rows, len(cols)),
                 "col_id": cols * len(rows),
                 "run": False}
            )
            st.success("Rows updated.")

    with c2:
        st.markdown("**Columns (modules)**")
        cols_edit = st.data_editor(
            st.session_state.cols,
            hide_index=True,
            use_container_width=True,
            column_config={
                "id": st.column_config.TextColumn("id", disabled=True),
                "label": st.column_config.TextColumn("Label"),
                "module": st.column_config.SelectboxColumn("Module", options=MODULES),
                "delete": st.column_config.CheckboxColumn("Delete"),
            },
            num_rows="dynamic",
            key="cols_editor",
        )
        if st.button("Apply column edits", type="secondary"):
            st.session_state.cols = cols_edit[~cols_edit["delete"]].drop(columns=["delete"], errors="ignore")
            # refresh matrix for new cols
            rows = st.session_state.rows["id"].tolist()
            cols = st.session_state.cols["id"].tolist()
            st.session_state.matrix = pd.DataFrame(
                {"row_id": np.repeat(rows, len(cols)),
                 "col_id": cols * len(rows),
                 "run": False}
            )
            st.success("Columns updated.")

    with c3:
        st.markdown("**Quick add**")
        if st.button("Add QoE Columns", help="Adds Unit Economics column if missing"):
            if not (st.session_state.cols["module"] == "Unit Economics (CSV)").any():
                st.session_state.cols.loc[len(st.session_state.cols)] = {
                    "id": _uid("col"),
                    "label": "Unit Economics",
                    "module": "Unit Economics (CSV)",
                    "delete": False
                }
                st.experimental_rerun()
        st.markdown("<div class='muted'>Use the editors to add/delete. Changes require *Apply*.</div>", unsafe_allow_html=True)

    with c4:
        st.markdown("**Hints**")
        st.markdown("- **tables** → CSV-based modules (Retention, NRR/GRR, Pricing, Unit Econ)\n- **pdf** → PDF KPIs")
        st.markdown("- Use **Apply** after edits to refresh mapping.")

    st.divider()

    # MATRIX
    st.markdown("**2) Map rows ↔ modules (checkbox means this cell runs)**")

    # Construct a pivot-like editor
    rows = st.session_state.rows.copy()
    cols = st.session_state.cols.copy()
    mat = st.session_state.matrix.copy()

    # Build a wide grid for visual mapping
    matrix_wide = pd.DataFrame({"row_id": rows["id"], "Alias": rows["alias"]}).copy()
    for _, col in cols.iterrows():
        col_id = col["id"]
        label = col["label"]
        # bring existing run state
        run_series = mat[mat["col_id"] == col_id].set_index("row_id")["run"]
        matrix_wide[label] = matrix_wide["row_id"].map(run_series).fillna(False).astype(bool)

    matrix_edit = st.data_editor(
        matrix_wide.drop(columns=["row_id"]),
        hide_index=True,
        use_container_width=True,
        key="matrix_editor",
        column_config={label: st.column_config.CheckboxColumn(label) for label in cols["label"]},
    )

    if st.button("Apply matrix selection", type="primary"):
        # convert matrix_edit back to st.session_state.matrix
        # align on Alias -> row_id
        alias_to_row = dict(zip(rows["alias"], rows["id"]))
        updates = []
        for _, row in matrix_edit.iterrows():
            row_id = alias_to_row.get(row["Alias"])
            if row_id is None:
                continue
            for _, col in cols.iterrows():
                col_id = col["id"]
                label = col["label"]
                run = bool(row.get(label, False))
                updates.append({"row_id": row_id, "col_id": col_id, "run": run})
        st.session_state.matrix = pd.DataFrame(updates)
        st.success("Matrix updated.")

# -----------------------------------------------------------------------------
# RUN TAB — run selection and see results
# -----------------------------------------------------------------------------
with tab_run:
    st.subheader("Run cells")

    rows_df = st.session_state.rows.copy()
    cols_df = st.session_state.cols.copy()
    mat = st.session_state.matrix.copy()

    # default selection = all currently checked in matrix
    available_rows = rows_df["alias"].tolist()
    available_cols = cols_df["label"].tolist()

    # Build default lists from matrix
    default_row_aliases = rows_df[rows_df["id"].isin(mat[mat["run"]]["row_id"])].alias.tolist()
    default_col_labels  = cols_df[cols_df["id"].isin(mat[mat["run"]]["col_id"])].label.tolist()

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        sel_rows = st.multiselect("Rows to run", options=available_rows, default=default_row_aliases, key="run_rows")
    with c2:
        sel_cols = st.multiselect("Columns to run", options=available_cols, default=default_col_labels, key="run_cols")
    with c3:
        force = st.toggle("Force re-run", value=False, help="Ignore cached cell results")

    def _resolve_row(alias: str) -> Dict[str, Any]:
        r = rows_df[rows_df["alias"] == alias].iloc[0].to_dict()
        # Attach data pointer
        src = r.get("source", "")
        if r["row_type"] == "table":
            path = os.path.join("data", src)
            df = _read_csv_safe(path)
            r["df"] = df
        else:
            r["df"] = pd.DataFrame()
        return r

    def _run_cell(row_rec: Dict[str, Any], col_rec: Dict[str, Any]) -> Dict[str, Any]:
        module = col_rec["module"]
        key = _matrix_key(row_rec["id"], col_rec["id"])

        if (not force) and (key in st.session_state.cells):
            return {**st.session_state.cells[key], "status": "cached"}

        # Dispatch to modules
        if module == "PDF KPIs (PDF)":
            out = mod_pdf_kpis_stub(None)
        elif module == "Cohort Retention (CSV)":
            out = mod_cohort_retention(row_rec["df"])
        elif module == "NRR/GRR (CSV)":
            out = mod_nrr_grr(DATA)
        elif module == "Pricing Power (CSV)":
            out = mod_pricing_power(row_rec["df"])
        elif module == "Unit Economics (CSV)":
            out = mod_unit_econ(DATA["qoe"])
        else:
            out = {"value": None, "summary": "Unknown module.", "artifacts": {}}

        out = {"status": "done", **out}
        st.session_state.cells[key] = out
        return out

    if st.button("Run selection", type="primary"):
        if not sel_rows or not sel_cols:
            st.warning("Pick at least one row and one column.")
        else:
            for r_alias in sel_rows:
                r = _resolve_row(r_alias)
                for c_label in sel_cols:
                    c = cols_df[cols_df["label"] == c_label].iloc[0].to_dict()
                    _ = _run_cell(r, c)
            st.success("Run complete.")

    st.divider()
    st.markdown("**Results (investor view)**")

    # Build results table
    rows_df = st.session_state.rows.copy()
    cols_df = st.session_state.cols.copy()
    res_rows = []
    for _, r in rows_df.iterrows():
        for _, c in cols_df.iterrows():
            key = _matrix_key(r["id"], c["id"])
            cell = st.session_state.cells.get(key)
            res_rows.append({
                "Row": r["alias"],
                "Column": c["label"],
                "Module": c["module"],
                "Status": cell["status"] if cell else "—",
                "Value": cell.get("value") if cell else None,
                "Summary": cell.get("summary") if cell else None,
            })
    res_df = pd.DataFrame(res_rows)
    st.dataframe(res_df, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# SHEET TAB — agentic spreadsheet (status by cell)
# -----------------------------------------------------------------------------
with tab_sheet:
    st.subheader("Agentic Spreadsheet (status by cell)")

    rows_df = st.session_state.rows.copy()
    cols_df = st.session_state.cols.copy()

    sheet = pd.DataFrame({"Row": rows_df["alias"]})
    for _, c in cols_df.iterrows():
        label = c["label"]
        col_vals = []
        for _, r in rows_df.iterrows():
            key = _matrix_key(r["id"], c["id"])
            status = _cell_status(r["id"], c["id"])
            col_vals.append(status)
        sheet[label] = col_vals
    st.dataframe(sheet, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# VISUALS TAB — charts that aren't crammed
# -----------------------------------------------------------------------------
with tab_visuals:
    st.subheader("Visualizations")
    # Pull artifacts if exist, otherwise compute lightweight defaults from DATA
    # Retention
    ret_curve = None
    for key, cell in st.session_state.cells.items():
        if "agg_curve" in cell.get("artifacts", {}):
            ret_curve = cell["artifacts"]["agg_curve"]
            break
    if ret_curve is None:
        # quick on-the-fly
        ret_curve = mod_cohort_retention(DATA["transactions"])["artifacts"].get("agg_curve", [1, 0.9, 0.75, 0.65, 0.6])

    # NRR/GRR
    nrr_art = None
    for _, cell in st.session_state.cells.items():
        art = cell.get("artifacts", {})
        if {"nrr", "grr", "month"}.issubset(art.keys()):
            nrr_art = art
            break
    if nrr_art is None:
        nrr_art = mod_nrr_grr(DATA)["artifacts"]

    # Pricing points
    pricing_art = None
    for _, cell in st.session_state.cells.items():
        art = cell.get("artifacts", {})
        if "points" in art:
            pricing_art = art
            break
    if pricing_art is None:
        tmp = mod_pricing_power(DATA["transactions"])
        pricing_art = tmp.get("artifacts", {"points": {"log_price": [], "log_qty": []}})

    # Layout charts in 2x2 clean grid
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("**Retention curve (agg)**")
        df_ret = pd.DataFrame({"period": list(range(len(ret_curve))), "retention": ret_curve})
        fig = px.line(df_ret, x="period", y="retention", markers=True, template="plotly_dark")
        fig.update_layout(height=280, margin=dict(l=10,r=10,t=30,b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.markdown("**NRR / GRR**")
        df_nrr = pd.DataFrame({"month": nrr_art["month"], "NRR": nrr_art["nrr"], "GRR": nrr_art["grr"]})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_nrr["month"], y=df_nrr["NRR"], mode="lines+markers", name="NRR"))
        fig.add_trace(go.Scatter(x=df_nrr["month"], y=df_nrr["GRR"], mode="lines+markers", name="GRR"))
        fig.update_layout(height=280, template="plotly_dark", margin=dict(l=10,r=10,t=30,b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("**Pricing power: log(Q) vs log(P)**")
        pts = pricing_art.get("points", {"log_price": [], "log_qty": []})
        df_pp = pd.DataFrame({"log_price": pts.get("log_price", []), "log_qty": pts.get("log_qty", [])})
        if len(df_pp) > 0:
            fig = px.scatter(df_pp, x="log_price", y="log_qty", trendline="ols", template="plotly_dark")
            fig.update_layout(height=280, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run Pricing Power on a row to see points.")

    with r2c2:
        st.markdown("**Cohort heatmap (demo)**")
        # If detailed curves exist we could render a true heatmap; show a demo gradient otherwise
        if "curves" in (st.session_state.cells[next(iter(st.session_state.cells))]["artifacts"] if st.session_state.cells else {}):
            # For simplicity, keep a placeholder; full cohort heatmap requires rebuilding from stored curves
            pass
        # simple block gradient placeholder
        z = np.array([[0.85, 0.8, 0.74, 0.68, 0.63],
                      [0.88, 0.82, 0.75, 0.69, 0.64],
                      [0.86, 0.81, 0.73, 0.66, 0.62]])
        fig = px.imshow(z, color_continuous_scale="Blues", aspect="auto", origin="lower")
        fig.update_layout(height=280, template="plotly_dark", margin=dict(l=10,r=10,t=30,b=10),
                          coloraxis_colorbar=dict(tickformat=".0%"))
        st.plotly_chart(fig, use_container_width=True)
