# pages/3_Diligence_Grid_Pro.py
# Competitive mock "intersection" grid: evidence upload -> schema mapping -> quant modules -> approve -> memo -> PDF
# No backend required. Uses session_state only. Ready to swap to HTTP backend later with the same contract.

import io, uuid, math, textwrap, json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

st.set_page_config(page_title="Transform AI — Diligence Grid (Pro)", layout="wide")
st.title("Diligence Grid (Pro) — mock, investor-style UX")
st.caption("Upload CSVs → Map schema (columns) → Add rows/columns → Run modules → Approve → Memo → Export PDF")

# ------------------------------------------------------------------------------------
# In-memory store (per user session)
# ------------------------------------------------------------------------------------
SS = st.session_state
SS.setdefault("tables", {})         # name -> DataFrame
SS.setdefault("schema_map", {})     # table_name -> {"customer","date","revenue","price","quantity"}
SS.setdefault("grid", {             # grid runtime
    "id": f"grid_{uuid.uuid4().hex[:6]}",
    "rows": [],                    # [{id,row_ref,source}]
    "columns": [],                 # [{id,name,tool,params}]
    "cells": [],                   # [{id,row_id,col_id,status,output,...,citations,figure}]
    "activities": []               # audit log
})

# ------------------------------------------------------------------------------------
# Helpers & logging
# ------------------------------------------------------------------------------------
def _log(action:str, detail:str=""):
    SS["grid"]["activities"].append({"id": uuid.uuid4().hex, "action": action, "detail": detail})

def _new_id(prefix): return f"{prefix}_{uuid.uuid4().hex[:8]}"

def _add_row_from_table(table_name: str):
    rid = _new_id("row")
    SS["grid"]["rows"].append({"id": rid, "row_ref": f"table:{table_name}", "source": table_name})
    _log("ROW_ADDED", table_name)
    return rid

def _add_column(name: str, tool: str, params: Optional[Dict[str,Any]] = None):
    cid = _new_id("col")
    SS["grid"]["columns"].append({"id": cid, "name": name, "tool": tool, "params": params or {}})
    _log("COLUMN_ADDED", f"{name} [{tool}]")
    return cid

def _ensure_cells():
    # materialize all row x column cells if missing
    ids = {(c["row_id"], c["col_id"]) for c in SS["grid"]["cells"]}
    for r in SS["grid"]["rows"]:
        for c in SS["grid"]["columns"]:
            key = (r["id"], c["id"])
            if key not in ids:
                SS["grid"]["cells"].append({
                    "id": _new_id("cell"),
                    "row_id": r["id"], "col_id": c["id"],
                    "status": "queued", "output_text": None,
                    "numeric_value": None, "units": None, "citations": [],
                    "confidence": None, "notes": []
                })

def _find(df: pd.DataFrame, col: str) -> Optional[str]:
    # case-insensitive helper to guess a column name
    cols = {c.lower(): c for c in df.columns}
    # exact or with _id suffix
    for k in (col.lower(), f"{col.lower()}_id"):
        if k in cols: return cols[k]
    # partial contains
    needles = {
        "customer": ["customer","user","buyer","account","client"],
        "date": ["date","timestamp","order_date","created_at","period","month"],
        "revenue": ["revenue","amount","net_revenue","sales","gmv","value"],
        "price": ["price","unit_price","avg_price","p"],
        "quantity": ["qty","quantity","units","volume","q"]
    }.get(col.lower(), [col.lower()])
    for n in needles:
        for k in cols:
            if n in k: return cols[k]
    return None

# ------------------------------------------------------------------------------------
# Quant modules (mock = local math; no LLM calls)
# ------------------------------------------------------------------------------------
@dataclass
class ModuleResult:
    kpis: Dict[str, Any]
    narrative: str
    citations: List[Dict[str, Any]]
    figure: Optional[Any] = None

def module_cohort_retention(df: pd.DataFrame,
                            customer_col: Optional[str]=None,
                            ts_col: Optional[str]=None,
                            revenue_col: Optional[str]=None) -> ModuleResult:
    # Guess columns if not provided
    customer_col = customer_col or _find(df, "customer")
    ts_col = ts_col or _find(df, "date")
    revenue_col = revenue_col or _find(df, "revenue")

    if not (isinstance(df, pd.DataFrame) and not df.empty):
        return ModuleResult(kpis={}, narrative="Empty dataset.", citations=[])

    if not (customer_col and ts_col):
        return ModuleResult(
            kpis={}, narrative="Missing customer/date columns; map schema to compute retention.",
            citations=[]
        )

    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce")
    d = d.dropna(subset=[ts_col, customer_col]).sort_values(ts_col)

    # cohort by first order month
    d["first_month"] = d.groupby(customer_col)[ts_col].transform("min").dt.to_period("M")
    d["age"] = (d[ts_col].dt.to_period("M") - d["first_month"]).apply(lambda p: p.n)

    cohort_sizes = d.drop_duplicates([customer_col, "first_month"]).groupby("first_month")[customer_col].count()
    active = d.groupby(["first_month", "age"])[customer_col].nunique()
    mat = (active / cohort_sizes).unstack(fill_value=0).sort_index()

    curve = mat.mean(axis=0) if not mat.empty else pd.Series(dtype=float)
    m3 = float(round(curve.get(3, np.nan), 4)) if not curve.empty else np.nan

    # simple 12m LTV proxy if revenue column exists
    ltv_12 = None
    if revenue_col and revenue_col in d.columns:
        rev = d.groupby([customer_col, d[ts_col].dt.to_period("M")])[revenue_col].sum().groupby(customer_col).sum()
        ltv_12 = float(round(float(rev.mean()), 2))

    fig = px.line(x=list(curve.index), y=list(curve.values),
                  labels={"x": "Months since first purchase", "y": "Retention"},
                  title="Average Retention Curve")

    narrative = f"Retention stabilizes by ~M3 at {m3:.0%}." if (m3==m3) else \
                "Could not compute M3 retention due to missing fields."
    if ltv_12:
        narrative += f" Average 12-month LTV proxy ≈ ${ltv_12:,.2f}."

    citations = [{"type":"table","ref":"(uploaded CSV)","selector":"all_rows"}]
    kpis = {"month_3_retention": m3, "ltv_12m": ltv_12, "cohort_count": int(cohort_sizes.shape[0]) if hasattr(cohort_sizes,'shape') else 0}
    return ModuleResult(kpis=kpis, narrative=narrative, citations=citations, figure=fig)

def module_pricing_power(df: pd.DataFrame,
                         price_col: Optional[str]=None,
                         qty_col: Optional[str]=None) -> ModuleResult:
    price_col = price_col or _find(df, "price")
    qty_col = qty_col or _find(df, "quantity")
    if not (isinstance(df, pd.DataFrame) and not df.empty):
        return ModuleResult(kpis={}, narrative="Empty dataset.", citations=[])
    if not (price_col and qty_col):
        return ModuleResult(kpis={}, narrative="Missing price/quantity columns; map schema to compute elasticity.", citations=[])

    d = df.copy()[[price_col, qty_col]].dropna()
    d = d[(d[price_col] > 0) & (d[qty_col] > 0)]
    if len(d) < 8:
        return ModuleResult(kpis={}, narrative="Not enough observations for elasticity (need ≥ 8 rows).", citations=[])

    # log-log regression for elasticity
    X = np.log(d[price_col].values)
    Y = np.log(d[qty_col].values)
    A = np.vstack([X, np.ones(len(X))]).T
    beta, intercept = np.linalg.lstsq(A, Y, rcond=None)[0]  # Y ≈ beta*X + intercept
    elasticity = float(beta)  # typically negative
    # R^2
    ss_res = float(np.sum((Y - (beta*X + intercept))**2))
    ss_tot = float(np.sum((Y - np.mean(Y))**2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    fig = px.scatter(d, x=price_col, y=qty_col, trendline="ols")
    narrative = f"Estimated own-price elasticity ≈ {elasticity:.2f} (R²={r2:.2f}). " + \
                ("Pricing power appears strong (|ε|<1)." if abs(elasticity) < 1 else "Demand is elastic (|ε|≥1).")
    kpis = {"elasticity": elasticity, "r2": r2}

    return ModuleResult(kpis=kpis, narrative=narrative,
                        citations=[{"type":"table","ref":"(uploaded CSV)","selector":"price/qty columns"}],
                        figure=fig)

MODULES = {
    "cohort_retention": {"title":"Cohort Retention", "fn": module_cohort_retention, "needs": ["customer","date"], "optional": ["revenue"]},
    "pricing_power": {"title":"Pricing Power", "fn": module_pricing_power, "needs": ["price","quantity"], "optional": []},
}

# ------------------------------------------------------------------------------------
# Evidence Sources (CSV Upload)
# ------------------------------------------------------------------------------------
st.subheader("1) Evidence Sources (CSV)")
up = st.file_uploader("Upload CSV(s) (transactions, price/qty, etc.)",
                      type=["csv"], accept_multiple_files=True)
if up:
    for f in up:
        name = f.name
        try:
            SS["tables"][name] = pd.read_csv(f)
            _log("SOURCE_ADDED", name)
            st.success(f"Loaded: {name} — {SS['tables'][name].shape[0]:,} rows")
            # initialize blank mapping
            SS["schema_map"].setdefault(name, {"customer": None, "date": None, "revenue": None, "price": None, "quantity": None})
        except Exception as e:
            st.error(f"{name}: {e}")

# ------------------------------------------------------------------------------------
# CSV Schema Mapper
# ------------------------------------------------------------------------------------
st.subheader("2) Map CSV Schema")

def _auto_guess_map(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "customer": _find(df, "customer"),
        "date": _find(df, "date"),
        "revenue": _find(df, "revenue"),
        "price": _find(df, "price"),
        "quantity": _find(df, "quantity"),
    }

def _apply_mapping_ui(table_name: str, df: pd.DataFrame):
    st.markdown(f"**{table_name}** — {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    cols = list(df.columns)
    cur = SS["schema_map"].get(table_name) or _auto_guess_map(df)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        customer = st.selectbox("Customer ID", options=["(none)"]+cols, index=(cols.index(cur["customer"])+1) if cur.get("customer") in cols else 0)
    with c2:
        date = st.selectbox("Date/Timestamp", options=["(none)"]+cols, index=(cols.index(cur["date"])+1) if cur.get("date") in cols else 0)
    with c3:
        revenue = st.selectbox("Revenue (optional)", options=["(none)"]+cols, index=(cols.index(cur["revenue"])+1) if cur.get("revenue") in cols else 0)
    with c4:
        price = st.selectbox("Price (optional)", options=["(none)"]+cols, index=(cols.index(cur["price"])+1) if cur.get("price") in cols else 0)
    with c5:
        qty = st.selectbox("Quantity (optional)", options=["(none)"]+cols, index=(cols.index(cur["quantity"])+1) if cur.get("quantity") in cols else 0)

    # preview
    prev_cols = [x for x in [customer, date, revenue, price, qty] if x != "(none)"]
    if prev_cols:
        st.dataframe(df[prev_cols].head(8), use_container_width=True)

    # save
    if st.button(f"Save mapping for {table_name}"):
        SS["schema_map"][table_name] = {
            "customer": None if customer=="(none)" else customer,
            "date": None if date=="(none)" else date,
            "revenue": None if revenue=="(none)" else revenue,
            "price": None if price=="(none)" else price,
            "quantity": None if qty=="(none)" else qty,
        }
        _log("SCHEMA_SAVED", table_name)
        st.success("Mapping saved.")

if SS["tables"]:
    # bulk auto-map
    if st.button("Auto-map all tables"):
        for name, df in SS["tables"].items():
            SS["schema_map"][name] = _auto_guess_map(df)
        st.success("Auto-mapped based on header heuristics.")

    # per-table mapping expanders
    for name, df in SS["tables"].items():
        with st.expander(f"Map schema — {name}", expanded=False):
            _apply_mapping_ui(name, df)
else:
    st.info("Upload CSVs above to map schema.")

# ------------------------------------------------------------------------------------
# Create rows/columns
# ------------------------------------------------------------------------------------
st.subheader("3) Define Grid")
c1, c2 = st.columns([1,4])
with c1:
    if st.button("Add rows from all tables"):
        for name in SS["tables"].keys():
            _add_row_from_table(name)
        _ensure_cells()
with c2:
    st.caption("Each uploaded CSV becomes a **row** (e.g., transactions → cohort retention).")

col_name = st.text_input("Column label", value="Cohort retention")
tool_key = st.selectbox("Module", options=list(MODULES.keys()), format_func=lambda k: MODULES[k]["title"])
if st.button("Add Column"):
    _add_column(col_name, tool_key, params={})
    _ensure_cells()
    st.success(f"Added column: {col_name} [{tool_key}]")

with st.expander("Rows & Columns (plan)", expanded=False):
    st.write("Rows:", SS["grid"]["rows"])
    st.write("Columns:", SS["grid"]["columns"])
    st.write("Schema map:", SS["schema_map"])

# ------------------------------------------------------------------------------------
# Run cells (uses schema mapping)
# ------------------------------------------------------------------------------------
st.subheader("4) Run Cells")
sel_rows = st.multiselect("Rows to run", options=[r["id"] for r in SS["grid"]["rows"]],
                          format_func=lambda rid: next((r["row_ref"] for r in SS["grid"]["rows"] if r["id"]==rid), rid))
sel_cols = st.multiselect("Columns to run", options=[c["id"] for c in SS["grid"]["columns"]],
                          format_func=lambda cid: next((c["name"] for c in SS["grid"]["columns"] if c["id"]==cid), cid))

def _run_cell(cell: Dict[str,Any]):
    row = next(x for x in SS["grid"]["rows"] if x["id"]==cell["row_id"])
    col = next(x for x in SS["grid"]["columns"] if x["id"]==cell["col_id"])
    cell["status"] = "running"
    table_name = row["source"]
    df = SS["tables"].get(table_name)
    mapping = SS["schema_map"].get(table_name, {})

    # route to module with mapped columns
    if col["tool"] == "cohort_retention":
        res = MODULES[col["tool"]]["fn"](
            df if isinstance(df, pd.DataFrame) else pd.DataFrame(),
            customer_col=mapping.get("customer"),
            ts_col=mapping.get("date"),
            revenue_col=mapping.get("revenue"),
        )
    elif col["tool"] == "pricing_power":
        res = MODULES[col["tool"]]["fn"](
            df if isinstance(df, pd.DataFrame) else pd.DataFrame(),
            price_col=mapping.get("price"),
            qty_col=mapping.get("quantity"),
        )
    else:
        res = ModuleResult(kpis={}, narrative=f"Unknown tool {col['tool']}", citations=[])

    # persist outputs
    cell["status"] = "done" if res.kpis else "needs_review"
    cell["output_text"] = res.narrative
    cell["numeric_value"] = (list(res.kpis.values())[0] if res.kpis else None)
    cell["units"] = None
    cell["citations"] = res.citations
    cell["figure"] = res.figure
    _log("CELL_RUN", f"{row['row_ref']} × {col['name']} → {cell['status']}")

if st.button("Run selection"):
    _ensure_cells()
    targets = [c for c in SS["grid"]["cells"]
               if (not sel_rows or c["row_id"] in sel_rows) and (not sel_cols or c["col_id"] in sel_cols)]
    for cell in targets:
        _run_cell(cell)
    st.success(f"Ran {len(targets)} cell(s).")

# Table of cells
cells_df = pd.DataFrame(SS["grid"]["cells"])
if not cells_df.empty:
    show = cells_df.drop(columns=[c for c in ["citations","figure","notes","confidence"] if c in cells_df.columns])
    st.dataframe(show, use_container_width=True, height=320)
else:
    st.info("No cells yet. Upload a CSV, map schema, add rows & a column, then Run.")

# ------------------------------------------------------------------------------------
# Review: cell details + Approve/Retry
# ------------------------------------------------------------------------------------
st.subheader("5) Review")
sel_cell_id = st.selectbox("Choose a cell", options=[c["id"] for c in SS["grid"]["cells"]], index=0 if SS["grid"]["cells"] else None)
if sel_cell_id:
    cell = next(c for c in SS["grid"]["cells"] if c["id"]==sel_cell_id)
    col = next(x for x in SS["grid"]["columns"] if x["id"]==cell["col_id"])
    st.markdown(f"**{col['name']}** — status: `{cell['status']}`")
    if cell.get("figure") is not None:
        st.plotly_chart(cell["figure"], use_container_width=True)
    if cell.get("output_text"):
        st.write(cell["output_text"])
    with st.expander("Citations"):
        st.json(cell.get("citations", []))
    a1, a2 = st.columns(2)
    with a1:
        if st.button("Approve"):
            cell["status"] = "approved"
            _log("CELL_APPROVE", sel_cell_id)
            st.success("Approved.")
    with a2:
        if st.button("Mark Needs-Review"):
            cell["status"] = "needs_review"
            _log("CELL_MARK_REVIEW", sel_cell_id)
            st.warning("Marked as needs review.")

# ------------------------------------------------------------------------------------
# Memo composer + Export PDF
# ------------------------------------------------------------------------------------
st.subheader("6) Compose Memo & Export")

def _build_memo() -> str:
    lines = [f"# Investment Memo — {SS['grid']['id']}", ""]
    # Executive Summary (toy)
    approved = [c for c in SS["grid"]["cells"] if c["status"]=="approved"]
    if approved:
        lines.append("## Executive Summary")
        lines.append(f"- Approved findings: {len(approved)}")
        for c in approved[:6]:
            col = next(x for x in SS["grid"]["columns"] if x["id"]==c["col_id"])
            row = next(x for x in SS["grid"]["rows"] if x["id"]==c["row_id"])
            lines.append(f"  - **{col['name']}** on _{row['row_ref']}_ → {c.get('numeric_value') or ''} — {c.get('output_text','')}")
        lines.append("")
    # Evidence
    lines.append("## Evidence Appendix")
    for c in approved:
        lines.append(f"- Cell {c['id']} — citations: {json.dumps(c.get('citations', []))}")
    return "\n".join(lines)

def _memo_pdf_bytes(markdown_text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER
    margin = 0.75*inch
    y = height - margin
    c.setFont("Helvetica-Bold", 14); c.drawString(margin, y, "Transform AI — Investment Memo"); y -= 18
    c.setFont("Helvetica", 9); c.drawString(margin, y, f"Grid: {SS['grid']['id']}"); y -= 14
    c.setFont("Helvetica", 11)
    for line in markdown_text.splitlines():
        for seg in textwrap.wrap(line, width=95) or [" "]:
            if y < margin: c.showPage(); y = height - margin; c.setFont("Helvetica", 11)
            c.drawString(margin, y, seg); y -= 14
    c.showPage(); c.save(); buf.seek(0)
    return buf.getvalue()

if st.button("Compose Memo"):
    memo_md = _build_memo()
    st.code(memo_md, language="markdown")
    SS["last_memo_md"] = memo_md

memo_md = SS.get("last_memo_md")
if memo_md:
    pdf_bytes = _memo_pdf_bytes(memo_md)
    st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name=f"TransformAI_Memo_{SS['grid']['id']}.pdf", mime="application/pdf")

# ------------------------------------------------------------------------------------
# Audit log
# ------------------------------------------------------------------------------------
with st.expander("Activity Log", expanded=False):
    st.json(SS["grid"]["activities"])

