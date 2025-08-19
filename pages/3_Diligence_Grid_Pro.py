# pages/3_Diligence_Grid_Pro.py
# TransformAI — Diligence Grid (Pro)
# One-page demo: CSV+PDF evidence → schema mapping → run modules → approve → memo → export → chat
# No backend required. Safe to later swap modules to FastAPI endpoints.

from __future__ import annotations
import io, uuid, re, textwrap, json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# --- Optional PDF support (install pypdf to enable) ---
try:
    from pypdf import PdfReader  # pip install pypdf
except Exception:
    PdfReader = None

# --------------------- PAGE SETUP ---------------------
st.set_page_config(page_title="TransformAI — Diligence (Pro)", layout="wide")
st.title("Transform AI — Diligence Grid (Pro)")
st.caption("Upload CSV / PDF → map schema → run modules → approve → compose memo → export PDF — all locally (mock).")

SS = st.session_state
SS.setdefault("tables", {})          # CSV name -> DataFrame
SS.setdefault("docs", {})            # PDF name -> list[page_text]
SS.setdefault("schema_map", {})      # CSV schema mapping
SS.setdefault("chat_history", [])    # simple chat transcript
SS.setdefault("grid", {
    "id": f"grid_{uuid.uuid4().hex[:6]}",
    "rows": [],                      # [{id,row_ref,source,type}]
    "columns": [],                   # [{id,name,tool,params}]
    "cells": [],                     # [{...}]
    "activities": []                 # audit log
})

# --------------------- UTILS ---------------------
def _log(action:str, detail:str=""):
    SS["grid"]["activities"].append({"id": uuid.uuid4().hex, "action": action, "detail": detail})

def _new_id(p): return f"{p}_{uuid.uuid4().hex[:8]}"

def _add_row_from_table(name:str):
    rid = _new_id("row")
    SS["grid"]["rows"].append({"id": rid, "row_ref": f"table:{name}", "source": name, "type": "table"})
    _log("ROW_ADDED", f"table:{name}")
    return rid

def _add_row_from_pdf(name:str):
    rid = _new_id("row")
    SS["grid"]["rows"].append({"id": rid, "row_ref": f"pdf:{name}", "source": name, "type": "pdf"})
    _log("ROW_ADDED", f"pdf:{name}")
    return rid

def _add_column(name:str, tool:str, params:Optional[Dict[str,Any]]=None):
    cid = _new_id("col")
    SS["grid"]["columns"].append({"id": cid, "name": name, "tool": tool, "params": params or {}})
    _log("COLUMN_ADDED", f"{name} [{tool}]")
    return cid

def _ensure_cells():
    have = {(c["row_id"], c["col_id"]) for c in SS["grid"]["cells"]}
    for r in SS["grid"]["rows"]:
        for c in SS["grid"]["columns"]:
            k = (r["id"], c["id"])
            if k not in have:
                SS["grid"]["cells"].append({
                    "id": _new_id("cell"),
                    "row_id": r["id"], "col_id": c["id"],
                    "status": "queued", "output_text": None,
                    "numeric_value": None, "units": None,
                    "citations": [], "confidence": None, "notes": [], "figure": None
                })

def _find(df: pd.DataFrame, key: str) -> Optional[str]:
    # robust-ish header guesser
    candidates = {
        "customer": ["customer","user","buyer","account","client","cust","cust_id"],
        "date":     ["date","timestamp","order_date","created_at","period","month"],
        "revenue":  ["revenue","amount","net_revenue","sales","gmv","value"],
        "price":    ["price","unit_price","avg_price","p"],
        "quantity": ["qty","quantity","units","volume","q"]
    }.get(key.lower(), [key.lower()])
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for needle in candidates:
        if needle in lower: return lower[needle]
    for needle in candidates:
        for c in cols:
            if needle in c.lower(): return c
    return None

# --------------------- MODULES ---------------------
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
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ModuleResult({}, "Empty dataset.", [])
    customer_col = customer_col or _find(df, "customer")
    ts_col = ts_col or _find(df, "date")
    revenue_col = revenue_col or _find(df, "revenue")
    if not (customer_col and ts_col):
        return ModuleResult({}, "Missing customer/date columns; map schema to compute retention.", [])
    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce")
    d = d.dropna(subset=[ts_col, customer_col]).sort_values(ts_col)
    d["first_month"] = d.groupby(customer_col)[ts_col].transform("min").dt.to_period("M")
    d["age"] = (d[ts_col].dt.to_period("M") - d["first_month"]).apply(lambda p: p.n)
    cohort_sizes = d.drop_duplicates([customer_col, "first_month"]).groupby("first_month")[customer_col].count()
    active = d.groupby(["first_month", "age"])[customer_col].nunique()
    mat = (active / cohort_sizes).unstack(fill_value=0).sort_index()
    curve = mat.mean(axis=0) if not mat.empty else pd.Series(dtype=float)
    m3 = float(round(curve.get(3, np.nan), 4)) if not curve.empty else np.nan
    ltv_12 = None
    if revenue_col and revenue_col in d.columns:
        rev = d.groupby([customer_col, d[ts_col].dt.to_period("M")])[revenue_col].sum().groupby(customer_col).sum()
        ltv_12 = float(round(float(rev.mean()), 2))
    fig = px.line(x=list(curve.index), y=list(curve.values),
                  labels={"x":"Months since first purchase","y":"Retention"},
                  title="Average Retention Curve")
    narrative = f"Retention stabilizes ~M3 at {m3:.0%}." if m3==m3 else "Not enough data to compute M3 retention."
    if ltv_12: narrative += f" Avg 12-month LTV proxy ≈ ${ltv_12:,.2f}."
    return ModuleResult(
        kpis={"month_3_retention": m3, "ltv_12m": ltv_12},
        narrative=narrative,
        citations=[{"type":"table","ref":"(uploaded CSV)","selector":"all_rows"}],
        figure=fig
    )

def module_pricing_power(df: pd.DataFrame,
                         price_col: Optional[str]=None,
                         qty_col: Optional[str]=None) -> ModuleResult:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ModuleResult({}, "Empty dataset.", [])
    price_col = price_col or _find(df, "price")
    qty_col   = qty_col   or _find(df, "quantity")
    if not (price_col and qty_col):
        return ModuleResult({}, "Missing price/quantity columns; map schema first.", [])
    d = df[[price_col, qty_col]].dropna()
    d = d[(d[price_col] > 0) & (d[qty_col] > 0)]
    if len(d) < 8:
        return ModuleResult({}, "Need ≥ 8 observations for elasticity regression.", [])
    # log-log regression
    X = np.log(d[price_col].values)
    Y = np.log(d[qty_col].values)
    A = np.vstack([X, np.ones(len(X))]).T
    beta, intercept = np.linalg.lstsq(A, Y, rcond=None)[0]   # Y ≈ beta*X + intercept
    # Plot scatter + fitted curve (no statsmodels dependency)
    fig = px.scatter(d, x=price_col, y=qty_col, title="Price vs Quantity")
    xs = np.linspace(float(d[price_col].min()), float(d[price_col].max()), 60)
    ys = np.exp(beta*np.log(xs) + intercept)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="fit"))
    # R^2
    yhat = beta*X + intercept
    ss_res = float(np.sum((Y - yhat)**2))
    ss_tot = float(np.sum((Y - np.mean(Y))**2))
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
    narrative = f"Own-price elasticity ≈ {beta:.2f} (R²={r2:.2f}). "
    narrative += "Inelastic (|ε|<1)." if abs(beta) < 1 else "Elastic (|ε|≥1)."
    return ModuleResult(
        kpis={"elasticity": float(beta), "r2": r2},
        narrative=narrative,
        citations=[{"type":"table","ref":"(uploaded CSV)","selector":"price/qty columns"}],
        figure=fig
    )

# --- PDF KPI extraction (regex-y, page-cited) ---
_money = re.compile(r"\$?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:billion|bn|million|m)?", re.I)
_pct   = re.compile(r"\d{1,3}(?:\.\d+)?\s*%")

def _parse_money(tok:str) -> Optional[float]:
    if not tok: return None
    t = tok.lower().replace("$","").replace(" ","")
    mult = 1.0
    if "billion" in t or "bn" in t: mult = 1_000_000_000
    elif "million" in t or t.endswith("m"): mult = 1_000_000
    t = re.sub(r"[a-z]", "", t)
    try: return float(t.replace(",",""))*mult
    except: return None

def _parse_pct(tok:str) -> Optional[float]:
    if not tok: return None
    t = tok.replace("%","").strip()
    try: return float(t)/100.0
    except: return None

def _scan_metric(pages: List[str], keywords: List[str], want: str) -> Optional[Dict[str,Any]]:
    for i, page in enumerate(pages):
        txt = page or ""
        low = txt.lower()
        for kw in keywords:
            for m in re.finditer(re.escape(kw.lower()), low):
                start = max(0, m.start()-80); end = min(len(txt), m.end()+80)
                window = txt[start:end]
                if want == "money":
                    n = _money.search(window)
                    if n:
                        val = _parse_money(n.group())
                        if val is not None:
                            return {"page": i+1, "snippet": window.strip(), "raw": n.group(), "value": val}
                else:
                    n = _pct.search(window)
                    if n:
                        val = _parse_pct(n.group())
                        if val is not None:
                            return {"page": i+1, "snippet": window.strip(), "raw": n.group(), "value": val}
    return None

def module_pdf_kpi(pages: List[str]) -> ModuleResult:
    if not pages:
        return ModuleResult({}, "Empty PDF.", [])
    rev = _scan_metric(pages, ["revenue","revenues","total revenue"], "money")
    ebt = _scan_metric(pages, ["ebitda","adj ebitda"], "money")
    gm  = _scan_metric(pages, ["gross margin","gm%","gm"], "pct")
    chn = _scan_metric(pages, ["churn","net churn"], "pct")
    found = {"revenue": rev, "ebitda": ebt, "gross_margin": gm, "churn": chn}
    kpis = {k: (v["value"] if v else None) for k,v in found.items()}
    parts = []
    if rev: parts.append(f"Revenue ≈ ${rev['value']:,.0f} (p.{rev['page']}).")
    if ebt: parts.append(f"EBITDA ≈ ${ebt['value']:,.0f} (p.{ebt['page']}).")
    if gm : parts.append(f"Gross margin ≈ {gm['value']:.0%} (p.{gm['page']}).")
    if chn: parts.append(f"Churn ≈ {chn['value']:.0%} (p.{chn['page']}).")
    narrative = " ".join(parts) or "No obvious KPIs found; try a clearer KPI pack."
    citations = []
    for _,v in found.items():
        if v: citations.append({"type":"pdf","page":v["page"],"excerpt":v["snippet"][:220]})
    return ModuleResult(kpis=kpis, narrative=narrative, citations=citations)

MODULES = {
    "cohort_retention": {"title":"Cohort Retention (CSV)", "fn": module_cohort_retention, "needs": ["customer","date"], "optional": ["revenue"]},
    "pricing_power":   {"title":"Pricing Power (CSV)",   "fn": module_pricing_power,   "needs": ["price","quantity"], "optional": []},
    "pdf_kpi_extract": {"title":"PDF KPI Extract",       "fn": module_pdf_kpi,          "needs": [], "optional": []},
}

# --------------------- 1) EVIDENCE SOURCES ---------------------
st.subheader("1) Evidence Sources")
c_csv, c_pdf = st.columns(2)

with c_csv:
    up = st.file_uploader("Upload CSV(s)", type=["csv"], accept_multiple_files=True, key="csv_up")
    if up:
        for f in up:
            try:
                df = pd.read_csv(f)
                SS["tables"][f.name] = df
                SS["schema_map"].setdefault(f.name, {"customer":None,"date":None,"revenue":None,"price":None,"quantity":None})
                _log("SOURCE_ADDED", f"csv:{f.name}")
                st.success(f"Loaded CSV: {f.name} ({df.shape[0]:,} rows)")
            except Exception as e:
                st.error(f"{f.name}: {e}")

with c_pdf:
    up_pdf = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True, key="pdf_up")
    if up_pdf and PdfReader is None:
        st.error("PDF parsing requires `pypdf` in requirements.txt (e.g., pypdf>=4.0.0).")
    if up_pdf and PdfReader is not None:
        for f in up_pdf:
            reader = PdfReader(f)
            pages = []
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    pages.append("")
            SS["docs"][f.name] = pages
            _log("SOURCE_ADDED", f"pdf:{f.name}")
            st.success(f"Loaded PDF: {f.name} ({len(pages)} pages)")

# --------------------- CSV SCHEMA MAPPING ---------------------
with st.expander("2) Map CSV Schema", expanded=False):
    def _auto_guess(df: pd.DataFrame) -> Dict[str, Optional[str]]:
        return {
            "customer": _find(df,"customer"),
            "date":     _find(df,"date"),
            "revenue":  _find(df,"revenue"),
            "price":    _find(df,"price"),
            "quantity": _find(df,"quantity"),
        }
    if st.button("Auto-map all tables"):
        for name, df in SS["tables"].items():
            SS["schema_map"][name] = _auto_guess(df)
        st.success("Guessed schemas.")
    for name, df in SS["tables"].items():
        st.markdown(f"**{name}** — {df.shape[0]:,} rows × {df.shape[1]:,} cols")
        cols = list(df.columns)
        cur = SS["schema_map"].get(name) or _auto_guess(df)
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1:
            customer = st.selectbox(f"{name}: Customer", ["(none)"]+cols, index=(cols.index(cur["customer"])+1) if cur.get("customer") in cols else 0, key=f"cust_{name}")
        with c2:
            date = st.selectbox(f"{name}: Date", ["(none)"]+cols, index=(cols.index(cur["date"])+1) if cur.get("date") in cols else 0, key=f"date_{name}")
        with c3:
            revenue = st.selectbox(f"{name}: Revenue", ["(none)"]+cols, index=(cols.index(cur["revenue"])+1) if cur.get("revenue") in cols else 0, key=f"rev_{name}")
        with c4:
            price = st.selectbox(f"{name}: Price", ["(none)"]+cols, index=(cols.index(cur["price"])+1) if cur.get("price") in cols else 0, key=f"price_{name}")
        with c5:
            qty = st.selectbox(f"{name}: Quantity", ["(none)"]+cols, index=(cols.index(cur["quantity"])+1) if cur.get("quantity") in cols else 0, key=f"qty_{name}")
        if st.button(f"Save mapping for {name}", key=f"save_{name}"):
            SS["schema_map"][name] = {
                "customer": None if customer=="(none)" else customer,
                "date":     None if date=="(none)" else date,
                "revenue":  None if revenue=="(none)" else revenue,
                "price":    None if price=="(none)" else price,
                "quantity": None if qty=="(none)" else qty,
            }
            _log("SCHEMA_SAVED", name)
            st.success("Saved.")

# --------------------- 3) BUILD GRID ---------------------
st.subheader("3) Build Grid")
b1,b2,b3 = st.columns(3)
with b1:
    if st.button("Add rows from all CSVs"):
        for name in SS["tables"].keys(): _add_row_from_table(name)
        _ensure_cells()
with b2:
    if st.button("Add rows from all PDFs"):
        for name in SS["docs"].keys(): _add_row_from_pdf(name)
        _ensure_cells()

col_name = st.text_input("Column label", value="PDF KPIs")
tool_key = st.selectbox("Module", options=list(MODULES.keys()), format_func=lambda k: MODULES[k]["title"])
if st.button("Add Column"):
    _add_column(col_name, tool_key, params={})
    _ensure_cells()
    st.success(f"Added column: {col_name} [{tool_key}]")

with st.expander("Plan", expanded=False):
    st.write("Rows:", SS["grid"]["rows"])
    st.write("Columns:", SS["grid"]["columns"])

# --------------------- 4) RUN CELLS ---------------------
st.subheader("4) Run Cells")
sel_rows = st.multiselect("Rows to run", options=[r["id"] for r in SS["grid"]["rows"]],
                          format_func=lambda rid: next((r["row_ref"] for r in SS["grid"]["rows"] if r["id"]==rid), rid))
sel_cols = st.multiselect("Columns to run", options=[c["id"] for c in SS["grid"]["columns"]],
                          format_func=lambda cid: next((c["name"] for c in SS["grid"]["columns"] if c["id"]==cid), cid))

def _run_cell(cell: Dict[str,Any]):
    row = next(x for x in SS["grid"]["rows"] if x["id"]==cell["row_id"])
    col = next(x for x in SS["grid"]["columns"] if x["id"]==cell["col_id"])
    cell["status"] = "running"
    if row["type"] == "table":
        df = SS["tables"].get(row["source"])
        mapping = SS["schema_map"].get(row["source"], {})
        if col["tool"] == "cohort_retention":
            res = module_cohort_retention(df, mapping.get("customer"), mapping.get("date"), mapping.get("revenue"))
        elif col["tool"] == "pricing_power":
            res = module_pricing_power(df, mapping.get("price"), mapping.get("quantity"))
        elif col["tool"] == "pdf_kpi_extract":
            res = ModuleResult({}, "PDF module requires a PDF row.", [])
        else:
            res = ModuleResult({}, f"Unknown tool {col['tool']}", [])
    else:  # pdf
        pages = SS["docs"].get(row["source"], [])
        if col["tool"] == "pdf_kpi_extract":
            res = module_pdf_kpi(pages)
        elif col["tool"] in ("cohort_retention","pricing_power"):
            res = ModuleResult({}, f"{col['tool']} applies to CSV rows.", [])
        else:
            res = ModuleResult({}, f"Unknown tool {col['tool']}", [])
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
    for cell in targets: _run_cell(cell)
    st.success(f"Ran {len(targets)} cell(s).")

cells_df = pd.DataFrame(SS["grid"]["cells"])
if not cells_df.empty:
    hide = [c for c in ["citations","figure","notes","confidence"] if c in cells_df.columns]
    st.dataframe(cells_df.drop(columns=hide), use_container_width=True, height=300)
else:
    st.info("No cells yet. Add rows & a column, then run.")

# --------------------- 5) REVIEW ---------------------
st.subheader("5) Review")
sel_cell_id = st.selectbox("Choose a cell", options=[c["id"] for c in SS["grid"]["cells"]], index=0 if SS["grid"]["cells"] else None)
if sel_cell_id:
    cell = next(c for c in SS["grid"]["cells"] if c["id"]==sel_cell_id)
    col  = next(x for x in SS["grid"]["columns"] if x["id"]==cell["col_id"])
    row  = next(x for x in SS["grid"]["rows"] if x["id"]==cell["row_id"])
    st.markdown(f"**{col['name']}** on _{row['row_ref']}_ — status: `{cell['status']}`")
    if cell.get("figure") is not None:
        st.plotly_chart(cell["figure"], use_container_width=True)
    if cell.get("output_text"): st.write(cell["output_text"])
    with st.expander("Citations"):
        st.json(cell.get("citations", []))
    a1,a2 = st.columns(2)
    with a1:
        if st.button("Approve"):
            cell["status"] = "approved"; _log("CELL_APPROVE", sel_cell_id); st.success("Approved.")
    with a2:
        if st.button("Mark Needs-Review"):
            cell["status"] = "needs_review"; _log("CELL_MARK_REVIEW", sel_cell_id); st.warning("Marked.")

# --------------------- 6) MEMO & EXPORT ---------------------
st.subheader("6) Compose Memo & Export")

def _build_memo() -> str:
    lines = [f"# Investment Memo — {SS['grid']['id']}", ""]
    approved = [c for c in SS["grid"]["cells"] if c["status"]=="approved"]
    if approved:
        lines += ["## Executive Summary", f"- Approved findings: {len(approved)}"]
        for c in approved[:8]:
            col = next(x for x in SS["grid"]["columns"] if x["id"]==c["col_id"])
            row = next(x for x in SS["grid"]["rows"] if x["id"]==c["row_id"])
            val = c.get("numeric_value"); val_s = f"{val:,.0f}" if isinstance(val,(int,float)) else (val or "")
            lines.append(f"  - **{col['name']}** on _{row['row_ref']}_ → {val_s} — {c.get('output_text','')}")
        lines.append("")
    lines.append("## Evidence Appendix")
    for c in approved: lines.append(f"- Cell {c['id']} citations: {json.dumps(c.get('citations', []))}")
    return "\n".join(lines)

def _memo_pdf_bytes(md: str) -> bytes:
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=LETTER)
    W,H = LETTER; M = 0.75*inch; y = H-M
    c.setFont("Helvetica-Bold", 14); c.drawString(M,y,"Transform AI — Investment Memo"); y -= 18
    c.setFont("Helvetica", 9); c.drawString(M,y,f"Grid: {SS['grid']['id']}"); y -= 14
    c.setFont("Helvetica", 11)
    for line in md.splitlines():
        for seg in textwrap.wrap(line, width=95) or [" "]:
            if y < M: c.showPage(); y = H-M; c.setFont("Helvetica", 11)
            c.drawString(M,y,seg); y -= 14
    c.showPage(); c.save(); buf.seek(0); return buf.getvalue()

if st.button("Compose Memo"):
    SS["last_memo_md"] = _build_memo()
    st.code(SS["last_memo_md"], language="markdown")

if SS.get("last_memo_md"):
    st.download_button("⬇️ Download PDF",
                       data=_memo_pdf_bytes(SS["last_memo_md"]),
                       file_name=f"TransformAI_Memo_{SS['grid']['id']}.pdf",
                       mime="application/pdf")

# --------------------- 7) EVIDENCE CHAT (BETA) ---------------------
st.subheader("7) Evidence Chat (beta)")

def _score(text: str, q: str) -> int:
    if not text: return 0
    score = 0
    q_terms = [t for t in re.findall(r"\w+", q.lower()) if len(t) > 2]
    t_lower = text.lower()
    for t in q_terms: score += t_lower.count(t)
    return score

def search_pdfs(q: str, topk: int = 3):
    results = []
    for name, pages in SS.get("docs", {}).items():
        for i, ptxt in enumerate(pages):
            s = _score(ptxt, q)
            if s > 0:
                idx = ptxt.lower().find(q.lower())
                if idx < 0: idx = 0
                snippet = (ptxt[max(0, idx-120): idx+120] or "").replace("\n", " ")
                results.append({"kind":"pdf","source":name,"page":i+1,"score":s,"snippet":snippet.strip()})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:topk]

def search_csvs(q: str, topk: int = 3, sample_rows: int = 5):
    results = []
    for name, df in SS.get("tables", {}).items():
        s_cols = sum(1 for c in df.columns if q.lower() in c.lower())
        s_cells = 0
        try:
            sample = df.head(200)
            for col in sample.columns:
                s_cells += sample[col].astype(str).str.lower().str.contains(q.lower(), na=False).sum()
        except Exception:
            pass
        score = s_cols*3 + s_cells
        if score > 0:
            prev = pd.DataFrame()
            try:
                mask = pd.Series(False, index=df.index)
                for col in df.columns:
                    mask = mask | df[col].astype(str).str.lower().str.contains(q.lower(), na=False)
                prev = df[mask].head(sample_rows)
            except Exception:
                pass
            results.append({"kind":"csv","source":name,"score":score,"preview":prev})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:topk]

def answers_from_grid(q: str):
    hits = []
    ql = q.lower()
    for c in SS["grid"]["cells"]:
        text = (c.get("output_text") or "").lower()
        if any(tok in text for tok in re.findall(r"\w+", ql)):
            col = next((x for x in SS["grid"]["columns"] if x["id"]==c["col_id"]), {"name":"(unknown)"})
            row = next((x for x in SS["grid"]["rows"] if x["id"]==c["row_id"]), {"row_ref":"(unknown)"})
            hits.append({
                "kind": "cell",
                "col_name": col["name"],
                "row_ref": row["row_ref"],
                "status": c.get("status"),
                "output_text": c.get("output_text"),
                "citations": c.get("citations", [])
            })
    return hits[:3]

# render previous chat
for role, content in SS["chat_history"]:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("Ask about your evidence (e.g., 'show EBITDA', 'pages about churn', 'which CSV has price?')")
if prompt:
    SS["chat_history"].append(("user", prompt))
    with st.chat_message("user"): st.markdown(prompt)

    pdf_hits = search_pdfs(prompt, topk=3)
    csv_hits = search_csvs(prompt, topk=3)
    grid_hits = answers_from_grid(prompt)

    parts = []
    if grid_hits:
        parts.append("**Existing answers (from grid):**")
        for h in grid_hits:
            parts.append(f"- _{h['row_ref']}_ • **{h['col_name']}** → {h['output_text']}")
            if h["citations"]: parts.append(f"  · Citations: {len(h['citations'])}")
    if pdf_hits:
        parts.append("**PDF matches:**")
        for h in pdf_hits:
            parts.append(f"- `{h['source']}` — p.{h['page']} · “{h['snippet']}”")
    if csv_hits:
        parts.append("**CSV matches:**")
        for h in csv_hits:
            row_count = h["preview"].shape[0] if isinstance(h["preview"], pd.DataFrame) else 0
            parts.append(f"- `{h['source']}` — matches in columns/rows (showing {row_count} rows below).")

    if not (grid_hits or pdf_hits or csv_hits):
        parts.append("_No obvious matches. Try a different term (e.g., 'revenue', 'margin', 'price')._")

    answer_md = "\n".join(parts) or "No results."
    SS["chat_history"].append(("assistant", answer_md))
    with st.chat_message("assistant"):
        st.markdown(answer_md)
        for h in csv_hits:
            if isinstance(h["preview"], pd.DataFrame) and not h["preview"].empty:
                st.dataframe(h["preview"], use_container_width=True)

# --------------------- ACTIVITY LOG ---------------------
with st.expander("Activity Log", expanded=False):
    st.json(SS["grid"]["activities"])


