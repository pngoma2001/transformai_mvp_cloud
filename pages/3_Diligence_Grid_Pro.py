# pages/3_Diligence_Grid_Pro.py
# TransformAI — Diligence Grid (Pro) — Sprint C upgrade
# - Starter Columns (matching rows only)
# - Filters (status / row type / column) + bulk approve/review
# - Export results CSV
# - Memo column picker (defaults to approved)
# - Light cache + Force re-run
# - Keeps type-guards, quick-manage, charts, memo cross-checks

from __future__ import annotations
import io, uuid, re, textwrap, hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

st.set_page_config(page_title="TransformAI — Diligence (Pro)", layout="wide")
st.title("Transform AI — Diligence Grid (Pro)")
st.caption("Upload CSV/PDF → map schema → run Cohorts, Pricing, NRR/GRR, PDF KPIs → approve → Export Investor Memo with cross-checks.")

SS = st.session_state
SS.setdefault("tables", {})
SS.setdefault("docs", {})
SS.setdefault("schema_map", {})
SS.setdefault("chat_history", [])
SS.setdefault("sel_rows", [])
SS.setdefault("sel_cols", [])
SS.setdefault("do_clear_selection", False)
SS.setdefault("cache", {})  # cell_id -> {"sig": ..., "result": ...}
SS.setdefault("memo_cols", None)  # list of column ids selected for memo (None=auto)
SS.setdefault("grid", {
    "id": f"grid_{uuid.uuid4().hex[:6]}",
    "rows": [],
    "columns": [],
    "cells": [],
    "activities": []
})

# -------------------- utils --------------------
def _log(action:str, detail:str=""):
    SS["grid"]["activities"].append({"id": uuid.uuid4().hex, "action": action, "detail": detail})

def _new_id(p): return f"{p}_{uuid.uuid4().hex[:8]}"

def _fmt_money(x: Optional[float]) -> str:
    if x is None or not isinstance(x,(int,float)) or x!=x: return "-"
    if abs(x) >= 1_000_000_000: return f"${x/1_000_000_000:,.2f}B"
    if abs(x) >= 1_000_000:     return f"${x/1_000_000:,.2f}M"
    return f"${x:,.2f}"

def _fmt_pct(x: Optional[float]) -> str:
    if x is None or not isinstance(x,(int,float)) or x!=x: return "-"
    return f"{x*100:.1f}%"

def _find(df: pd.DataFrame, key: str) -> Optional[str]:
    bank = {
        "customer": ["customer","user","buyer","account","client","cust","cust_id","customer_id"],
        "date":     ["date","timestamp","order_date","created_at","period","month"],
        "revenue":  ["revenue","amount","net_revenue","sales","gmv","value"],
        "price":    ["price","unit_price","avg_price","p"],
        "quantity": ["qty","quantity","units","volume","q"],
        "segment":  ["segment","sku","product","category","plan","region","cohort","family"]
    }.get(key.lower(), [key.lower()])
    cols = list(df.columns); lower = {c.lower(): c for c in cols}
    for needle in bank:
        if needle in lower: return lower[needle]
    for needle in bank:
        for c in cols:
            if needle in c.lower(): return c
    return None

# -------------------- grid helpers --------------------
def _add_row_from_table(name:str):
    rid = _new_id("row")
    SS["grid"]["rows"].append({"id": rid, "row_ref": f"table:{name}", "source": name, "type": "table", "alias": f"{name} (Transactions)"})
    _log("ROW_ADDED", f"table:{name}")
    return rid

def _add_row_from_pdf(name:str):
    rid = _new_id("row")
    SS["grid"]["rows"].append({"id": rid, "row_ref": f"pdf:{name}", "source": name, "type": "pdf", "alias": f"{name} (KPI Pack)"})
    _log("ROW_ADDED", f"pdf:{name}")
    return rid

def _add_column(name:str, tool:str, params:Optional[Dict[str,Any]]=None):
    cid = _new_id("col")
    SS["grid"]["columns"].append({"id": cid, "name": name, "tool": tool, "params": params or {}})
    _log("COLUMN_ADDED", f"{name} [{tool}]")
    return cid

def delete_row(row_id: str):
    SS["grid"]["rows"] = [r for r in SS["grid"]["rows"] if r["id"] != row_id]
    SS["grid"]["cells"] = [c for c in SS["grid"]["cells"] if c["row_id"] != row_id]
    SS["cache"].pop(row_id, None)
    _log("ROW_DELETED", row_id)

def delete_col(col_id: str):
    SS["grid"]["columns"] = [c for c in SS["grid"]["columns"] if c["id"] != col_id]
    SS["grid"]["cells"] = [c for c in SS["grid"]["cells"] if c["col_id"] != col_id]
    _log("COLUMN_DELETED", col_id)

def move_col(col_id: str, direction: str):
    cols = SS["grid"]["columns"]
    idx = next((i for i,c in enumerate(cols) if c["id"]==col_id), None)
    if idx is None: return
    if direction=="up" and idx>0: cols[idx-1], cols[idx] = cols[idx], cols[idx-1]
    if direction=="down" and idx < len(cols)-1: cols[idx+1], cols[idx] = cols[idx], cols[idx+1]
    _log("COLUMN_MOVED", f"{col_id}:{direction}")

# -------------------- modules --------------------
@dataclass
class ModuleResult:
    kpis: Dict[str, Any]
    narrative: str
    citations: List[Dict[str, Any]]
    figure: Optional[Any] = None
    units_hint: Optional[str] = None
    figure2: Optional[Any] = None

def module_cohort_retention(df: pd.DataFrame,
                            customer_col: Optional[str]=None,
                            ts_col: Optional[str]=None,
                            revenue_col: Optional[str]=None) -> ModuleResult:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ModuleResult({}, "Empty dataset.", [], units_hint="pct")
    customer_col = customer_col or _find(df, "customer")
    ts_col = ts_col or _find(df, "date")
    revenue_col = revenue_col or _find(df, "revenue")
    if not (customer_col and ts_col):
        return ModuleResult({}, "Missing customer/date columns; map schema to compute retention.", [], units_hint="pct")

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

    fig_curve = px.line(x=list(curve.index), y=list(curve.values),
                        labels={"x":"Months since first purchase","y":"Retention"},
                        title="Average Retention Curve")

    if not mat.empty:
        hm = mat.copy()
        hm.index = hm.index.astype(str)
        hm.columns = [int(c) for c in hm.columns]
        fig_heat = px.imshow(hm.values,
                             x=list(hm.columns),
                             y=list(hm.index),
                             aspect="auto",
                             origin="upper",
                             labels=dict(x="Months since first purchase", y="Cohort (first month)", color="Retention"),
                             title="Cohort Heatmap")
    else:
        fig_heat = None

    narrative = f"Retention stabilizes ~M3 at {m3:.0%}." if m3==m3 else "Not enough data to compute M3 retention."
    if ltv_12: narrative += f" Avg 12-month LTV proxy ≈ {_fmt_money(ltv_12)}."
    return ModuleResult(
        kpis={"month_3_retention": m3, "ltv_12m": ltv_12},
        narrative=narrative,
        citations=[{"type":"table","ref":"(uploaded CSV)","selector":"all_rows"}],
        figure=fig_curve,
        units_hint="pct",
        figure2=fig_heat
    )

def _ols_loglog(x: np.ndarray, y: np.ndarray) -> Tuple[float,float,Optional[float]]:
    X = np.log(x); Y = np.log(y)
    A = np.vstack([X, np.ones(len(X))]).T
    beta, intercept = np.linalg.lstsq(A, Y, rcond=None)[0]
    yhat = beta*X + intercept
    ss_res = float(np.sum((Y - yhat)**2)); ss_tot = float(np.sum((Y - np.mean(Y))**2))
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else None
    return float(beta), float(intercept), r2

def _guess_segment_col(df: pd.DataFrame) -> Optional[str]:
    return _find(df, "segment")

def module_pricing_power(df: pd.DataFrame,
                         price_col: Optional[str]=None,
                         qty_col: Optional[str]=None) -> ModuleResult:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ModuleResult({}, "Empty dataset.", [])
    price_col = price_col or _find(df, "price")
    qty_col   = qty_col   or _find(df, "quantity")
    if not (price_col and qty_col):
        return ModuleResult({}, "Missing price/quantity columns; map schema first.", [])

    d = df[[price_col, qty_col] + [c for c in df.columns if c not in [price_col, qty_col]]].dropna(subset=[price_col, qty_col]).copy()
    d = d[(d[price_col] > 0) & (d[qty_col] > 0)]
    if len(d) < 8:
        return ModuleResult({}, "Need ≥ 8 observations for elasticity regression.", [])

    beta, intercept, r2 = _ols_loglog(d[price_col].values, d[qty_col].values)
    fig_scatter = px.scatter(d, x=price_col, y=qty_col, title="Price vs Quantity")
    xs = np.linspace(float(d[price_col].min()), float(d[price_col].max()), 60)
    ys = np.exp(beta*np.log(xs) + intercept)
    fig_scatter.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="fit"))
    narrative = f"Own-price elasticity ≈ {beta:.2f} (R²={r2:.2f}). "
    narrative += "Inelastic (|ε|<1)." if abs(beta) < 1 else "Elastic (|ε|≥1)."

    seg_col = _guess_segment_col(d)
    seg_results = []
    fig_segments = None
    if seg_col and seg_col in d.columns:
        topk = d[seg_col].value_counts().index.tolist()[:8]
        bars_x, bars_y = [], []
        for seg in topk:
            sd = d[d[seg_col]==seg]
            if len(sd) >= 6:
                b, _int, _r2 = _ols_loglog(sd[price_col].values, sd[qty_col].values)
                seg_results.append({"segment": str(seg), "elasticity": float(b), "r2": _r2})
                bars_x.append(str(seg)); bars_y.append(float(b))
        if seg_results:
            fig_segments = go.Figure()
            fig_segments.add_trace(go.Bar(x=bars_x, y=bars_y, name="Elasticity by segment"))
            fig_segments.update_layout(title=f"Segment Elasticities (by {seg_col})",
                                       xaxis_title=seg_col, yaxis_title="Elasticity (log-log slope)")

    kpis = {"elasticity": float(beta), "r2": r2}
    if seg_results: kpis["segments"] = seg_results

    return ModuleResult(
        kpis=kpis,
        narrative=narrative,
        citations=[{"type":"table","ref":"(uploaded CSV)","selector":"price/qty (+ segment if available)"}],
        figure=fig_scatter,
        figure2=fig_segments
    )

def module_nrr_grr(df: pd.DataFrame,
                   customer_col: Optional[str]=None,
                   ts_col: Optional[str]=None,
                   revenue_col: Optional[str]=None) -> ModuleResult:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ModuleResult({}, "Empty dataset.", [], units_hint="pct")
    customer_col = customer_col or _find(df, "customer")
    ts_col       = ts_col       or _find(df, "date")
    revenue_col  = revenue_col  or _find(df, "revenue")
    if not (customer_col and ts_col and revenue_col):
        return ModuleResult({}, "Need customer/date/revenue columns; map schema first.", [], units_hint="pct")
    d = df[[customer_col, ts_col, revenue_col]].copy()
    d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce")
    d = d.dropna(subset=[customer_col, ts_col, revenue_col])
    d["month"] = d[ts_col].dt.to_period("M")
    gp = d.groupby([customer_col, "month"], as_index=False)[revenue_col].sum()
    pivot = gp.pivot(index=customer_col, columns="month", values=revenue_col).fillna(0.0).sort_index(axis=1)
    months = list(pivot.columns)
    if len(months) < 2: return ModuleResult({}, "Need at least two months of data.", [], units_hint="pct")
    labels, grr_list, nrr_list = [], [], []
    churn_rate, contraction_rate, expansion_rate = [], [], []
    last_pair = None
    for i in range(1, len(months)):
        prev_m, curr_m = months[i-1], months[i]
        prev_rev, curr_rev = pivot[prev_m], pivot[curr_m]
        base_mask = prev_rev > 0
        start = float(prev_rev[base_mask].sum())
        if start <= 0: continue
        curr_base = curr_rev[base_mask]
        churn_amt = float(prev_rev[base_mask & (curr_base == 0)].sum())
        contraction_amt = float(((prev_rev[base_mask] - curr_base).clip(lower=0.0).sum()) - churn_amt)
        contraction_amt = max(contraction_amt, 0.0)
        expansion_amt = float((curr_base - prev_rev[base_mask]).clip(lower=0.0).sum())
        grr = (start - churn_amt - contraction_amt) / start if start else np.nan
        nrr = (start - churn_amt - contraction_amt + expansion_amt) / start if start else np.nan
        labels.append(str(curr_m))
        grr_list.append(grr); nrr_list.append(nrr)
        churn_rate.append(churn_amt / start)
        contraction_rate.append(contraction_amt / start)
        expansion_rate.append(expansion_amt / start)
        last_pair = (str(prev_m), str(curr_m), start, churn_amt, contraction_amt, expansion_amt)

    if not labels: return ModuleResult({}, "Insufficient overlap to compute NRR/GRR.", [], units_hint="pct")

    fig_lines = go.Figure()
    fig_lines.add_trace(go.Scatter(x=labels, y=nrr_list, mode="lines+markers", name="NRR"))
    fig_lines.add_trace(go.Scatter(x=labels, y=grr_list, mode="lines+markers", name="GRR"))
    ymax = float(np.nanmax(nrr_list + grr_list)) if len(nrr_list + grr_list) else 1.0
    fig_lines.update_layout(title="Monthly NRR & GRR", xaxis_title="Month", yaxis_title="Rate",
                            yaxis=dict(range=[0, max(1.2, ymax)]))

    fig_wf = None
    if last_pair:
        pm, cm, start, churn_amt, contr_amt, exp_amt = last_pair
        data = [
            dict(name="Start", measure="absolute", y=start),
            dict(name="Churn", measure="relative", y=-churn_amt),
            dict(name="Contraction", measure="relative", y=-contr_amt),
            dict(name="Expansion", measure="relative", y=exp_amt),
            dict(name="End", measure="total", y=start - churn_amt - contr_amt + exp_amt),
        ]
        fig_wf = go.Figure(go.Waterfall(
            name=f"{pm}→{cm}",
            orientation="v",
            measure=[d["measure"] for d in data],
            x=[d["name"] for d in data],
            y=[d["y"] for d in data]
        ))
        fig_wf.update_layout(title=f"Revenue Waterfall ({pm} → {cm})")

    last_label = labels[-1]
    kpis = {
        "month": last_label,
        "grr": float(round(grr_list[-1], 4)),
        "nrr": float(round(nrr_list[-1], 4)),
        "churn_rate": float(round(churn_rate[-1], 4)),
        "contraction_rate": float(round(contraction_rate[-1], 4)),
        "expansion_rate": float(round(expansion_rate[-1], 4)),
    }
    narrative = (f"Latest ({last_label}): GRR {kpis['grr']:.0%}, NRR {kpis['nrr']:.0%} "
                 f"(expansion {kpis['expansion_rate']:.0%}, contraction {kpis['contraction_rate']:.0%}, churn {kpis['churn_rate']:.0%}).")
    return ModuleResult(kpis=kpis, narrative=narrative,
                        citations=[{"type":"table","ref":"(uploaded CSV)","selector":"monthly revenue by customer"}],
                        figure=fig_lines, units_hint="pct", figure2=fig_wf)

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
        txt = page or ""; low = txt.lower()
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
    if not pages: return ModuleResult({}, "Empty PDF.", [])
    rev = _scan_metric(pages, ["revenue","revenues","total revenue"], "money")
    ebt = _scan_metric(pages, ["ebitda","adj ebitda"], "money")
    gm  = _scan_metric(pages, ["gross margin","gm%","gm"], "pct")
    chn = _scan_metric(pages, ["churn","net churn"], "pct")
    found = {"revenue": rev, "ebitda": ebt, "gross_margin": gm, "churn": chn}
    kpis = {k: (v["value"] if v else None) for k,v in found.items()}
    parts = []
    if rev: parts.append(f"Revenue ≈ {_fmt_money(rev['value'])} (p.{rev['page']}).")
    if ebt: parts.append(f"EBITDA ≈ {_fmt_money(ebt['value'])} (p.{ebt['page']}).")
    if gm : parts.append(f"Gross margin ≈ {_fmt_pct(gm['value'])} (p.{gm['page']}).")
    if chn: parts.append(f"Churn ≈ {_fmt_pct(chn['value'])} (p.{chn['page']}).")
    narrative = " ".join(parts) or "No obvious KPIs found; try a clearer KPI pack."
    citations = []
    for _,v in found.items():
        if v: citations.append({"type":"pdf","page":v["page"],"excerpt":(v["snippet"] or "")[:220]})
    return ModuleResult(kpis=kpis, narrative=narrative, citations=citations)

MODULES = {
    "cohort_retention": {
        "title": "Cohort Retention (CSV)",
        "fn": module_cohort_retention,
        "needs": ["customer", "date"],
        "optional": ["revenue"],
        "applies_to": ["table"],
    },
    "pricing_power": {
        "title": "Pricing Power (CSV)",
        "fn": module_pricing_power,
        "needs": ["price", "quantity"],
        "optional": [],
        "applies_to": ["table"],
    },
    "nrr_grr": {
        "title": "NRR/GRR (CSV)",
        "fn": module_nrr_grr,
        "needs": ["customer", "date", "revenue"],
        "optional": [],
        "applies_to": ["table"],
    },
    "pdf_kpi_extract": {
        "title": "PDF KPI Extract",
        "fn": module_pdf_kpi,
        "needs": [],
        "optional": [],
        "applies_to": ["pdf"],
    },
}

# -------------------- 1) Evidence Sources --------------------
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
                try: pages.append(p.extract_text() or "")
                except Exception: pages.append("")
            SS["docs"][f.name] = pages
            _log("SOURCE_ADDED", f"pdf:{f.name}")
            st.success(f"Loaded PDF: {f.name} ({len(pages)} pages)")

# -------------------- 2) Map CSV Schema --------------------
with st.expander("2) Map CSV Schema", expanded=False):
    def _auto_guess(df: pd.DataFrame) -> Dict[str, Optional[str]]:
        return {"customer": _find(df,"customer"), "date": _find(df,"date"),
                "revenue": _find(df,"revenue"), "price": _find(df,"price"), "quantity": _find(df,"quantity")}
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
            _log("SCHEMA_SAVED", name); st.success("Saved.")

# -------------------- prune + ensure cells --------------------
def _prune_incompatible_cells():
    rows_by_id = {r["id"]: r for r in SS["grid"]["rows"]}
    cols_by_id = {c["id"]: c for c in SS["grid"]["columns"]}
    kept = []
    for cell in SS["grid"]["cells"]:
        r = rows_by_id.get(cell["row_id"]); c = cols_by_id.get(cell["col_id"])
        if not r or not c: continue
        applies = MODULES.get(c["tool"], {}).get("applies_to", ["table","pdf"])
        if r["type"] in applies: kept.append(cell)
    if len(kept) != len(SS["grid"]["cells"]):
        SS["grid"]["cells"] = kept; _log("CELLS_PRUNED", f"{len(kept)} kept")

def _ensure_cells():
    _prune_incompatible_cells()
    have = {(c["row_id"], c["col_id"]) for c in SS["grid"]["cells"]}
    for r in SS["grid"]["rows"]:
        rtype = r["type"]
        for c in SS["grid"]["columns"]:
            applies = MODULES.get(c["tool"], {}).get("applies_to", ["table","pdf"])
            if rtype not in applies: continue
            if (r["id"], c["id"]) not in have:
                SS["grid"]["cells"].append({
                    "id": _new_id("cell"), "row_id": r["id"], "col_id": c["id"],
                    "status": "queued", "output_text": None,
                    "numeric_value": None, "units": None,
                    "kpis": {}, "citations": [],
                    "confidence": None, "notes": [],
                    "figure": None, "figure2": None
                })

# -------------------- 3) Build Grid --------------------
st.subheader("3) Build Grid")
b1,b2,b3,b4 = st.columns(4)
with b1:
    if st.button("Add rows from all CSVs"):
        for name in SS["tables"].keys(): _add_row_from_table(name)
        _ensure_cells()
with b2:
    if st.button("Add rows from all PDFs"):
        for name in SS["docs"].keys(): _add_row_from_pdf(name)
        _ensure_cells()
with b3:
    if st.button("Add Starter Columns (all 4)"):
        for name, tool in [("Cohort Retention","cohort_retention"),
                           ("Pricing Power","pricing_power"),
                           ("NRR/GRR","nrr_grr"),
                           ("PDF KPIs","pdf_kpi_extract")]:
            _add_column(name, tool, {})
        _ensure_cells(); st.success("Added 4 starter columns.")
with b4:
    if st.button("Add Starter Columns (matching rows)"):
        has_csv = any(r["type"]=="table" for r in SS["grid"]["rows"])
        has_pdf = any(r["type"]=="pdf" for r in SS["grid"]["rows"])
        existing_tools = {c["tool"] for c in SS["grid"]["columns"]}
        if has_csv:
            for name,tool in [("Cohort Retention","cohort_retention"),
                              ("Pricing Power","pricing_power"),
                              ("NRR/GRR","nrr_grr")]:
                if tool not in existing_tools: _add_column(name, tool, {})
        if has_pdf and "pdf_kpi_extract" not in existing_tools:
            _add_column("PDF KPIs","pdf_kpi_extract",{})
        _ensure_cells(); st.success("Added matching starter columns.")

col_name = st.text_input("Column label", value="PDF KPIs")
tool_key = st.selectbox("Module", options=list(MODULES.keys()), format_func=lambda k: MODULES[k]["title"])
c_add1, _ = st.columns([0.25, 0.75])
with c_add1:
    if st.button("Add Column"):
        _add_column(col_name, tool_key, params={}); _ensure_cells()
        st.success(f"Added column: {col_name} [{tool_key}]")

with st.expander("Manage rows & columns", expanded=False):
    st.markdown("**Rows**")
    for r in list(SS["grid"]["rows"]):
        cols = st.columns([0.58, 0.22, 0.10, 0.10])
        with cols[0]:
            r["alias"] = st.text_input(f"Alias for {r['row_ref']}", r.get("alias", r["row_ref"]), key=f"alias_{r['id']}")
        with cols[1]:
            st.caption(f"type: {r['type']}")
        with cols[2]:
            if st.button("⬆️ Up", key=f"r_up_{r['id']}"):
                idx = next((i for i,x in enumerate(SS["grid"]["rows"]) if x["id"]==r["id"]), None)
                if idx is not None and idx>0:
                    SS["grid"]["rows"][idx-1], SS["grid"]["rows"][idx] = SS["grid"]["rows"][idx], SS["grid"]["rows"][idx-1]
                    try: st.rerun()
                    except: st.experimental_rerun()
        with cols[3]:
            if st.button("🗑️ Delete", key=f"delrow_{r['id']}"):
                delete_row(r["id"]); _ensure_cells()
                try: st.rerun()
                except: st.experimental_rerun()

    st.markdown("---")
    st.markdown("**Columns**")
    for c in list(SS["grid"]["columns"]):
        cols = st.columns([0.50, 0.14, 0.14, 0.10, 0.12])
        with cols[0]:
            c["name"] = st.text_input(f"Column label ({c['tool']})", c["name"], key=f"colname_{c['id']}")
        with cols[1]:
            if st.button("⬆️ Up", key=f"up_{c['id']}"): move_col(c["id"], "up"); _ensure_cells(); st.experimental_rerun()
        with cols[2]:
            if st.button("⬇️ Down", key=f"dn_{c['id']}"): move_col(c["id"], "down"); _ensure_cells(); st.experimental_rerun()
        with cols[3]:
            st.caption("csv" if MODULES[c["tool"]]["applies_to"]==["table"] else "pdf")
        with cols[4]:
            if st.button("🗑️ Delete", key=f"delcol_{c['id']}"): delete_col(c["id"]); _ensure_cells(); st.experimental_rerun()

# -------------------- 4) Run Cells --------------------
st.subheader("4) Run Cells")

force_rerun = st.toggle("Force re-run (ignore cache)", value=False)

if SS.get("do_clear_selection"):
    SS["sel_rows"] = []; SS["sel_cols"] = []; SS["do_clear_selection"] = False

sel_rows = st.multiselect(
    "Rows to run",
    options=[r["id"] for r in SS["grid"]["rows"]],
    format_func=lambda rid: next((r.get("alias") or r["row_ref"] for r in SS["grid"]["rows"] if r["id"]==rid), rid),
    key="sel_rows"
)
sel_cols = st.multiselect(
    "Columns to run",
    options=[c["id"] for c in SS["grid"]["columns"]],
    format_func=lambda cid: next((c["name"] for c in SS["grid"]["columns"] if c["id"]==cid), cid),
    key="sel_cols"
)

def _hash_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _sig_for_row(row: Dict[str,Any]) -> str:
    if row["type"]=="table":
        df = SS["tables"].get(row["source"])
        if df is None or df.empty: return "table:empty"
        sample = df.head(60).to_csv(index=False)
        cols = ",".join(df.columns.astype(str).tolist())
        return _hash_text(f"table|{row['source']}|{df.shape}|{cols}|{sample[:5000]}")
    else:
        pages = SS["docs"].get(row["source"], [])
        head = "".join((pages[i] if i < len(pages) else "")[:300] for i in range(min(5, len(pages))))
        return _hash_text(f"pdf|{row['source']}|{len(pages)}|{len(head)}|{head[:1500]}")

def _run_cell(cell: Dict[str,Any], force: bool=False):
    row = next(x for x in SS["grid"]["rows"] if x["id"]==cell["row_id"])
    col = next(x for x in SS["grid"]["columns"] if x["id"]==cell["col_id"])

    # cache key / signature
    sig = f"{_sig_for_row(row)}|{col['tool']}|{str(SS['schema_map'].get(row['source'], {}))}"
    cached = SS["cache"].get(cell["id"])
    if cached and (cached.get("sig")==sig) and not force:
        res: ModuleResult = cached["result"]
    else:
        cell["status"] = "running"
        if row["type"] == "table":
            df = SS["tables"].get(row["source"])
            mapping = SS["schema_map"].get(row["source"], {})
            if col["tool"] == "cohort_retention":
                res = module_cohort_retention(df, mapping.get("customer"), mapping.get("date"), mapping.get("revenue"))
            elif col["tool"] == "pricing_power":
                res = module_pricing_power(df, mapping.get("price"), mapping.get("quantity"))
            elif col["tool"] == "nrr_grr":
                res = module_nrr_grr(df, mapping.get("customer"), mapping.get("date"), mapping.get("revenue"))
            elif col["tool"] == "pdf_kpi_extract":
                res = ModuleResult({}, "PDF module requires a PDF row.", [])
            else:
                res = ModuleResult({}, f"Unknown tool {col['tool']}", [])
        else:
            pages = SS["docs"].get(row["source"], [])
            if col["tool"] == "pdf_kpi_extract":
                res = module_pdf_kpi(pages)
            elif col["tool"] in ("cohort_retention","pricing_power","nrr_grr"):
                res = ModuleResult({}, f"{MODULES[col['tool']]['title']} applies to CSV rows.", [])
            else:
                res = ModuleResult({}, f"Unknown tool {col['tool']}", [])
        SS["cache"][cell["id"]] = {"sig": sig, "result": res}

    # write result into cell
    cell["status"] = "done" if res.kpis else "needs_review"
    cell["output_text"] = res.narrative
    v = None
    if res.kpis:
        for _k,_v in res.kpis.items():
            if isinstance(_v,(int,float)) and _v==_v:
                v = float(_v); break
    cell["numeric_value"] = v
    cell["units"] = res.units_hint
    cell["kpis"] = res.kpis
    cell["citations"] = res.citations
    cell["figure"] = res.figure
    cell["figure2"] = res.figure2
    _log("CELL_RUN", f"{row['row_ref']} × {col['name']} → {cell['status']}")

c1,c2,c3 = st.columns(3)
with c1:
    if st.button("Run selection"):
        _ensure_cells()
        targets = [c for c in SS["grid"]["cells"]
                   if (not sel_rows or c["row_id"] in sel_rows) and (not sel_cols or c["col_id"] in sel_cols)]
        for cell in targets: _run_cell(cell, force=force_rerun)
        st.success(f"Ran {len(targets)} cell(s).")
with c2:
    if st.button("Run ALL"):
        _ensure_cells()
        for cell in SS["grid"]["cells"]: _run_cell(cell, force=force_rerun)
        st.success(f"Ran {len(SS['grid']['cells'])} cell(s).")
with c3:
    def _request_clear():
        st.session_state["do_clear_selection"] = True
        try: st.rerun()
        except: st.experimental_rerun()
    st.button("Clear selection", on_click=_request_clear)

# -------------------- 4.1 Results with filters + bulk actions --------------------
st.subheader("4.1 Results (investor view)")

def _cells_df():
    g = SS["grid"]
    if not g["cells"]: return pd.DataFrame()
    row_map = {r["id"]: (r.get("alias") or f'{r["type"]}:{r["source"]}') for r in g["rows"]}
    row_type = {r["id"]: r["type"] for r in g["rows"]}
    col_map = {c["id"]: c["name"] for c in g["columns"]}
    col_tool= {c["id"]: c["tool"] for c in g["columns"]}
    df = pd.DataFrame(g["cells"]).copy()
    df["Row"] = df["row_id"].map(row_map)
    df["row_type"] = df["row_id"].map(row_type)
    df["Column"] = df["col_id"].map(col_map)
    df["tool"] = df["col_id"].map(col_tool)
    def _val(row):
        v,u = row.get("numeric_value"), row.get("units")
        if v is None: return "-"
        return _fmt_pct(v) if u=="pct" else (f"{v:,.4f}" if abs(v) < 1 else f"{v:,.2f}")
    df["Value"] = df.apply(_val, axis=1)
    df["Summary"] = df.get("output_text", "").astype(str).str.slice(0, 160)
    return df

df_all = _cells_df()
if df_all.empty:
    st.info("No cells yet. Add rows & columns, then run.")
else:
    statuses = sorted(df_all["status"].dropna().unique().tolist())
    row_types = sorted(df_all["row_type"].dropna().unique().tolist())
    col_names = sorted(df_all["Column"].dropna().unique().tolist())

    f1,f2,f3,f4 = st.columns([0.32,0.24,0.30,0.14])
    with f1:
        sel_status = st.multiselect("Filter by status", options=statuses, default=statuses, key="flt_status")
    with f2:
        sel_rowtype = st.multiselect("Row type", options=row_types, default=row_types, key="flt_rtype")
    with f3:
        sel_colname = st.multiselect("Columns", options=col_names, default=col_names, key="flt_cols")
    with f4:
        st.caption("")

    df_view = df_all[
        df_all["status"].isin(sel_status)
        & df_all["row_type"].isin(sel_rowtype)
        & df_all["Column"].isin(sel_colname)
    ].copy()

    st.dataframe(df_view[["Row","row_type","Column","status","Value","Summary"]], hide_index=True, use_container_width=True)

    bA,bB,bC = st.columns([0.25,0.25,0.50])
    with bA:
        if st.button("Approve all in view"):
            ids = set(df_view["id"].tolist())
            for c in SS["grid"]["cells"]:
                if c["id"] in ids: c["status"]="approved"
            _log("BULK_APPROVE", f"{len(ids)}"); st.success("Approved.")
    with bB:
        if st.button("Mark needs-review in view"):
            ids = set(df_view["id"].tolist())
            for c in SS["grid"]["cells"]:
                if c["id"] in ids: c["status"]="needs_review"
            _log("BULK_REVIEW", f"{len(ids)}"); st.warning("Marked needs-review.")
    with bC:
        csv_bytes = df_view.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export results (filtered CSV)", data=csv_bytes, file_name="grid_results_filtered.csv", mime="text/csv")

    # ---------- Quick manage inline ----------
    with st.expander("Quick manage (add/delete without leaving this page)", expanded=False):
        row_alias = lambda rid: next((r.get("alias") or r["row_ref"] for r in SS["grid"]["rows"] if r["id"]==rid), rid)
        col_name  = lambda cid: next((c["name"] for c in SS["grid"]["columns"] if c["id"]==cid), cid)
        st.markdown("**Add rows**")
        existing_table_sources = {r["source"] for r in SS["grid"]["rows"] if r["type"]=="table"}
        existing_pdf_sources   = {r["source"] for r in SS["grid"]["rows"] if r["type"]=="pdf"}
        add_tables = st.multiselect("From CSVs", options=[n for n in SS["tables"].keys() if n not in existing_table_sources])
        add_pdfs   = st.multiselect("From PDFs", options=[n for n in SS["docs"].keys() if n not in existing_pdf_sources])
        if st.button("Add selected rows"):
            for n in add_tables: _add_row_from_table(n)
            for n in add_pdfs:   _add_row_from_pdf(n)
            _ensure_cells(); st.experimental_rerun()
        st.markdown("---")
        st.markdown("**Delete rows**")
        del_rows = st.multiselect("Pick rows to delete", options=[r["id"] for r in SS["grid"]["rows"]], format_func=row_alias, key="inline_del_rows")
        if st.button("Delete selected row(s)"):
            for rid in del_rows: delete_row(rid)
            _ensure_cells(); st.experimental_rerun()
        st.markdown("**Delete columns**")
        del_cols = st.multiselect("Pick columns to delete", options=[c["id"] for c in SS["grid"]["columns"]], format_func=col_name, key="inline_del_cols")
        if st.button("Delete selected column(s)"):
            for cid in del_cols: delete_col(cid)
            _ensure_cells(); st.experimental_rerun()

# -------------------- 5) Review --------------------
st.subheader("5) Review")
if SS["grid"]["cells"]:
    sel_cell_id = st.selectbox("Choose a cell", options=[c["id"] for c in SS["grid"]["cells"]], index=0)
    cell = next(c for c in SS["grid"]["cells"] if c["id"]==sel_cell_id)
    col  = next(x for x in SS["grid"]["columns"] if x["id"]==cell["col_id"])
    row  = next(x for x in SS["grid"]["rows"] if x["id"]==cell["row_id"])
    st.markdown(f"**{col['name']}** on _{row.get('alias') or row['row_ref']}_ — status: `{cell['status']}`")

    tabs = st.tabs(["Chart 1", "Chart 2", "Details"])
    with tabs[0]:
        if cell.get("figure") is not None: st.plotly_chart(cell["figure"], use_container_width=True)
        else: st.info("No chart available.")
    with tabs[1]:
        if cell.get("figure2") is not None: st.plotly_chart(cell["figure2"], use_container_width=True)
        else: st.info("No second chart.")
    with tabs[2]:
        if cell.get("output_text"): st.write(cell["output_text"])
        with st.expander("KPIs"): st.json(cell.get("kpis", {}))
        with st.expander("Citations"): st.json(cell.get("citations", []))

    if next((True for cc in SS["grid"]["columns"] if cc["id"]==cell["col_id"] and cc["tool"]=="pricing_power"), False):
        st.markdown("### 💡 Pricing Uplift Simulator")
        eps = cell.get("kpis", {}).get("elasticity")
        if isinstance(eps,(int,float)):
            pct = st.slider("Proposed average price change (%)", min_value=-30, max_value=30, value=5, step=1)
            dp = pct/100.0; rev_change = (1+dp)**(1+eps) - 1
            st.write(f"Estimated revenue change: **{_fmt_pct(rev_change)}** given ε≈{eps:.2f}")
        else:
            st.info("Run Pricing Power to compute elasticity first.")

    if next((True for cc in SS["grid"]["columns"] if cc["id"]==cell["col_id"] and cc["tool"]=="nrr_grr"), False):
        k = cell.get("kpis", {})
        if k:
            st.markdown(f"**Latest month** · GRR {_fmt_pct(k.get('grr'))}, NRR {_fmt_pct(k.get('nrr'))} · "
                        f"Expansion {_fmt_pct(k.get('expansion_rate'))}, Contraction {_fmt_pct(k.get('contraction_rate'))}, Churn {_fmt_pct(k.get('churn_rate'))}")

    a1,a2 = st.columns(2)
    with a1:
        if st.button("Approve"): cell["status"]="approved"; _log("CELL_APPROVE", sel_cell_id); st.success("Approved.")
    with a2:
        if st.button("Mark Needs-Review"): cell["status"]="needs_review"; _log("CELL_MARK_REVIEW", sel_cell_id); st.warning("Marked.")
else:
    st.info("No cells to review yet.")

# -------------------- 6) Memo & Export --------------------
st.subheader("6) Compose Memo & Export (with cross-checks)")

def _csv_revenue_total() -> Optional[float]:
    total = 0.0; seen = False
    for name, df in SS["tables"].items():
        rev_col = SS["schema_map"].get(name, {}).get("revenue")
        if rev_col and rev_col in df.columns:
            try: total += float(pd.to_numeric(df[rev_col], errors="coerce").fillna(0).sum()); seen=True
            except Exception: pass
    return total if seen else None

def _nrr_last_churn() -> Optional[float]:
    cells = [c for c in SS["grid"]["cells"] if c.get("status")=="approved"]
    nrr_cells = [c for c in cells if next((col for col in SS["grid"]["columns"] if col["id"]==c["col_id"] and col["tool"]=="nrr_grr"), None)]
    if not nrr_cells: return None
    k = nrr_cells[-1].get("kpis", {})
    return k.get("churn_rate")

def _first_pdf_kpis() -> Dict[str, Any]:
    for c in SS["grid"]["cells"]:
        if c.get("status")=="approved":
            col = next((x for x in SS["grid"]["columns"] if x["id"]==c["col_id"]), None)
            row = next((x for x in SS["grid"]["rows"] if x["id"]==c["row_id"]), None)
            if col and row and col["tool"]=="pdf_kpi_extract" and row["type"]=="pdf":
                return c.get("kpis", {})
    return {}

def _cross_checks() -> List[Tuple[str,str]]:
    checks = []
    pdf_kpis = _first_pdf_kpis()
    pdf_rev = pdf_kpis.get("revenue")
    csv_rev = _csv_revenue_total()
    if pdf_rev is not None and csv_rev is not None and csv_rev > 0:
        rel_diff = abs(pdf_rev - csv_rev) / max(pdf_rev, csv_rev)
        status = "✅ Revenue matches (≤10% delta)" if rel_diff<=0.10 else ("🟡 Revenue close (10–25% delta)" if rel_diff<=0.25 else "🔴 Revenue mismatch (>25% delta)")
        checks.append(("Revenue", f"{status}: PDF {_fmt_money(pdf_rev)} vs CSV {_fmt_money(csv_rev)}"))
    elif pdf_rev is not None:
        checks.append(("Revenue", f"ℹ️ PDF {_fmt_money(pdf_rev)} (no CSV revenue to compare)"))
    elif csv_rev is not None:
        checks.append(("Revenue", f"ℹ️ CSV {_fmt_money(csv_rev)} (no PDF revenue to compare)"))
    pdf_churn = pdf_kpis.get("churn"); csv_churn = _nrr_last_churn()
    if (pdf_churn is not None) and (csv_churn is not None):
        rel_diff = abs(pdf_churn - csv_churn) / max(pdf_churn, csv_churn, 1e-9)
        status = "✅ Churn matches (≤10% delta)" if rel_diff<=0.10 else ("🟡 Churn close (10–25% delta)" if rel_diff<=0.25 else "🔴 Churn mismatch (>25% delta)")
        checks.append(("Churn", f"{status}: PDF {_fmt_pct(pdf_churn)} vs CSV {_fmt_pct(csv_churn)}"))
    elif pdf_churn is not None:
        checks.append(("Churn", f"ℹ️ PDF {_fmt_pct(pdf_churn)} (no CSV churn to compare)"))
    elif csv_churn is not None:
        checks.append(("Churn", f"ℹ️ CSV {_fmt_pct(csv_churn)} (no PDF churn to compare)"))
    return checks

def export_memo_pdf_friendly(grid: Dict[str,Any], cells: List[Dict[str,Any]]) -> bytes:
    rows = {r["id"]: r for r in grid["rows"]}
    cols = {c["id"]: c for c in grid["columns"]}

    # memo column picker (defaults to all columns)
    selected_cols = SS.get("memo_cols")
    approved = [c for c in cells if c.get("status") == "approved" and (selected_cols is None or c["col_id"] in selected_cols)]

    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=LETTER)
    W,H = LETTER; M = 0.75*inch; y = H - M
    def line(txt, size=11, bold=False, leading=14):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        for seg in textwrap.wrap(str(txt), width=95) or [" "]:
            if y < M: c.showPage(); y = H - M; c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
            c.drawString(M, y, seg); y -= leading

    line("Transform AI — Investment Memo", 14, True, 16); line(f"Grid: {grid.get('id','unknown')}", 9); y -= 6

    checks = _cross_checks()
    if checks:
        line("Cross-checks (PDF vs CSV)", 12, True)
        for label,msg in checks: line(f"• {label}: {msg}", 10, leading=12)
        y -= 4

    line("Executive Summary", 12, True)
    if not approved: line("No approved findings yet.", 11)
    for ccell in approved:
        r = rows[ccell["row_id"]]; row_label = r.get("alias") or f"{r['type']}:{r['source']}"
        col_title = cols[ccell["col_id"]]["name"]
        val = ccell.get("numeric_value"); u = ccell.get("units")
        val_str = _fmt_pct(val) if (u=="pct" and isinstance(val,(int,float))) else (f"{val:,.2f}" if isinstance(val,(int,float)) else "")
        line(f"• {col_title} on {row_label}" + (f" — value: {val_str}" if val_str else ""), 11)
        if ccell.get("output_text"): line(f"   {ccell['output_text']}", 10, leading=12)
        y -= 4

    y -= 4; line("Evidence Appendix", 12, True)
    for ccell in approved:
        r = rows[ccell["row_id"]]; row_label = r.get("alias") or f"{r['type']}:{r['source']}"
        col_title = cols[ccell["col_id"]]["name"]
        citations = ccell.get("citations") or []
        if not citations: line(f"• {col_title} on {row_label}: (no citations captured)", 10); continue
        line(f"• {col_title} on {row_label}:", 10)
        for cit in citations[:6]:
            if cit.get("type")=="pdf":
                page = cit.get("page","?"); snip = (cit.get("excerpt","") or "").replace("\n"," ")
                line(f"   - PDF p.{page}: “{snip[:120]}”", 9)
            elif cit.get("type")=="table":
                sel = cit.get("selector",""); line(f"   - CSV selection: {sel}", 9)
            else:
                line(f"   - {cit}", 9)

    c.showPage(); c.save(); buf.seek(0); return buf.getvalue()

left,right = st.columns([1,1])

# Column picker UI
all_cols = SS["grid"]["columns"]
with left:
    st.markdown("**Memo columns**")
    memo_cols = st.multiselect(
        "Choose which columns feed the memo (default: all).",
        options=[c["id"] for c in all_cols],
        default=[c["id"] for c in all_cols] if SS["memo_cols"] is None else SS["memo_cols"],
        format_func=lambda cid: next((c["name"] for c in all_cols if c["id"]==cid), cid),
        key="memo_cols_picker"
    )
    if st.button("Apply memo columns"):
        SS["memo_cols"] = memo_cols
        st.success("Memo columns updated.")

with right:
    pdf_bytes = export_memo_pdf_friendly(SS["grid"], SS["grid"]["cells"])
    st.download_button("📄 Download Investor Memo (clean)", data=pdf_bytes,
                       file_name=f"TransformAI_Memo_{SS['grid']['id']}.pdf", mime="application/pdf")

# Compose markdown preview (optional)
if st.button("Compose (show markdown)"):
    approved = [c for c in SS["grid"]["cells"] if c.get("status")=="approved"
                and (SS["memo_cols"] is None or c["col_id"] in SS["memo_cols"])]
    lines = [f"# Investment Memo — {SS['grid']['id']}", ""]
    checks = _cross_checks()
    if checks:
        lines.append("## Cross-checks (PDF vs CSV)")
        for label,msg in checks: lines.append(f"- **{label}**: {msg}")
        lines.append("")
    lines.append("## Executive Summary")
    for ccell in approved:
        col = next(x for x in SS["grid"]["columns"] if x["id"]==ccell["col_id"])
        row = next(x for x in SS["grid"]["rows"] if x["id"]==ccell["row_id"])
        val = ccell.get("numeric_value"); u = ccell.get("units")
        val_s = _fmt_pct(val) if (u=="pct" and isinstance(val,(int,float))) else (f"{val:,.2f}" if isinstance(val,(int,float)) else "")
        lines.append(f"- **{col['name']}** on _{row.get('alias') or row['row_ref']}_ → {val_s} — {ccell.get('output_text','')}")
    SS["last_memo_md"] = "\n".join(lines) or "# (no approved findings)"
    st.code(SS["last_memo_md"], language="markdown")

# -------------------- 7) Evidence Chat (beta) --------------------
st.subheader("7) Evidence Chat (beta)")
scope = st.radio("Search scope", options=["All","PDFs","CSVs"], horizontal=True)

def _score(text: str, q: str) -> int:
    if not text: return 0
    score = 0; q_terms = [t for t in re.findall(r"\w+", q.lower()) if len(t) > 2]; t_lower = text.lower()
    for t in q_terms: score += t_lower.count(t)
    return score

def search_pdfs(q: str, topk: int = 3):
    results = []
    if scope in ("All","PDFs"):
        for name, pages in SS.get("docs", {}).items():
            for i, ptxt in enumerate(pages):
                s = _score(ptxt, q)
                if s > 0:
                    idx = ptxt.lower().find(q.lower()); idx = max(idx, 0)
                    snippet = (ptxt[max(0, idx-160): idx+160] or "").replace("\n", " ")
                    results.append({"kind":"pdf","source":name,"page":i+1,"score":s,"snippet":snippet.strip()})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:topk]

def search_csvs(q: str, topk: int = 3, sample_rows: int = 5):
    results = []
    if scope in ("All","CSVs"):
        for name, df in SS.get("tables", {}).items():
            s_cols = sum(1 for c in df.columns if q.lower() in c.lower())
            s_cells = 0
            try:
                sample = df.head(300)
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
            row = next((x for x in SS["grid"]["rows"] if x["id"]==c["row_id"]), {"row_ref":"(unknown)","alias":"(unknown)"})
            hits.append({"kind":"cell","col_name": col["name"],"row_ref": row.get("alias") or row["row_ref"],
                         "status": c.get("status"),"output_text": c.get("output_text"),"citations": c.get("citations", [])})
    return hits[:3]

for role, content in SS["chat_history"]:
    with st.chat_message(role): st.markdown(content)

prompt = st.chat_input("Ask about your evidence (e.g., 'show EBITDA', 'pages about churn', 'which CSV has price?' )")
if prompt:
    SS["chat_history"].append(("user", prompt)); with st.chat_message("user"): st.markdown(prompt)
    pdf_hits = search_pdfs(prompt, topk=3); csv_hits = search_csvs(prompt, topk=3); grid_hits = answers_from_grid(prompt)
    parts = []
    if grid_hits:
        parts.append("**Existing answers (from grid):**")
        for h in grid_hits:
            parts.append(f"- _{h['row_ref']}_ • **{h['col_name']}** → {h['output_text']}")
            if h["citations"]: parts.append(f"  · Citations: {len(h['citations'])}")
    if pdf_hits:
        parts.append("**PDF matches:**")
        for h in pdf_hits: parts.append(f"- `{h['source']}` — p.{h['page']} · “{h['snippet']}”")
    if csv_hits:
        parts.append("**CSV matches:**")
        for h in csv_hits:
            row_count = h["preview"].shape[0] if isinstance(h["preview"], pd.DataFrame) else 0
            parts.append(f"- `{h['source']}` — column/row matches (showing {row_count} rows below).")
    if not (grid_hits or pdf_hits or csv_hits):
        parts.append("_No obvious matches. Try a different term (e.g., 'revenue', 'margin', 'price')._")

    answer_md = "\n".join(parts) or "No results."
    SS["chat_history"].append(("assistant", answer_md))
    with st.chat_message("assistant"):
        st.markdown(answer_md)
        for h in csv_hits:
            if isinstance(h["preview"], pd.DataFrame) and not h["preview"].empty:
                st.dataframe(h["preview"], use_container_width=True)

with st.expander("Activity Log (debug)", expanded=False):
    st.json(SS["grid"]["activities"])
