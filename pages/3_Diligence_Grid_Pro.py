# pages/3_Diligence_Grid_Pro.py
# TransformAI — Diligence Grid (Pro, Tabbed UI) + Drive/Box/SharePoint connector stub
from __future__ import annotations
import io, uuid, re, textwrap, hashlib, json, copy
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

st.set_page_config(page_title="TransformAI — Diligence (Pro)", layout="wide")

# ---------- session ----------
SS = st.session_state
SS.setdefault("tables", {})           # name -> DataFrame
SS.setdefault("docs", {})             # name -> List[str] (page texts)
SS.setdefault("schema_map", {})       # table_name -> mapping dict
SS.setdefault("cache", {})            # cell_id -> {"sig":..., "result": ModuleResult}
SS.setdefault("views", {})            # saved result filters
SS.setdefault("memo_cols", None)      # selected column IDs used in memo (None=all)
SS.setdefault("row_run_selection", [])# from manager
SS.setdefault("history", []); SS.setdefault("history_idx", -1)

SS.setdefault("remote_sources", [])   # connector stub list of {"provider","hint","status","kind","name"}
SS.setdefault("grid", {
    "id": f"grid_{uuid.uuid4().hex[:6]}",
    "rows": [],         # {id,row_ref,source,type,alias}
    "columns": [],      # {id,name,tool,params}
    "cells": [],        # {id,row_id,col_id,status,output_text,numeric_value,units,kpis,citations,figure,figure2}
    "activities": []
})

# ---------- small utils ----------
def _log(action:str, detail:str=""):
    SS["grid"]["activities"].append({"id": uuid.uuid4().hex, "action": action, "detail": detail})

def _new_id(p): return f"{p}_{uuid.uuid4().hex[:8]}"

def _snapshot():
    snap = {"grid": copy.deepcopy(SS["grid"]), "schema_map": copy.deepcopy(SS["schema_map"])}
    if SS["history_idx"] < len(SS["history"]) - 1:
        SS["history"] = SS["history"][:SS["history_idx"]+1]
    SS["history"].append(snap); SS["history_idx"] += 1

def _undo():
    if SS["history_idx"] <= 0: return False
    SS["history_idx"] -= 1
    snap = SS["history"][SS["history_idx"]]
    SS["grid"] = copy.deepcopy(snap["grid"]); SS["schema_map"] = copy.deepcopy(snap["schema_map"]); SS["cache"].clear()
    return True

def _redo():
    if SS["history_idx"] >= len(SS["history"]) - 1: return False
    SS["history_idx"] += 1
    snap = SS["history"][SS["history_idx"]]
    SS["grid"] = copy.deepcopy(snap["grid"]); SS["schema_map"] = copy.deepcopy(snap["schema_map"]); SS["cache"].clear()
    return True

def _fmt_money(x):
    if x is None or not isinstance(x,(int,float)) or x!=x: return "-"
    if abs(x) >= 1_000_000_000: return f"${x/1_000_000_000:,.2f}B"
    if abs(x) >= 1_000_000:     return f"${x/1_000_000:,.2f}M"
    return f"${x:,.2f}"

def _fmt_pct(x):
    if x is None or not isinstance(x,(int,float)) or x!=x: return "-"
    return f"{x*100:.1f}%"

# ---------- grid helpers ----------
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

def _prune_incompatible_cells() -> int:
    rows_by_id = {r["id"]: r for r in SS["grid"]["rows"]}
    cols_by_id = {c["id"]: c for c in SS["grid"]["columns"]}
    kept, dropped = [], 0
    for cell in SS["grid"]["cells"]:
        r = rows_by_id.get(cell["row_id"]); c = cols_by_id.get(cell["col_id"])
        if not r or not c: dropped += 1; continue
        applies = MODULES.get(c["tool"], {}).get("applies_to", ["table","pdf"])
        if r["type"] in applies: kept.append(cell)
        else: dropped += 1
    if len(kept) != len(SS["grid"]["cells"]): SS["grid"]["cells"] = kept
    return dropped

def _ensure_cells():
    _prune_incompatible_cells()
    existing = {(c["row_id"], c["col_id"]) for c in SS["grid"]["cells"]}
    for r in SS["grid"]["rows"]:
        for c in SS["grid"]["columns"]:
            applies = MODULES.get(c["tool"], {}).get("applies_to", ["table","pdf"])
            if r["type"] not in applies: continue
            if (r["id"], c["id"]) not in existing:
                SS["grid"]["cells"].append({
                    "id": _new_id("cell"), "row_id": r["id"], "col_id": c["id"],
                    "status": "queued", "output_text": None, "numeric_value": None,
                    "units": None, "kpis": {}, "citations": [], "figure": None, "figure2": None
                })

# ---------- module helpers ----------
@dataclass
class ModuleResult:
    kpis: Dict[str, Any]
    narrative: str
    citations: List[Dict[str, Any]]
    figure: Optional[Any] = None
    units_hint: Optional[str] = None
    figure2: Optional[Any] = None

def _find(df: pd.DataFrame, key: str) -> Optional[str]:
    bank = {
        "customer":["customer","user","buyer","account","client","cust","customer_id"],
        "date":["date","order_date","created_at","timestamp","period","month"],
        "revenue":["revenue","amount","sales","net_revenue","gmv","value"],
        "price":["price","unit_price","avg_price","p"],
        "quantity":["qty","quantity","units","volume","q"],
        "segment":["segment","sku","product","category","plan","region","cohort","family"],
        # Unit Econ additions
        "cogs":["cogs","cost_of_goods","cos","cost"],
        "variable_cost":["variable_cost","var_cost","transaction_fees","shipping","processing_fees","fulfillment"],
        "fixed_cost":["fixed_cost","overhead","rent","payroll","opex","sga","sg&a"],
        "marketing_spend":["marketing_spend","marketing","ad_spend","paid","s&m","sales_marketing"],
        "new_customers":["new_customers","acquisitions","new_logos","signups"],
        "active_customers":["active_customers","active_users","active_accounts"],
    }.get(key.lower(), [key.lower()])
    cols = list(df.columns); lower = {c.lower(): c for c in cols}
    for needle in bank:
        if needle in lower: return lower[needle]
    for needle in bank:
        for c in cols:
            if needle in c.lower(): return c
    return None

# ---------- modules (same logic you had, trimmed for space) ----------
def module_cohort_retention(df: pd.DataFrame,
                            customer_col: Optional[str]=None,
                            ts_col: Optional[str]=None,
                            revenue_col: Optional[str]=None) -> ModuleResult:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ModuleResult({}, "Empty dataset.", [], units_hint="pct")
    customer_col = customer_col or _find(df,"customer")
    ts_col       = ts_col or _find(df,"date")
    revenue_col  = revenue_col or _find(df,"revenue")
    if not (customer_col and ts_col):
        return ModuleResult({}, "Missing customer/date; map schema first.", [], units_hint="pct")
    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce")
    d = d.dropna(subset=[ts_col, customer_col]).sort_values(ts_col)
    d["first_month"] = d.groupby(customer_col)[ts_col].transform("min").dt.to_period("M")
    d["age"] = (d[ts_col].dt.to_period("M") - d["first_month"]).apply(lambda p: p.n)
    cohort_sizes = d.drop_duplicates([customer_col,"first_month"]).groupby("first_month")[customer_col].count()
    active = d.groupby(["first_month","age"])[customer_col].nunique()
    mat = (active / cohort_sizes).unstack(fill_value=0).sort_index()
    curve = mat.mean(axis=0) if not mat.empty else pd.Series(dtype=float)
    m3 = float(round(curve.get(3, np.nan), 4)) if not curve.empty else np.nan
    ltv_12=None
    if revenue_col and revenue_col in d.columns:
        rev = d.groupby([customer_col, d[ts_col].dt.to_period("M")])[revenue_col].sum().groupby(customer_col).sum()
        ltv_12 = float(round(float(rev.mean()),2))
    fig_curve = px.line(x=list(curve.index), y=list(curve.values),
                        labels={"x":"Months since first purchase","y":"Retention"},
                        title="Average Retention Curve")
    narrative = f"Retention stabilizes ~M3 at {m3:.0%}." if m3==m3 else "Not enough data to compute M3 retention."
    if ltv_12: narrative += f" Avg 12-month LTV proxy ≈ {_fmt_money(ltv_12)}."
    return ModuleResult(
        kpis={"month_3_retention": m3, "ltv_12m": ltv_12},
        narrative=narrative,
        citations=[{"type":"table","ref":"(uploaded CSV)","selector":"all_rows"}],
        figure=fig_curve,
        units_hint="pct"
    )

def _ols_loglog(x, y):
    X = np.log(x); Y = np.log(y)
    A = np.vstack([X, np.ones(len(X))]).T
    beta, intercept = np.linalg.lstsq(A, Y, rcond=None)[0]
    yhat = beta*X + intercept
    ss_res = float(np.sum((Y-yhat)**2)); ss_tot = float(np.sum((Y-np.mean(Y))**2))
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else None
    return float(beta), float(intercept), r2

def module_pricing_power(df: pd.DataFrame,
                         price_col: Optional[str]=None,
                         qty_col: Optional[str]=None) -> ModuleResult:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ModuleResult({}, "Empty dataset.", [])
    price_col = price_col or _find(df,"price")
    qty_col   = qty_col or _find(df,"quantity")
    if not (price_col and qty_col):
        return ModuleResult({}, "Missing price/quantity; map schema first.", [])
    d = df.dropna(subset=[price_col, qty_col]).copy()
    d = d[(d[price_col] > 0) & (d[qty_col] > 0)]
    if len(d) < 8:
        return ModuleResult({}, "Need ≥ 8 observations for elasticity.", [])
    beta, intercept, r2 = _ols_loglog(d[price_col].values, d[qty_col].values)
    fig = px.scatter(d, x=price_col, y=qty_col, title="Price vs Quantity")
    xs = np.linspace(float(d[price_col].min()), float(d[price_col].max()), 60)
    ys = np.exp(beta*np.log(xs) + intercept)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="fit"))
    narrative = f"Own-price elasticity ≈ {beta:.2f} (R²={r2:.2f}). "
    narrative += "Inelastic (|ε|<1)." if abs(beta) < 1 else "Elastic (|ε|≥1)."
    return ModuleResult(kpis={"elasticity":float(beta),"r2":r2}, narrative=narrative,
                        citations=[{"type":"table","ref":"(uploaded CSV)","selector":"price/qty"}],
                        figure=fig)

def module_nrr_grr(df: pd.DataFrame,
                   customer_col: Optional[str]=None,
                   ts_col: Optional[str]=None,
                   revenue_col: Optional[str]=None) -> ModuleResult:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ModuleResult({}, "Empty dataset.", [], units_hint="pct")
    customer_col = customer_col or _find(df,"customer")
    ts_col       = ts_col or _find(df,"date")
    revenue_col  = revenue_col or _find(df,"revenue")
    if not (customer_col and ts_col and revenue_col):
        return ModuleResult({}, "Need customer/date/revenue; map schema first.", [], units_hint="pct")
    d = df[[customer_col, ts_col, revenue_col]].copy()
    d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce")
    d = d.dropna(subset=[customer_col, ts_col, revenue_col])
    d["month"] = d[ts_col].dt.to_period("M")
    gp = d.groupby([customer_col,"month"], as_index=False)[revenue_col].sum()
    pivot = gp.pivot(index=customer_col, columns="month", values=revenue_col).fillna(0.0).sort_index(axis=1)
    months = list(pivot.columns)
    if len(months) < 2: return ModuleResult({}, "Need at least two months.", [], units_hint="pct")
    labels, grr_list, nrr_list, churn_rate, contraction_rate, expansion_rate = [], [], [], [], [], []
    last_pair = None
    for i in range(1,len(months)):
        prev_m, curr_m = months[i-1], months[i]
        prev_rev, curr_rev = pivot[prev_m], pivot[curr_m]
        mask = prev_rev > 0
        start = float(prev_rev[mask].sum())
        if start <= 0: continue
        curr_base = curr_rev[mask]
        churn_amt = float(prev_rev[mask & (curr_base == 0)].sum())
        contr_amt = float(((prev_rev[mask] - curr_base).clip(lower=0.0).sum()) - churn_amt); contr_amt = max(contr_amt, 0.0)
        exp_amt = float((curr_base - prev_rev[mask]).clip(lower=0.0).sum())
        grr = (start - churn_amt - contr_amt)/start if start else np.nan
        nrr = (start - churn_amt - contr_amt + exp_amt)/start if start else np.nan
        labels.append(str(curr_m)); grr_list.append(grr); nrr_list.append(nrr)
        churn_rate.append(churn_amt/start); contraction_rate.append(contr_amt/start); expansion_rate.append(exp_amt/start)
        last_pair = (str(prev_m), str(curr_m), start, churn_amt, contr_amt, exp_amt)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=nrr_list, mode="lines+markers", name="NRR"))
    fig.add_trace(go.Scatter(x=labels, y=grr_list, mode="lines+markers", name="GRR"))
    ymax = float(np.nanmax(nrr_list+grr_list)) if (nrr_list or grr_list) else 1.0
    fig.update_layout(title="Monthly NRR & GRR", yaxis=dict(range=[0, max(1.2, ymax)]))
    last_label = labels[-1]
    k = {
        "month": last_label,
        "grr": float(round(grr_list[-1],4)),
        "nrr": float(round(nrr_list[-1],4)),
        "churn_rate": float(round(churn_rate[-1],4)),
        "contraction_rate": float(round(contraction_rate[-1],4)),
        "expansion_rate": float(round(expansion_rate[-1],4)),
    }
    narrative = (f"Latest ({last_label}): GRR {k['grr']:.0%}, NRR {k['nrr']:.0%} "
                 f"(expansion {k['expansion_rate']:.0%}, contraction {k['contraction_rate']:.0%}, churn {k['churn_rate']:.0%}).")
    return ModuleResult(kpis=k, narrative=narrative,
                        citations=[{"type":"table","ref":"(uploaded CSV)","selector":"monthly revenue by customer"}],
                        figure=fig, units_hint="pct")

# ---- PDF KPI extraction (simple keywords) ----
_money = re.compile(r"\$?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:billion|bn|million|m)?", re.I)
_pct   = re.compile(r"\d{1,3}(?:\.\d+)?\s*%")

def _parse_money(tok):
    if not tok: return None
    t = tok.lower().replace("$","").replace(" ",""); mult=1.0
    if "billion" in t or "bn" in t: mult=1_000_000_000
    elif "million" in t or t.endswith("m"): mult=1_000_000
    t = re.sub(r"[a-z]", "", t)
    try: return float(t.replace(",",""))*mult
    except: return None

def _parse_pct(tok):
    if not tok: return None
    t = tok.replace("%","").strip()
    try: return float(t)/100.0
    except: return None

def _scan_metric(pages, keywords, want):
    for i, page in enumerate(pages):
        txt = page or ""; low = txt.lower()
        for kw in keywords:
            for m in re.finditer(re.escape(kw.lower()), low):
                start = max(0, m.start()-80); end = min(len(txt), m.end()+80)
                win = txt[start:end]
                n = _money.search(win) if want=="money" else _pct.search(win)
                if n:
                    val = _parse_money(n.group()) if want=="money" else _parse_pct(n.group())
                    if val is not None:
                        return {"page": i+1, "snippet": win.strip(), "raw": n.group(), "value": val}
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
    narrative = " ".join(parts) or "No obvious KPIs found."
    cits = []
    for _,v in found.items():
        if v: cits.append({"type":"pdf","page":v["page"],"excerpt":(v["snippet"] or "")[:220]})
    return ModuleResult(kpis=kpis, narrative=narrative, citations=cits)

# ---- Unit Economics (CSV) ----
def module_unit_economics(df: pd.DataFrame,
                          ts_col=None, revenue_col=None, cogs_col=None, var_col=None,
                          fixed_col=None, mkt_col=None, new_cust_col=None,
                          customer_col=None, active_col=None) -> ModuleResult:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ModuleResult({}, "Empty dataset.", [])
    ts_col       = ts_col       or _find(df,"date")
    revenue_col  = revenue_col  or _find(df,"revenue")
    cogs_col     = cogs_col     or _find(df,"cogs")
    var_col      = var_col      or _find(df,"variable_cost")
    fixed_col    = fixed_col    or _find(df,"fixed_cost")
    mkt_col      = mkt_col      or _find(df,"marketing_spend")
    new_cust_col = new_cust_col or _find(df,"new_customers")
    customer_col = customer_col or _find(df,"customer")
    active_col   = active_col   or _find(df,"active_customers")
    if not (ts_col and revenue_col):
        return ModuleResult({}, "Need date + revenue columns; map schema first.", [])
    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce")
    d = d.dropna(subset=[ts_col, revenue_col]); d["month"] = d[ts_col].dt.to_period("M")
    agg = d.groupby("month", as_index=False)[revenue_col].sum().rename(columns={revenue_col:"revenue"})
    for col,name in [(cogs_col,"cogs"),(var_col,"variable_cost"),(fixed_col,"fixed_cost"),(mkt_col,"marketing_spend")]:
        if col and col in d.columns:
            s = d.groupby("month", as_index=False)[col].sum().rename(columns={col:name})
            agg = agg.merge(s, on="month", how="left")
    for k in ["cogs","variable_cost","fixed_cost","marketing_spend"]:
        if k not in agg.columns: agg[k]=0.0
    if active_col and active_col in d.columns:
        active = d.groupby("month", as_index=False)[active_col].sum().rename(columns={active_col:"active_customers"})
        agg = agg.merge(active, on="month", how="left")
    elif customer_col and customer_col in d.columns:
        act = d.groupby(["month"])[customer_col].nunique().reset_index().rename(columns={customer_col:"active_customers"})
        agg = agg.merge(act, on="month", how="left")
    else:
        agg["active_customers"]=np.nan
    if new_cust_col and new_cust_col in d.columns:
        newc = d.groupby("month", as_index=False)[new_cust_col].sum().rename(columns={new_cust_col:"new_customers"})
        agg = agg.merge(newc, on="month", how="left")
    elif customer_col and customer_col in d.columns:
        first = d.groupby(customer_col)[ts_col].min().dt.to_period("M")
        d_first = pd.DataFrame({customer_col:first.index, "first_month": first.values})
        newc = d_first.groupby("first_month")[customer_col].count().rename("new_customers").reset_index().rename(columns={"first_month":"month"})
        agg = agg.merge(newc, on="month", how="left")
    else:
        agg["new_customers"]=np.nan
    agg["cm1"] = agg["revenue"] - agg["cogs"]
    agg["cm2"] = agg["cm1"] - agg["variable_cost"]
    agg["ebitda_proxy"] = agg["cm2"] - agg["fixed_cost"]
    agg["gm_pct"] = np.where(agg["revenue"]>0, agg["cm1"]/agg["revenue"], np.nan)
    agg["cm2_pct"] = np.where(agg["revenue"]>0, agg["cm2"]/agg["revenue"], np.nan)
    agg["arpu"] = np.where(agg["active_customers"]>0, agg["revenue"]/agg["active_customers"], np.nan)
    agg["cac"] = np.where(agg["new_customers"]>0, agg["marketing_spend"]/agg["new_customers"], np.nan)
    agg["ltv_proxy"] = 12 * agg["arpu"] * agg["gm_pct"]
    agg["payback_months"] = np.where((agg["arpu"]*agg["gm_pct"])>0, agg["cac"]/(agg["arpu"]*agg["gm_pct"]), np.nan)
    agg = agg.sort_values("month")
    latest = agg[agg["revenue"]>0].tail(1)
    if latest.empty: return ModuleResult({}, "No revenue rows to compute Unit Economics.", [])
    L = latest.iloc[0].to_dict(); mlabel = str(L["month"])
    fig_wf = go.Figure(go.Waterfall(
        name=f"Unit Econ {mlabel}", orientation="v",
        measure=["absolute","relative","relative","relative","total"],
        x=["Revenue","COGS","Variable","Fixed","EBITDA*"],
        y=[float(L["revenue"]), -float(L["cogs"]), -float(L["variable_cost"]), -float(L["fixed_cost"]), float(L["ebitda_proxy"])]
    ))
    fig_wf.update_layout(title=f"Unit Economics — {mlabel}", yaxis_title="USD")
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(x=[str(x) for x in agg["month"]], y=agg["gm_pct"], mode="lines+markers", name="CM1 margin"))
    fig_m.add_trace(go.Scatter(x=[str(x) for x in agg["month"]], y=agg["cm2_pct"], mode="lines+markers", name="CM2 margin"))
    fig_m.update_layout(title="Margins over time", yaxis=dict(tickformat=".0%"))
    kpis = {
        "month": mlabel,
        "cm1_margin": float(L["gm_pct"]) if L["gm_pct"]==L["gm_pct"] else None,
        "cm2_margin": float(L["cm2_pct"]) if L["cm2_pct"]==L["cm2_pct"] else None,
        "ebitda_proxy": float(L["ebitda_proxy"]),
        "arpu": float(L["arpu"]) if L["arpu"]==L["arpu"] else None,
        "cac": float(L["cac"]) if L["cac"]==L["cac"] else None,
        "ltv_proxy": float(L["ltv_proxy"]) if L["ltv_proxy"]==L["ltv_proxy"] else None,
        "payback_months": float(L["payback_months"]) if L["payback_months"]==L["payback_months"] else None,
        "revenue": float(L["revenue"])
    }
    msg = [f"{mlabel}: CM1 {_fmt_pct(kpis['cm1_margin']) if kpis['cm1_margin'] is not None else '-'};"
           f" CM2 {_fmt_pct(kpis['cm2_margin']) if kpis['cm2_margin'] is not None else '-'};"
           f" EBITDA* {_fmt_money(kpis['ebitda_proxy'])}."]
    if kpis["cac"] is not None: msg.append(f" CAC ~ {_fmt_money(kpis['cac'])}.")
    if kpis["ltv_proxy"] is not None: msg.append(f" LTV proxy ~ {_fmt_money(kpis['ltv_proxy'])}.")
    if kpis["payback_months"] is not None: msg.append(f" Payback ~ {kpis['payback_months']:.1f} mo.")
    narrative = " ".join(msg) + " (EBITDA* is a proxy; depends on mapping quality.)"
    return ModuleResult(kpis=kpis, narrative=narrative,
                        citations=[{"type":"table","ref":"(uploaded CSV)","selector":"revenue/costs/new_customers"}],
                        figure=fig_wf, figure2=fig_m)

MODULES = {
    "cohort_retention": {"title":"Cohort Retention (CSV)", "fn":module_cohort_retention, "applies_to":["table"]},
    "pricing_power":    {"title":"Pricing Power (CSV)",    "fn":module_pricing_power,    "applies_to":["table"]},
    "nrr_grr":          {"title":"NRR/GRR (CSV)",          "fn":module_nrr_grr,          "applies_to":["table"]},
    "unit_economics":   {"title":"Unit Economics (CSV)",   "fn":module_unit_economics,   "applies_to":["table"]},
    "pdf_kpi_extract":  {"title":"PDF KPI Extract",        "fn":module_pdf_kpi,          "applies_to":["pdf"]},
}

# ---------- page header ----------
st.markdown("### Transform AI — Diligence Grid (Pro)")
st.caption("**Quick path**: 1) Data → 2) Grid → 3) Run → 4) Review → 5) Memo.  •  Templates: **CDD** (customer DD) or **QoE (Quality of Earnings)**.")

# =======================================
#                TABS
# =======================================
tab_data, tab_grid, tab_run, tab_review, tab_memo = st.tabs(
    ["1) Data", "2) Grid", "3) Run", "4) Review", "5) Memo"]
)

# ------------ 1) DATA ------------
with tab_data:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Upload CSV(s)")
        up = st.file_uploader("Drag/drop or select CSV files", type=["csv"], accept_multiple_files=True, key="csv_up")
        if up:
            for f in up:
                try:
                    df = pd.read_csv(f)
                    SS["tables"][f.name] = df
                    SS["schema_map"].setdefault(f.name, {
                        "customer":None,"date":None,"revenue":None,"price":None,"quantity":None,
                        "cogs":None,"variable_cost":None,"fixed_cost":None,"marketing_spend":None,
                        "new_customers":None,"active_customers":None
                    })
                    _log("SOURCE_ADDED", f"csv:{f.name}")
                    st.success(f"Loaded CSV: {f.name} ({df.shape[0]:,} rows)")
                    _snapshot()
                except Exception as e:
                    st.error(f"{f.name}: {e}")
    with c2:
        st.subheader("Upload PDF(s)")
        up_pdf = st.file_uploader("Drag/drop or select PDF files", type=["pdf"], accept_multiple_files=True, key="pdf_up")
        if up_pdf and PdfReader is None:
            st.error("PDF parsing requires `pypdf` in requirements.txt (pypdf>=4).")
        if up_pdf and PdfReader is not None:
            for f in up_pdf:
                try:
                    reader = PdfReader(f); pages=[]
                    for p in reader.pages:
                        try: pages.append(p.extract_text() or "")
                        except Exception: pages.append("")
                    SS["docs"][f.name] = pages
                    _log("SOURCE_ADDED", f"pdf:{f.name}")
                    st.success(f"Loaded PDF: {f.name} ({len(pages)} pages)")
                    _snapshot()
                except Exception as e:
                    st.error(f"{f.name}: {e}")

    st.markdown("---")
    with st.expander("Map CSV Schema", expanded=False):
        def _auto_guess(df: pd.DataFrame) -> Dict[str, Optional[str]]:
            return {
                "customer": _find(df,"customer"), "date": _find(df,"date"),
                "revenue": _find(df,"revenue"), "price": _find(df,"price"),
                "quantity": _find(df,"quantity"), "cogs": _find(df,"cogs"),
                "variable_cost": _find(df,"variable_cost"), "fixed_cost": _find(df,"fixed_cost"),
                "marketing_spend": _find(df,"marketing_spend"), "new_customers": _find(df,"new_customers"),
                "active_customers": _find(df,"active_customers")
            }
        if st.button("Auto-map all tables"):
            for name, df in SS["tables"].items():
                SS["schema_map"][name] = _auto_guess(df)
            _snapshot(); st.success("Guessed schemas.")

        for name, df in SS["tables"].items():
            st.markdown(f"**{name}** — {df.shape[0]:,} rows × {df.shape[1]:,} cols")
            cols = list(df.columns)
            cur = SS["schema_map"].get(name) or _auto_guess(df)
            g1,g2,g3,g4,g5 = st.columns(5)
            with g1:
                customer = st.selectbox(f"{name}: Customer", ["(none)"]+cols, index=(cols.index(cur.get("customer"))+1) if cur.get("customer") in cols else 0, key=f"cust_{name}")
            with g2:
                date = st.selectbox(f"{name}: Date", ["(none)"]+cols, index=(cols.index(cur.get("date"))+1) if cur.get("date") in cols else 0, key=f"date_{name}")
            with g3:
                revenue = st.selectbox(f"{name}: Revenue", ["(none)"]+cols, index=(cols.index(cur.get("revenue"))+1) if cur.get("revenue") in cols else 0, key=f"rev_{name}")
            with g4:
                price = st.selectbox(f"{name}: Price", ["(none)"]+cols, index=(cols.index(cur.get("price"))+1) if cur.get("price") in cols else 0, key=f"price_{name}")
            with g5:
                qty = st.selectbox(f"{name}: Quantity", ["(none)"]+cols, index=(cols.index(cur.get("quantity"))+1) if cur.get("quantity") in cols else 0, key=f"qty_{name}")
            h1,h2,h3,h4,h5 = st.columns(5)
            with h1:
                cogs = st.selectbox(f"{name}: COGS", ["(none)"]+cols, index=(cols.index(cur.get("cogs"))+1) if cur.get("cogs") in cols else 0, key=f"cogs_{name}")
            with h2:
                varc = st.selectbox(f"{name}: Variable cost", ["(none)"]+cols, index=(cols.index(cur.get("variable_cost"))+1) if cur.get("variable_cost") in cols else 0, key=f"var_{name}")
            with h3:
                fixc = st.selectbox(f"{name}: Fixed cost", ["(none)"]+cols, index=(cols.index(cur.get("fixed_cost"))+1) if cur.get("fixed_cost") in cols else 0, key=f"fix_{name}")
            with h4:
                mkt = st.selectbox(f"{name}: Marketing spend", ["(none)"]+cols, index=(cols.index(cur.get("marketing_spend"))+1) if cur.get("marketing_spend") in cols else 0, key=f"mkt_{name}")
            with h5:
                newc = st.selectbox(f"{name}: New customers", ["(none)"]+cols, index=(cols.index(cur.get("new_customers"))+1) if cur.get("new_customers") in cols else 0, key=f"newc_{name}")
            j1,_ = st.columns([0.5,0.5])
            with j1:
                active = st.selectbox(f"{name}: Active customers", ["(none)"]+cols, index=(cols.index(cur.get("active_customers"))+1) if cur.get("active_customers") in cols else 0, key=f"active_{name}")
            if st.button(f"Save mapping for {name}", key=f"save_{name}"):
                SS["schema_map"][name] = {
                    "customer":None if customer=="(none)" else customer, "date":None if date=="(none)" else date,
                    "revenue":None if revenue=="(none)" else revenue, "price":None if price=="(none)" else price,
                    "quantity":None if qty=="(none)" else qty, "cogs":None if cogs=="(none)" else cogs,
                    "variable_cost":None if varc=="(none)" else varc, "fixed_cost":None if fixc=="(none)" else fixc,
                    "marketing_spend":None if mkt=="(none)" else mkt, "new_customers":None if newc=="(none)" else newc,
                    "active_customers":None if active=="(none)" else active
                }
                _snapshot(); st.success("Saved.")

    st.markdown("---")
    st.subheader("Drive / Box / SharePoint (Connector Stub)")
    st.caption("Paste a public CSV/PDF link or file ID; we’ll try to fetch it. If fetch fails, a placeholder is created (for demo UI only).")
    cc1, cc2, cc3 = st.columns([0.28, 0.52, 0.20])
    with cc1:
        provider = st.selectbox("Provider", ["Google Drive","Box","SharePoint","Generic URL"])
        kind = st.selectbox("Kind", ["CSV","PDF"])
    with cc2:
        hint = st.text_input("Share link or file ID")
    with cc3:
        if st.button("Add from connector"):
            name = f"{provider}_{kind}_{uuid.uuid4().hex[:6]}"
            status = "placeholder"
            if kind=="CSV":
                try:
                    df = pd.read_csv(hint)
                    SS["tables"][name] = df
                    SS["schema_map"].setdefault(name, {"customer":None,"date":None,"revenue":None,"price":None,"quantity":None,
                                                       "cogs":None,"variable_cost":None,"fixed_cost":None,"marketing_spend":None,
                                                       "new_customers":None,"active_customers":None})
                    status = "loaded"
                except Exception:
                    pass
            else:
                if PdfReader is not None:
                    try:
                        # very naive attempt if a direct URL
                        import requests  # allowed in Streamlit cloud; ignore if blocked
                        content = requests.get(hint, timeout=10).content
                        reader = PdfReader(io.BytesIO(content)); pages=[]
                        for p in reader.pages:
                            try: pages.append(p.extract_text() or "")
                            except Exception: pages.append("")
                        SS["docs"][name] = pages; status = "loaded"
                    except Exception:
                        pass
            SS["remote_sources"].append({"provider":provider,"hint":hint,"status":status,"kind":kind,"name":name})
            _snapshot()
            st.success(f"Added {provider} {kind}: {name} ({status})")

    if SS["remote_sources"]:
        df_rem = pd.DataFrame(SS["remote_sources"])
        st.dataframe(df_rem, use_container_width=True, hide_index=True)

# ------------ 2) GRID ------------
with tab_grid:
    st.subheader("Build Grid")
    t1, t2 = st.columns([0.6, 0.4])
    with t1:
        b1,b2,b3 = st.columns(3)
        with b1:
            if st.button("Add CSV rows"):
                for name in SS["tables"].keys(): _add_row_from_table(name)
                _ensure_cells(); _snapshot(); st.success("Added CSV rows.")
        with b2:
            if st.button("Add PDF rows"):
                for name in SS["docs"].keys(): _add_row_from_pdf(name)
                _ensure_cells(); _snapshot(); st.success("Added PDF rows.")
        with b3:
            preset = st.selectbox("Add preset columns", options=["(choose)","CDD (Cohorts, NRR, Unit Econ, PDF)","QoE (Quality of Earnings)"])
            if st.button("Apply preset"):
                want = []
                if preset=="CDD (Cohorts, NRR, Unit Econ, PDF)":
                    want = [("Cohort Retention","cohort_retention"),("NRR/GRR","nrr_grr"),
                            ("Unit Economics","unit_economics"),("PDF KPIs","pdf_kpi_extract")]
                elif preset=="QoE (Quality of Earnings)":
                    want = [("Unit Economics","unit_economics"),("Pricing Power","pricing_power"),
                            ("PDF KPIs","pdf_kpi_extract"),("NRR/GRR","nrr_grr")]
                have = {c["tool"] for c in SS["grid"]["columns"]}
                added = 0
                for label,tool in want:
                    if tool not in have: _add_column(label, tool, {}); added += 1
                _ensure_cells(); _snapshot()
                st.success(f"Added {added} column(s).")

        st.markdown("**Add a single column**")
        col_name = st.text_input("Column label", value="Unit Economics", key="add_col_name")
        tool_key = st.selectbox("Module", options=list(MODULES.keys()), format_func=lambda k: MODULES[k]["title"])
        if st.button("Add Column"):
            _add_column(col_name, tool_key, {}); _ensure_cells(); _snapshot(); st.success("Column added.")

    with t2:
        st.markdown("**Structure controls**")
        c1,c2 = st.columns(2)
        with c1:
            if st.button("↩️ Undo"): 
                if _undo(): st.experimental_rerun()
                else: st.info("Nothing to undo.")
        with c2:
            if st.button("↪️ Redo"):
                if _redo(): st.experimental_rerun()
                else: st.info("Nothing to redo.")

    st.markdown("---")
    st.markdown("**Rows**")
    if SS["grid"]["rows"]:
        rm_df = pd.DataFrame([{"Row ID": r["id"], "Alias": r.get("alias") or r["row_ref"], "Type": r["type"], "Source": r["source"], "Delete?": False} for r in SS["grid"]["rows"]])
        edited = st.data_editor(rm_df, num_rows="dynamic", use_container_width=True, key="row_mgr")
        to_del = edited[edited["Delete?"]==True]["Row ID"].tolist() if "Delete?" in edited else []
        cdel, csel = st.columns([0.35,0.65])
        with cdel:
            if st.button("Delete selected row(s)"):
                for rid in to_del: delete_row(rid)
                _ensure_cells(); _snapshot(); st.experimental_rerun()
        with csel:
            SS["row_run_selection"] = st.multiselect("Rows to run (quick pick)", options=[r["id"] for r in SS["grid"]["rows"]],
                                                     format_func=lambda rid: next((r.get("alias") or r["row_ref"] for r in SS["grid"]["rows"] if r["id"]==rid), rid),
                                                     default=SS.get("row_run_selection", []))
    else:
        st.info("No rows yet. Add CSV/PDF rows above.")

    st.markdown("---")
    st.markdown("**Columns**")
    for c in list(SS["grid"]["columns"]):
        cols = st.columns([0.50, 0.12, 0.12, 0.10, 0.16])
        with cols[0]:
            c["name"] = st.text_input(f"Label ({c['tool']})", c["name"], key=f"colname_{c['id']}")
        with cols[1]:
            if st.button("⬆️", key=f"up_{c['id']}"): move_col(c["id"], "up"); _ensure_cells(); _snapshot(); st.experimental_rerun()
        with cols[2]:
            if st.button("⬇️", key=f"dn_{c['id']}"): move_col(c["id"], "down"); _ensure_cells(); _snapshot(); st.experimental_rerun()
        with cols[3]:
            st.caption("csv" if MODULES[c["tool"]]["applies_to"]==["table"] else "pdf")
        with cols[4]:
            if st.button("🗑️ Delete", key=f"del_{c['id']}"): delete_col(c["id"]); _ensure_cells(); _snapshot(); st.experimental_rerun()

# ------------ 3) RUN ------------
with tab_run:
    st.subheader("Run Cells")
    force_rerun = st.toggle("Force re-run (ignore cache)", value=False)
    sel_rows = st.multiselect("Rows to run", options=[r["id"] for r in SS["grid"]["rows"]],
                              format_func=lambda rid: next((r.get("alias") or r["row_ref"] for r in SS["grid"]["rows"] if r["id"]==rid), rid))
    sel_cols = st.multiselect("Columns to run", options=[c["id"] for c in SS["grid"]["columns"]],
                              format_func=lambda cid: next((c["name"] for c in SS["grid"]["columns"] if c["id"]==cid), cid))

    def _hash_text(s): return hashlib.md5(s.encode("utf-8")).hexdigest()
    def _sig_for_row(row):
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

    def _run_cell(cell, force=False):
        row = next(x for x in SS["grid"]["rows"] if x["id"]==cell["row_id"])
        col = next(x for x in SS["grid"]["columns"] if x["id"]==cell["col_id"])
        sig = f"{_sig_for_row(row)}|{col['tool']}|{str(SS['schema_map'].get(row['source'], {}))}"
        cached = SS["cache"].get(cell["id"])
        if cached and (cached.get("sig")==sig) and not force:
            res: ModuleResult = cached["result"]
        else:
            cell["status"]="running"
            try:
                if row["type"]=="table":
                    df = SS["tables"].get(row["source"])
                    m = SS["schema_map"].get(row["source"], {})
                    if col["tool"]=="cohort_retention":
                        res = module_cohort_retention(df, m.get("customer"), m.get("date"), m.get("revenue"))
                    elif col["tool"]=="pricing_power":
                        res = module_pricing_power(df, m.get("price"), m.get("quantity"))
                    elif col["tool"]=="nrr_grr":
                        res = module_nrr_grr(df, m.get("customer"), m.get("date"), m.get("revenue"))
                    elif col["tool"]=="unit_economics":
                        res = module_unit_economics(df, m.get("date"), m.get("revenue"),
                                                    m.get("cogs"), m.get("variable_cost"), m.get("fixed_cost"),
                                                    m.get("marketing_spend"), m.get("new_customers"),
                                                    m.get("customer"), m.get("active_customers"))
                    else:
                        res = ModuleResult({}, f"Tool {col['tool']} not for CSV row.", [])
                else:
                    if col["tool"]=="pdf_kpi_extract":
                        pages = SS["docs"].get(row["source"], []); res = module_pdf_kpi(pages)
                    else:
                        res = ModuleResult({}, f"{MODULES[col['tool']]['title']} applies to CSV rows.", [])
                SS["cache"][cell["id"]] = {"sig": sig, "result": res}
            except Exception as e:
                res = ModuleResult({}, f"Error: {e}", [])
                cell["status"]="error"
        if cell.get("status")!="error":
            cell["status"] = "done" if (res.kpis or res.narrative) else "needs_review"
        cell["output_text"] = res.narrative
        v=None
        if res.kpis:
            for _k,_v in res.kpis.items():
                if isinstance(_v,(int,float)) and _v==_v: v=float(_v); break
        cell["numeric_value"]=v; cell["units"]=res.units_hint; cell["kpis"]=res.kpis
        cell["citations"]=res.citations; cell["figure"]=res.figure; cell["figure2"]=res.figure2

    cA,cB,cC = st.columns(3)
    with cA:
        if st.button("Run selection"):
            _ensure_cells()
            targets = [c for c in SS["grid"]["cells"] if (not sel_rows or c["row_id"] in sel_rows) and (not sel_cols or c["col_id"] in sel_cols)]
            for cell in targets: _run_cell(cell, force=force_rerun)
            st.success(f"Ran {len(targets)} cell(s).")
    with cB:
        if st.button("Run ALL"):
            _ensure_cells()
            for cell in SS["grid"]["cells"]: _run_cell(cell, force=force_rerun)
            st.success(f"Ran {len(SS['grid']['cells'])} cell(s).")
    with cC:
        if st.button("Run selected Rows (from Grid tab)"):
            _ensure_cells()
            row_ids = SS.get("row_run_selection", [])
            targets = [c for c in SS["grid"]["cells"] if c["row_id"] in row_ids and (not sel_cols or c["col_id"] in sel_cols)]
            for cell in targets: _run_cell(cell, force=force_rerun)
            st.success(f"Ran {len(targets)} cell(s).")

    # results + filters
    st.markdown("---")
    st.subheader("Results (filtered)")

    def _cells_df():
        g = SS["grid"]
        if not g["cells"]: return pd.DataFrame()
        row_map = {r["id"]: (r.get("alias") or f'{r["type"]}:{r["source"]}') for r in g["rows"]}
        row_type = {r["id"]: r["type"] for r in g["rows"]}
        col_map = {c["id"]: c["name"] for c in g["columns"]}
        col_tool= {c["id"]: c["tool"] for c in g["columns"]}
        df = pd.DataFrame(g["cells"]).copy()
        df["Row"] = df["row_id"].map(row_map); df["row_type"] = df["row_id"].map(row_type)
        df["Column"] = df["col_id"].map(col_map); df["tool"] = df["col_id"].map(col_tool)
        def _val(row):
            v,u=row.get("numeric_value"),row.get("units")
            if v is None: return "-"
            return _fmt_pct(v) if u=="pct" else (f"{v:,.4f}" if abs(v)<1 else f"{v:,.2f}")
        df["Value"] = df.apply(_val,axis=1)
        df["Summary"] = df.get("output_text","").astype(str).str.slice(0,160)
        return df

    df_all = _cells_df()
    if df_all.empty:
        st.info("No results yet. Run some cells.")
    else:
        statuses = sorted(df_all["status"].dropna().unique().tolist())
        row_types = sorted(df_all["row_type"].dropna().unique().tolist())
        col_names = sorted(df_all["Column"].dropna().unique().tolist())
        f1,f2,f3,f4,f5 = st.columns([0.24,0.22,0.28,0.14,0.12])
        with f1: st.session_state["flt_status"]=st.multiselect("Status", options=statuses, default=statuses)
        with f2: st.session_state["flt_rtype"]=st.multiselect("Row type", options=row_types, default=row_types)
        with f3: st.session_state["flt_cols"]=st.multiselect("Columns", options=col_names, default=col_names)
        with f4: att = st.toggle("Attention only", value=False)
        with f5:
            view_ops=["(none)"]+sorted(SS["views"].keys())
            chosen = st.selectbox("Saved view", options=view_ops)
            if chosen and chosen!="(none)" and st.button("Apply view"):
                v=SS["views"][chosen]
                st.session_state["flt_status"]=v["status"]; st.session_state["flt_rtype"]=v["row_type"]; st.session_state["flt_cols"]=v["columns"]
                st.experimental_rerun()

        df_view = df_all[
            df_all["status"].isin(st.session_state["flt_status"])
            & df_all["row_type"].isin(st.session_state["flt_rtype"])
            & df_all["Column"].isin(st.session_state["flt_cols"])
        ].copy()
        if att: df_view = df_view[df_view["status"].isin(["needs_review","error"])]
        st.dataframe(df_view[["Row","row_type","Column","status","Value","Summary"]],
                     hide_index=True, use_container_width=True)

        sv1, sv2, sv3 = st.columns([0.35,0.25,0.40])
        with sv1:
            new_name = st.text_input("Save current filters as view", value="")
            if st.button("Save view") and new_name.strip():
                SS["views"][new_name.strip()] = {"status":st.session_state["flt_status"],"row_type":st.session_state["flt_rtype"],"columns":st.session_state["flt_cols"]}
                st.success(f"Saved '{new_name.strip()}'")
        with sv2:
            del_name = st.selectbox("Delete view", options=["(choose)"]+sorted(SS["views"].keys()))
            if st.button("Delete view") and del_name and del_name!="(choose)":
                SS["views"].pop(del_name, None); st.success(f"Deleted '{del_name}'")
        with sv3:
            csv_bytes = df_view.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Export filtered CSV", data=csv_bytes, file_name="grid_results_filtered.csv", mime="text/csv")

# ------------ 4) REVIEW ------------
with tab_review:
    st.subheader("Review & Approve")
    if not SS["grid"]["cells"]:
        st.info("Nothing to review yet.")
    else:
        sel_cell_id = st.selectbox("Choose a cell", options=[c["id"] for c in SS["grid"]["cells"]], index=0)
        cell = next(c for c in SS["grid"]["cells"] if c["id"]==sel_cell_id)
        col  = next(x for x in SS["grid"]["columns"] if x["id"]==cell["col_id"])
        row  = next(x for x in SS["grid"]["rows"] if x["id"]==cell["row_id"])
        st.markdown(f"**{col['name']}** on _{row.get('alias') or row['row_ref']}_ — status: `{cell['status']}`")

        tabs = st.tabs(["Chart 1","Chart 2","Details"])
        with tabs[0]:
            if cell.get("figure") is not None: st.plotly_chart(cell["figure"], use_container_width=True)
            else: st.info("No chart.")
        with tabs[1]:
            if cell.get("figure2") is not None: st.plotly_chart(cell["figure2"], use_container_width=True)
            else: st.info("No second chart.")
        with tabs[2]:
            if cell.get("output_text"): st.write(cell["output_text"])
            with st.expander("KPIs"): st.json(cell.get("kpis", {}))
            with st.expander("Citations"): st.json(cell.get("citations", []))

        if col["tool"]=="pricing_power":
            st.markdown("### Pricing Uplift Simulator")
            eps = cell.get("kpis", {}).get("elasticity")
            if isinstance(eps,(int,float)):
                pct = st.slider("Proposed average price change (%)", -30, 30, 5, 1)
                dp = pct/100.0; rev_change = (1+dp)**(1+eps) - 1
                st.write(f"Estimated revenue change: **{_fmt_pct(rev_change)}** (ε≈{eps:.2f})")
            else:
                st.info("Run Pricing Power first.")

        a1,a2 = st.columns(2)
        with a1:
            if st.button("Approve"): cell["status"]="approved"; st.success("Approved.")
        with a2:
            if st.button("Mark Needs-Review"): cell["status"]="needs_review"; st.warning("Marked.")

# ------------ 5) MEMO ------------
with tab_memo:
    st.subheader("Compose Memo & Export")

    def _csv_revenue_total():
        total=0.0; seen=False
        for name, df in SS["tables"].items():
            rev_col = SS["schema_map"].get(name, {}).get("revenue")
            if rev_col and rev_col in df.columns:
                try: total += float(pd.to_numeric(df[rev_col], errors="coerce").fillna(0).sum()); seen=True
                except Exception: pass
        return total if seen else None

    def _first_pdf_kpis():
        for c in SS["grid"]["cells"]:
            if c.get("status")=="approved":
                col = next((x for x in SS["grid"]["columns"] if x["id"]==c["col_id"]), None)
                row = next((x for x in SS["grid"]["rows"] if x["id"]==c["row_id"]), None)
                if col and row and col["tool"]=="pdf_kpi_extract" and row["type"]=="pdf":
                    return c.get("kpis", {})
        return {}

    def _cross_checks():
        checks=[]
        pdf_kpis = _first_pdf_kpis(); pdf_rev = pdf_kpis.get("revenue")
        csv_rev = _csv_revenue_total()
        if pdf_rev is not None and csv_rev is not None and csv_rev>0:
            rel = abs(pdf_rev - csv_rev) / max(pdf_rev, csv_rev)
            status = "✅ Revenue matches (≤10% delta)" if rel<=0.10 else ("🟡 Close (10–25%)" if rel<=0.25 else "🔴 Mismatch (>25%)")
            checks.append(("Revenue", f"{status}: PDF {_fmt_money(pdf_rev)} vs CSV {_fmt_money(csv_rev)}"))
        elif pdf_rev is not None:
            checks.append(("Revenue", f"ℹ️ PDF {_fmt_money(pdf_rev)} (no CSV)"))
        elif csv_rev is not None:
            checks.append(("Revenue", f"ℹ️ CSV {_fmt_money(csv_rev)} (no PDF)"))
        return checks

    def export_memo_pdf(grid, cells) -> bytes:
        rows = {r["id"]: r for r in grid["rows"]}
        cols = {c["id"]: c for c in grid["columns"]}
        selected_cols = SS.get("memo_cols")
        approved = [c for c in cells if c.get("status")=="approved" and (selected_cols is None or c["col_id"] in selected_cols)]
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
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
            line("Cross-checks", 12, True)
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
            if ccell.get("output_text"): line(f"   {ccell['output_text']}", 10, leading=12); y -= 4
        y -= 4; line("Evidence Appendix", 12, True)
        for ccell in approved:
            r = rows[ccell["row_id"]]; row_label = r.get("alias") or f"{r['type']}:{r['source']}"
            col_title = cols[ccell["col_id"]]["name"]; citations = ccell.get("citations") or []
            if not citations: line(f"• {col_title} on {row_label}: (no citations captured)", 10); continue
            line(f"• {col_title} on {row_label}:", 10)
            for cit in citations[:6]:
                if cit.get("type")=="pdf":
                    page = cit.get("page","?"); snip = (cit.get("excerpt","") or "").replace("\n"," ")
                    line(f"   - PDF p.{page}: “{snip[:120]}”", 9)
                elif cit.get("type")=="table":
                    sel = cit.get("selector",""); line(f"   - CSV selection: {sel}", 9)
        c.showPage(); c.save(); buf.seek(0); return buf.getvalue()

    all_cols = SS["grid"]["columns"]
    memo_cols = st.multiselect("Columns to include", options=[c["id"] for c in all_cols],
                               default=[c["id"] for c in all_cols] if SS["memo_cols"] is None else SS["memo_cols"],
                               format_func=lambda cid: next((c["name"] for c in all_cols if c["id"]==cid), cid))
    if st.button("Apply selection"): SS["memo_cols"] = memo_cols; st.success("Memo columns updated.")

    pdf_bytes = export_memo_pdf(SS["grid"], SS["grid"]["cells"])
    st.download_button("📄 Download Investor Memo (PDF)", data=pdf_bytes,
                       file_name=f"TransformAI_Memo_{SS['grid']['id']}.pdf", mime="application/pdf")

# ------------- FOOTER: activity (collapsed) -------------
with st.expander("Activity Log (debug)"):
    st.json(SS["grid"]["activities"])
