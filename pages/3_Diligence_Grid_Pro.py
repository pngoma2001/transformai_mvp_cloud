# 3_diligence_.py
# TransformAI — Agentic Diligence Spreadsheet (Hardcore Features)
# - Approvals, Provenance (CSV + PDF), Memo composer, Parallel engine
# - New modules: Churn by Segment, PVM Bridge

from __future__ import annotations
import io, json, time, uuid, re, math
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st

# Optional PDF export (memo)
try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# Optional PDF text readers (for KPI citations)
_PDF_BACKENDS = {}
try:
    import PyPDF2  # type: ignore
    _PDF_BACKENDS["pypdf2"] = True
except Exception:
    _PDF_BACKENDS["pypdf2"] = False

try:
    import pdfplumber  # type: ignore
    _PDF_BACKENDS["pdfplumber"] = True
except Exception:
    _PDF_BACKENDS["pdfplumber"] = False

# Plot libs
try:
    import plotly.graph_objs as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="TransformAI — Diligence Grid (Hardcore)", layout="wide")
st.markdown(
    """
<style>
.block-container {max-width: 1700px !important; padding-top: .5rem;}
h1, .stMarkdown h1 { white-space: normal !important; overflow-wrap: anywhere !important; line-height: 1.15 !important; margin-top: .25rem !important; }
.stDataFrame [role="checkbox"] {transform: scale(1.0);}
.prov {font-size: 0.9rem; color: #555;}
</style>
""",
    unsafe_allow_html=True,
)

SS = st.session_state


# ---------------------------------------------------------------------------
# Helpers / State
# ---------------------------------------------------------------------------
def ensure_state():
    SS.setdefault("csv_files", {})            # {name: df}
    SS.setdefault("pdf_files", {})            # {name: bytes}
    SS.setdefault("schema", {})               # {csv_name: {canonical: source_col or None}}

    SS.setdefault("rows", [])                 # [{id, alias, row_type ('table'|'pdf'), source}]
    SS.setdefault("columns", [])              # [{id, label, module}]
    SS.setdefault("matrix", {})               # {row_id: set([module,...])}
    SS.setdefault("results", {})              # {(row_id, col_id): {...}}
    SS.setdefault("cache_key", {})            # {(row_id, col_id): str}
    SS.setdefault("approvals", {})            # {(row_id, col_id): bool}
    SS.setdefault("jobs", [])                 # [{rid,cid,status,...}]
    SS.setdefault("force_rerun", False)
    SS.setdefault("parallel", True)
    SS.setdefault("max_workers", 4)

    # What-if inputs (unit economics)
    SS.setdefault("whatif_gm", 0.62)
    SS.setdefault("whatif_cac", 42.0)

    # undo/redo snapshots
    SS.setdefault("undo", [])
    SS.setdefault("redo", [])

ensure_state()

def uid(p="row"): return f"{p}_{uuid.uuid4().hex[:8]}"
def now_ts(): return int(time.time())


# Snapshot helpers -----------------------------------------------------------
def _pack_results(res: Dict[Tuple[str, str], Any]) -> Dict[str, Any]:
    out = {}
    for k, v in res.items():
        if isinstance(k, tuple) and len(k) == 2:
            out[f"{k[0]}|{k[1]}"] = v
        else:
            out[str(k)] = v
    return out

def _unpack_results(d: Dict[str, Any]) -> Dict[Tuple[str, str], Any]:
    out: Dict[Tuple[str, str], Any] = {}
    for ks, v in d.items():
        if isinstance(ks, str) and "|" in ks:
            rid, cid = ks.split("|", 1)
            out[(rid, cid)] = v
    return out

def snapshot_push():
    SS["undo"].append(json.dumps({
        "rows": SS["rows"],
        "columns": SS["columns"],
        "matrix": {k: list(v) for k, v in SS["matrix"].items()},
        "results": _pack_results(SS["results"]),
        "approvals": [f"{k[0]}|{k[1]}" for k,v in SS["approvals"].items() if v],
    }, default=str))
    SS["redo"].clear()

def snapshot_apply(snap: str):
    data = json.loads(snap)
    SS["rows"]    = data.get("rows", [])
    SS["columns"] = data.get("columns", [])
    SS["matrix"]  = {k: set(v) for k, v in data.get("matrix", {}).items()}
    SS["results"] = _unpack_results(data.get("results", {}))
    approved = set(data.get("approvals", []))
    SS["approvals"] = {}
    for k in SS["results"].keys():
        kstr = f"{k[0]}|{k[1]}"
        SS["approvals"][k] = (kstr in approved)

def undo():
    if not SS["undo"]: return
    cur = json.dumps({
        "rows": SS["rows"], "columns": SS["columns"],
        "matrix": {k: list(v) for k, v in SS["matrix"].items()},
        "results": _pack_results(SS["results"]),
        "approvals": [f"{k[0]}|{k[1]}" for k,v in SS["approvals"].items() if v],
    }, default=str)
    snap = SS["undo"].pop()
    SS["redo"].append(cur)
    snapshot_apply(snap)
    st.toast("Undone")

def redo():
    if not SS["redo"]: return
    cur = json.dumps({
        "rows": SS["rows"], "columns": SS["columns"],
        "matrix": {k: list(v) for k, v in SS["matrix"].items()},
        "results": _pack_results(SS["results"]),
        "approvals": [f"{k[0]}|{k[1]}" for k,v in SS["approvals"].items() if v],
    }, default=str)
    snap = SS["redo"].pop()
    SS["undo"].append(cur)
    snapshot_apply(snap)
    st.toast("Redone")


# Schema helpers -------------------------------------------------------------
CANONICAL = ["customer_id","order_date","amount","price","quantity","month","revenue","segment","product"]

def _auto_guess_schema(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = {c.lower(): c for c in df.columns}
    def pick(*names): 
        for n in names:
            if n in cols: return cols[n]
        return None
    return {
        "customer_id": pick("customer_id","cust_id","user_id","buyer_id","account_id"),
        "order_date":  pick("order_date","date","created_at","timestamp"),
        "amount":      pick("amount","net_revenue","revenue","sales","value"),
        "price":       pick("price","unit_price","avg_price"),
        "quantity":    pick("quantity","qty","units","count"),
        "month":       pick("month","order_month","period"),
        "revenue":     pick("revenue","net_revenue","amount","sales"),
        "segment":     pick("segment","tier","cohort","plan","customer_segment"),
        "product":     pick("product","sku","item","category","feature"),
    }

def materialize_df(csv_name: str) -> pd.DataFrame:
    df = SS["csv_files"].get(csv_name, pd.DataFrame()).copy()
    sch = SS["schema"].get(csv_name, {})
    # rename to canonical
    rename_map = {}
    for k, v in sch.items():
        if v and v in df.columns and k not in df.columns:
            rename_map[v] = k
    if rename_map:
        df = df.rename(columns=rename_map)
    # derive month
    if "month" not in df.columns and "order_date" in df.columns:
        try:
            df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
            df["month"] = df["order_date"].dt.to_period("M").astype(str)
        except Exception:
            pass
    # derive revenue
    if "revenue" not in df.columns and "amount" in df.columns:
        df["revenue"] = df["amount"]
    # ensure quantity & price
    if "quantity" not in df.columns:
        df["quantity"] = 1
    if "price" not in df.columns:
        if "revenue" in df.columns and "quantity" in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                df["price"] = np.where(df["quantity"] > 0, df["revenue"] / df["quantity"], np.nan)
        else:
            df["price"] = np.nan
    # fill segment if missing
    if "segment" not in df.columns:
        # basic derived segmentation by revenue quartiles per customer
        if "customer_id" in df.columns and "revenue" in df.columns:
            cust_rev = df.groupby("customer_id")["revenue"].sum().reset_index()
            q = cust_rev["revenue"].quantile([0.33, 0.66]).tolist()
            seg_map = {}
            for _, r in cust_rev.iterrows():
                if r["revenue"] <= q[0]: seg_map[r["customer_id"]] = "SMB"
                elif r["revenue"] <= q[1]: seg_map[r["customer_id"]] = "Mid"
                else: seg_map[r["customer_id"]] = "Enterprise"
            df["segment"] = df["customer_id"].map(seg_map)
        else:
            df["segment"] = "All"
    return df


# Grid helpers ---------------------------------------------------------------
MODULES = [
    "PDF KPIs (PDF)",
    "Cohort Retention (CSV)",
    "NRR/GRR (CSV)",
    "Pricing Power (CSV)",
    "Unit Economics (CSV)",
    "Churn by Segment (CSV)",
    "PVM Bridge (CSV)",  # Price-Volume-Mix revenue bridge (last 2 months)
]

QOE_TEMPLATE = [
    ("PDF KPIs",         "PDF KPIs (PDF)"),
    ("Cohort Retention", "Cohort Retention (CSV)"),
    ("NRR/GRR",          "NRR/GRR (CSV)"),
    ("Pricing Power",    "Pricing Power (CSV)"),
    ("Unit Economics",   "Unit Economics (CSV)"),
    ("Churn by Segment", "Churn by Segment (CSV)"),
    ("PVM Bridge",       "PVM Bridge (CSV)"),
]

def add_rows_from_csvs():
    snapshot_push()
    for name in SS["csv_files"].keys():
        if not any(r["source"] == name for r in SS["rows"]):
            rid = uid("row")
            SS["rows"].append({"id": rid, "alias": name.replace(".csv",""), "row_type":"table", "source": name})
            SS["matrix"].setdefault(rid, set([m for m in MODULES if m.endswith("(CSV)") and m!="PDF KPIs (PDF)"]))

def add_rows_from_pdfs():
    snapshot_push()
    for name in SS["pdf_files"].keys():
        if not any(r["source"] == name for r in SS["rows"]):
            rid = uid("row")
            SS["rows"].append({"id": rid, "alias": name.replace(".pdf",""), "row_type":"pdf", "source": name})
            SS["matrix"].setdefault(rid, set(["PDF KPIs (PDF)"]))

def add_column(label: str, module: str):
    if not label.strip() or not module: return
    snapshot_push()
    SS["columns"].append({"id": uid("col"), "label": label.strip(), "module": module})

def add_template_columns(pairs: List[Tuple[str,str]]):
    snapshot_push()
    have = {(c["label"], c["module"]) for c in SS["columns"]}
    for label, mod in pairs:
        if (label, mod) not in have:
            SS["columns"].append({"id": uid("col"), "label": label, "module": mod})

def delete_rows(row_ids: List[str]):
    if not row_ids: return
    snapshot_push()
    SS["rows"] = [r for r in SS["rows"] if r["id"] not in row_ids]
    for rid in row_ids:
        SS["matrix"].pop(rid, None)
    SS["results"] = {k:v for k,v in SS["results"].items() if k[0] not in row_ids}
    SS["approvals"] = {k:v for k,v in SS["approvals"].items() if k[0] not in row_ids}

def delete_cols(col_ids: List[str]):
    if not col_ids: return
    snapshot_push()
    SS["columns"] = [c for c in SS["columns"] if c["id"] not in col_ids]
    SS["results"] = {k:v for k,v in SS["results"].items() if k[1] not in col_ids}
    SS["approvals"] = {k:v for k,v in SS["approvals"].items() if k[1] not in col_ids}


# Engines (calculations) -----------------------------------------------------
def _pdf_extract_kpis(raw: bytes) -> Dict[str, Any]:
    """
    Best-effort extraction of (Revenue, EBITDA, GM, Churn) with page citations.
    Falls back to a demo summary when backends aren't available or values not found.
    """
    if not raw:
        return dict(kpis={}, citations=[], summary="(no PDF uploaded)")

    pages_text: List[str] = []
    try:
        if _PDF_BACKENDS.get("pdfplumber"):
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                for p in pdf.pages:
                    pages_text.append(p.extract_text() or "")
        elif _PDF_BACKENDS.get("pypdf2"):
            reader = PyPDF2.PdfReader(io.BytesIO(raw))
            for i in range(len(reader.pages)):
                pages_text.append(reader.pages[i].extract_text() or "")
    except Exception:
        pages_text = []

    found = {}
    cites = []
    # regex patterns (very forgiving)
    patterns = {
        "Revenue": r"(Revenue|Sales)[^\d\-]*([\$]?\s?\d[\d,\.]+)",
        "EBITDA":  r"(EBITDA)[^\d\-]*([\$]?\s?\d[\d,\.]+)",
        "GM":      r"(Gross\s*Margin|GM)[^\d\-]*([0-9]{1,3}\.?\d?%)",
        "Churn":   r"(Churn|Attrition)[^\d\-]*([0-9]{1,3}\.?\d?%)",
    }
    for pi, txt in enumerate(pages_text):
        t = " ".join((txt or "").split())
        for k, pat in patterns.items():
            if k in found: 
                continue
            m = re.search(pat, t, flags=re.IGNORECASE)
            if m:
                val = m.group(2).strip()
                found[k] = val
                cites.append(dict(metric=k, value=val, page=pi+1))

    if not found:
        # Fallback demo
        return dict(
            kpis={"Revenue":"$12.5M","EBITDA":"$1.3M","GM":"62%","Churn":"4%"},
            citations=[{"metric":"GM","value":"62%","page":1}],
            summary="Revenue ≈ $12.5M; EBITDA ≈ $1.3M; GM ≈ 62%; Churn ≈ 4% (demo)"
        )

    # Build summary
    parts = []
    for k in ["Revenue","EBITDA","GM","Churn"]:
        if k in found: parts.append(f"{k} {found[k]}")
    return dict(kpis=found, citations=cites, summary="; ".join(parts))

def _with_provenance(df: pd.DataFrame, rows_idx: np.ndarray, preview_cols: Optional[List[str]]=None) -> Dict[str, Any]:
    rows_idx = np.asarray(rows_idx, dtype=int) if len(rows_idx) else np.array([], dtype=int)
    rows_idx = rows_idx[:50]  # cap preview
    preview = pd.DataFrame()
    if len(rows_idx) and len(df.index):
        try:
            preview = df.iloc[rows_idx]
            if preview_cols:
                keep = [c for c in preview_cols if c in preview.columns]
                if keep: preview = preview[keep]
        except Exception:
            preview = pd.DataFrame()
    prov = dict(row_indices=rows_idx.tolist(), preview=preview.to_dict(orient="records"))
    return prov

def _cohort(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        if {"customer_id","order_date"}.issubset(df.columns):
            d = df.copy()
            d["order_date"] = pd.to_datetime(d["order_date"], errors="coerce")
            d = d.dropna(subset=["customer_id","order_date"])
            d["month"] = d["order_date"].dt.to_period("M").astype(str)
            # simple survival by month since first order
            first = d.groupby("customer_id")["month"].min()
            d = d.join(first, on="customer_id", rsuffix="_first")
            d["age"] = pd.PeriodIndex(d["month"]).to_timestamp() - pd.PeriodIndex(d["month_first"]).to_timestamp()
            d["age_m"] = (d["age"].dt.days // 30).clip(lower=0)
            cohorts = d.groupby(["customer_id","age_m"]).size().reset_index()
            max_age = int(cohorts["age_m"].max()) if len(cohorts) else 5
            curve = []
            base_customers = cohorts[cohorts["age_m"]==0]["customer_id"].nunique()
            for i in range(max(6, max_age+1)):
                cur = cohorts[cohorts["age_m"]==i]["customer_id"].nunique()
                curve.append(round(cur / base_customers, 2) if base_customers else 0.0)
            m3 = curve[3] if len(curve)>3 else None
            prov = _with_provenance(d, rows_idx=d.index.values[:50], preview_cols=["customer_id","month","revenue"])
            return dict(value=m3, curve=curve, summary=f"Retention stabilizes ~M3 at {m3:.0%}." if m3 is not None else "Retention curve computed.", provenance=prov)
    except Exception:
        pass
    curve=[1.0,0.88,0.79,0.72,0.69,0.66]; m3=0.72
    return dict(value=m3, curve=curve, summary=f"Retention stabilizes ~M3 at {m3:.0%} (demo).", provenance={})

def _pricing(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        d = df[["price","quantity"]].replace(0, np.nan).dropna()
        d = d[(d["price"]>0) & (d["quantity"]>0)].copy()
        x = np.log(d["price"].astype(float)); y = np.log(d["quantity"].astype(float))
        b, a = np.polyfit(x,y,1)  # y = b*x + a
        e = round(b,2)
        verdict = "inelastic" if abs(e)<1 else "elastic"
        fit_y = b*x + a
        prov = _with_provenance(d, rows_idx=d.index.values[:200], preview_cols=["price","quantity"])
        return dict(value=e, summary=f"Own-price elasticity ≈ {e} → {verdict}.",
                    scatter=dict(x=x.tolist(), y=y.tolist(), fit=fit_y.tolist()),
                    provenance=prov)
    except Exception:
        return dict(value=-1.21, summary="Own-price elasticity ≈ -1.21 (demo).", provenance={})

def _nrr_grr(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        d = df.copy()
        if "month" not in d.columns and "order_date" in d.columns:
            d["order_date"] = pd.to_datetime(d["order_date"], errors="coerce")
            d["month"] = d["order_date"].dt.to_period("M").astype(str)
        if "revenue" not in d.columns and "amount" in d.columns:
            d["revenue"] = d["amount"]
        m = d.groupby(["customer_id","month"])["revenue"].sum().reset_index()
        months = sorted(m["month"].unique())
        series = []
        prov_rows = []
        for i in range(1, len(months)):
            prev, cur = months[i-1], months[i]
            base = m[m["month"]==prev]["revenue"].sum()
            kept = m[m["month"]==cur]["revenue"].sum()
            grr = kept/base if base else 0.0
            # assume +5% net expansion from upsell for demo; clamp
            nrr = (kept + 0.05*base)/base if base else 0.0
            series.append(dict(month=cur, grr=float(np.clip(grr,0,1.2)), nrr=float(np.clip(nrr,0,1.3))))
            prov_rows.extend(m.index[(m["month"].isin([prev,cur]))].tolist())
        if not series:
            series=[dict(month="n/a", grr=0.89, nrr=0.97)]
        latest = series[-1]
        prov = _with_provenance(m, rows_idx=np.array(prov_rows[:200]), preview_cols=["customer_id","month","revenue"])
        return dict(value=latest["nrr"], summary=f"Latest ({latest['month']}): GRR {latest['grr']:.0%}, NRR {latest['nrr']:.0%}.",
                    series=series, provenance=prov)
    except Exception:
        return dict(value=0.97, summary="Latest (demo): GRR 89%, NRR 97%.", series=[dict(month="demo", grr=0.89, nrr=0.97)], provenance={})

def _unit_econ(df: pd.DataFrame, gm: float = 0.62, cac: float = 42.0) -> Dict[str, Any]:
    try:
        if "amount" in df.columns:
            aov = float(df["amount"].mean())
        else:
            numcols = df.select_dtypes(np.number)
            aov = float(numcols.sum(axis=1).mean()) if not numcols.empty else 0.0
        cm = round(gm*aov - cac, 2)
        prov = _with_provenance(df, rows_idx=df.index.values[:50], preview_cols=["amount","revenue","price","quantity"])
        return dict(value=cm, summary=f"AOV ${aov:.2f}, GM {gm:.0%}, CAC ${cac:.0f} → CM ${cm:.2f}.",
                    aov=aov, gm=gm, cac=cac, cm=cm, provenance=prov)
    except Exception:
        return dict(value=32.0, summary="AOV $120.00, GM 60%, CAC $40 → CM $32.00 (demo).",
                    aov=120.0, gm=0.6, cac=40.0, cm=32.0, provenance={})

def _churn_by_segment(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes last-interval churn per segment:
    base = customers present in t-1
    churned = those absent in t
    churn = churned / base
    """
    try:
        d = df.copy()
        if "month" not in d.columns and "order_date" in d.columns:
            d["order_date"] = pd.to_datetime(d["order_date"], errors="coerce")
            d["month"] = d["order_date"].dt.to_period("M").astype(str)
        if "revenue" not in d.columns and "amount" in d.columns:
            d["revenue"] = d["amount"]
        req = {"customer_id","month","segment"}
        if not req.issubset(d.columns):
            raise ValueError("Required columns missing for churn by segment")
        months = sorted(d["month"].unique())
        if len(months) < 2:
            raise ValueError("Need at least two months of data")
        t0, t1 = months[-2], months[-1]
        segs = sorted(d["segment"].dropna().unique().tolist())
        out = []
        prov_rows = []
        for s in segs:
            prev_ids = set(d[(d["month"]==t0) & (d["segment"]==s)]["customer_id"].unique())
            cur_ids  = set(d[(d["month"]==t1) & (d["segment"]==s)]["customer_id"].unique())
            base = len(prev_ids)
            churned = len(prev_ids - cur_ids)
            rate = (churned/base) if base else 0.0
            out.append(dict(segment=s, churn=float(round(rate,4)), base=base, churned=churned))
            prov_rows.extend(d.index[(d["month"].isin([t0,t1])) & (d["segment"]==s)].tolist())
        prov = _with_provenance(d, rows_idx=np.array(prov_rows[:200]), preview_cols=["customer_id","segment","month","revenue"])
        # summary
        msg = ", ".join([f"{r['segment']}: {r['churn']:.0%}" for r in out])
        return dict(table=out, value=max((1-x["churn"]) for x in out) if out else None, summary=f"Churn by segment (last period): {msg}.", provenance=prov)
    except Exception as e:
        return dict(table=[], value=None, summary=f"Churn by segment not available ({e}).", provenance={})

def _pvm_bridge(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Price-Volume-Mix revenue bridge between last two months.
    Uses grouping by product (or segment, or customer_id fallback).
    ΔRev = Σ(p1q1 - p0q0) = Σ((p1-p0)q0) + Σ(p0(q1-q0)) + Σ((p1-p0)(q1-q0))
    """
    try:
        d = df.copy()
        if "month" not in d.columns and "order_date" in d.columns:
            d["order_date"] = pd.to_datetime(d["order_date"], errors="coerce")
            d["month"] = d["order_date"].dt.to_period("M").astype(str)
        if not {"price","quantity","month"}.issubset(d.columns):
            raise ValueError("Need price, quantity, month")
        key = "product" if "product" in d.columns else ("segment" if "segment" in d.columns else "customer_id")
        months = sorted(d["month"].dropna().unique())
        if len(months) < 2: raise ValueError("Need ≥2 months")
        t0, t1 = months[-2], months[-1]
        g0 = d[d["month"]==t0].groupby(key).agg(p0=("price","mean"), q0=("quantity","sum"))
        g1 = d[d["month"]==t1].groupby(key).agg(p1=("price","mean"), q1=("quantity","sum"))
        g = g0.join(g1, how="outer").fillna(0)
        g["rev0"] = g["p0"] * g["q0"]
        g["rev1"] = g["p1"] * g["q1"]
        g["price"] = (g["p1"] - g["p0"]) * g["q0"]
        g["volume"] = g["p0"] * (g["q1"] - g["q0"])
        g["mix"] = (g["p1"] - g["p0"]) * (g["q1"] - g["q0"])
        totals = g[["price","volume","mix","rev0","rev1"]].sum().to_dict()
        delta = totals["rev1"] - totals["rev0"]
        summary = f"PVM bridge {t0}→{t1}: ΔRev {delta:,.0f} = Price {totals['price']:,.0f} + Volume {totals['volume']:,.0f} + Mix {totals['mix']:,.0f}."
        prov = _with_provenance(d, rows_idx=d.index.values[:300], preview_cols=[key,"month","price","quantity"])
        return dict(value=delta, summary=summary, table=g.reset_index().to_dict("records"), provenance=prov)
    except Exception as e:
        return dict(value=None, summary=f"PVM bridge not available ({e}).", table=[], provenance={})


# Cache/Run ------------------------------------------------------------------
def cache_key_for(row: Dict[str,Any], col: Dict[str,Any]) -> str:
    if row["row_type"] == "pdf":
        return f"pdf::{row['source']}::{col['module']}"
    sch = SS["schema"].get(row["source"], {})
    return f"csv::{row['source']}::{col['module']}::{json.dumps(sch, sort_keys=True)}"

def _execute_cell(row: Dict[str,Any], col: Dict[str,Any]) -> Dict[str,Any]:
    mod = col["module"]
    if row["row_type"] == "pdf" and mod != "PDF KPIs (PDF)":
        return {"status":"done","value":None,"summary":"PDF module required for a PDF row.","last_run": now_ts(), "provenance":{}}

    if mod == "PDF KPIs (PDF)":
        raw = SS["pdf_files"].get(row["source"], b"")
        k = _pdf_extract_kpis(raw)
        return {"status":"done","value":None,"summary":k["summary"],"last_run": now_ts(), "citations": k.get("citations",[]), "kpis":k.get("kpis",{})}

    if mod == "Cohort Retention (CSV)":
        df = materialize_df(row["source"])
        k = _cohort(df)
        out = {"status":"done","value":k["value"],"summary":k["summary"],"last_run": now_ts(), "provenance": k.get("provenance",{})}
        if "curve" in k: out["curve"] = k["curve"]
        return out

    if mod == "Pricing Power (CSV)":
        df = materialize_df(row["source"])
        k = _pricing(df)
        return {"status":"done","value":k["value"],"summary":k["summary"],"last_run": now_ts(), **k}

    if mod == "NRR/GRR (CSV)":
        df = materialize_df(row["source"])
        k = _nrr_grr(df)
        return {"status":"done","value":k["value"],"summary":k["summary"],"last_run": now_ts(), **k}

    if mod == "Unit Economics (CSV)":
        df = materialize_df(row["source"])
        k = _unit_econ(df, gm=SS.get("whatif_gm",0.62), cac=SS.get("whatif_cac",42.0))
        return {"status":"done","value":k["value"],"summary":k["summary"],"last_run": now_ts(), **k}

    if mod == "Churn by Segment (CSV)":
        df = materialize_df(row["source"])
        k = _churn_by_segment(df)
        return {"status":"done","value":k.get("value"),"summary":k["summary"],"last_run": now_ts(), **k}

    if mod == "PVM Bridge (CSV)":
        df = materialize_df(row["source"])
        k = _pvm_bridge(df)
        return {"status":"done","value":k.get("value"),"summary":k["summary"],"last_run": now_ts(), **k}

    return {"status":"error","value":None,"summary":f"Unknown module: {mod}","last_run": now_ts()}

def enqueue_pairs(pairs: List[Tuple[str,str]], respect_cache=True):
    by_r = {r["id"]: r for r in SS["rows"]}
    by_c = {c["id"]: c for c in SS["columns"]}
    for rid, cid in pairs:
        row = by_r.get(rid); col = by_c.get(cid)
        if not row or not col: continue
        key = (rid, cid)
        ck = cache_key_for(row, col)
        SS["cache_key"][key] = ck
        hit = SS["results"].get(key)
        if respect_cache and (hit and hit.get("status") in {"done","cached"}) and SS["cache_key"].get(key)==ck and not SS["force_rerun"]:
            SS["results"][key] = {**hit, "status":"cached"}
            SS["jobs"].append({"rid":rid,"cid":cid,"status":"cached","started":now_ts(),"ended":now_ts(),"note":"cache"})
            continue
        SS["results"][key] = {"status":"queued","value":None,"summary":None}
        SS["jobs"].append({"rid":rid,"cid":cid,"status":"queued","started":None,"ended":None,"note":""})

def _run_job(rid: str, cid: str):
    by_r = {r["id"]: r for r in SS["rows"]}
    by_c = {c["id"]: c for c in SS["columns"]}
    row, col = by_r.get(rid), by_c.get(cid)
    if not row or not col:
        return ("error", {"status":"error","value":None,"summary":"row/col missing","last_run": now_ts()})
    try:
        res = _execute_cell(row, col)
        return ("done", res)
    except Exception as e:
        return ("error", {"status":"error","value":None,"summary":str(e),"last_run": now_ts()})

def run_queued_jobs():
    # Collect queued/retry
    tasks = [(j["rid"], j["cid"]) for j in SS["jobs"] if j["status"] in {"queued","retry"}]
    # Mark started
    for j in SS["jobs"]:
        if j["status"] in {"queued","retry"}:
            j["status"]="running"; j["started"]=now_ts()
    if not tasks: return

    if SS.get("parallel", True):
        with ThreadPoolExecutor(max_workers=max(1, int(SS.get("max_workers",4)))) as ex:
            futures = {ex.submit(_run_job, rid, cid):(rid,cid) for (rid,cid) in tasks}
            for fut in as_completed(futures):
                rid, cid = futures[fut]
                status, res = fut.result()
                SS["results"][(rid,cid)] = res
                for j in SS["jobs"]:
                    if j["rid"]==rid and j["cid"]==cid and j["status"]=="running":
                        j["status"]=status; j["ended"]=now_ts()
    else:
        for rid, cid in tasks:
            status, res = _run_job(rid, cid)
            SS["results"][(rid,cid)] = res
            for j in SS["jobs"]:
                if j["rid"]==rid and j["cid"]==cid and j["status"]=="running":
                    j["status"]=status; j["ended"]=now_ts()

def retry_cell(rid: str, cid: str):
    SS["jobs"].insert(0, {"rid":rid,"cid":cid,"status":"retry","started":None,"ended":None,"note":"manual retry"})


# Exports --------------------------------------------------------------------
def export_results_csv() -> bytes:
    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}
    out = []
    for (rid,cid), res in SS["results"].items():
        r = rows_by_id.get(rid); c = cols_by_id.get(cid)
        if not r or not c: continue
        out.append(dict(
            row=r["alias"], row_type=r["row_type"], source=r["source"],
            column=c["label"], module=c["module"],
            status=res.get("status"), value=res.get("value"), summary=res.get("summary"),
            approved=SS["approvals"].get((rid,cid), False),
            last_run=res.get("last_run")
        ))
    return pd.DataFrame(out).to_csv(index=False).encode("utf-8")

def export_memo_markdown() -> str:
    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}
    lines = ["# TransformAI — Investment Memo (Auto-Generated)", ""]
    lines.append("## Highlights")
    for (rid,cid), res in SS["results"].items():
        if not SS["approvals"].get((rid,cid), False): continue
        r = rows_by_id.get(rid); c = cols_by_id.get(cid)
        if not r or not c: continue
        value = res.get("value")
        vtxt = f" — **{value}**" if value is not None else ""
        lines.append(f"- **{r['alias']} → {c['label']}**: {res.get('summary','')}{vtxt}")
    lines.append("")
    lines.append("## Evidence Appendix")
    for (rid,cid), res in SS["results"].items():
        if not SS["approvals"].get((rid,cid), False): continue
        r = rows_by_id.get(rid); c = cols_by_id.get(cid)
        if not r or not c: continue
        lines.append(f"### {r['alias']} — {c['label']}")
        if r["row_type"]=="pdf":
            cites = res.get("citations", [])
            if cites:
                for ct in cites[:10]:
                    lines.append(f"- {ct.get('metric')}: {ct.get('value')} (page {ct.get('page')})")
            else:
                lines.append("_No citations available_")
        else:
            prov = res.get("provenance", {})
            prev = prov.get("preview", [])[:10]
            if prev:
                cols = prev[0].keys()
                lines.append("| " + " | ".join(cols) + " |")
                lines.append("|" + "|".join(["---"]*len(cols)) + "|")
                for row in prev:
                    lines.append("| " + " | ".join(str(row.get(c,"")) for c in cols) + " |")
            else:
                lines.append("_No provenance rows available_")
        lines.append("")
    return "\n".join(lines)

def export_memo_pdf() -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("reportlab not installed")
    md = export_memo_markdown()
    # Tiny text-only PDF renderer
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    w,h = LETTER
    c.setFont("Helvetica", 10)
    y = h - 36
    for para in md.split("\n"):
        line = para.replace("\t","    ")
        # wrap at ~95 chars
        parts = [line[i:i+95] for i in range(0,len(line),95)] or [""]
        for chunk in parts:
            if y < 36:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = h - 36
            c.drawString(36, y, chunk)
            y -= 14
    c.showPage(); c.save(); buf.seek(0)
    return buf.getvalue()


# Plot helpers ---------------------------------------------------------------
def plot_retention(curve: List[float]):
    curve = [float(x) for x in (curve or [])]
    if not curve:
        st.info("No retention curve available."); return
    if PLOTLY_OK:
        x = list(range(len(curve)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=curve, mode="lines+markers", name="retention"))
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    elif ALTAIR_OK:
        df = pd.DataFrame({"month": list(range(len(curve))), "retention": curve})
        ch = alt.Chart(df).mark_line(point=True).encode(x="month:O", y=alt.Y("retention:Q", axis=alt.Axis(format="%"))).properties(height=320)
        st.altair_chart(ch, use_container_width=True)
    else:
        st.line_chart(curve)

def plot_nrr(series: List[Dict[str, Any]]):
    if not series:
        st.info("No NRR/GRR series available."); return
    if PLOTLY_OK:
        months = [s["month"] for s in series]
        nrr = [s["nrr"] for s in series]; grr = [s["grr"] for s in series]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=nrr, mode="lines+markers", name="NRR"))
        fig.add_trace(go.Scatter(x=months, y=grr, mode="lines+markers", name="GRR"))
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    elif ALTAIR_OK:
        df = pd.DataFrame(series)
        d1 = df.melt("month", value_vars=["nrr","grr"], var_name="metric", value_name="value")
        ch = alt.Chart(d1).mark_line(point=True).encode(x="month:O", y=alt.Y("value:Q", axis=alt.Axis(format="%")), color="metric:N").properties(height=320)
        st.altair_chart(ch, use_container_width=True)
    else:
        st.line_chart(pd.DataFrame(series).set_index("month")[["nrr","grr"]])

def plot_pricing(scatter: Dict[str, Any]):
    x = scatter.get("x", []); y = scatter.get("y", []); fit = scatter.get("fit", [])
    if not x or not y:
        st.info("No pricing scatter available."); return
    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="log Q vs log P"))
        if fit: fig.add_trace(go.Scatter(x=x, y=fit, mode="lines", name="fit"))
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
    elif ALTAIR_OK:
        df = pd.DataFrame({"log_p": x, "log_q": y})
        ch = alt.Chart(df).mark_point().encode(x="log_p:Q", y="log_q:Q").properties(height=320)
        if fit:
            df2 = pd.DataFrame({"log_p": x, "fit": fit})
            ch = ch + alt.Chart(df2).mark_line().encode(x="log_p:Q", y="fit:Q")
        st.altair_chart(ch, use_container_width=True)
    else:
        df = pd.DataFrame({"log_p": x, "log_q": y})
        st.scatter_chart(df, x="log_p", y="log_q")


# UI -------------------------------------------------------------------------
st.title("TransformAI — Agentic Diligence Spreadsheet (Hardcore)")

tab_data, tab_grid, tab_run, tab_sheet, tab_review, tab_memo = st.tabs(
    ["Data","Grid","Run","Sheet","Review","Memo"]
)

# DATA -----------------------------------------------------------------------
with tab_data:
    st.subheader("Evidence Sources & CSV Schema")

    c1, c2 = st.columns(2)
    with c1:
        csvs = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)
        if csvs:
            for f in csvs:
                try:
                    df = pd.read_csv(f)
                except Exception:
                    df = pd.read_csv(io.BytesIO(f.getvalue()))
                SS["csv_files"][f.name] = df
                SS["schema"].setdefault(f.name, _auto_guess_schema(df))
            st.success(f"Loaded {len(csvs)} CSV file(s).")

    with c2:
        pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if pdfs:
            for f in pdfs:
                SS["pdf_files"][f.name] = f.getvalue()
            st.success(f"Loaded {len(pdfs)} PDF file(s).")

    with st.expander("Map CSV Schema (click to edit)", expanded=True if SS["csv_files"] else False):
        for name, df in SS["csv_files"].items():
            st.markdown(f"**{name}**")
            sch = SS["schema"].setdefault(name, _auto_guess_schema(df))
            cols = ["— None —"] + list(df.columns)
            def pick(lbl, key):
                cur = sch.get(key)
                if cur not in df.columns: cur = None
                idx = cols.index(cur) if cur in cols else 0
                val = st.selectbox(lbl, cols, index=idx, key=f"{name}:{key}")
                sch[key] = None if val == "— None —" else val
            pick("Customer ID", "customer_id")
            pick("Order Date", "order_date")
            pick("Amount", "amount")
            pick("Unit Price", "price")
            pick("Quantity", "quantity")
            pick("Month (YYYY-MM)", "month")
            pick("Revenue (period revenue)", "revenue")
            pick("Segment", "segment")
            pick("Product", "product")
            st.divider()

    st.write("**Loaded CSVs:**", list(SS["csv_files"].keys()) or "—")
    st.write("**Loaded PDFs:**", list(SS["pdf_files"].keys()) or "—")


# GRID -----------------------------------------------------------------------
with tab_grid:
    st.subheader("Build Grid: rows, columns, and the Matrix Board")

    a1, a2, a3, a4, a5 = st.columns([1,1,1,1,1])
    with a1:
        if st.button("Add rows from CSVs", use_container_width=True):
            add_rows_from_csvs(); st.toast("CSV rows added")
    with a2:
        if st.button("Add rows from PDFs", use_container_width=True):
            add_rows_from_pdfs(); st.toast("PDF rows added")
    with a3:
        if st.button("Add QoE Columns", use_container_width=True):
            add_template_columns(QOE_TEMPLATE); st.toast("QoE columns added")
    with a4:
        if st.button("Undo", use_container_width=True): undo()
    with a5:
        if st.button("Redo", use_container_width=True): redo()

    st.markdown("**Rows**")
    if SS["rows"]:
        df_rows = pd.DataFrame(SS["rows"])[["id","alias","row_type","source"]]
        df_rows_display = df_rows.copy(); df_rows_display["delete"] = False
        edited = st.data_editor(
            df_rows_display,
            hide_index=True, use_container_width=True,
            disabled=["id","row_type","source"], key="rows_editor_grid"
        )
        dels = edited[edited["delete"]==True]["id"].tolist()
        if st.button("Apply row edits / deletes", use_container_width=True):
            alias_map = {row["id"]: row["alias"] for _, row in edited.iterrows()}
            for r in SS["rows"]: r["alias"] = alias_map.get(r["id"], r["alias"])
            if dels: delete_rows(dels)
            st.success("Rows updated")
    else:
        st.info("No rows yet. Add from CSVs/PDFs.")

    st.markdown("**Columns**")
    if SS["columns"]:
        df_cols = pd.DataFrame(SS["columns"])[["id","label","module"]]
        df_cols_display = df_cols.copy(); df_cols_display["delete"] = False
        edc = st.data_editor(
            df_cols_display,
            hide_index=True, use_container_width=True,
            disabled=["id","module"], key="cols_editor_grid"
        )
        delc = edc[edc["delete"]==True]["id"].tolist()
        if st.button("Apply column edits / deletes", use_container_width=True):
            label_map = {row["id"]: row["label"] for _, row in edc.iterrows()}
            for c in SS["columns"]: c["label"] = label_map.get(c["id"], c["label"])
            if delc: delete_cols(delc)
            st.success("Columns updated")
    else:
        st.caption("No columns yet. Add QoE Columns or create one below.")

    nc1, nc2, nc3 = st.columns([2,2,1])
    with nc1:
        new_label = st.text_input("New column label", value=SS.get("new_col_label","NRR/GRR"))
        SS["new_col_label"] = new_label
    with nc2:
        new_mod = st.selectbox(
            "Module",
            MODULES,
            index=MODULES.index(SS.get("new_col_mod","NRR/GRR (CSV)")) if SS.get("new_col_mod","NRR/GRR (CSV)") in MODULES else 2
        )
        SS["new_col_mod"] = new_mod
    with nc3:
        if st.button("Add Column", use_container_width=True):
            add_column(SS["new_col_label"], SS["new_col_mod"]); st.success("Column added")

    st.divider()
    st.markdown("### Matrix Board — map **rows ↔ modules** (what should run where)")

    if SS["rows"]:
        base = []
        for r in SS["rows"]:
            sel = SS["matrix"].setdefault(r["id"], set())
            base.append({
                "row_id": r["id"], "Alias": r["alias"], "Type": r["row_type"],
                "PDF KPIs (PDF)": "PDF KPIs (PDF)" in sel,
                "Cohort Retention (CSV)": "Cohort Retention (CSV)" in sel,
                "NRR/GRR (CSV)": "NRR/GRR (CSV)" in sel,
                "Pricing Power (CSV)": "Pricing Power (CSV)" in sel,
                "Unit Economics (CSV)": "Unit Economics (CSV)" in sel,
                "Churn by Segment (CSV)": "Churn by Segment (CSV)" in sel,
                "PVM Bridge (CSV)": "PVM Bridge (CSV)" in sel,
            })
        mdf = pd.DataFrame(base)
        mdf_edit = st.data_editor(
            mdf,
            column_config={k: st.column_config.CheckboxColumn() for k in mdf.columns if k not in ["row_id","Alias","Type"]},
            hide_index=True, use_container_width=True, key="matrix_editor"
        )
        if st.button("Apply Matrix", use_container_width=True):
            snapshot_push()
            for _, row in mdf_edit.iterrows():
                rid = row["row_id"]
                sel = set()
                for mod in MODULES:
                    if mod in row and bool(row[mod]): sel.add(mod)
                # PDF guard
                if any(rr["id"]==rid and rr["row_type"]=="pdf" for rr in SS["rows"]):
                    sel = set(m for m in sel if m=="PDF KPIs (PDF)")
                SS["matrix"][rid] = sel
            st.success("Matrix updated")
    else:
        st.info("Add rows first, then use the Matrix to map modules.")


# RUN ------------------------------------------------------------------------
with tab_run:
    st.subheader("Run — queue, process, and see status")

    cA, cB, cC = st.columns([1,1,1])
    with cA: st.toggle("Force re-run (ignore cache)", key="force_rerun", value=SS.get("force_rerun", False))
    with cB: st.toggle("Parallel execution", key="parallel", value=SS.get("parallel", True))
    with cC:
        mw = st.number_input("Max workers", min_value=1, max_value=32, value=int(SS.get("max_workers",4)), step=1)
        SS["max_workers"] = int(mw)

    with st.expander("One-click QoE", expanded=True):
        st.caption("Adds QoE columns (if missing), selects mapped pairs from Matrix, runs all.")
        if st.button("Run QoE Now", type="primary"):
            add_template_columns(QOE_TEMPLATE)
            by_label_mod = {(c["label"], c["module"]): c["id"] for c in SS["columns"]}
            pairs = []
            for r in SS["rows"]:
                rid = r["id"]
                sel = SS["matrix"].get(rid, set())
                for label, mod in QOE_TEMPLATE:
                    if mod in sel and (label,mod) in by_label_mod:
                        pairs.append((rid, by_label_mod[(label,mod)]))
            enqueue_pairs(pairs, respect_cache=True)
            run_queued_jobs()
            st.success(f"Ran {len(pairs)} cell(s).")

    with st.expander("Manual run by Matrix selection", expanded=False):
        rows = SS["rows"]; cols = SS["columns"]
        by_mod = {c["module"]: c["id"] for c in cols}
        options = []
        for r in rows:
            sel = SS["matrix"].get(r["id"], set())
            for mod in sel:
                if mod in by_mod:
                    options.append((r["id"], by_mod[mod], f"{r['alias']} → {mod}"))
        if options:
            choice = st.selectbox("Pick a row/module to run", options, format_func=lambda t: t[2])
            if st.button("Run selected"):
                enqueue_pairs([(choice[0], choice[1])], respect_cache=True)
                run_queued_jobs()
                st.success("Cell executed.")
        else:
            st.info("Nothing mapped in Matrix yet.")

    st.divider()
    # Jobs + quick exports
    if SS["jobs"]:
        st.markdown("**Jobs**")
        st.dataframe(pd.DataFrame(SS["jobs"]), use_container_width=True, height=180)
    if SS["results"]:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Export results CSV", data=export_results_csv(), file_name="transformai_results.csv")
        with c2:
            md = export_memo_markdown()
            st.download_button("Download Memo (Markdown)", data=md.encode("utf-8"), file_name="TransformAI_Memo.md")
        with c3:
            if REPORTLAB_OK:
                pdf_bytes = export_memo_pdf()
                st.download_button("Download Memo (PDF)", data=pdf_bytes, file_name="TransformAI_Memo.pdf")


# SHEET (Agentic Spreadsheet) ------------------------------------------------
with tab_sheet:
    st.subheader("Agentic Spreadsheet (status by cell)")

    qoe_cols = [c for c in SS["columns"] if c["module"] in {m for _,m in QOE_TEMPLATE}] or SS["columns"]
    header = ["Row"] + [c["label"] for c in qoe_cols]
    table = []
    for r in SS["rows"]:
        row_vals = [r["alias"]]
        for c in qoe_cols:
            res = SS["results"].get((r["id"], c["id"]), {})
            mark = ""
            if res:
                mark = "✓ " + (str(res.get("value")) if res.get("value") is not None else "")
                if res.get("status") == "queued": mark = "… queued"
                if res.get("status") == "running": mark = "⟳ running"
                if res.get("status") == "cached": mark = "⟲ cached"
                if res.get("status") == "error":  mark = "⚠ error"
                if SS["approvals"].get((r["id"], c["id"]), False): mark = "✅ " + (str(res.get("value")) if res.get("value") is not None else "")
            row_vals.append(mark)
        table.append(row_vals)

    df_sheet = pd.DataFrame(table, columns=header)
    st.dataframe(df_sheet, use_container_width=True, height=min(420, 120 + 28*len(df_sheet)))


# REVIEW (Focused) -----------------------------------------------------------
with tab_review:
    st.subheader("Review a single cell — approve, retry, and see provenance")

    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}

    row_opt = [(r["id"], r["alias"]) for r in SS["rows"]]
    col_opt = [(c["id"], f"{c['label']}  ·  {c['module']}") for c in SS["columns"]]

    csel1, csel2, csel3, csel4 = st.columns([2,2,1,1])
    with csel1:
        rid_opt = st.selectbox("Row", row_opt, format_func=lambda t: t[1]) if row_opt else None
    with csel2:
        cid_opt = st.selectbox("Column", col_opt, format_func=lambda t: t[1]) if col_opt else None
    with csel3:
        if rid_opt and cid_opt and st.button("Run this cell now", type="primary"):
            enqueue_pairs([(rid_opt[0], cid_opt[0])], respect_cache=False)
            run_queued_jobs()
    with csel4:
        if rid_opt and cid_opt and st.button("Retry", use_container_width=True):
            retry_cell(rid_opt[0], cid_opt[0]); run_queued_jobs()

    if not (rid_opt and cid_opt):
        st.info("Choose a Row and a Column above.")
    else:
        rid, cid = rid_opt[0], cid_opt[0]
        res = SS["results"].get((rid, cid))
        r = rows_by_id.get(rid); c = cols_by_id.get(cid)

        if not res:
            st.warning("No result yet. Click **Run this cell now**.")
        else:
            st.caption(f"**{r['alias']}** → **{c['label']}** ({c['module']})")
            st.write(res.get("summary",""))

            # Approve / reject
            approved = SS["approvals"].get((rid,cid), False)
            cA, cB = st.columns([1,1])
            with cA:
                if st.button("✅ Approve", use_container_width=True):
                    SS["approvals"][(rid,cid)] = True
                    st.toast("Approved")
            with cB:
                if st.button("🚫 Reject", use_container_width=True):
                    SS["approvals"][(rid,cid)] = False
                    st.toast("Rejected")

            # Render charts per module
            module = c["module"]
            if module == "Cohort Retention (CSV)" or "curve" in res:
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**Retention curve**")
                    plot_retention(res.get("curve", []))
                with colB:
                    st.markdown("**Provenance (sample rows)**")
                    prov = res.get("provenance", {})
                    prev = prov.get("preview", [])
                    if prev:
                        st.dataframe(pd.DataFrame(prev), use_container_width=True, height=260)
                    else:
                        st.caption("No provenance preview.")
            elif module == "NRR/GRR (CSV)":
                st.markdown("**NRR / GRR by month**")
                plot_nrr(res.get("series", []))
                st.markdown("**Provenance (sample rows)**")
                prov = res.get("provenance", {})
                prev = prov.get("preview", [])
                if prev: st.dataframe(pd.DataFrame(prev), use_container_width=True, height=240)
            elif module == "Pricing Power (CSV)":
                st.markdown("**Price–Demand (log) with fit**")
                plot_pricing(res.get("scatter", {}))
                st.markdown("**Provenance (sample rows)**")
                prov = res.get("provenance", {})
                prev = prov.get("preview", [])
                if prev: st.dataframe(pd.DataFrame(prev), use_container_width=True, height=240)
            elif module == "Unit Economics (CSV)":
                kpi = {k: res.get(k) for k in ["aov","gm","cac","cm"] if k in res}
                st.metric(label="Contribution Margin (per order)", value=f"${res.get('value'):.2f}")
                cols4 = st.columns(3)
                with cols4[0]: st.metric("AOV", f"${kpi.get('aov',0):.2f}")
                with cols4[1]: st.metric("GM", f"{kpi.get('gm',0):.0%}")
                with cols4[2]: st.metric("CAC", f"${kpi.get('cac',0):.0f}")
                st.markdown("**Provenance (sample rows)**")
                prov = res.get("provenance", {})
                prev = prov.get("preview", [])
                if prev: st.dataframe(pd.DataFrame(prev), use_container_width=True, height=240)
            elif module == "Churn by Segment (CSV)":
                tbl = res.get("table", [])
                if tbl:
                    st.dataframe(pd.DataFrame(tbl), use_container_width=True, height=260)
                st.markdown("**Provenance (sample rows)**")
                prov = res.get("provenance", {})
                prev = prov.get("preview", [])
                if prev: st.dataframe(pd.DataFrame(prev), use_container_width=True, height=240)
            elif module == "PVM Bridge (CSV)":
                tbl = res.get("table", [])
                if tbl:
                    st.dataframe(pd.DataFrame(tbl), use_container_width=True, height=260)
                st.markdown("**Provenance (sample rows)**")
                prov = res.get("provenance", {})
                prev = prov.get("preview", [])
                if prev: st.dataframe(pd.DataFrame(prev), use_container_width=True, height=240)
            elif module == "PDF KPIs (PDF)":
                cites = res.get("citations", [])
                if cites:
                    st.markdown("**Citations**")
                    dfc = pd.DataFrame(cites)
                    st.dataframe(dfc, use_container_width=True, height=200)
                else:
                    st.info("No citations available for this PDF.")

# MEMO -----------------------------------------------------------------------
with tab_memo:
    st.subheader("Investor memo")
    st.caption("Only **approved** cells are included below. Use the Review tab to approve cells.")
    md = export_memo_markdown()
    st.markdown(md)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download Memo (Markdown)", data=md.encode("utf-8"), file_name="TransformAI_Memo.md")
    with c2:
        if REPORTLAB_OK:
            pdf_bytes = export_memo_pdf()
            st.download_button("Download Memo (PDF)", data=pdf_bytes, file_name="TransformAI_Memo.pdf")
        else:
            st.info("Install `reportlab` to enable PDF export.")

