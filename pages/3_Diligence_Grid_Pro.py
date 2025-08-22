# pages/3_Diligence_Grid_Pro.py
# TransformAI — Diligence Grid (Pro, wide + matrix UI + Agentic Spreadsheet)
from __future__ import annotations
import io, json, time, uuid, math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional PDF export
try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# Plotly for charts
try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# -----------------------------------------------------------------------------
# Page & CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="TransformAI — Diligence Grid (Pro)", layout="wide")
st.markdown(
    """
<style>
/* Give us a wider canvas and prevent any header clipping */
.block-container { max-width: 2000px !important; padding-top: 0.35rem; }
h1, h2, h3 { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
[data-testid="stMetricValue"] { font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)

SS = st.session_state

# -----------------------------------------------------------------------------
# Helpers / State
# -----------------------------------------------------------------------------
def ensure_state():
    SS.setdefault("csv_files", {})            # {name: df}
    SS.setdefault("pdf_files", {})            # {name: bytes}
    SS.setdefault("schema", {})               # {csv_name: {canonical: source_col or None}}

    SS.setdefault("rows", [])                 # [{id, alias, row_type ('table'|'pdf'), source}]
    SS.setdefault("columns", [])              # [{id, label, module}]
    SS.setdefault("matrix", {})               # {row_id: set([module,...])}

    # RESULTS live as {(row_id, col_id): {...}} in-memory
    SS.setdefault("results", {})              # {(row_id, col_id): {...}}
    SS.setdefault("cache_key", {})            # {(row_id, col_id): str}

    SS.setdefault("jobs", [])
    SS.setdefault("force_rerun", False)

    SS.setdefault("whatif_gm", 0.62)
    SS.setdefault("whatif_cac", 42.0)

    SS.setdefault("undo", [])
    SS.setdefault("redo", [])

ensure_state()

def uid(p="row"): return f"{p}_{uuid.uuid4().hex[:8]}"
def now_ts(): return int(time.time())

# ----------------------- tuple-safe pack/unpack for results -------------------
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
        elif isinstance(ks, str) and ks.startswith("(") and ks.endswith(")"):
            try:
                tup = eval(ks, {"__builtins__": {}}, {})
                if isinstance(tup, tuple) and len(tup) == 2:
                    out[(str(tup[0]), str(tup[1]))] = v
            except Exception:
                pass
    return out

# ----------------------------- snapshots (tuple-safe) -------------------------
def snapshot_push():
    SS["undo"].append(json.dumps({
        "rows": SS["rows"],
        "columns": SS["columns"],
        "matrix": {k: list(v) for k, v in SS["matrix"].items()},
        "results": _pack_results(SS["results"]),
    }, default=str))
    SS["redo"].clear()

def snapshot_apply(snap: str):
    data = json.loads(snap)
    SS["rows"]    = data.get("rows", [])
    SS["columns"] = data.get("columns", [])
    SS["matrix"]  = {k: set(v) for k, v in data.get("matrix", {}).items()}
    SS["results"] = _unpack_results(data.get("results", {}))

def undo():
    if not SS.get("undo"):
        return
    cur = json.dumps({
        "rows": SS["rows"],
        "columns": SS["columns"],
        "matrix": {k: list(v) for k, v in SS["matrix"].items()},
        "results": _pack_results(SS["results"]),
    }, default=str)
    snap = SS["undo"].pop()
    SS["redo"].append(cur)
    snapshot_apply(snap)
    st.toast("Undone")

def redo():
    if not SS.get("redo"):
        return
    cur = json.dumps({
        "rows": SS["rows"],
        "columns": SS["columns"],
        "matrix": {k: list(v) for k, v in SS["matrix"].items()},
        "results": _pack_results(SS["results"]),
    }, default=str)
    snap = SS["redo"].pop()
    SS["undo"].append(cur)
    snapshot_apply(snap)
    st.toast("Redone")

# -----------------------------------------------------------------------------
# Schema helpers
# -----------------------------------------------------------------------------
CANONICAL = ["customer_id","order_date","amount","price","quantity","month","revenue"]

def _auto_guess_schema(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    return {
        "customer_id": pick("customer_id","cust_id","user_id","buyer_id"),
        "order_date":  pick("order_date","date","created_at","timestamp"),
        "amount":      pick("amount","net_revenue","revenue","sales","value"),
        "price":       pick("price","unit_price","avg_price"),
        "quantity":    pick("quantity","qty","units"),
        "month":       pick("month","order_month","period"),
        "revenue":     pick("revenue","net_revenue","amount","sales"),
    }

def materialize_df(csv_name: str) -> pd.DataFrame:
    df = SS["csv_files"].get(csv_name, pd.DataFrame()).copy()
    if df.empty:
        return df
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
    return df

# -----------------------------------------------------------------------------
# Grid helpers
# -----------------------------------------------------------------------------
MODULES = [
    "PDF KPIs (PDF)",
    "Cohort Retention (CSV)",
    "Pricing Power (CSV)",
    "NRR/GRR (CSV)",
    "Unit Economics (CSV)",
]

QOE_TEMPLATE = [
    ("PDF KPIs",        "PDF KPIs (PDF)"),
    ("Unit Economics",  "Unit Economics (CSV)"),
    ("NRR/GRR",         "NRR/GRR (CSV)"),
    ("Pricing Power",   "Pricing Power (CSV)"),
    ("Cohort Retention","Cohort Retention (CSV)"),
]

def add_rows_from_csvs():
    snapshot_push()
    for name in SS["csv_files"].keys():
        if not any(r["source"] == name for r in SS["rows"]):
            rid = uid("row")
            SS["rows"].append({"id": rid, "alias": name.replace(".csv",""), "row_type":"table", "source": name})
            SS["matrix"].setdefault(rid, set(["Cohort Retention (CSV)","Pricing Power (CSV)","NRR/GRR (CSV)","Unit Economics (CSV)"]))

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

def delete_cols(col_ids: List[str]):
    if not col_ids: return
    snapshot_push()
    SS["columns"] = [c for c in SS["columns"] if c["id"] not in col_ids]
    SS["results"] = {k:v for k,v in SS["results"].items() if k[1] not in col_ids}

# -----------------------------------------------------------------------------
# Engines
# -----------------------------------------------------------------------------
def _pdf_kpis(_raw: bytes) -> Dict[str, Any]:
    # Stub narrative; PDF parsing can be wired to your backend later.
    return dict(summary="Revenue ≈ $12.5M; EBITDA ≈ $3.2M; Gross margin ≈ 64%; Churn ≈ 4% (H2).")

def _cohort(df: pd.DataFrame) -> Dict[str, Any]:
    # Proper cohort matrix + average curve + heatmap payload
    d = df.copy()
    if "customer_id" not in d.columns or ("order_date" not in d.columns and "month" not in d.columns):
        return dict(value=None, summary="Map schema: customer_id + order_date/month required.")
    if "order_date" in d.columns:
        d["order_date"] = pd.to_datetime(d["order_date"], errors="coerce")
        d["month"] = d["order_date"].dt.to_period("M").astype(str)
    d = d.dropna(subset=["customer_id","month"])
    first = (d.groupby("customer_id")["month"].min()).rename("cohort")
    d = d.merge(first, on="customer_id", how="left")
    # lag in months between txn month and cohort month
    d["lag"] = (
        pd.to_datetime(d["month"] + "-01") - pd.to_datetime(d["cohort"] + "-01")
    ).dt.days // 30
    counts = d.groupby(["cohort","lag"])["customer_id"].nunique().unstack(fill_value=0).sort_index()
    if 0 not in counts.columns:
        return dict(value=None, summary="Could not form cohorts (no lag 0).")
    sizes = counts[0].replace(0, np.nan)
    retention = (counts.div(sizes, axis=0)).fillna(0.0)
    avg_curve = retention.mean(axis=0).to_list()[:6]
    m3 = avg_curve[3] if len(avg_curve) > 3 else None
    # heatmap payload
    heat = {
        "z": retention.values.tolist(),
        "x": [int(c) for c in retention.columns.tolist()],
        "y": retention.index.tolist()
    }
    return dict(
        value=m3,
        curve=avg_curve,
        heatmap=heat,
        summary=f"Avg retention stabilizes ~M3 at {m3:.0%}." if m3 is not None else "Retention computed."
    )

def _pricing(df: pd.DataFrame) -> Dict[str, Any]:
    d = df.copy()
    if "price" not in d.columns or "quantity" not in d.columns:
        return dict(value=None, summary="Map schema: price + quantity required.")
    d = d.replace([np.inf, -np.inf, 0], np.nan).dropna(subset=["price","quantity"])
    d = d[(d["price"]>0) & (d["quantity"]>0)]
    if len(d) < 8:
        return dict(value=None, summary="Not enough observations for elasticity.")
    lp = np.log(d["price"].astype(float))
    lq = np.log(d["quantity"].astype(float))
    b, a = np.polyfit(lp, lq, 1)  # y = b*x + a
    e = round(b, 2)
    verdict = "inelastic" if abs(e) < 1 else "elastic"
    fit = b*lp + a
    return dict(
        value=e,
        summary=f"Own-price elasticity ≈ {e} → {verdict}.",
        scatter=dict(x=lp.tolist(), y=lq.tolist(), fit=fit.tolist())
    )

def _nrr_grr(df: pd.DataFrame) -> Dict[str, Any]:
    d = df.copy()
    # Path A: explicit subscription schedule
    if {"month","mrr_begin","new_mrr","expansion_mrr","contraction_mrr","churn_mrr","mrr_end"}.issubset(d.columns):
        ser = []
        for _, r in d.iterrows():
            base = float(r["mrr_begin"]) if r["mrr_begin"] else 0.0
            if base <= 0: continue
            grr = (base - float(r["contraction_mrr"]) - float(r["churn_mrr"])) / base
            nrr = (base - float(r["contraction_mrr"]) - float(r["churn_mrr"]) + float(r["expansion_mrr"]) + float(r["new_mrr"])) / base
            ser.append({"month": str(r["month"]), "grr": float(grr), "nrr": float(nrr)})
        if not ser:
            return dict(value=None, summary="No valid GRR/NRR rows.")
        latest = ser[-1]
        return dict(value=latest["nrr"], summary=f"Latest ({latest['month']}): GRR {latest['grr']:.0%}, NRR {latest['nrr']:.0%}.", series=ser)

    # Path B: derive from revenue by month (crude proxy)
    if {"customer_id","month","revenue"}.issubset(d.columns):
        g = d.groupby(["customer_id","month"])["revenue"].sum().reset_index()
        months = sorted(g["month"].unique())
        ser = []
        for i in range(1, len(months)):
            prev, cur = months[i-1], months[i]
            base = g[g["month"]==prev]["revenue"].sum()
            kept = g[g["month"]==cur]["revenue"].sum()
            grr = kept/base if base else 0.0
            nrr = min(1.3, grr + 0.05)  # toy uplift
            ser.append({"month": cur, "grr": float(grr), "nrr": float(nrr)})
        if not ser:
            return dict(value=None, summary="Need at least two months of revenue.")
        latest = ser[-1]
        return dict(value=latest["nrr"], summary=f"Latest ({latest['month']}): GRR {latest['grr']:.0%}, NRR {latest['nrr']:.0%}.", series=ser)

    return dict(value=None, summary="Map schema for NRR/GRR: (month + MRR schedule) or (customer_id + month + revenue).")

def _unit_econ(df: pd.DataFrame, gm: float = 0.62, cac: float = 42.0) -> Dict[str, Any]:
    # Recognize QoE monthly P&L OR AR/Inventory OR generic orders
    cols = set(df.columns)
    if {"month","revenue","cogs","ebitda","ebitda_margin_pct"}.issubset(cols):
        m = float(df["ebitda_margin_pct"].tail(3).mean())
        e = float(df["ebitda"].iloc[-1])
        return dict(value=m, summary=f"EBITDA margin (3-mo avg) ≈ {m:.0%}; latest EBITDA ${e:,.0f}.",
                    pnl={"months": df["month"].tolist(), "ebitda_margin": df["ebitda_margin_pct"].tolist()})
    if {"current","30d","60d","90d"}.issubset(cols):
        ar = df[["current","30d","60d","90d"]].sum()
        total = float(ar.sum()) or 1.0
        dso = (0*ar["current"] + 30*ar["30d"] + 60*ar["60d"] + 90*ar["90d"]) / total
        return dict(value=dso, summary=f"Est. DSO ≈ {dso:.0f} days from aging buckets.",
                    aging={"buckets":["current","30d","60d","90d"], "values":[float(ar[x]) for x in ["current","30d","60d","90d"]]})
    # generic orders proxy
    aov = float(df["revenue"].mean()) if "revenue" in cols else float(df.select_dtypes(np.number).sum(axis=1).mean())
    cm = gm*aov - cac
    return dict(value=cm, summary=f"AOV ${aov:.2f}, GM {gm:.0%}, CAC ${cac:.0f} → CM ${cm:.2f}.",
                cm_breakdown={"aov":aov,"gm":gm,"cac":cac,"cm":cm})

# -----------------------------------------------------------------------------
# Cache/Run
# -----------------------------------------------------------------------------
def cache_key_for(row: Dict[str,Any], col: Dict[str,Any]) -> str:
    if row["row_type"] == "pdf":
        return f"pdf::{row['source']}::{col['module']}"
    sch = SS["schema"].get(row["source"], {})
    return f"csv::{row['source']}::{col['module']}::{json.dumps(sch, sort_keys=True)}"

def execute_cell(row: Dict[str,Any], col: Dict[str,Any]) -> Dict[str,Any]:
    mod = col["module"]
    if row["row_type"] == "pdf" and mod != "PDF KPIs (PDF)":
        return {"status":"done","value":None,"summary":"Use **PDF KPIs (PDF)** for PDF rows.","last_run": now_ts()}
    if mod == "PDF KPIs (PDF)":
        raw = SS["pdf_files"].get(row["source"], b"")
        k = _pdf_kpis(raw)
        return {"status":"done","value":None,"summary":k["summary"],"last_run": now_ts()}
    if mod == "Cohort Retention (CSV)":
        df = materialize_df(row["source"])
        k = _cohort(df)
        return {"status":"done","last_run": now_ts(), **k}
    if mod == "Pricing Power (CSV)":
        df = materialize_df(row["source"])
        k = _pricing(df)
        return {"status":"done","last_run": now_ts(), **k}
    if mod == "NRR/GRR (CSV)":
        df = materialize_df(row["source"])
        k = _nrr_grr(df)
        return {"status":"done","last_run": now_ts(), **k}
    if mod == "Unit Economics (CSV)":
        df = materialize_df(row["source"])
        k = _unit_econ(df, gm=SS.get("whatif_gm",0.62), cac=SS.get("whatif_cac",42.0))
        return {"status":"done","last_run": now_ts(), **k}
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

def run_queued_jobs():
    by_r = {r["id"]: r for r in SS["rows"]}
    by_c = {c["id"]: c for c in SS["columns"]}
    for j in SS["jobs"]:
        if j["status"] not in {"queued","retry"}: continue
        rid, cid = j["rid"], j["cid"]
        row, col = by_r.get(rid), by_c.get(cid)
        if not row or not col:
            j["status"]="error"; j["note"]="row/col missing"; j["ended"]=now_ts(); continue
        j["status"]="running"; j["started"]=now_ts()
        try:
            SS["results"][(rid,cid)] = execute_cell(row, col)
            j["status"]="done"; j["ended"]=now_ts()
        except Exception as e:
            SS["results"][(rid,cid)]={"status":"error","value":None,"summary":str(e),"last_run": now_ts()}
            j["status"]="error"; j["note"]=str(e); j["ended"]=now_ts()

def retry_cell(rid: str, cid: str):
    SS["jobs"].insert(0, {"rid":rid,"cid":cid,"status":"retry","started":None,"ended":None,"note":"manual retry"})

# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------
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
            status=res.get("status"), value=res.get("value"), summary=res.get("summary"), last_run=res.get("last_run")
        ))
    return pd.DataFrame(out).to_csv(index=False).encode("utf-8")

def export_results_pdf() -> bytes:
    if not REPORTLAB_OK: raise RuntimeError("reportlab not installed")
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    w,h = LETTER
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, h-72, "TransformAI — Evidence Grid Summary")
    y = h-100; c.setFont("Helvetica", 10)
    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}
    for (rid,cid),res in list(SS["results"].items()):
        r = rows_by_id.get(rid); cdef = cols_by_id.get(cid)
        if not r or not cdef: continue
        line = f"{r['alias']} → {cdef['label']}: {res.get('summary')}"
        for chunk in [line[i:i+95] for i in range(0,len(line),95)]:
            if y<72: c.showPage(); y=h-72; c.setFont("Helvetica",10)
            c.drawString(72,y,chunk); y-=14
    c.showPage(); c.save(); buf.seek(0)
    return buf.getvalue()

# -----------------------------------------------------------------------------
# UI — Tabs
# -----------------------------------------------------------------------------
st.title("TransformAI — Diligence Grid (Pro)")
tab_data, tab_grid, tab_run, tab_sheet, tab_review, tab_memo = st.tabs(
    ["Data","Grid","Run","Sheet","Review","Memo"]
)

# --------------------------- DATA ---------------------------
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

    with st.expander("Map CSV Schema (click to edit)", expanded=bool(SS["csv_files"])):
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
            st.divider()

    st.write("**Loaded CSVs:**", list(SS["csv_files"].keys()) or "—")
    st.write("**Loaded PDFs:**", list(SS["pdf_files"].keys()) or "—")

# --------------------------- GRID ---------------------------
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

    # Inline rows
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

    # Inline columns
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

    # New column
    nc1, nc2, nc3 = st.columns([2,2,1])
    with nc1:
        new_label = st.text_input("New column label", value=SS.get("new_col_label","NRR/GRR"))
        SS["new_col_label"] = new_label
    with nc2:
        new_mod = st.selectbox("Module", MODULES, index=MODULES.index(SS.get("new_col_mod","NRR/GRR (CSV)")) if SS.get("new_col_mod","NRR/GRR (CSV)") in MODULES else 3)
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
                "Pricing Power (CSV)": "Pricing Power (CSV)" in sel,
                "NRR/GRR (CSV)": "NRR/GRR (CSV)" in sel,
                "Unit Economics (CSV)": "Unit Economics (CSV)" in sel,
            })
        mdf = pd.DataFrame(base)
        mdf_edit = st.data_editor(
            mdf,
            column_config={
                "PDF KPIs (PDF)": st.column_config.CheckboxColumn(),
                "Cohort Retention (CSV)": st.column_config.CheckboxColumn(),
                "Pricing Power (CSV)": st.column_config.CheckboxColumn(),
                "NRR/GRR (CSV)": st.column_config.CheckboxColumn(),
                "Unit Economics (CSV)": st.column_config.CheckboxColumn(),
            },
            hide_index=True, use_container_width=True, key="matrix_editor"
        )
        if st.button("Apply Matrix", use_container_width=True):
            snapshot_push()
            for _, row in mdf_edit.iterrows():
                rid = row["row_id"]
                sel = set()
                for mod in MODULES:
                    if mod in row and bool(row[mod]): sel.add(mod)
                if any(rr["id"]==rid and rr["row_type"]=="pdf" for rr in SS["rows"]):
                    sel = set(m for m in sel if m=="PDF KPIs (PDF)")
                SS["matrix"][rid] = sel
            st.success("Matrix updated")
    else:
        st.info("Add rows first, then use the Matrix to map modules.")

# --------------------------- RUN ----------------------------
with tab_run:
    st.subheader("Run — queue, process, and see status")
    colA, colB, colC = st.columns([1,1,6])
    colA.toggle("Force re-run (ignore cache)", key="force_rerun", value=SS.get("force_rerun", False))

    # One-click QoE
    with st.expander("One-click QoE (recommended demo)", expanded=True):
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
            if not pairs:
                st.warning("Matrix is empty. Go to Grid → Apply Matrix first.")
            else:
                enqueue_pairs(pairs, respect_cache=True)
                run_queued_jobs()
                st.success(f"Ran {len(pairs)} cell(s).")

    # Manual: run all mapped
    if st.button("Run All Mapped (Manual)"):
        by_mod = {c["module"]: c["id"] for c in SS["columns"]}
        pairs = []
        for r in SS["rows"]:
            sel = SS["matrix"].get(r["id"], set())
            for m in sel:
                cid = by_mod.get(m)
                if cid: pairs.append((r["id"], cid))
        if not pairs:
            st.warning("No mapped pairs. Configure Matrix first.")
        else:
            enqueue_pairs(pairs, respect_cache=True)
            run_queued_jobs()
            st.success(f"Ran {len(pairs)} cell(s).")

    # Jobs table
    if SS["jobs"]:
        st.markdown("**Job History**")
        st.dataframe(pd.DataFrame(SS["jobs"]), use_container_width=True, height=220)

# --------------------------- SHEET --------------------------
with tab_sheet:
    st.subheader("Agentic Spreadsheet (status by cell)")
    if SS["rows"] and SS["columns"]:
        rows_by_id = {r["id"]: r for r in SS["rows"]}
        cols_by_id = {c["id"]: c for c in SS["columns"]}
        # build a wide status sheet
        df_sheet = []
        for r in SS["rows"]:
            row = {"Row": r["alias"]}
            for c in SS["columns"]:
                res = SS["results"].get((r["id"], c["id"]), {})
                stt = res.get("status")
                if stt == "done":
                    val = res.get("value")
                    if isinstance(val, float):
                        val_str = f"{val:.2f}"
                    elif val is None:
                        val_str = "✓"
                    else:
                        val_str = str(val)
                    row[c["label"]] = f"✓ {val_str}"
                elif stt == "cached":
                    row[c["label"]] = "⟳ cached"
                elif stt == "running":
                    row[c["label"]] = "… running"
                elif stt == "queued":
                    row[c["label"]] = "⏳ queued"
                elif stt == "error":
                    row[c["label"]] = "⚠ error"
                else:
                    row[c["label"]] = ""
            df_sheet.append(row)
        st.dataframe(pd.DataFrame(df_sheet), use_container_width=True, height=280)
    else:
        st.info("After you add rows/columns and run, you’ll see a sheet of cell statuses here.")

# --------------------------- REVIEW -------------------------
with tab_review:
    st.subheader("Investor View — results & visuals")

    # Results table
    if SS["results"]:
        rows_by_id = {r["id"]: r for r in SS["rows"]}
        cols_by_id = {c["id"]: c for c in SS["columns"]}
        view = []
        for (rid, cid), res in SS["results"].items():
            r = rows_by_id.get(rid); c = cols_by_id.get(cid)
            if not r or not c: continue
            view.append({
                "Row": r["alias"], "Column": c["label"], "Module": c["module"],
                "Status": res.get("status"), "Value": res.get("value"),
                "Summary": res.get("summary")
            })
        st.dataframe(pd.DataFrame(view).sort_values(["Row","Column"]), use_container_width=True, height=260)
    else:
        st.info("Run cells first to see results.")
    
    st.markdown("### Visualizations")

    # Compose multi-plot figure if Plotly available (spacing + height tuned to prevent overlap)
    if PLOTLY_OK and SS["results"]:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Retention curve","NRR/GRR","Pricing scatter","Cohort heatmap"),
            specs=[[{"type":"xy"},{"type":"xy"}],
                   [{"type":"xy"},{"type":"heatmap"}]],
            horizontal_spacing=0.09,
            vertical_spacing=0.16
        )

        # Retention curve & heatmap
        rcells = [v for v in SS["results"].values() if isinstance(v, dict) and ("curve" in v or "heatmap" in v)]
        if rcells:
            v = rcells[-1]
            if "curve" in v and v["curve"]:
                x = list(range(len(v["curve"])))
                fig.add_trace(go.Scatter(x=x, y=v["curve"], mode="lines+markers", name="Avg retention"), row=1, col=1)
                fig.update_yaxes(tickformat=".0%", row=1, col=1)
            if "heatmap" in v:
                hm = v["heatmap"]
                fig.add_trace(go.Heatmap(z=hm["z"], x=hm["x"], y=hm["y"], colorbar=dict(title="retention")), row=2, col=2)

        # NRR/GRR
        nrrs = [v for v in SS["results"].values() if "series" in v]
        if nrrs:
            s = nrrs[-1]["series"]
            fig.add_trace(go.Scatter(x=[d["month"] for d in s], y=[d["grr"] for d in s], mode="lines+markers", name="GRR"), row=1, col=2)
            fig.add_trace(go.Scatter(x=[d["month"] for d in s], y=[d["nrr"] for d in s], mode="lines+markers", name="NRR"), row=1, col=2)
            fig.update_yaxes(tickformat=".0%", row=1, col=2)

        # Pricing scatter
        pr = [v for v in SS["results"].values() if "scatter" in v]
        if pr:
            s = pr[-1]["scatter"]
            fig.add_trace(go.Scatter(x=s["x"], y=s["y"], mode="markers", name="log Q vs log P"), row=2, col=1)
            fig.add_trace(go.Scatter(x=s["x"], y=s["fit"], mode="lines", name="fit"), row=2, col=1)
            fig.update_xaxes(title="ln(price)", row=2, col=1)
            fig.update_yaxes(title="ln(quantity)", row=2, col=1)

        fig.update_annotations(font_size=12)
        fig.update_layout(
            height=900,
            showlegend=True,
            margin=dict(l=20, r=20, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=-0.14, xanchor="center", x=0.5),
            hovermode="x unified"
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False, "modeBarButtonsToAdd": ["zoom2d","pan2d","resetScale2d"]}
        )
    else:
        st.caption("Charts appear here after you run cells (Plotly recommended).")

# --------------------------- MEMO ---------------------------
with tab_memo:
    st.subheader("Memo / Export")
    c1, c2, c3, c4 = st.columns([1,1,1,6])
    c1.number_input("What-if Gross Margin", 0.0, 1.0, key="whatif_gm", step=0.01)
    c2.number_input("What-if CAC ($)", 0.0, 5000.0, key="whatif_cac", step=1.0)
    if c3.button("Recompute Unit Econ on latest CSV row"):
        # run Unit Economics on the last table row if available
        tables = [r for r in SS["rows"] if r["row_type"]=="table"]
        cols_unit = [c for c in SS["columns"] if c["module"]=="Unit Economics (CSV)"]
        if tables and cols_unit:
            rid = tables[-1]["id"]; cid = cols_unit[-1]["id"]
            SS["results"][(rid,cid)] = execute_cell(tables[-1], cols_unit[-1])
            st.success("Recomputed Unit Economics.")
        else:
            st.warning("Need at least one CSV row and a Unit Economics column.")
    st.divider()
    ec1, ec2 = st.columns(2)
    with ec1:
        if st.button("Export Results (CSV)"):
            st.download_button("Download CSV", data=export_results_csv(), file_name="TransformAI_results.csv", mime="text/csv")
    with ec2:
        if REPORTLAB_OK and st.button("Export Summary (PDF)"):
            try:
                st.download_button("Download PDF", data=export_results_pdf(), file_name="TransformAI_summary.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF export failed: {e}")
