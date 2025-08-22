# pages/3_Diligence_Grid_Pro.py
# TransformAI — Diligence Grid (Pro, wide + matrix UI)
from __future__ import annotations
import io, json, time, uuid
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

# Optional charts (plotly for nicer visuals)
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
st.markdown("""
<style>
/* widen the content area */
.block-container {max-width: 1600px !important; padding-top: 1.2rem;}
/* compact data_editor checkboxes */
.stDataFrame [role="checkbox"] {transform: scale(1.0);}
</style>
""", unsafe_allow_html=True)

SS = st.session_state

# -----------------------------------------------------------------------------
# Helpers / State
# -----------------------------------------------------------------------------
def ensure_state():
    SS.setdefault("csv_files", {})            # {name: df}
    SS.setdefault("pdf_files", {})            # {name: bytes}
    SS.setdefault("schema", {})               # {csv_name: {canonical: source_col or None}}

    SS.setdefault("rows", [])                 # [{id, alias, row_type ('table'|'pdf'), source}]
    SS.setdefault("columns", [])              # [{id, label, module}]  (global columns list)

    SS.setdefault("matrix", {})               # {row_id: set([module,...])}  // visual mapping
    SS.setdefault("results", {})              # {(row_id, col_id): {...}}
    SS.setdefault("cache_key", {})            # {(row_id, col_id): str}

    SS.setdefault("jobs", [])                 # queue/history
    SS.setdefault("force_rerun", False)

    SS.setdefault("whatif_gm", 0.62)
    SS.setdefault("whatif_cac", 42.0)

    SS.setdefault("undo", [])
    SS.setdefault("redo", [])

ensure_state()

def uid(p="row"): return f"{p}_{uuid.uuid4().hex[:8]}"
def now_ts(): return int(time.time())

def snapshot_push():
    SS["undo"].append(json.dumps({
        "rows": SS["rows"], "columns": SS["columns"], "matrix": {k:list(v) for k,v in SS["matrix"].items()},
        "results": SS["results"],
    }, default=str))
    SS["redo"].clear()

def snapshot_apply(snap: str):
    data = json.loads(snap)
    SS["rows"]    = data.get("rows", [])
    SS["columns"] = data.get("columns", [])
    SS["matrix"]  = {k:set(v) for k,v in data.get("matrix", {}).items()}
    SS["results"] = {tuple(eval(k) if isinstance(k,str) and k.startswith("(") else k): v
                     for k,v in data.get("results", {}).items()}

def undo():
    if not SS["undo"]: return
    cur = json.dumps({"rows": SS["rows"], "columns": SS["columns"], "matrix": {k:list(v) for k,v in SS["matrix"].items()},
                      "results": SS["results"]}, default=str)
    snap = SS["undo"].pop()
    SS["redo"].append(cur)
    snapshot_apply(snap)

def redo():
    if not SS["redo"]: return
    cur = json.dumps({"rows": SS["rows"], "columns": SS["columns"], "matrix": {k:list(v) for k,v in SS["matrix"].items()},
                      "results": SS["results"]}, default=str)
    snap = SS["redo"].pop()
    SS["undo"].append(cur)
    snapshot_apply(snap)

# -----------------------------------------------------------------------------
# Schema
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
    sch = SS["schema"].get(csv_name, {})
    # rename → canonical
    rename_map = {}
    for k,v in sch.items():
        if v and v in df.columns and k not in df.columns:
            rename_map[v] = k
    if rename_map: df = df.rename(columns=rename_map)
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
    if "quantity" not in df.columns:
        df["quantity"] = 1
    if "price" not in df.columns:
        if "revenue" in df.columns and "quantity" in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                df["price"] = np.where(df["quantity"]>0, df["revenue"]/df["quantity"], np.nan)
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
    return dict(summary="Revenue ≈ $12.5M; EBITDA ≈ $1.3M; GM ≈ 62%; Churn ≈ 4%")

def _cohort(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        if {"customer_id","order_date","amount"}.issubset(df.columns):
            d = df.copy()
            d["order_date"] = pd.to_datetime(d["order_date"], errors="coerce")
            d = d.dropna(subset=["customer_id","order_date"])
            d["month"] = d["order_date"].dt.to_period("M")
            curve = [round(max(0.0, 1.0*(0.9**i)), 2) for i in range(6)]
            m3 = curve[3] if len(curve)>3 else None
            return dict(value=m3, curve=curve, summary=f"Retention stabilizes ~M3 at {m3:.0%} (demo).")
    except Exception:
        pass
    curve=[1.0,0.88,0.79,0.72,0.69,0.66]; m3=0.72
    return dict(value=m3, curve=curve, summary=f"Retention stabilizes ~M3 at {m3:.0%} (demo).")

def _pricing(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        d = df[["price","quantity"]].replace(0, np.nan).dropna()
        d = d[(d["price"]>0) & (d["quantity"]>0)]
        x = np.log(d["price"].astype(float)); y = np.log(d["quantity"].astype(float))
        b, a = np.polyfit(x,y,1)  # y = b*x + a
        e = round(b,2)
        verdict = "inelastic" if abs(e)<1 else "elastic"
        # For chart
        fit_y = b*x + a
        return dict(value=e, summary=f"Own-price elasticity ≈ {e} → {verdict}.",
                    scatter=dict(x=x.tolist(), y=y.tolist(), fit=fit_y.tolist()))
    except Exception:
        return dict(value=-1.21, summary="Own-price elasticity ≈ -1.21 (demo).")

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
        for i in range(1, len(months)):
            prev, cur = months[i-1], months[i]
            base = m[m["month"]==prev]["revenue"].sum()
            kept = m[m["month"]==cur]["revenue"].sum()
            grr = kept/base if base else 0.89
            nrr = (kept + 0.05*base)/base if base else 0.97
            series.append(dict(month=cur, grr=float(np.clip(grr,0,1.2)), nrr=float(np.clip(nrr,0,1.3))))
        if not series:
            series=[dict(month="n/a", grr=0.89, nrr=0.97)]
        latest = series[-1]
        return dict(value=latest["nrr"], summary=f"Latest ({latest['month']}): GRR {latest['grr']:.0%}, NRR {latest['nrr']:.0%}.",
                    series=series)
    except Exception:
        return dict(value=0.97, summary="Latest (demo): GRR 89%, NRR 97%.", series=[dict(month="demo", grr=0.89, nrr=0.97)])

def _unit_econ(df: pd.DataFrame, gm: float = 0.62, cac: float = 42.0) -> Dict[str, Any]:
    try:
        aov = float(df["amount"].mean()) if "amount" in df.columns else float(df.select_dtypes(np.number).sum(axis=1).mean())
        cm = round(gm*aov - cac, 2)
        return dict(value=cm, summary=f"AOV ${aov:.2f}, GM {gm:.0%}, CAC ${cac:.0f} → CM ${cm:.2f}.",
                    aov=aov, gm=gm, cac=cac, cm=cm)
    except Exception:
        return dict(value=32.0, summary="AOV $120.00, GM 60%, CAC $40 → CM $32.00 (demo).",
                    aov=120.0, gm=0.6, cac=40.0, cm=32.0)

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
        return {"status":"done","value":None,"summary":"PDF module required for a PDF row.","last_run": now_ts()}

    if mod == "PDF KPIs (PDF)":
        raw = SS["pdf_files"].get(row["source"], b"")
        k = _pdf_kpis(raw)
        return {"status":"done","value":None,"summary":k["summary"],"last_run": now_ts()}

    if mod == "Cohort Retention (CSV)":
        df = materialize_df(row["source"])
        k = _cohort(df)
        out = {"status":"done","value":k["value"],"summary":k["summary"],"last_run": now_ts()}
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
    c.drawString(72, h-72, "TransformAI — QoE Summary (Demo)")
    y = h-100; c.setFont("Helvetica", 10)
    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}
    for (rid,cid),res in list(SS["results"].items())[:28]:
        r = rows_by_id.get(rid); cdef = cols_by_id.get(cid)
        if not r or not cdef: continue
        line = f"{r['alias']} → {cdef['label']}: {res.get('summary')}"
        for chunk in [line[i:i+95] for i in range(0,len(line),95)]:
            if y<72: c.showPage(); y=h-72; c.setFont("Helvetica",10)
            c.drawString(72,y,chunk); y-=14
    c.showPage(); c.save(); buf.seek(0)
    return buf.getvalue()

# -----------------------------------------------------------------------------
# UI — Tabs (wide)
# -----------------------------------------------------------------------------
st.title("Transform AI — Diligence Grid (Pro)")
tab_data, tab_grid, tab_run, tab_review, tab_memo = st.tabs(["Data","Grid","Run","Review","Memo"])

# --------------------------- DATA ---------------------------
with tab_data:
    st.subheader("Evidence Sources & CSV Schema")

    c1, c2 = st.columns(2)
    with c1:
        csvs = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)
        if csvs:
            for f in csvs:
                try: df = pd.read_csv(f)
                except Exception: df = pd.read_csv(io.BytesIO(f.getvalue()))
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
            st.divider()

    st.write("**Loaded CSVs:**", list(SS["csv_files"].keys()) or "—")
    st.write("**Loaded PDFs:**", list(SS["pdf_files"].keys()) or "—")

# --------------------------- GRID ---------------------------
with tab_grid:
    st.subheader("Build Grid: rows, columns, and the Matrix Board")

    a1, a2, a3, a4 = st.columns([1,1,1,1])
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
        if st.button("Undo / Redo", use_container_width=True):
            if SS["undo"]: undo(); st.toast("Undone")  # one tap undo

    c3, c4 = st.columns([1,1])
    with c3:
        if st.button("Redo", use_container_width=True): 
            if SS["redo"]: redo(); st.toast("Redone")
    with c4:
        pass

    # Inline Rows
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

    # Inline Columns
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
        # Build matrix df: one row per grid row, boolean columns for each module
        base = []
        rid_by_alias = {}
        for r in SS["rows"]:
            rid_by_alias[r["alias"]] = r["id"]
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
                # enforce type compatibility
                if any(r["id"]==rid and r["row_type"]=="pdf" for r in SS["rows"]):
                    sel = set(m for m in sel if m=="PDF KPIs (PDF)")
                SS["matrix"][rid] = sel
            st.success("Matrix updated")

    else:
        st.info("Add rows first, then use the Matrix to map modules.")

# --------------------------- RUN ----------------------------
with tab_run:
    st.subheader("Run — queue, process, and see status")
    st.toggle("Force re-run (ignore cache)", key="force_rerun", value=SS.get("force_rerun", False))

    # One-click QoE
    with st.expander("One-click QoE", expanded=True):
        st.caption("Adds QoE columns (if missing), selects mapped pairs from Matrix, runs all.")
        if st.button("Run QoE Now", type="primary"):
            add_template_columns(QOE_TEMPLATE)
            # build pairs from matrix
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

    # Manual selection by Matrix
    with st.expander("Manual run by Matrix selection", expanded=False):
        rows = SS["rows"]
        cols = SS["columns"]
        by_mod = {c["module"]: c["id"] for c in cols}
        options = []
        for r in rows:
            sel = SS["matrix"].get(r["id"], set())
            for mod in sel:
                cid = by_mod.get(mod)
                if cid:
                    options.append((r["id"], cid, f"{r['alias']} → {mod}"))
        if options:
            labels = [o[2] for o in options]
            picks = st.multiselect("Pick cell pairs to run", options=list(range(len(options))), default=list(range(len(options)))[:8], format_func=lambda i: labels[i])
            if st.button("Queue + Run selection"):
                pairs = [(options[i][0], options[i][1]) for i in picks]
                enqueue_pairs(pairs, respect_cache=True)
                run_queued_jobs()
                st.success(f"Ran {len(pairs)} cell(s).")
        else:
            st.info("No mapped pairs in Matrix yet.")

    # Job table
    if SS["jobs"]:
        rows_by_id = {r["id"]: r for r in SS["rows"]}
        cols_by_id = {c["id"]: c for c in SS["columns"]}
        out=[]
        for j in SS["jobs"]:
            r=rows_by_id.get(j["rid"],{"alias":"(deleted)"})
            c=cols_by_id.get(j["cid"],{"label":"(deleted)"})
            out.append(dict(Row=r["alias"], Column=c["label"], Status=j["status"], Started=j["started"], Ended=j["ended"], Note=j.get("note","")))
        st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True)
    else:
        st.info("No jobs yet.")

    # Results snapshot
    st.markdown("**Results snapshot**")
    if SS["results"]:
        rows_by_id = {r["id"]: r for r in SS["rows"]}
        cols_by_id = {c["id"]: c for c in SS["columns"]}
        tbl=[]
        for (rid,cid),res in SS["results"].items():
            r=rows_by_id.get(rid); c=cols_by_id.get(cid)
            if not r or not c: continue
            status=res.get("status"); emoji={"done":"✅","cached":"🟢","queued":"⏳","running":"🟡","error":"🔴","needs_review":"🟠"}.get(status,"•")
            tbl.append(dict(Row=r["alias"], Column=c["label"], Module=c["module"], Status=f"{emoji} {status}", Value=res.get("value"), Summary=res.get("summary")))
        st.dataframe(pd.DataFrame(tbl), hide_index=True, use_container_width=True)
    else:
        st.caption("Run something to populate results.")

# -------------------------- REVIEW --------------------------
with tab_review:
    st.subheader("Review — inspect a single cell with charts & what-ifs")

    if not SS["results"]:
        st.info("No results yet.")
    else:
        rows_by_id = {r["id"]: r for r in SS["rows"]}
        cols_by_id = {c["id"]: c for c in SS["columns"]}
        keys = list(SS["results"].keys())
        labels=[]
        for (rid,cid) in keys:
            r=rows_by_id.get(rid,{"alias":rid}); c=cols_by_id.get(cid,{"label":cid,"module":"?"})
            status=SS["results"][(rid,cid)].get("status")
            emoji={"done":"✅","cached":"🟢","queued":"⏳","running":"🟡","error":"🔴","needs_review":"🟠"}.get(status,"•")
            labels.append(f"{emoji} {r['alias']} → {c['label']} ({c['module']})")
        idx = st.selectbox("Pick result", list(range(len(keys))), index=0, format_func=lambda i: labels[i])
        rid,cid = keys[idx]
        res = SS["results"].get((rid,cid),{})
        col_def = cols_by_id.get(cid,{}); row_def = rows_by_id.get(rid,{})

        st.write("**Status**:", res.get("status"))
        st.write("**Summary**:", res.get("summary") or "—")

        # Charts
        if col_def.get("module") == "Cohort Retention (CSV)" and "curve" in res:
            curve = res["curve"]
            if PLOTLY_OK:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(curve))), y=curve, mode="lines+markers", name="Retention"))
                fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(pd.DataFrame({"x":range(len(curve)),"y":curve}).set_index("x"))

        if col_def.get("module") == "Pricing Power (CSV)" and "scatter" in res and PLOTLY_OK:
            sc = res["scatter"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sc["x"], y=sc["y"], mode="markers", name="log(q) vs log(p)"))
            fig.add_trace(go.Scatter(x=sc["x"], y=sc["fit"], mode="lines", name="fit"))
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10), xaxis_title="log(price)", yaxis_title="log(quantity)")
            st.plotly_chart(fig, use_container_width=True)

        if col_def.get("module") == "NRR/GRR (CSV)" and "series" in res and PLOTLY_OK:
            s = res["series"]
            fig = make_subplots(rows=1, cols=1)
            fig.add_trace(go.Bar(x=[r["month"] for r in s], y=[r["grr"] for r in s], name="GRR"))
            fig.add_trace(go.Scatter(x=[r["month"] for r in s], y=[r["nrr"] for r in s], mode="lines+markers", name="NRR"))
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10), yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        if col_def.get("module") == "Unit Economics (CSV)":
            st.markdown("**What-ifs**")
            c1,c2 = st.columns(2)
            with c1:
                SS["whatif_gm"] = st.slider("Gross Margin %", 0.2, 0.9, SS.get("whatif_gm",0.62), 0.01)
            with c2:
                SS["whatif_cac"] = st.slider("CAC ($)", 0.0, 200.0, SS.get("whatif_cac",42.0), 1.0)
            # temp recompute
            df = materialize_df(row_def["source"]) if row_def.get("row_type")=="table" else pd.DataFrame()
            k = _unit_econ(df, gm=SS["whatif_gm"], cac=SS["whatif_cac"])
            st.info(f"What-if → {k['summary']}")
            if PLOTLY_OK:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=["AOV","CAC","CM"], y=[k["aov"], k["cac"], k["cm"]]))
                fig.update_layout(height=280, margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            if st.button("Retry", use_container_width=True):
                retry_cell(rid,cid); run_queued_jobs(); st.success("Retried.")
        with c2:
            if st.button("Mark needs_review", use_container_width=True):
                SS["results"][(rid,cid)]["status"]="needs_review"; st.toast("Marked.")
        with c3:
            if st.button("Clear needs_review", use_container_width=True):
                if SS["results"][(rid,cid)].get("status")=="needs_review":
                    SS["results"][(rid,cid)]["status"]="done"; st.toast("Cleared.")

# --------------------------- MEMO ---------------------------
with tab_memo:
    st.subheader("Memo / Export")
    if SS["results"]:
        st.download_button("Download Results (CSV)", export_results_csv(), file_name="transformai_results.csv", mime="text/csv")
        if REPORTLAB_OK:
            try:
                st.download_button("Download QoE Summary (PDF)", export_results_pdf(), file_name="TransformAI_QoE_Summary.pdf", mime="application/pdf")
            except Exception as e:
                st.warning(f"PDF export unavailable: {e}")
    else:
        st.info("Run some cells to export.")
