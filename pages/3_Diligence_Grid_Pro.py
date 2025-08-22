# pages/3_Diligence_Grid_Pro.py
# TransformAI — Diligence Grid (Pro)
# Classic grid + CSV schema mapper + Run Plan (QoE template), inline row/column editing,
# queue/progress, retry, caching, review (with Unit Econ what-ifs), and CSV/PDF export (graceful fallback).

from __future__ import annotations
import io
import json
import time
import uuid
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

# ============================================================
# ----------------- Session, Utils, Safe Widgets -------------
# ============================================================

SS = st.session_state

def _as_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set)): return list(x)
    return [x]

def _keep_only(options, default):
    opts = set(options or [])
    return [v for v in _as_list(default) if v in opts]

def ui_selectbox(label, options, default=None, **kwargs):
    opt_list = list(options or [])
    dv = _keep_only(opt_list, default)
    dv = dv[0] if dv else None
    idx = opt_list.index(dv) if dv in opt_list else (0 if opt_list else None)
    return st.selectbox(label, opt_list, index=idx, **kwargs)

def ui_multiselect(label, options, default=None, **kwargs):
    opt_list = list(options or [])
    dv = _keep_only(opt_list, default)
    return st.multiselect(label, opt_list, default=dv, **kwargs)

def rerun():
    st.rerun()

def uid(prefix="row"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def now_ts():
    return int(time.time())

def ensure_state():
    # Data
    SS.setdefault("csv_files", {})            # {name: DataFrame}
    SS.setdefault("pdf_files", {})            # {name: bytes}
    SS.setdefault("schema", {})               # {csv_name: {canonical: source_col or None}}

    # Grid
    SS.setdefault("rows", [])                 # [{id, alias, row_type ('table'|'pdf'), source}]
    SS.setdefault("columns", [])              # [{id, label, module}]

    # Results + cache
    SS.setdefault("results", {})              # {(row_id, col_id): {status,value,summary,curve?,last_run,err?}}
    SS.setdefault("cache_key", {})            # {(row_id,col_id): str}

    # Selections
    SS.setdefault("row_run_selection", [])
    SS.setdefault("col_run_selection", [])

    # Queue
    SS.setdefault("jobs", [])                 # list of {"rid","cid","status","started","ended","note"}

    # Undo/Redo
    SS.setdefault("undo", [])
    SS.setdefault("redo", [])

    # Stepper
    SS.setdefault("step_idx", 0)              # 0..4

    # What-ifs (review)
    SS.setdefault("whatif_gm", 0.62)
    SS.setdefault("whatif_cac", 42.0)
    SS.setdefault("force_rerun", False)

ensure_state()

def snapshot_push():
    SS["undo"].append(json.dumps({
        "rows": SS["rows"], "columns": SS["columns"], "results": SS["results"]
    }, default=str))
    SS["redo"].clear()

def snapshot_apply(snap: str):
    data = json.loads(snap)
    SS["rows"]    = data.get("rows", [])
    SS["columns"] = data.get("columns", [])
    SS["results"] = {tuple(eval(k) if isinstance(k, str) and k.startswith("(") else k): v
                     for k, v in data.get("results", {}).items()}

def undo():
    if not SS["undo"]: return
    cur = json.dumps({"rows": SS["rows"], "columns": SS["columns"], "results": SS["results"]}, default=str)
    snap = SS["undo"].pop()
    SS["redo"].append(cur)
    snapshot_apply(snap)

def redo():
    if not SS["redo"]: return
    cur = json.dumps({"rows": SS["rows"], "columns": SS["columns"], "results": SS["results"]}, default=str)
    snap = SS["redo"].pop()
    SS["undo"].append(cur)
    snapshot_apply(snap)

# ============================================================
# ---------------------- CSV Schema Map ----------------------
# ============================================================

CANONICAL = ["customer_id", "order_date", "amount", "price", "quantity", "month", "revenue"]

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

    # rename selected mappings → canonical
    rename_map = {}
    for k, v in sch.items():
        if v and v in df.columns and k not in df.columns:
            rename_map[v] = k
    if rename_map:
        df = df.rename(columns=rename_map)

    # derive month if missing
    if "month" not in df.columns and "order_date" in df.columns:
        try:
            df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
            df["month"] = df["order_date"].dt.to_period("M").astype(str)
        except Exception:
            pass

    # derive revenue if missing
    if "revenue" not in df.columns and "amount" in df.columns:
        df["revenue"] = df["amount"]

    # fill quantity/price if missing (soft defaults)
    if "quantity" not in df.columns:
        df["quantity"] = 1
    if "price" not in df.columns:
        if "revenue" in df.columns and "quantity" in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                df["price"] = np.where(df["quantity"]>0, df["revenue"]/df["quantity"], np.nan)
        else:
            df["price"] = np.nan

    return df

# ============================================================
# -------------------------- Grid Ops ------------------------
# ============================================================

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
            SS["rows"].append({"id": uid("row"), "alias": name.replace(".csv",""),
                               "row_type":"table", "source": name})

def add_rows_from_pdfs():
    snapshot_push()
    for name in SS["pdf_files"].keys():
        if not any(r["source"] == name for r in SS["rows"]):
            SS["rows"].append({"id": uid("row"), "alias": name.replace(".pdf",""),
                               "row_type":"pdf", "source": name})

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
    SS["results"] = {k:v for k,v in SS["results"].items() if k[0] not in row_ids}
    SS["row_run_selection"] = _keep_only([r["id"] for r in SS["rows"]], SS["row_run_selection"])

def delete_cols(col_ids: List[str]):
    if not col_ids: return
    snapshot_push()
    SS["columns"] = [c for c in SS["columns"] if c["id"] not in col_ids]
    SS["results"] = {k:v for k,v in SS["results"].items() if k[1] not in col_ids}
    SS["col_run_selection"] = _keep_only([c["id"] for c in SS["columns"]], SS["col_run_selection"])

# ============================================================
# ------------------ Mini “Engines” (MVP) --------------------
# ============================================================

def _pdf_kpis(_raw: bytes) -> Dict[str, Any]:
    return dict(summary="Revenue ≈ $12.5M; EBITDA ≈ $1.3M; GM ≈ 62%; Churn ≈ 4%")

def _cohort(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        if {"customer_id","order_date","amount"}.issubset(df.columns):
            d = df.copy()
            d["order_date"] = pd.to_datetime(d["order_date"], errors="coerce")
            d = d.dropna(subset=["customer_id","order_date"])
            d["month"] = d["order_date"].dt.to_period("M")
            # naive retention curve (demo)
            curve = [round(max(0.0, 1.0*(0.9**i)), 2) for i in range(5)]
            m3 = curve[3] if len(curve)>3 else None
            return dict(value=m3, curve=curve, summary=f"Retention stabilizes ~M3 at {m3:.0%} (demo).")
    except Exception:
        pass
    curve=[1.0,0.86,0.78,0.72,0.69]; m3=0.72
    return dict(value=m3, curve=curve, summary=f"Retention stabilizes ~M3 at {m3:.0%} (demo).")

def _pricing(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        d = df[["price","quantity"]].replace(0, np.nan).dropna()
        d = d[(d["price"]>0) & (d["quantity"]>0)]
        x = np.log(d["price"].astype(float)); y = np.log(d["quantity"].astype(float))
        b = np.polyfit(x,y,1)[0]
        e = round(b,2)
        verdict = "inelastic" if abs(e)<1 else "elastic"
        return dict(value=e, summary=f"Own-price elasticity ≈ {e} → {verdict}.")
    except Exception:
        return dict(value=-1.21, summary="Own-price elasticity ≈ -1.21 (demo).")

def _nrr_grr(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        if "month" not in df.columns and "order_date" in df.columns:
            df = df.copy()
            df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
            df["month"] = df["order_date"].dt.to_period("M").astype(str)
        if "revenue" not in df.columns and "amount" in df.columns:
            df["revenue"] = df["amount"]
        m = df.groupby(["customer_id","month"])["revenue"].sum().reset_index()
        months = sorted(m["month"].unique())
        if len(months) < 2: return dict(value=0.97, summary="Latest (n/a): GRR 89%, NRR 97% (demo).")
        prev, cur = months[-2], months[-1]
        base = m[m["month"]==prev]["revenue"].sum()
        kept = m[m["month"]==cur]["revenue"].sum()
        grr = round(min(kept/base,1.2),2) if base else 0.89
        nrr = round(min((kept+0.05*base)/base,1.3),2) if base else 0.97
        return dict(value=nrr, summary=f"Latest ({cur}): GRR {grr:.0%}, NRR {nrr:.0%}.")
    except Exception:
        return dict(value=0.97, summary="Latest (demo): GRR 89%, NRR 97%.")

def _unit_econ(df: pd.DataFrame, gm: float = 0.62, cac: float = 42.0) -> Dict[str, Any]:
    try:
        aov = float(df["amount"].mean()) if "amount" in df.columns else float(df.select_dtypes(np.number).sum(axis=1).mean())
        cm = round(gm*aov - cac, 2)
        return dict(value=cm, summary=f"AOV ${aov:.2f}, GM {gm:.0%}, CAC ${cac:.0f} → CM ${cm:.2f}.",
                    aov=aov, gm=gm, cac=cac, cm=cm)
    except Exception:
        return dict(value=32.0, summary="AOV $120.00, GM 60%, CAC $40 → CM $32.00 (demo).",
                    aov=120.0, gm=0.6, cac=40.0, cm=32.0)

# ============================================================
# ---------------------- Runner & Cache ----------------------
# ============================================================

def cache_key_for(row: Dict[str,Any], col: Dict[str,Any]) -> str:
    # Deterministic cache key: row source + module + schema snapshot
    if row["row_type"] == "pdf":
        base = f"pdf::{row['source']}::{col['module']}"
    else:
        sch = SS["schema"].get(row["source"], {})
        sch_str = json.dumps(sch, sort_keys=True)
        base = f"csv::{row['source']}::{col['module']}::{sch_str}"
    return base

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
        return {"status":"done","value":k["value"],"summary":k["summary"],"last_run": now_ts()}

    if mod == "NRR/GRR (CSV)":
        df = materialize_df(row["source"])
        k = _nrr_grr(df)
        return {"status":"done","value":k["value"],"summary":k["summary"],"last_run": now_ts()}

    if mod == "Unit Economics (CSV)":
        df = materialize_df(row["source"])
        k = _unit_econ(df, gm=SS.get("whatif_gm",0.62), cac=SS.get("whatif_cac",42.0))
        return {"status":"done","value":k["value"],"summary":k["summary"],"last_run": now_ts(), **k}

    return {"status":"error","value":None,"summary":f"Unknown module: {mod}","last_run": now_ts()}

def enqueue_selection(row_ids: List[str], col_ids: List[str], respect_cache=True):
    if not row_ids or not col_ids: return
    by_r = {r["id"]: r for r in SS["rows"]}
    by_c = {c["id"]: c for c in SS["columns"]}

    for rid in row_ids:
        for cid in col_ids:
            row = by_r.get(rid); col = by_c.get(cid)
            if not row or not col: continue
            key = (rid, cid)
            ck = cache_key_for(row, col)
            SS["cache_key"][key] = ck
            already = SS["results"].get(key)
            if respect_cache and (already and already.get("status") in {"done","cached"}) and SS["cache_key"].get(key) == ck and not SS["force_rerun"]:
                # visible cache hit
                SS["results"][key] = {**already, "status":"cached"}
                SS["jobs"].append({"rid": rid, "cid": cid, "status":"cached", "started": now_ts(), "ended": now_ts(), "note":"cache"})
                continue
            SS["results"][key] = {"status":"queued","value":None,"summary":None}
            SS["jobs"].append({"rid": rid, "cid": cid, "status":"queued", "started": None, "ended": None, "note":""})

def run_queued_jobs():
    by_r = {r["id"]: r for r in SS["rows"]}
    by_c = {c["id"]: c for c in SS["columns"]}

    for job in SS["jobs"]:
        if job["status"] not in {"queued","retry"}:
            continue
        rid, cid = job["rid"], job["cid"]
        row = by_r.get(rid); col = by_c.get(cid)
        if not row or not col:
            job["status"] = "error"; job["note"] = "row/col missing"; job["ended"] = now_ts()
            continue
        job["status"] = "running"; job["started"] = now_ts()
        key = (rid, cid)
        try:
            res = execute_cell(row, col)
            SS["results"][key] = res
            job["status"] = "done"; job["ended"] = now_ts()
        except Exception as e:
            SS["results"][key] = {"status":"error","value":None,"summary":str(e), "last_run": now_ts()}
            job["status"] = "error"; job["note"] = str(e); job["ended"] = now_ts()

def retry_cell(rid: str, cid: str):
    SS["jobs"].insert(0, {"rid": rid, "cid": cid, "status":"retry", "started": None, "ended": None, "note":"manual retry"})

def job_table():
    if not SS["jobs"]:
        st.info("No jobs yet. Use Run Plan or Run buttons below.")
        return
    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}
    out = []
    for j in SS["jobs"]:
        r = rows_by_id.get(j["rid"], {"alias":"(deleted)"})
        c = cols_by_id.get(j["cid"], {"label":"(deleted)"})
        out.append(dict(
            Row=r["alias"], Column=c["label"], Status=j["status"],
            Started=j["started"], Ended=j["ended"], Note=j.get("note","")
        ))
    st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True)

# ============================================================
# ---------------------- Export Helpers ----------------------
# ============================================================

def export_results_csv() -> bytes:
    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}
    out = []
    for (rid,cid), res in SS["results"].items():
        r = rows_by_id.get(rid); c = cols_by_id.get(cid)
        if not r or not c: continue
        out.append(dict(
            row_id=rid, row_alias=r["alias"], row_type=r["row_type"], source=r["source"],
            col_id=cid, col_label=c["label"], module=c["module"],
            status=res.get("status"), value=res.get("value"), summary=res.get("summary"),
            last_run=res.get("last_run")
        ))
    df = pd.DataFrame(out)
    return df.to_csv(index=False).encode("utf-8")

def export_results_pdf() -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab not installed")
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height-72, "TransformAI — QoE Summary (Demo)")
    c.setFont("Helvetica", 10)
    y = height - 100
    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c2["id"]: c2 for c2 in SS["columns"]}
    for (rid, cid), res in list(SS["results"].items())[:28]:
        r = rows_by_id.get(rid); cdef = cols_by_id.get(cid)
        if not r or not cdef: continue
        line = f"{r['alias']} → {cdef['label']}: {res.get('summary')}"
        for chunk in [line[i:i+95] for i in range(0, len(line), 95)]:
            if y < 72:
                c.showPage(); y = height-72; c.setFont("Helvetica", 10)
            c.drawString(72, y, chunk)
            y -= 14
    c.showPage(); c.save()
    buf.seek(0)
    return buf.getvalue()

# ============================================================
# --------------------------- UI -----------------------------
# ============================================================

st.title("Transform AI — Diligence Grid (Pro)")
# Stepper (fallback to radio if segmented_control missing)
try:
    steps = ["Data", "Grid", "Run", "Review", "Memo"]
    SS["step_idx"] = st.segmented_control("Workflow", steps, selection=steps[SS.get("step_idx", 0)], key="seg_steps")
except Exception:
    steps = ["Data", "Grid", "Run", "Review", "Memo"]
    SS["step_idx"] = steps.index(st.radio("Workflow", steps, index=SS.get("step_idx", 0), horizontal=True))

# ------------------------- DATA -----------------------------
if SS["step_idx"] == 0:
    st.subheader("1) Evidence Sources & CSV Schema")

    c1, c2 = st.columns(2)
    with c1:
        csv_files = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)
        if csv_files:
            for f in csv_files:
                try:
                    df = pd.read_csv(f)
                except Exception:
                    df = pd.read_csv(io.BytesIO(f.getvalue()))
                SS["csv_files"][f.name] = df
                SS["schema"].setdefault(f.name, _auto_guess_schema(df))
            st.success(f"Loaded {len(csv_files)} CSV file(s).")

    with c2:
        pdf_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if pdf_files:
            for f in pdf_files:
                SS["pdf_files"][f.name] = f.getvalue()
            st.success(f"Loaded {len(pdf_files)} PDF file(s).")

    with st.expander("Drive/Box/SharePoint Link (stub)", expanded=False):
        link = st.text_input("Paste a public/shared link (ends with .csv or .pdf)")
        fname = st.text_input("Save as (e.g., data.csv or pack.pdf)")
        if st.button("Fetch & add"):
            try:
                import requests
                r = requests.get(link, timeout=15)
                r.raise_for_status()
                if fname.lower().endswith(".csv"):
                    SS["csv_files"][fname] = pd.read_csv(io.BytesIO(r.content))
                    SS["schema"].setdefault(fname, _auto_guess_schema(SS["csv_files"][fname]))
                    st.success(f"Fetched CSV: {fname}")
                elif fname.lower().endswith(".pdf"):
                    SS["pdf_files"][fname] = r.content
                    st.success(f"Fetched PDF: {fname}")
                else:
                    st.error("File extension must be .csv or .pdf")
            except Exception as e:
                st.error(f"Fetch failed: {e}")

    if SS["csv_files"]:
        with st.expander("Map CSV Schema (per file)", expanded=True):
            for name, df in SS["csv_files"].items():
                st.markdown(f"**{name}**")
                sch = SS["schema"].setdefault(name, _auto_guess_schema(df))
                cols = ["— None —"] + list(df.columns)

                def pick(lbl, key):
                    current = sch.get(key)
                    if current not in df.columns: current = None
                    chosen = ui_selectbox(lbl, cols, default=current if current else "— None —", key=f"{name}:{key}")
                    sch[key] = None if chosen == "— None —" else chosen

                pick("Customer ID", "customer_id")
                pick("Order Date (timestamp)", "order_date")
                pick("Amount (txn value)", "amount")
                pick("Unit Price", "price")
                pick("Quantity", "quantity")
                pick("Month (YYYY-MM)", "month")
                pick("Revenue (period revenue)", "revenue")
                st.divider()
    else:
        st.info("Upload at least one CSV to map schema.")

    st.write("**Loaded CSVs**:", list(SS["csv_files"].keys()) or "—")
    st.write("**Loaded PDFs**:", list(SS["pdf_files"].keys()) or "—")

# ------------------------- GRID -----------------------------
if SS["step_idx"] == 1:
    st.subheader("2) Build Grid")

    # Top actions
    b1, b2, b3, b4 = st.columns([1,1,1,1])
    with b1:
        if st.button("Add rows from CSVs", use_container_width=True):
            add_rows_from_csvs(); st.toast("CSV rows added"); rerun()
    with b2:
        if st.button("Add rows from PDFs", use_container_width=True):
            add_rows_from_pdfs(); st.toast("PDF rows added"); rerun()
    with b3:
        if st.button("Add QoE Columns", use_container_width=True):
            add_template_columns(QOE_TEMPLATE); st.toast("QoE columns added"); rerun()
    with b4:
        if st.button("Undo / Redo help", use_container_width=True):
            st.info("Use the Undo/Redo buttons below to revert structural changes.")

    c3, c4 = st.columns([1,1])
    with c3:
        if st.button("Undo", use_container_width=True): undo(); rerun()
    with c4:
        if st.button("Redo", use_container_width=True): redo(); rerun()

    # Inline row editor
    st.markdown("**Rows (inline editable)**")
    if SS["rows"]:
        df_rows = pd.DataFrame(SS["rows"])[["id","alias","row_type","source"]]
        df_rows_display = df_rows.copy()
        df_rows_display["delete"] = False
        edited = st.data_editor(
            df_rows_display,
            hide_index=True,
            use_container_width=True,
            disabled=["id","row_type","source"],
            key="rows_editor"
        )
        del_selection = edited[edited["delete"] == True]["id"].tolist()
        if st.button("Apply row changes"):
            # Apply alias edits
            alias_map = {row["id"]: row["alias"] for _, row in edited.iterrows()}
            for r in SS["rows"]:
                r["alias"] = alias_map.get(r["id"], r["alias"])
            # Deletes
            if del_selection:
                delete_rows(del_selection)
            st.success("Rows updated"); rerun()
    else:
        st.info("No rows yet. Use the buttons above to add from CSV/PDF.")

    # Inline column editor
    st.markdown("**Columns (inline editable)**")
    if SS["columns"]:
        df_cols = pd.DataFrame(SS["columns"])[["id","label","module"]]
        df_cols_display = df_cols.copy()
        df_cols_display["delete"] = False
        edited_cols = st.data_editor(
            df_cols_display,
            hide_index=True,
            use_container_width=True,
            disabled=["id","module"],  # you can allow module edits if you want
            key="cols_editor"
        )
        del_cols = edited_cols[edited_cols["delete"] == True]["id"].tolist()
        if st.button("Apply column changes"):
            # Apply label edits
            label_map = {row["id"]: row["label"] for _, row in edited_cols.iterrows()}
            for c in SS["columns"]:
                c["label"] = label_map.get(c["id"], c["label"])
            # Deletes
            if del_cols:
                delete_cols(del_cols)
            st.success("Columns updated"); rerun()
    else:
        st.info("No columns yet. Add QoE columns or a custom one below.")

    st.markdown("**Add a single column**")
    ac1, ac2, ac3 = st.columns([2,2,1])
    with ac1:
        new_label = st.text_input("Column label", value=SS.get("new_col_label","NRR/GRR"))
        SS["new_col_label"] = new_label
    with ac2:
        new_mod = ui_selectbox("Module", MODULES, default=SS.get("new_col_mod","NRR/GRR (CSV)"))
        SS["new_col_mod"] = new_mod
    with ac3:
        if st.button("Add Column", use_container_width=True):
            add_column(SS["new_col_label"], SS["new_col_mod"]); st.success("Column added"); rerun()

# -------------------------- RUN -----------------------------
if SS["step_idx"] == 2:
    st.subheader("3) Run")

    st.toggle("Force re-run (ignore cache)", key="force_rerun", value=SS.get("force_rerun", False))

    with st.expander("Run Plan & Templates", expanded=True):
        plan = st.radio("Template", ["QoE (one-click)", "Custom selection"], horizontal=True)
        if plan.startswith("QoE"):
            st.caption("Adds QoE columns (if missing), selects **all rows**, and runs everything in one click.")
            if st.button("Apply QoE & Run All", type="primary"):
                add_template_columns(QOE_TEMPLATE)
                row_ids = [r["id"] for r in SS["rows"]]
                col_ids = [c["id"] for c in SS["columns"] if (c["label"], c["module"]) in QOE_TEMPLATE]
                SS["row_run_selection"] = row_ids
                SS["col_run_selection"] = col_ids
                enqueue_selection(row_ids, col_ids, respect_cache=True)
                run_queued_jobs()
                rerun()
        else:
            st.caption("Pick any rows and columns below, preview the queue, then run.")

        st.markdown("**Job Queue / History**")
        job_table()

    # Custom selectors
    st.markdown("**Custom Run**")
    row_ids = [r["id"] for r in SS["rows"]]
    col_ids = [c["id"] for c in SS["columns"]]
    SS["row_run_selection"] = ui_multiselect("Rows", options=row_ids, default=SS.get("row_run_selection", []))
    SS["col_run_selection"] = ui_multiselect("Columns", options=col_ids, default=SS.get("col_run_selection", []))

    rc1, rc2, rc3 = st.columns([1,1,1])
    with rc1:
        if st.button("Queue selection", use_container_width=True):
            enqueue_selection(SS["row_run_selection"], SS["col_run_selection"], respect_cache=True)
            st.toast("Selection queued"); rerun()
    with rc2:
        if st.button("Run queued now", use_container_width=True, type="primary"):
            run_queued_jobs(); st.toast("Queue processed"); rerun()
    with rc3:
        if st.button("Clear queue", use_container_width=True):
            SS["jobs"].clear(); st.toast("Queue cleared"); rerun()

    # Results table (status/last run)
    st.markdown("**Results (investor view)**")
    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}
    table = []
    for (rid, cid), res in SS["results"].items():
        r = rows_by_id.get(rid); c = cols_by_id.get(cid)
        if not r or not c: continue
        status = res.get("status")
        emoji = {"done":"✅","cached":"🟢","queued":"⏳","running":"🟡","error":"🔴","needs_review":"🟠"}.get(status, "•")
        table.append(dict(
            Row=r["alias"], Column=c["label"], Module=c["module"],
            Status=f"{emoji} {status}", Value=res.get("value"), Summary=res.get("summary"),
            LastRun=res.get("last_run")
        ))
    if table:
        st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    else:
        st.info("No results yet. Queue & run a selection or use QoE one-click.")

# ------------------------- REVIEW ---------------------------
if SS["step_idx"] == 3:
    st.subheader("4) Review & What-ifs")

    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}
    keys = list(SS["results"].keys())
    if not keys:
        st.info("No cell results yet.")
    else:
        labels = []
        for (rid, cid) in keys:
            r = rows_by_id.get(rid, {"alias":rid})
            c = cols_by_id.get(cid, {"label":cid})
            status = SS["results"][(rid,cid)].get("status")
            emoji = {"done":"✅","cached":"🟢","queued":"⏳","running":"🟡","error":"🔴","needs_review":"🟠"}.get(status, "•")
            labels.append(f"{emoji} {r['alias']} → {c['label']}")
        idx = ui_selectbox("Pick a cell", list(range(len(keys))), default=0,
                           format_func=lambda i: labels[i] if 0 <= i < len(labels) else "—")
        rid, cid = keys[idx]
        res = SS["results"].get((rid,cid), {})
        st.write("**Status**:", res.get("status"))
        st.write("**Summary**:", res.get("summary") or "—")
        if "curve" in res:
            st.write("Average Retention Curve")
            curve = res["curve"]
            chart_df = pd.DataFrame({"x": list(range(len(curve))), "y": curve})
            st.line_chart(chart_df, x="x", y="y", height=240)

        # Unit Econ what-ifs (visible for Unit Economics)
        col_def = cols_by_id.get(cid, {})
        if col_def.get("module") == "Unit Economics (CSV)":
            st.markdown("**What-ifs (Unit Economics)**")
            c1, c2 = st.columns(2)
            with c1:
                SS["whatif_gm"] = st.slider("Gross Margin %", 0.2, 0.9, SS.get("whatif_gm", 0.62), 0.01)
            with c2:
                SS["whatif_cac"] = st.slider("CAC ($)", 0.0, 200.0, SS.get("whatif_cac", 42.0), 1.0)
            # Recompute this cell with what-ifs (temp display)
            row = rows_by_id.get(rid)
            if row:
                df = materialize_df(row["source"])
                k = _unit_econ(df, gm=SS["whatif_gm"], cac=SS["whatif_cac"])
                st.info(f"What-if → {k['summary']}")

        cc1, cc2, cc3 = st.columns([1,1,1])
        with cc1:
            if st.button("Retry this cell", use_container_width=True):
                retry_cell(rid, cid); run_queued_jobs(); st.success("Retried."); rerun()
        with cc2:
            if st.button("Mark needs_review", use_container_width=True):
                SS["results"][(rid,cid)]["status"] = "needs_review"; st.toast("Marked as needs_review"); rerun()
        with cc3:
            if st.button("Clear needs_review", use_container_width=True):
                if SS["results"][(rid,cid)].get("status") == "needs_review":
                    SS["results"][(rid,cid)]["status"] = "done"; st.toast("Marked as done"); rerun()

# -------------------------- MEMO ----------------------------
if SS["step_idx"] == 4:
    st.subheader("5) Memo / Export")

    # CSV export
    out_csv = export_results_csv() if SS["results"] else None
    if out_csv:
        st.download_button("Download Results (CSV)", out_csv, file_name="transformai_results.csv", mime="text/csv")
    else:
        st.info("No results yet to export.")

    # Simple PDF export (summary page)
    if REPORTLAB_OK and SS["results"]:
        try:
            pdf_bytes = export_results_pdf()
            st.download_button("Download QoE Summary (PDF)", pdf_bytes, file_name="TransformAI_QoE_Summary.pdf", mime="application/pdf")
            st.caption("PDF is a compact summary of the current grid results (demo format).")
        except Exception as e:
            st.warning(f"PDF export unavailable: {e}")
    elif not REPORTLAB_OK:
        st.caption("Install reportlab to enable PDF export: pip install reportlab")
