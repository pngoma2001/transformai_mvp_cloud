# pages/3_Diligence_Grid_Pro.py
# Transform AI — Diligence Grid (Pro)
# Wide layout + Matrix mapping + Agentic Spreadsheet + Focused Review viz
# Adds: Real cohort engine, PDF evidence viewer, Save/Load project

from __future__ import annotations
import io, json, re, time, uuid
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

# Plotly for charts (primary renderer)
try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots  # noqa
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# Altair for charts (fallback renderer so retention still shows)
try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False

# Optional PDF parsers for evidence viewer
PDF_PARSE_OK = False
try:
    # Prefer pdfminer.six if available
    from pdfminer.high_level import extract_text
    PDF_PARSE_OK = True
except Exception:
    try:
        # Fallback to PyPDF2 text extraction if present
        import PyPDF2  # type: ignore
        PDF_PARSE_OK = True
    except Exception:
        PDF_PARSE_OK = False


# ---------------------------------------------------------------------------
# Page & CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Transform AI — Diligence Grid (Pro)", layout="wide")
st.markdown(
    """
<style>
/* widen canvas and keep generous breathing room */
.block-container {max-width: 1700px !important; padding-top: 0.5rem;}
/* Fix H1 being visually cut */
h1, .stMarkdown h1 {
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  line-height: 1.15 !important;
  margin-top: .25rem !important;
}
/* soften table checkboxes a bit */
.stDataFrame [role="checkbox"] {transform: scale(1.0);}
/* compact buttons in grid toolbar */
.grid-toolbar .stButton>button {width: 100%;}
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

    # RESULTS live as {(row_id, col_id): {...}} in-memory
    SS.setdefault("results", {})              # {(row_id, col_id): {...}}
    SS.setdefault("cache_key", {})            # {(row_id, col_id): str}

    SS.setdefault("jobs", [])
    SS.setdefault("force_rerun", False)

    # What-if inputs used by Unit Economics
    SS.setdefault("whatif_gm", 0.62)
    SS.setdefault("whatif_cac", 42.0)

    # undo/redo snapshots
    SS.setdefault("undo", [])
    SS.setdefault("redo", [])

    # project name for save/load
    SS.setdefault("project_name", "My Diligence Project")

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


# ----------------------------- snapshots -------------------------------------
def snapshot_push():
    SS["undo"].append(json.dumps({
        "rows": SS["rows"],
        "columns": SS["columns"],
        "matrix": {k: list(v) for k, v in SS["matrix"].items()},
        "results": _pack_results(SS["results"]),
        "project_name": SS.get("project_name", "My Diligence Project"),
    }, default=str))
    SS["redo"].clear()

def snapshot_apply(snap: str):
    data = json.loads(snap)
    SS["rows"]    = data.get("rows", [])
    SS["columns"] = data.get("columns", [])
    SS["matrix"]  = {k: set(v) for k, v in data.get("matrix", {}).items()}
    SS["results"] = _unpack_results(data.get("results", {}))
    SS["project_name"] = data.get("project_name", "My Diligence Project")

def undo():
    if not SS["undo"]:
        return
    cur = json.dumps({
        "rows": SS["rows"],
        "columns": SS["columns"],
        "matrix": {k: list(v) for k, v in SS["matrix"].items()},
        "results": _pack_results(SS["results"]),
        "project_name": SS.get("project_name", "My Diligence Project"),
    }, default=str)
    snap = SS["undo"].pop()
    SS["redo"].append(cur)
    snapshot_apply(snap)
    st.toast("Undone")

def redo():
    if not SS["redo"]:
        return
    cur = json.dumps({
        "rows": SS["rows"],
        "columns": SS["columns"],
        "matrix": {k: list(v) for k, v in SS["matrix"].items()},
        "results": _pack_results(SS["results"]),
        "project_name": SS.get("project_name", "My Diligence Project"),
    }, default=str)
    snap = SS["redo"].pop()
    SS["undo"].append(cur)
    snapshot_apply(snap)
    st.toast("Redone")


# ---------------------------------------------------------------------------
# Save / Load (Project persistence)
# ---------------------------------------------------------------------------
def save_project_bytes() -> bytes:
    payload = {
        "version": 1,
        "project_name": SS.get("project_name", "My Diligence Project"),
        "rows": SS["rows"],
        "columns": SS["columns"],
        "matrix": {k: list(v) for k, v in SS["matrix"].items()},
        "results": _pack_results(SS["results"]),
        "schema": SS["schema"],
        # NOTE: We do NOT serialize file bytes to keep files small.
        # After load, if a referenced CSV/PDF is missing, results can still be viewed,
        # but new runs will require re-uploading the sources.
        "csv_names": list(SS["csv_files"].keys()),
        "pdf_names": list(SS["pdf_files"].keys()),
        "saved_at": now_ts(),
    }
    return json.dumps(payload, indent=2, default=str).encode("utf-8")

def load_project_json(raw: bytes):
    try:
        data = json.loads(raw.decode("utf-8"))
        if data.get("version") != 1:
            st.warning("Unknown project version; attempting best-effort load.")
        SS["project_name"] = data.get("project_name", "My Diligence Project")
        SS["rows"] = data.get("rows", [])
        SS["columns"] = data.get("columns", [])
        SS["matrix"] = {k: set(v) for k, v in data.get("matrix", {}).items()}
        SS["results"] = _unpack_results(data.get("results", {}))
        SS["schema"] = data.get("schema", {})
        # Keep current csv_files/pdf_files; warn if missing
        missing_csv = [n for n in data.get("csv_names", []) if n not in SS["csv_files"]]
        missing_pdf = [n for n in data.get("pdf_names", []) if n not in SS["pdf_files"]]
        if missing_csv or missing_pdf:
            st.info(f"Loaded project. Missing sources → CSV: {missing_csv or '—'}, PDF: {missing_pdf or '—'}. "
                    f"Re-upload if you want to re-run cells.")
        st.success("Project loaded.")
    except Exception as e:
        st.error(f"Failed to load project: {e}")


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Engines (calculations)
# ---------------------------------------------------------------------------
_KPI_PATTERNS = {
    "revenue": re.compile(r"\b(revenue|sales)\b[:\s\-–]*\$?([\d\.,]+[mbkMBK]?)", re.I),
    "ebitda": re.compile(r"\b(ebitda)\b[:\s\-–]*\$?([\d\.,]+[mbkMBK]?)", re.I),
    "gross_margin": re.compile(r"\b(gross\s*margin|gm)\b[:\s\-–]*([\d\.]+%)", re.I),
    "churn": re.compile(r"\b(churn|attrition)\b[:\s\-–]*([\d\.]+%)", re.I),
}

def _parse_pdf_text_all_pages(raw: bytes) -> List[str]:
    pages: List[str] = []
    if not PDF_PARSE_OK:
        return pages
    # Try pdfminer first
    try:
        text = extract_text(io.BytesIO(raw))
        # crude split by form feed, fallback by heuristic
        if "\f" in text:
            pages = text.split("\f")
        else:
            # single string; best effort page slicing
            pages = [text]
        return [p or "" for p in pages]
    except Exception:
        pass
    # Fallback to PyPDF2
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(raw))  # type: ignore
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
    except Exception:
        pages = []
    return pages

def _extract_kpi_hits_from_pages(pages: List[str]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for pi, text in enumerate(pages):
        if not text: continue
        lines = text.splitlines()
        for ln in lines:
            s = ln.strip()
            if not s: continue
            for kpi, pat in _KPI_PATTERNS.items():
                m = pat.search(s)
                if m:
                    val = m.group(2)
                    hits.append({"kpi": kpi, "page": pi+1, "snippet": s[:240], "value": val})
    return hits

def _pdf_kpis(raw: bytes) -> Dict[str, Any]:
    """Return narrative + evidence hits (if possible)."""
    evidence: List[Dict[str, Any]] = []
    summary = "PDF scanned: add KPIs with page-level citations."
    if raw and PDF_PARSE_OK:
        pages = _parse_pdf_text_all_pages(raw)
        evidence = _extract_kpi_hits_from_pages(pages)
        if evidence:
            # Build a small narrative from first occurrences
            rev = next((h["value"] for h in evidence if h["kpi"]=="revenue"), None)
            ebd = next((h["value"] for h in evidence if h["kpi"]=="ebitda"), None)
            gm  = next((h["value"] for h in evidence if h["kpi"]=="gross_margin"), None)
            ch  = next((h["value"] for h in evidence if h["kpi"]=="churn"), None)
            parts = []
            if rev: parts.append(f"Revenue ≈ {rev}")
            if ebd: parts.append(f"EBITDA ≈ {ebd}")
            if gm:  parts.append(f"GM ≈ {gm}")
            if ch:  parts.append(f"Churn ≈ {ch}")
            if parts: summary = "; ".join(parts)
    return {"summary": summary, "evidence": evidence}

def _cohort_true(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a true cohort table:
        - cohort = first purchase month per customer
        - age    = months since cohort (0..N)
        - value  = % of cohort active in that age (unique buyers)
    Returns curve (avg retention), cohort_matrix (2D list), cohorts (list), ages (list), and meta stats.
    """
    d = df.copy()
    if "customer_id" not in d.columns or ("order_date" not in d.columns and "month" not in d.columns):
        # fallback synthetic curve
        curve=[1.0,0.88,0.79,0.72,0.69,0.66]; m3=0.72
        return dict(value=m3, curve=curve, summary=f"Retention stabilizes ~M3 at {m3:.0%} (demo).",
                    cohort_matrix=[[v for v in curve]], cohorts=["demo"], ages=list(range(len(curve))),
                    meta={"m3": m3, "m6": (curve[6] if len(curve)>6 else None), "m12": (curve[12] if len(curve)>12 else None)})

    if "order_date" in d.columns:
        d["order_date"] = pd.to_datetime(d["order_date"], errors="coerce")
        d = d.dropna(subset=["customer_id","order_date"])
        d["period"] = d["order_date"].dt.to_period("M")
    else:
        d = d.dropna(subset=["customer_id","month"])
        try:
            # to_period is robust on yyyy-mm
            d["period"] = pd.PeriodIndex(d["month"], freq="M")
        except Exception:
            # fallback: parse to datetime
            d["period"] = pd.to_datetime(d["month"], errors="coerce").dt.to_period("M")

    # cohort per customer
    first = d.groupby("customer_id")["period"].min().rename("cohort")
    d = d.join(first, on="customer_id")
    # age in months
    d["age"] = (d["period"] - d["cohort"]).apply(lambda x: x.n)

    # For retention, count unique active customers per (cohort, age)
    cohort_size = first.groupby(first).size().rename("cohort_size")
    active = d.groupby(["cohort","age"])["customer_id"].nunique().rename("active")
    table = active.to_frame().join(cohort_size, on="cohort")
    table["retention"] = table["active"] / table["cohort_size"]

    cohorts = sorted(table.index.get_level_values(0).unique())
    ages = sorted(table.index.get_level_values(1).unique())
    # Build dense matrix
    matrix = []
    for c in cohorts:
        row = []
        for a in ages:
            val = table.loc[(c, a), "retention"] if (c, a) in table.index else np.nan
            row.append(float(val) if pd.notnull(val) else np.nan)
        matrix.append(row)

    # overall curve: average retention across cohorts per age
    curve = []
    for j, a in enumerate(ages):
        col_vals = [matrix[i][j] for i in range(len(cohorts)) if not np.isnan(matrix[i][j])]
        curve.append(float(np.nanmean(col_vals)) if col_vals else np.nan)
    # summary stats
    def safe_pick(arr, idx):
        try:
            v = arr[idx]
            return v if v == v else None  # not NaN
        except Exception:
            return None
    m3 = safe_pick(curve, 3)
    m6 = safe_pick(curve, 6)
    m12 = safe_pick(curve, 12)
    summary = f"Retention curve M3 {m3:.0%}" if (m3 is not None) else "Retention computed."

    return dict(
        value=(m3 if m3 is not None else (curve[3] if len(curve) > 3 and curve[3]==curve[3] else None)),
        curve=[float(x) if x==x else None for x in curve],
        summary=summary,
        cohort_matrix=[[float(x) if x==x else None for x in row] for row in matrix],
        cohorts=[str(c) for c in cohorts],
        ages=[int(a) for a in ages],
        meta={"m3": m3, "m6": m6, "m12": m12, "cohort_sizes": cohort_size.astype(int).to_dict()},
    )

def _pricing(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        d = df[["price","quantity"]].replace(0, np.nan).dropna()
        d = d[(d["price"]>0) & (d["quantity"]>0)]
        x = np.log(d["price"].astype(float)); y = np.log(d["quantity"].astype(float))
        b, a = np.polyfit(x,y,1)  # y = b*x + a
        e = round(b,2)
        verdict = "inelastic" if abs(e)<1 else "elastic"
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


# ---------------------------------------------------------------------------
# Cache/Run
# ---------------------------------------------------------------------------
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
        # keep existing evidence if any manual edits were done
        prev = SS["results"].get((row["id"], col["id"]), {})
        manual_evd = prev.get("evidence_manual", [])
        return {"status":"done","value":None,"summary":k["summary"],"evidence":k.get("evidence",[]),
                "evidence_manual": manual_evd, "last_run": now_ts()}

    if mod == "Cohort Retention (CSV)":
        df = materialize_df(row["source"])
        k = _cohort_true(df)
        out = {"status":"done","value":k["value"],"summary":k["summary"],"last_run": now_ts(),
               "curve": k.get("curve",[]), "cohort_matrix": k.get("cohort_matrix",[]),
               "cohorts": k.get("cohorts",[]), "ages": k.get("ages",[]), "meta": k.get("meta",{})}
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


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
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

def export_cohort_csv(res: Dict[str, Any]) -> bytes:
    matrix = res.get("cohort_matrix", [])
    cohorts = res.get("cohorts", [])
    ages = res.get("ages", [])
    if not matrix or not cohorts or not ages:
        return b""
    df = pd.DataFrame(matrix, index=cohorts, columns=[f"M{a}" for a in ages])
    df.index.name = "Cohort"
    return df.to_csv().encode("utf-8")


# ---------------------------------------------------------------------------
# Plot helpers (review page)
# ---------------------------------------------------------------------------
def plot_retention(curve: List[float]):
    curve = [float(x) for x in (curve or []) if x is not None]
    if not curve:
        st.info("No retention curve available.")
        return
    if PLOTLY_OK:
        x = list(range(len(curve)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=curve, mode="lines+markers", name="retention"))
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    elif ALTAIR_OK:
        df = pd.DataFrame({"month": list(range(len(curve))), "retention": curve})
        ch = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(x="month:O", y=alt.Y("retention:Q", axis=alt.Axis(format="%")))
            .properties(height=320)
        )
        st.altair_chart(ch, use_container_width=True)
    else:
        st.line_chart(curve)

def plot_retention_heatmap(matrix: List[List[Optional[float]]], cohorts: List[str], ages: List[int]):
    if not matrix or not cohorts or not ages:
        st.info("No cohort heatmap available.")
        return
    z = [[(v if v is not None else np.nan) for v in row] for row in matrix]
    if PLOTLY_OK:
        fig = go.Figure(data=go.Heatmap(
            z=z, x=[f"M{a}" for a in ages], y=cohorts, colorscale="Blues", colorbar=dict(tickformat=".0%")
        ))
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
    elif ALTAIR_OK:
        df = pd.DataFrame(z, index=cohorts, columns=[f"M{a}" for a in ages]).reset_index().melt("index", var_name="age", value_name="value")
        df = df.rename(columns={"index":"cohort"})
        ch = alt.Chart(df).mark_rect().encode(x="age:O", y="cohort:O", color=alt.Color("value:Q", scale=alt.Scale(scheme="blues")))
        st.altair_chart(ch, use_container_width=True)
    else:
        st.write(pd.DataFrame(z, index=cohorts, columns=[f"M{a}" for a in ages]))

def plot_nrr(series: List[Dict[str, Any]]):
    if not series:
        st.info("No NRR/GRR series available.")
        return
    if PLOTLY_OK:
        months = [s["month"] for s in series]
        nrr = [s["nrr"] for s in series]
        grr = [s["grr"] for s in series]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=nrr, mode="lines+markers", name="NRR"))
        fig.add_trace(go.Scatter(x=months, y=grr, mode="lines+markers", name="GRR"))
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    elif ALTAIR_OK:
        df = pd.DataFrame(series)
        d1 = df.melt("month", value_vars=["nrr","grr"], var_name="metric", value_name="value")
        ch = (
            alt.Chart(d1)
            .mark_line(point=True)
            .encode(x="month:O", y=alt.Y("value:Q", axis=alt.Axis(format="%")), color="metric:N")
            .properties(height=320)
        )
        st.altair_chart(ch, use_container_width=True)
    else:
        st.line_chart(pd.DataFrame(series).set_index("month")[["nrr","grr"]])

def plot_pricing(scatter: Dict[str, Any]):
    x = scatter.get("x", [])
    y = scatter.get("y", [])
    fit = scatter.get("fit", [])
    if not x or not y:
        st.info("No pricing scatter available.")
        return
    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="log Q vs log P"))
        if fit:
            fig.add_trace(go.Scatter(x=x, y=fit, mode="lines", name="fit"))
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


# ---------------------------------------------------------------------------
# UI — Tabs
# ---------------------------------------------------------------------------
st.title("Transform AI — Diligence Grid (Pro)")
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

    pn = st.text_input("Project name", value=SS.get("project_name","My Diligence Project"))
    SS["project_name"] = pn

    st.write("**Loaded CSVs:**", list(SS["csv_files"].keys()) or "—")
    st.write("**Loaded PDFs:**", list(SS["pdf_files"].keys()) or "—")


# --------------------------- GRID ---------------------------
with tab_grid:
    st.subheader("Build Grid: rows, columns, and the Matrix Board")
    st.caption("Tip: Use the Matrix to map which modules run on each row (CSV rows → quant modules, PDF rows → PDF KPIs).")

    with st.container():
        a1, a2, a3, a4 = st.columns([1,1,1,1], gap="small")
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
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Undo", use_container_width=True): undo()
            with b2:
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
        new_mod = st.selectbox(
            "Module",
            MODULES,
            index=MODULES.index(SS.get("new_col_mod","NRR/GRR (CSV)")) if SS.get("new_col_mod","NRR/GRR (CSV)") in MODULES else 3
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
                # type guard: PDF row → only PDF KPIs
                if any(rr["id"]==rid and rr["row_type"]=="pdf" for rr in SS["rows"]):
                    sel = set(m for m in sel if m=="PDF KPIs (PDF)")
                SS["matrix"][rid] = sel
            st.success("Matrix updated")
    else:
        st.info("Add rows first, then use the Matrix to map modules.")


# --------------------------- RUN ----------------------------
with tab_run:
    st.subheader("Run — queue, process, and see status")
    st.toggle("Force re-run (ignore cache)", key="force_rerun", value=SS.get("force_rerun", False))

    # Save / Load project
    st.markdown("#### Save / Load Project")
    csl1, csl2 = st.columns([1,1])
    with csl1:
        st.download_button("💾 Save Project (.tfa.json)", data=save_project_bytes(),
                           file_name=f"{SS.get('project_name','project')}.tfa.json", use_container_width=True)
    with csl2:
        proj = st.file_uploader("Load Project (.tfa.json)", type=["json"], accept_multiple_files=False, key="load_proj")
        if proj and st.button("Load", use_container_width=True):
            load_project_json(proj.getvalue())

    # One-click QoE
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

    # Manual by Matrix
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
        c1, c2 = st.columns(2)
        with c1:
            if st.download_button("Export results CSV", data=export_results_csv(), file_name="transformai_results.csv"):
                pass
        with c2:
            if REPORTLAB_OK and st.download_button("Export memo PDF (demo)", data=export_results_pdf(), file_name="TransformAI_Memo_demo.pdf"):
                pass


# --------------------------- SHEET (Agentic Spreadsheet) ---------------------
with tab_sheet:
    st.subheader("Agentic Spreadsheet (status by cell)")
    # pick QoE columns first; fall back to all columns if none
    qoe_cols = [c for c in SS["columns"] if c["module"] in {m for _,m in QOE_TEMPLATE}] or SS["columns"]

    header = ["Row"] + [c["label"] for c in qoe_cols]
    table = []
    for r in SS["rows"]:
        row_vals = [r["alias"]]
        for c in qoe_cols:
            res = SS["results"].get((r["id"], c["id"]), {})
            mark = "✓ " + (str(res.get("value")) if res.get("value") is not None else "")
            if res.get("status") == "queued": mark = "… queued"
            if res.get("status") == "running": mark = "⟳ running"
            if res.get("status") == "cached": mark = "⟲ cached"
            if res.get("status") == "error":  mark = "⚠ error"
            if not res: mark = ""
            row_vals.append(mark)
        table.append(row_vals)

    df_sheet = pd.DataFrame(table, columns=header)
    st.dataframe(df_sheet, use_container_width=True, height=min(400, 120 + 28*len(df_sheet)))


# --------------------------- REVIEW (focused viz by cell) --------------------
with tab_review:
    st.subheader("Review a single cell — charts/evidence render for your selection")

    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}

    # Row and Column selectors
    row_opt = [(r["id"], r["alias"]) for r in SS["rows"]]
    col_opt = [(c["id"], f"{c['label']}  ·  {c['module']}") for c in SS["columns"]]

    csel1, csel2, csel3 = st.columns([2,2,1])
    with csel1:
        rid = st.selectbox("Row", row_opt, format_func=lambda t: t[1]) if row_opt else None
    with csel2:
        cid = st.selectbox("Column", col_opt, format_func=lambda t: t[1]) if col_opt else None
    with csel3:
        if rid and cid and st.button("Retry"):
            retry_cell(rid[0], cid[0]); run_queued_jobs()

    if not (rid and cid):
        st.info("Choose a Row and a Column above.")
    else:
        rid, cid = rid[0], cid[0]
        res = SS["results"].get((rid, cid))
        r = rows_by_id.get(rid); c = cols_by_id.get(cid)

        # Action to (re)run on demand
        if st.button("Run this cell now", type="primary"):
            enqueue_pairs([(rid, cid)], respect_cache=False)
            run_queued_jobs()
            res = SS["results"].get((rid, cid))

        if not res:
            st.warning("No result yet. Click **Run this cell now**.")
        else:
            st.caption(f"**{r['alias']}** → **{c['label']}** ({c['module']})")
            st.write(res.get("summary", ""))
            module = c["module"]

            # Render only the charts/evidence relevant to this cell/module
            if module == "Cohort Retention (CSV)" or "curve" in res:
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**Retention curve**")
                    plot_retention(res.get("curve", []))
                with colB:
                    st.markdown("**Cohort heatmap**")
                    plot_retention_heatmap(res.get("cohort_matrix", []), res.get("cohorts", []), res.get("ages", []))

                with st.expander("View cohort table / Export CSV", expanded=False):
                    matrix = res.get("cohort_matrix", [])
                    cohorts = res.get("cohorts", [])
                    ages = res.get("ages", [])
                    if matrix and cohorts and ages:
                        dfc = pd.DataFrame(matrix, index=cohorts, columns=[f"M{a}" for a in ages])
                        dfc.index.name = "Cohort"
                        st.dataframe(dfc, use_container_width=True, height=320)
                        st.download_button("Download cohort table CSV", data=export_cohort_csv(res),
                                           file_name=f"{r['alias'].replace(' ','_')}_cohort_table.csv")
                    else:
                        st.info("No cohort table available.")

            elif module == "NRR/GRR (CSV)":
                st.markdown("**NRR / GRR by month**")
                plot_nrr(res.get("series", []))

            elif module == "Pricing Power (CSV)":
                st.markdown("**Price–Demand (log) with fit**")
                plot_pricing(res.get("scatter", {}))

            elif module == "Unit Economics (CSV)":
                # simple numeric panel, no chart needed
                kpi = {k: res.get(k) for k in ["aov","gm","cac","cm"] if k in res}
                st.metric(label="Contribution Margin (per order)", value=f"${res.get('value'):.2f}")
                cols4 = st.columns(3)
                with cols4[0]: st.metric("AOV", f"${kpi.get('aov',0):.2f}")
                with cols4[1]: st.metric("GM", f"{kpi.get('gm',0):.0%}")
                with cols4[2]: st.metric("CAC", f"${kpi.get('cac',0):.0f}")

            elif module == "PDF KPIs (PDF)":
                st.markdown("**Evidence viewer**")
                auto_hits: List[Dict[str, Any]] = res.get("evidence", []) or []
                manual_hits: List[Dict[str, Any]] = res.get("evidence_manual", []) or []

                kpis_available = sorted(set([h["kpi"] for h in auto_hits] + [h["kpi"] for h in manual_hits]))
                if not kpis_available:
                    st.info("No parsed KPI hits found. Add manual citations below.")
                else:
                    sel_kpi = st.selectbox("Select KPI", kpis_available, index=0)
                    colL, colR = st.columns([1,2])
                    with colL:
                        st.write("**All hits**")
                        hits = [h for h in auto_hits + manual_hits if h["kpi"] == sel_kpi]
                        if hits:
                            dfh = pd.DataFrame(hits)[["page","snippet"]]
                            st.dataframe(dfh, use_container_width=True, height=260)
                        else:
                            st.info("No hits for this KPI.")
                    with colR:
                        st.write("**Selected KPI preview**")
                        for h in hits[:5]:
                            st.caption(f"p.{h['page']}: {h['snippet'][:200]}")

                st.markdown("**Add manual citation**")
                kc1, kc2 = st.columns([1,1])
                with kc1:
                    kpi_name = st.selectbox("KPI", ["revenue","ebitda","gross_margin","churn","other"], index=0, key="manual_kpi")
                    page_no = st.number_input("Page #", min_value=1, value=1, step=1, key="manual_page")
                with kc2:
                    snip = st.text_input("Snippet / quote", key="manual_snip")
                    if st.button("Add citation"):
                        res.setdefault("evidence_manual", []).append({"kpi": kpi_name, "page": int(page_no), "snippet": snip[:240]})
                        SS["results"][(rid, cid)] = res
                        st.success("Added.")

                if not PDF_PARSE_OK:
                    st.info("PDF text parser not installed. Auto-detection limited. Manual citations still work.")

            else:
                st.write("No renderer for this module yet.")


# --------------------------- MEMO (placeholder) ------------------------------
with tab_memo:
    st.subheader("Investor memo (demo placeholder)")
    st.caption("Approved cells would be assembled into memo sections here.")
    if REPORTLAB_OK:
        st.write("Use **Run → Export memo PDF (demo)** to preview.")
    else:
        st.info("Install `reportlab` to enable PDF export.")
