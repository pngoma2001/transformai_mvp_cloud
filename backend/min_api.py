# backend/min_api.py
# Minimal FastAPI backend for TransformAI Diligence Grid (demo)
# - Ingest CSV/PDF (multipart)
# - Create grid (rows/columns) and auto-make cells
# - Run cells for modules: cohort_retention, pricing_power, nrr_grr, pdf_kpi_extract
# - List cells, approve/retry
# - Compose memo and export as PDF (streaming)
# - No auth, in-memory storage, CORS open for easy frontends

from __future__ import annotations
import io, json, re, textwrap, uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# Optional PDF text extractor
try:
    from pypdf import PdfReader  # pip install pypdf
except Exception:
    PdfReader = None

# PDF export (memo)
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

app = FastAPI(title="TransformAI Mini API", version="0.1")

# CORS: open
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ---------------- In-memory stores ----------------
DATA = {
    "tables": {},      # name -> pandas.DataFrame
    "docs": {},        # name -> List[str] (page texts)
    "schema": {},      # table_name -> {"customer","date","revenue","price","quantity"}
}

GRIDS: Dict[str, Dict[str, Any]] = {}  # grid_id -> grid dict
CELLS: Dict[str, Dict[str, Any]] = {}  # cell_id -> cell dict
RUNS:  Dict[str, Dict[str, Any]] = {}  # run_id -> run dict

# ---------------- Utilities ----------------
def _new_id(prefix="id"): return f"{prefix}_{uuid.uuid4().hex[:8]}"

def _find(df: pd.DataFrame, key: str) -> Optional[str]:
    """Best-effort column guesser."""
    candidates = {
        "customer": ["customer","user","buyer","account","client","cust","cust_id","customer_id"],
        "date":     ["date","timestamp","order_date","created_at","period","month"],
        "revenue":  ["revenue","amount","net_revenue","sales","gmv","value"],
        "price":    ["price","unit_price","avg_price","p"],
        "quantity": ["qty","quantity","units","volume","q"]
    }.get(key.lower(), [key.lower()])
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for n in candidates:
        if n in lower: return lower[n]
    for n in candidates:
        for c in cols:
            if n in c.lower(): return c
    return None

def _get_mapping(table_name: str) -> Dict[str, Optional[str]]:
    m = DATA["schema"].get(table_name) or {}
    # auto-guess if not set
    if table_name in DATA["tables"]:
        df = DATA["tables"][table_name]
        m = {
            "customer": m.get("customer") or _find(df,"customer"),
            "date":     m.get("date")     or _find(df,"date"),
            "revenue":  m.get("revenue")  or _find(df,"revenue"),
            "price":    m.get("price")    or _find(df,"price"),
            "quantity": m.get("quantity") or _find(df,"quantity"),
        }
    return m

# ---------------- Modules ----------------
def module_cohort_retention(df: pd.DataFrame, customer: Optional[str], date: Optional[str], revenue: Optional[str]):
    if df is None or df.empty:
        return {"kpis":{}, "narrative":"Empty dataset.", "citations":[]}
    if not (customer and date):
        return {"kpis":{}, "narrative":"Missing customer/date columns; set schema.", "citations":[]}
    d = df.copy()
    d[date] = pd.to_datetime(d[date], errors="coerce")
    d = d.dropna(subset=[date, customer]).sort_values(date)
    d["first_month"] = d.groupby(customer)[date].transform("min").dt.to_period("M")
    d["age"] = (d[date].dt.to_period("M") - d["first_month"]).apply(lambda p: p.n)
    cohort_sizes = d.drop_duplicates([customer, "first_month"]).groupby("first_month")[customer].count()
    active = d.groupby(["first_month", "age"])[customer].nunique()
    mat = (active / cohort_sizes).unstack(fill_value=0).sort_index()
    curve = mat.mean(axis=0) if not mat.empty else pd.Series(dtype=float)
    m3 = float(round(curve.get(3, np.nan), 4)) if not curve.empty else np.nan
    ltv_12 = None
    if revenue and revenue in d.columns:
        rev = d.groupby([customer, d[date].dt.to_period("M")])[revenue].sum().groupby(customer).sum()
        ltv_12 = float(round(float(rev.mean()), 2))
    narrative = f"Retention stabilizes ~M3 at {m3:.0%}." if m3==m3 else "Not enough data to compute M3 retention."
    if ltv_12: narrative += f" Avg 12-month LTV proxy ≈ ${ltv_12:,.2f}."
    return {
        "kpis": {"month_3_retention": m3, "ltv_12m": ltv_12},
        "narrative": narrative,
        "citations":[{"type":"table","ref":"(uploaded CSV)","selector":"all_rows"}]
    }

def module_pricing_power(df: pd.DataFrame, price: Optional[str], qty: Optional[str]):
    if df is None or df.empty:
        return {"kpis":{}, "narrative":"Empty dataset.", "citations":[]}
    if not (price and qty):
        return {"kpis":{}, "narrative":"Missing price/quantity columns; set schema.", "citations":[]}
    d = df[[price, qty]].dropna()
    d = d[(d[price] > 0) & (d[qty] > 0)]
    if len(d) < 8:
        return {"kpis":{}, "narrative":"Need ≥ 8 observations for elasticity regression.", "citations":[]}
    X = np.log(d[price].values)
    Y = np.log(d[qty].values)
    A = np.vstack([X, np.ones(len(X))]).T
    beta, intercept = np.linalg.lstsq(A, Y, rcond=None)[0]   # Y ≈ beta*X + intercept
    yhat = beta*X + intercept
    ss_res = float(np.sum((Y - yhat)**2))
    ss_tot = float(np.sum((Y - np.mean(Y))**2))
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
    narrative = f"Own-price elasticity ≈ {beta:.2f} (R²={r2:.2f}). "
    narrative += "Inelastic (|ε|<1)." if abs(beta) < 1 else "Elastic (|ε|≥1)."
    return {
        "kpis": {"elasticity": float(beta), "r2": r2},
        "narrative": narrative,
        "citations":[{"type":"table","ref":"(uploaded CSV)","selector":"price/qty columns"}]
    }

def module_nrr_grr(df: pd.DataFrame, customer: Optional[str], date: Optional[str], revenue: Optional[str]):
    if df is None or df.empty:
        return {"kpis":{}, "narrative":"Empty dataset.", "citations":[]}
    if not (customer and date and revenue):
        return {"kpis":{}, "narrative":"Need customer/date/revenue columns; set schema.", "citations":[]}
    d = df[[customer, date, revenue]].copy()
    d[date] = pd.to_datetime(d[date], errors="coerce")
    d = d.dropna(subset=[customer, date, revenue])
    d["month"] = d[date].dt.to_period("M")
    gp = d.groupby([customer, "month"], as_index=False)[revenue].sum()
    pivot = gp.pivot(index=customer, columns="month", values=revenue).fillna(0.0).sort_index(axis=1)
    months = list(pivot.columns)
    if len(months) < 2:
        return {"kpis":{}, "narrative":"Need at least two months of data.", "citations":[]}
    labels, grr_list, nrr_list = [], [], []
    churn_rate, contraction_rate, expansion_rate = [], [], []
    for i in range(1, len(months)):
        prev_m, curr_m = months[i-1], months[i]
        prev_rev = pivot[prev_m]; curr_rev = pivot[curr_m]
        base_mask = prev_rev > 0
        start = float(prev_rev[base_mask].sum())
        if start <= 0: continue
        curr_base = curr_rev[base_mask]
        churn_amt = float(prev_rev[base_mask & (curr_base == 0)].sum())
        contraction_amt = float(((prev_rev[base_mask] - curr_base).clip(lower=0.0).sum()) - churn_amt); contraction_amt = max(contraction_amt, 0.0)
        expansion_amt = float((curr_base - prev_rev[base_mask]).clip(lower=0.0).sum())
        grr = (start - churn_amt - contraction_amt) / start if start else np.nan
        nrr = (start - churn_amt - contraction_amt + expansion_amt) / start if start else np.nan
        labels.append(str(curr_m)); grr_list.append(grr); nrr_list.append(nrr)
        churn_rate.append(churn_amt/start); contraction_rate.append(contraction_amt/start); expansion_rate.append(expansion_amt/start)
    if not labels:
        return {"kpis":{}, "narrative":"Insufficient overlap to compute retention.", "citations":[]}
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
    return {"kpis":kpis, "narrative":narrative, "citations":[{"type":"table","ref":"(uploaded CSV)","selector":"monthly revenue by customer"}]}

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

def module_pdf_kpi(pages: List[str]):
    if not pages:
        return {"kpis":{}, "narrative":"Empty PDF.", "citations":[]}
    def _scan_metric(keywords: List[str], want: str):
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
                            if val is not None: return {"page": i+1, "snippet": window.strip(), "raw": n.group(), "value": val}
                    else:
                        n = _pct.search(window)
                        if n:
                            val = _parse_pct(n.group())
                            if val is not None: return {"page": i+1, "snippet": window.strip(), "raw": n.group(), "value": val}
        return None
    rev = _scan_metric(["revenue","revenues","total revenue"], "money")
    ebt = _scan_metric(["ebitda","adj ebitda"], "money")
    gm  = _scan_metric(["gross margin","gm%","gm"], "pct")
    chn = _scan_metric(["churn","net churn"], "pct")
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
    return {"kpis":kpis, "narrative":narrative, "citations":citations}

MODULES = {
    "cohort_retention": {"title":"Cohort Retention (CSV)", "fn": module_cohort_retention, "needs": ["customer","date"], "optional": ["revenue"]},
    "pricing_power":   {"title":"Pricing Power (CSV)",     "fn": module_pricing_power,   "needs": ["price","quantity"], "optional": []},
    "nrr_grr":         {"title":"NRR/GRR (CSV)",           "fn": module_nrr_grr,         "needs": ["customer","date","revenue"], "optional": []},
    "pdf_kpi_extract": {"title":"PDF KPI Extract",         "fn": module_pdf_kpi,         "needs": [], "optional": []},
}

# ---------------- Schemas ----------------
from pydantic import BaseModel, Field

class RowCreate(BaseModel):
    row_ref: str  # "table:<name>" or "pdf:<name>"

class ColCreate(BaseModel):
    name: str
    tool: str
    params: Dict[str, Any] = Field(default_factory=dict)

class GridCreate(BaseModel):
    name: str = "Deal Grid"
    rows: List[RowCreate]
    columns: List[ColCreate]

class RunSelection(BaseModel):
    rows: Optional[List[str]] = None  # row_ids
    cols: Optional[List[str]] = None  # col_ids

class SchemaSet(BaseModel):
    table: str
    map: Dict[str, Optional[str]]

# ---------------- Ingest ----------------
@app.post("/ingest/csv")
async def ingest_csv(files: List[UploadFile] = File(...)):
    loaded = []
    for f in files:
        try:
            df = pd.read_csv(io.BytesIO(await f.read()))
            DATA["tables"][f.filename] = df
            loaded.append({"name": f.filename, "rows": int(df.shape[0]), "cols": int(df.shape[1])})
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"{f.filename}: {e}")
    return {"ok": True, "tables": loaded}

@app.post("/ingest/pdf")
async def ingest_pdf(files: List[UploadFile] = File(...)):
    if PdfReader is None:
        raise HTTPException(status_code=400, detail="pypdf not installed on server.")
    loaded = []
    for f in files:
        data = await f.read()
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = []
            for p in reader.pages:
                try: pages.append(p.extract_text() or "")
                except Exception: pages.append("")
            DATA["docs"][f.filename] = pages
            loaded.append({"name": f.filename, "pages": len(pages)})
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"{f.filename}: {e}")
    return {"ok": True, "docs": loaded}

@app.post("/schema")
def set_schema(body: SchemaSet):
    DATA["schema"][body.table] = body.map
    return {"ok": True, "table": body.table, "map": body.map}

# ---------------- Grid ----------------
@app.post("/grid")
def create_grid(body: GridCreate):
    gid = _new_id("grid")
    grid = {"id": gid, "name": body.name, "rows": [], "columns": [], "cells": [], "activities": []}

    # rows
    for r in body.rows:
        kind, name = r.row_ref.split(":", 1)
        if kind not in ("table","pdf"):
            raise HTTPException(status_code=400, detail=f"row_ref must start with table: or pdf:")
        if kind=="table" and name not in DATA["tables"]:
            raise HTTPException(status_code=400, detail=f"unknown table {name}")
        if kind=="pdf" and name not in DATA["docs"]:
            raise HTTPException(status_code=400, detail=f"unknown pdf {name}")
        rid = _new_id("row")
        grid["rows"].append({"id": rid, "row_ref": r.row_ref, "source": name, "type": kind})

    # columns
    for c in body.columns:
        if c.tool not in MODULES:
            raise HTTPException(status_code=400, detail=f"unknown tool {c.tool}")
        cid = _new_id("col")
        grid["columns"].append({"id": cid, "name": c.name, "tool": c.tool, "params": c.params})

    # cells
    for r in grid["rows"]:
        for c in grid["columns"]:
            cell_id = _new_id("cell")
            cell = {
                "id": cell_id, "grid_id": gid, "row_id": r["id"], "col_id": c["id"],
                "status": "queued", "output_text": None, "numeric_value": None, "units": None,
                "citations": [], "confidence": None, "notes": [], "created_at": datetime.utcnow().isoformat()
            }
            CELLS[cell_id] = cell
            grid["cells"].append(cell_id)

    GRIDS[gid] = grid
    return grid

@app.get("/grid/{grid_id}")
def get_grid(grid_id: str):
    grid = GRIDS.get(grid_id)
    if not grid: raise HTTPException(status_code=404, detail="grid not found")
    # expand cells
    cells = [CELLS[cid] for cid in grid["cells"]]
    return {**grid, "cells": cells}

# ---------------- Run cells ----------------
@app.post("/grid/{grid_id}/run")
def run_grid(grid_id: str, body: RunSelection):
    grid = GRIDS.get(grid_id)
    if not grid: raise HTTPException(status_code=404, detail="grid not found")
    # select cells
    targets = []
    for cid in grid["cells"]:
        c = CELLS[cid]
        if body.rows and c["row_id"] not in set(body.rows): continue
        if body.cols and c["col_id"] not in set(body.cols): continue
        targets.append(c)
    # execute sequentially (demo)
    ran = 0
    for cell in targets:
        row = next(r for r in grid["rows"] if r["id"]==cell["row_id"])
        col = next(c for c in grid["columns"] if c["id"]==cell["col_id"])
        cell["status"] = "running"
        # route to modules
        if row["type"]=="table":
            df = DATA["tables"].get(row["source"])
            mapping = _get_mapping(row["source"])
            if col["tool"]=="cohort_retention":
                res = module_cohort_retention(df, mapping.get("customer"), mapping.get("date"), mapping.get("revenue"))
            elif col["tool"]=="pricing_power":
                res = module_pricing_power(df, mapping.get("price"), mapping.get("quantity"))
            elif col["tool"]=="nrr_grr":
                res = module_nrr_grr(df, mapping.get("customer"), mapping.get("date"), mapping.get("revenue"))
            elif col["tool"]=="pdf_kpi_extract":
                res = {"kpis":{}, "narrative":"PDF module requires a PDF row.", "citations":[]}
            else:
                res = {"kpis":{}, "narrative":f"Unknown tool {col['tool']}", "citations":[]}
        else:
            pages = DATA["docs"].get(row["source"], [])
            if col["tool"]=="pdf_kpi_extract":
                res = module_pdf_kpi(pages)
            else:
                res = {"kpis":{}, "narrative":f"{col['tool']} applies to CSV rows.", "citations":[]}
        # finalize
        cell["status"] = "done" if res["kpis"] else "needs_review"
        cell["output_text"] = res["narrative"]
        cell["numeric_value"] = (next(iter(res["kpis"].values())) if res["kpis"] else None)
        cell["units"] = None
        cell["citations"] = res["citations"]
        ran += 1
    return {"ok": True, "ran": ran}

# ---------------- Cells list / approve / retry ----------------
@app.get("/cells")
def list_cells(grid_id: str = Query(...)):
    grid = GRIDS.get(grid_id)
    if not grid: raise HTTPException(status_code=404, detail="grid not found")
    return [CELLS[cid] for cid in grid["cells"]]

@app.post("/cells/{cell_id}/approve")
def approve_cell(cell_id: str):
    cell = CELLS.get(cell_id)
    if not cell: raise HTTPException(status_code=404, detail="cell not found")
    cell["status"] = "approved"
    return {"ok": True, "cell": cell}

@app.post("/cells/{cell_id}/retry")
def retry_cell(cell_id: str):
    cell = CELLS.get(cell_id)
    if not cell: raise HTTPException(status_code=404, detail="cell not found")
    cell["status"] = "queued"; cell["output_text"]=None; cell["numeric_value"]=None; cell["citations"]=[]
    return {"ok": True, "cell": cell}

# ---------------- Memo & Export ----------------
@app.get("/memo")
def compose_memo(grid_id: str):
    grid = GRIDS.get(grid_id)
    if not grid: raise HTTPException(status_code=404, detail="grid not found")
    cells = [CELLS[cid] for cid in grid["cells"] if CELLS[cid]["status"]=="approved"]
    exec_sum = []
    for c in cells[:8]:
        col = next(x for x in grid["columns"] if x["id"]==c["col_id"])
        row = next(x for x in grid["rows"] if x["id"]==c["row_id"])
        val = c.get("numeric_value")
        vs = f"{val:,.0f}" if isinstance(val,(int,float)) else (val or "")
        exec_sum.append({"title": col["name"], "row": row["row_ref"], "value": vs, "text": c.get("output_text")})
    appendix = [{"cell_id": c["id"], "citations": c.get("citations", [])} for c in cells]
    return {"grid_id": grid_id, "executive_summary": exec_sum, "evidence_appendix": appendix}

@app.post("/export")
def export_pdf(grid_id: str = Body(..., embed=True)):
    memo = compose_memo(grid_id)
    # render simple PDF
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=LETTER)
    W,H = LETTER; M = 0.75*inch; y = H-M
    c.setFont("Helvetica-Bold", 14); c.drawString(M,y,"Transform AI — Investment Memo"); y -= 18
    c.setFont("Helvetica", 9); c.drawString(M,y,f"Grid: {grid_id}  •  Generated: {datetime.utcnow().isoformat()}Z"); y -= 16
    c.setFont("Helvetica-Bold", 12); c.drawString(M,y,"Executive Summary"); y -= 14
    c.setFont("Helvetica", 11)
    for item in memo["executive_summary"]:
        line = f"- {item['title']} on {item['row']}: {item['value']} — {item['text'] or ''}"
        for seg in textwrap.wrap(line, width=95):
            if y < M: c.showPage(); y = H-M; c.setFont("Helvetica", 11)
            c.drawString(M,y,seg); y -= 14
    y -= 8; c.setFont("Helvetica-Bold", 12); c.drawString(M,y,"Evidence Appendix"); y -= 14; c.setFont("Helvetica", 9)
    for entry in memo["evidence_appendix"]:
        line = f"Cell {entry['cell_id']} citations: {json.dumps(entry['citations'])[:300]}"
        for seg in textwrap.wrap(line, width=95):
            if y < M: c.showPage(); y = H-M; c.setFont("Helvetica", 9)
            c.drawString(M,y,seg); y -= 12
    c.showPage(); c.save(); buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="TransformAI_Memo.pdf"'})
