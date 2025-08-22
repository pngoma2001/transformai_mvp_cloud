# pages/3_Diligence_Grid_Pro.py
# Transform AI — Diligence Grid (Pro)
# NOTE: This patch ONLY upgrades the Cohort Retention math to a realistic implementation.
# All other features/UX remain unchanged: Data, Grid/Matrix, Run, Sheet, Review, Memo,
# modules (PDF KPIs, NRR/GRR, Pricing, Unit Econ, PVM Bridge), approvals, exports, etc.

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

# Plotly for charts (primary renderer)
try:
    import plotly.graph_objs as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# Altair for charts (fallback)
try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False


# -----------------------------------------------------------------------------
# Page & CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Transform AI — Diligence Grid (Pro)", layout="wide")
st.markdown(
    """
<style>
.block-container {max-width: 1700px !important; padding-top: 0.5rem;}
/* Prevent title clipping on some screens */
h1, .stMarkdown h1 {
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  line-height: 1.15 !important;
  margin-top: .25rem !important;
}
.stDataFrame [role="checkbox"] {transform: scale(1.0);}
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

    # Keep Unit Econ inputs (no sidebar UI here)
    SS.setdefault("whatif_gm", 0.62)
    SS.setdefault("whatif_cac", 42.0)

    # Undo/redo snapshots
    SS.setdefault("undo", [])
    SS.setdefault("redo", [])

    # Last focused cell to preselect in Review
    SS.setdefault("last_focus", None)

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
    }, default=str))
    SS["redo"].clear()

def snapshot_apply(snap: str):
    data = json.loads(snap)
    SS["rows"]    = data.get("rows", [])
    SS["columns"] = data.get("columns", [])
    SS["matrix"]  = {k: set(v) for k, v in data.get("matrix", {}).items()}
    SS["results"] = _unpack_results(data.get("results", {}))

def undo():
    if not SS["undo"]:
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
    if not SS["redo"]:
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
CANONICAL = ["customer_id","order_date","amount","price","quantity","month","revenue","product"]

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
        "product":     pick("product","sku","item","name"),
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
    # ensure product exists for PVM bridge math; safe default
    if "product" not in df.columns:
        df["product"] = "ALL"
    return df


# -----------------------------------------------------------------------------
# Grid helpers
# -----------------------------------------------------------------------------
MODULES = [
    "PDF KPIs (PDF)",
    "Cohort Retention (CSV)",
    "NRR/GRR (CSV)",
    "Pricing Power (CSV)",
    "Unit Economics (CSV)",
    "PVM Bridge (CSV)",
]

QOE_TEMPLATE = [
    ("PDF KPIs",        "PDF KPIs (PDF)"),
    ("Unit Economics",  "Unit Economics (CSV)"),
    ("NRR/GRR",         "NRR/GRR (CSV)"),
    ("Pricing Power",   "Pricing Power (CSV)"),
    # Cohort/PVM can be added ad-hoc in your grid; leaving template minimal on purpose
]

def add_rows_from_csvs():
    snapshot_push()
    for name in SS["csv_files"].keys():
        if not any(r["source"] == name for r in SS["rows"]):
            rid = uid("row")
            SS["rows"].append({"id": rid, "alias": name.replace(".csv",""), "row_type":"table", "source": name})
            SS["matrix"].setdefault(rid, set([
                "Cohort Retention (CSV)","NRR/GRR (CSV)","Pricing Power (CSV)","Unit Economics (CSV)","PVM Bridge (CSV)"
            ]))

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
# Engines (calculations)
# -----------------------------------------------------------------------------
def _pdf_kpis(_raw: bytes) -> Dict[str, Any]:
    # (Demo) Extracted values + quotes/pages would be attached as 'evidence'
    return dict(summary="Revenue ≈ $12.5M; EBITDA ≈ $1.3M; GM ≈ 62%; Churn ≈ 4%")

def _cohort_retention_real(df: pd.DataFrame, min_cohort_size: int = 10, max_horizon: int = 12) -> Dict[str, Any]:
    """
    Realistic cohort retention:
      - Cohort = customer's FIRST purchase month
      - Retained in month k if customer has ANY order in cohort_month + k
      - Returns:
          curve: average retention across cohorts (0..K)
          heatmap: 2D array [cohort x month_index] (% retained)
          cohorts: list of cohort month strings
          counts: list of cohort sizes (M0 customers)
          m3: month-3 retention (avg)
          evidence: small table with cohort sizes and m3 by cohort
    """
    d = df.copy()
    # Require these columns
    need = {"customer_id"}
    if not need.issubset(set(d.columns)):
        raise ValueError("Cohort needs at least 'customer_id' and 'order_date' or 'month'.")
    # Ensure month
    if "month" not in d.columns:
        if "order_date" not in d.columns:
            raise ValueError("Provide 'order_date' or precomputed 'month' for cohort retention.")
        d["order_date"] = pd.to_datetime(d["order_date"], errors="coerce")
        d = d.dropna(subset=["order_date"])
        d["month"] = d["order_date"].dt.to_period("M").astype(str)

    # First purchase month per customer (cohort key)
    first = (
        d.groupby("customer_id")["month"]
        .min()
        .rename("cohort")
        .to_frame()
        .reset_index()
    )
    d = d.merge(first, on="customer_id", how="left")

    # Normalize months to integer indices
    all_months = pd.period_range(start=min(d["month"]), end=max(d["month"]), freq="M")
    month_to_idx = {str(p): i for i, p in enumerate(all_months.astype(str))}
    d["cohort_idx"] = d["cohort"].map(month_to_idx)
    d["month_idx"] = d["month"].map(month_to_idx)
    d["age"] = d["month_idx"] - d["cohort_idx"]
    d = d[(d["age"] >= 0) & (d["age"] <= max_horizon)]

    # Cohort sizes at age 0
    m0 = d[d["age"] == 0].groupby("cohort")["customer_id"].nunique().rename("m0").to_frame()
    m0 = m0[m0["m0"] > 0]

    # Active customers by cohort x age
    active = (
        d.groupby(["cohort", "age"])["customer_id"]
        .nunique()
        .rename("active")
        .to_frame()
        .reset_index()
    )
    # Merge sizes to compute retention
    ret = active.merge(m0, on="cohort", how="left")
    ret = ret[ret["m0"] >= min_cohort_size]  # exclude tiny cohorts
    if ret.empty:
        # Fallback for thin data; synthetic pattern
        curve = [1.0, 0.88, 0.79, 0.72, 0.69, 0.66][:max_horizon+1]
        return dict(
            value=curve[3] if len(curve) > 3 else None,
            curve=curve,
            heatmap=[curve]*min(3, max_horizon+1),
            cohorts=["demo"]*min(3, max_horizon+1),
            counts=[100]*min(3, max_horizon+1),
            summary=f"Retention stabilizes ~M3 at {curve[3]:.0%} (demo; insufficient cohort depth).",
            evidence=[{"cohort":"demo","m0":100,"m3":round(curve[3],2)}],
        )

    ret["retention"] = ret["active"] / ret["m0"]

    # Build heatmap matrix
    cohorts = sorted(ret["cohort"].unique())
    ages = list(range(0, max_horizon+1))
    heat = np.zeros((len(cohorts), len(ages)), dtype=float)
    heat[:] = np.nan
    for i, c in enumerate(cohorts):
        sub = ret[ret["cohort"] == c]
        for _, row in sub.iterrows():
            a = int(row["age"])
            if 0 <= a <= max_horizon:
                heat[i, a] = float(row["retention"])

    # Average curve across cohorts (ignore NaNs per age)
    curve = np.nanmean(heat, axis=0)
    curve = np.where(np.isnan(curve), np.nan, curve)
    # Replace missing leading entries conservatively
    if np.isnan(curve[0]): curve[0] = 1.0
    for k in range(1, len(curve)):
        if np.isnan(curve[k]):
            curve[k] = curve[k-1] if not np.isnan(curve[k-1]) else 0.0
    curve = [float(round(max(0.0, min(1.0, v)), 4)) for v in curve.tolist()]

    m3 = curve[3] if len(curve) > 3 else None

    # Evidence table: cohort size and m3 where available
    evidence = []
    m3_rows = ret[ret["age"] == 3][["cohort","retention"]].rename(columns={"retention":"m3"})
    e = m0.merge(m3_rows, on="cohort", how="left").fillna({"m3": np.nan})
    e = e.sort_values("cohort").head(12)
    for _, r in e.iterrows():
        evidence.append({"cohort": r["cohort"], "m0": int(r["m0"]), "m3": (None if np.isnan(r["m3"]) else float(round(r["m3"],4)))})

    # Cohort sizes list (aligned to cohorts order)
    counts_map = m0["m0"].to_dict()
    counts = [int(counts_map.get(c, 0)) for c in cohorts]

    # Clip heat to [0,1] and replace NaNs with 0 for display
    heat_display = np.where(np.isnan(heat), 0.0, np.clip(heat, 0.0, 1.0)).tolist()

    summary = f"Avg M3 retention ≈ {m3:.0%} across {len(cohorts)} cohorts." if m3 is not None else \
              f"Average retention curve computed across {len(cohorts)} cohorts."

    return dict(
        value=m3,
        curve=curve,
        heatmap=heat_display,
        cohorts=cohorts,
        counts=counts,
        summary=summary,
        evidence=evidence,
    )

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

def _unit_econ(df: pd.DataFrame, gm: float = 0.62, cac: float = 42.0) -> Dict[str, Any]:
    try:
        aov = float(df["amount"].mean()) if "amount" in df.columns else float(df.select_dtypes(np.number).sum(axis=1).mean())
        cm = round(gm*aov - cac, 2)
        return dict(value=cm, summary=f"AOV ${aov:.2f}, GM {gm:.0%}, CAC ${cac:.0f} → CM ${cm:.2f}.",
                    aov=aov, gm=gm, cac=cac, cm=cm)
    except Exception:
        return dict(value=32.0, summary="AOV $120.00, GM 60%, CAC $40 → CM $32.00 (demo).",
                    aov=120.0, gm=0.6, cac=40.0, cm=32.0)

def _pvm_bridge(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Simple Price-Volume-Mix bridge from earliest to latest month.
    Requires price, quantity, product, and month (derived if order_date exists).
    """
    d = df.copy()
    if "month" not in d.columns and "order_date" in d.columns:
        d["order_date"] = pd.to_datetime(d["order_date"], errors="coerce")
        d["month"] = d["order_date"].dt.to_period("M").astype(str)
    if not {"price","quantity","product","month"}.issubset(d.columns):
        return dict(summary="Insufficient columns for PVM (need price, quantity, product, month).", bars=[])
    g = d.groupby(["product","month"]).agg(price=("price","mean"), qty=("quantity","sum")).reset_index()
    months = sorted(g["month"].unique())
    if len(months) < 2:
        return dict(summary="Need at least two months for a bridge.", bars=[])
    m0, m1 = months[0], months[-1]
    base = g[g["month"]==m0].set_index("product")
    new  = g[g["month"]==m1].set_index("product")
    products = sorted(set(base.index).union(new.index))
    base = base.reindex(products).fillna(0.0); new = new.reindex(products).fillna(0.0)

    rev0 = float((base["price"]*base["qty"]).sum())
    rev1 = float((new["price"]*new["qty"]).sum())
    price_effect  = float(((new["price"]-base["price"])*base["qty"]).sum())
    volume_effect = float((base["price"]*(new["qty"]-base["qty"])).sum())
    mix_effect    = rev1 - rev0 - price_effect - volume_effect

    bars = [
        {"label": f"{m0}", "value": rev0},
        {"label": "Price", "value": price_effect},
        {"label": "Volume", "value": volume_effect},
        {"label": "Mix", "value": mix_effect},
        {"label": f"{m1}", "value": rev1},
    ]
    delta = rev1 - rev0
    sign = "↑" if delta>=0 else "↓"
    summary = f"PVM bridge {m0} → {m1}: ΔRevenue {sign} ${abs(delta):,.0f} (Price {price_effect:+,.0f}, Volume {volume_effect:+,.0f}, Mix {mix_effect:+,.0f})."
    return dict(summary=summary, bars=bars)


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
        k = _cohort_retention_real(df)
        out = {
            "status":"done","value":k["value"],"summary":k["summary"],"last_run": now_ts(),
            "curve": k.get("curve", []),
            "heatmap": k.get("heatmap", []),
            "cohorts": k.get("cohorts", []),
            "counts": k.get("counts", []),
            "evidence": k.get("evidence", []),
        }
        return out

    if mod == "NRR/GRR (CSV)":
        df = materialize_df(row["source"])
        k = _nrr_grr(df)
        return {"status":"done","value":k["value"],"summary":k["summary"],"last_run": now_ts(), **k}

    if mod == "Pricing Power (CSV)":
        df = materialize_df(row["source"])
        k = _pricing(df)
        return {"status":"done","value":k["value"],"summary":k["summary"],"last_run": now_ts(), **k}

    if mod == "Unit Economics (CSV)":
        df = materialize_df(row["source"])
        k = _unit_econ(df, gm=SS.get("whatif_gm",0.62), cac=SS.get("whatif_cac",42.0))
        return {"status":"done","value":k["value"],"summary":k["summary"],"last_run": now_ts(), **k}

    if mod == "PVM Bridge (CSV)":
        df = materialize_df(row["source"])
        k = _pvm_bridge(df)
        return {"status":"done","value":None,"summary":k["summary"],"last_run": now_ts(), **k}

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
            SS["last_focus"] = (rid, cid)
            continue
        SS["results"][key] = {"status":"queued","value":None,"summary":None}
        SS["jobs"].append({"rid":rid,"cid":cid,"status":"queued","started":None,"ended":None,"note":""})

def run_queued_jobs():
    by_r = {r["id"]: r for r in SS["rows"]}
    by_c = {c["id"]: c for c in SS["columns"]}
    last_done = None
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
            last_done = (rid, cid)
        except Exception as e:
            SS["results"][(rid,cid)]={"status":"error","value":None,"summary":str(e),"last_run": now_ts()}
            j["status"]="error"; j["note"]=str(e); j["ended"]=now_ts()
    if last_done:
        SS["last_focus"] = last_done

def retry_cell(rid: str, cid: str):
    SS["jobs"].insert(0, {"rid":rid,"cid":cid,"status":"retry","started":None,"ended":None,"note":"manual retry"})


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------
def export_results_csv(approved_only: bool = False) -> bytes:
    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}
    out = []
    for (rid,cid), res in SS["results"].items():
        if approved_only and not res.get("approved", False):
            continue
        r = rows_by_id.get(rid); c = cols_by_id.get(cid)
        if not r or not c: continue
        out.append(dict(
            row=r["alias"], row_type=r["row_type"], source=r["source"],
            column=c["label"], module=c["module"],
            status=res.get("status"), value=res.get("value"),
            summary=res.get("summary"), last_run=res.get("last_run"),
            approved=bool(res.get("approved", False)),
        ))
    return pd.DataFrame(out).to_csv(index=False).encode("utf-8")

def export_results_pdf(approved_only: bool = True) -> bytes:
    if not REPORTLAB_OK: raise RuntimeError("reportlab not installed")
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    w,h = LETTER
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, h-72, "TransformAI — Memo (Demo)")
    y = h-100; c.setFont("Helvetica", 10)
    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}
    for (rid,cid),res in list(SS["results"].items()):
        if approved_only and not res.get("approved", False):
            continue
        r = rows_by_id.get(rid); cdef = cols_by_id.get(cid)
        if not r or not cdef: continue
        line = f"{r['alias']} → {cdef['label']}: {res.get('summary')}"
        for chunk in [line[i:i+95] for i in range(0,len(line),95)]:
            if y<72: c.showPage(); y=h-72; c.setFont("Helvetica",10)
            c.drawString(72,y,chunk); y-=14
    c.showPage(); c.save(); buf.seek(0)
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Plot helpers (focused Review)
# -----------------------------------------------------------------------------
def plot_retention(curve: List[float]):
    curve = [float(x) for x in (curve or [])]
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

def plot_retention_heatmap(z: List[List[float]]):
    if not z:
        st.info("No cohort heatmap available.")
        return
    arr = np.array(z, dtype=float)
    if PLOTLY_OK:
        fig = go.Figure(data=go.Heatmap(z=arr, colorscale="Blues"))
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
    elif ALTAIR_OK:
        df = pd.DataFrame(arr)
        df = df.reset_index().melt("index", var_name="age", value_name="retention")
        df = df.rename(columns={"index": "cohort_idx"})
        ch = (
            alt.Chart(df)
            .mark_rect()
            .encode(x="age:O", y="cohort_idx:O", color="retention:Q")
            .properties(height=320)
        )
        st.altair_chart(ch, use_container_width=True)
    else:
        st.write(pd.DataFrame(arr))

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


# -----------------------------------------------------------------------------
# UI — Tabs (kept as-is)
# -----------------------------------------------------------------------------
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
            pick("Product/SKU", "product")
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
            row = {
                "row_id": r["id"], "Alias": r["alias"], "Type": r["row_type"],
                "PDF KPIs (PDF)": "PDF KPIs (PDF)" in sel,
                "Cohort Retention (CSV)": "Cohort Retention (CSV)" in sel,
                "NRR/GRR (CSV)": "NRR/GRR (CSV)" in sel,
                "Pricing Power (CSV)": "Pricing Power (CSV)" in sel,
                "Unit Economics (CSV)": "Unit Economics (CSV)" in sel,
                "PVM Bridge (CSV)": "PVM Bridge (CSV)" in sel,
            }
            base.append(row)
        mdf = pd.DataFrame(base)
        mdf_edit = st.data_editor(
            mdf,
            column_config={
                "PDF KPIs (PDF)": st.column_config.CheckboxColumn(),
                "Cohort Retention (CSV)": st.column_config.CheckboxColumn(),
                "NRR/GRR (CSV)": st.column_config.CheckboxColumn(),
                "Pricing Power (CSV)": st.column_config.CheckboxColumn(),
                "Unit Economics (CSV)": st.column_config.CheckboxColumn(),
                "PVM Bridge (CSV)": st.column_config.CheckboxColumn(),
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
                # PDF safety: only allow PDF KPIs on PDF rows
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

    # Manual single run
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
    if SS["jobs"]:
        st.markdown("**Jobs**")
        st.dataframe(pd.DataFrame(SS["jobs"]), use_container_width=True, height=180)

    if SS["results"]:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Export ALL results (CSV)", data=export_results_csv(False), file_name="transformai_results_all.csv")
        with c2:
            st.download_button("Export APPROVED results (CSV)", data=export_results_csv(True), file_name="transformai_results_approved.csv")
        with c3:
            if REPORTLAB_OK:
                st.download_button("Export memo PDF (approved only)", data=export_results_pdf(True), file_name="TransformAI_Memo_demo.pdf")


# --------------------------- SHEET (Agentic Spreadsheet) ---------------------
with tab_sheet:
    st.subheader("Agentic Spreadsheet (status by cell)")
    header = ["Row"] + [c["label"] for c in SS["columns"]]
    table = []
    for r in SS["rows"]:
        row_vals = [r["alias"]]
        for c in SS["columns"]:
            res = SS["results"].get((r["id"], c["id"]), {})
            mark = "✓ " + (str(res.get("value")) if res.get("value") is not None else "")
            if res.get("status") == "queued": mark = "… queued"
            if res.get("status") == "running": mark = "⟳ running"
            if res.get("status") == "cached": mark = "⟲ cached"
            if res.get("status") == "error":  mark = "⚠ error"
            if not res: mark = ""
            if res.get("approved", False): mark = "✅ " + (str(res.get("value")) if res.get("value") is not None else "")
            row_vals.append(mark)
        table.append(row_vals)

    df_sheet = pd.DataFrame(table, columns=header)
    st.dataframe(df_sheet, use_container_width=True, height=min(420, 120 + 28*len(df_sheet)))


# --------------------------- REVIEW (focused viz by cell) --------------------
with tab_review:
    st.subheader("Review a single cell — charts render only for your selection")

    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}

    # default select last_focus if available
    row_opt = [(r["id"], r["alias"]) for r in SS["rows"]]
    col_opt = [(c["id"], f"{c['label']}  ·  {c['module']}") for c in SS["columns"]]

    default_row = 0
    default_col = 0
    if SS.get("last_focus"):
        rid0, cid0 = SS["last_focus"]
        try:
            default_row = [i for i,(rid,_) in enumerate(row_opt) if rid == rid0][0]
            default_col = [i for i,(cid,_) in enumerate(col_opt) if cid == cid0][0]
        except Exception:
            pass

    csel1, csel2, csel3, csel4 = st.columns([2,2,1,1])
    with csel1:
        rid = st.selectbox("Row", row_opt, format_func=lambda t: t[1], index=default_row if row_opt else 0) if row_opt else None
    with csel2:
        cid = st.selectbox("Column", col_opt, format_func=lambda t: t[1], index=default_col if col_opt else 0) if col_opt else None
    with csel3:
        if rid and cid and st.button("Run"):
            enqueue_pairs([(rid[0], cid[0])], respect_cache=False)
            run_queued_jobs()
    with csel4:
        if rid and cid:
            key = (rid[0], cid[0])
            res = SS["results"].get(key, {})
            approved = st.toggle("Approved", value=bool(res.get("approved", False)))
            if approved != bool(res.get("approved", False)):
                SS["results"][key] = {**res, "approved": approved}

    if not (rid and cid):
        st.info("Choose a Row and a Column above.")
    else:
        rid, cid = rid[0], cid[0]
        res = SS["results"].get((rid, cid))
        r = rows_by_id.get(rid); c = cols_by_id.get(cid)

        if not res:
            st.warning("No result yet. Click **Run**.")
        else:
            st.caption(f"**{r['alias']}** → **{c['label']}** ({c['module']})")
            st.write(res.get("summary", ""))
            module = c["module"]

            # Render only the charts relevant to this cell/module
            if module == "Cohort Retention (CSV)":
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**Retention curve (avg across cohorts)**")
                    plot_retention(res.get("curve", []))
                with colB:
                    st.markdown("**Cohort heatmap (cohort × age)**")
                    plot_retention_heatmap(res.get("heatmap", []))
                with st.expander("Evidence (cohort sizes & M3)", expanded=False):
                    ev = pd.DataFrame(res.get("evidence", []))
                    if not ev.empty:
                        st.dataframe(ev, use_container_width=True, height=180)
                    else:
                        st.caption("No evidence rows available.")

            elif module == "NRR/GRR (CSV)":
                st.markdown("**NRR / GRR by month**")
                plot_nrr(res.get("series", []))

            elif module == "Pricing Power (CSV)":
                st.markdown("**Price–Demand (log) with fit**")
                plot_pricing(res.get("scatter", {}))

            elif module == "Unit Economics (CSV)":
                kpi = {k: res.get(k) for k in ["aov","gm","cac","cm"] if k in res}
                st.metric(label="Contribution Margin (per order)", value=f"${res.get('value'):.2f}")
                cols4 = st.columns(3)
                with cols4[0]: st.metric("AOV", f"${kpi.get('aov',0):.2f}")
                with cols4[1]: st.metric("GM", f"{kpi.get('gm',0):.0%}")
                with cols4[2]: st.metric("CAC", f"${kpi.get('cac',0):.0f}")

            elif module == "PVM Bridge (CSV)":
                bars = res.get("bars", [])
                if PLOTLY_OK and bars:
                    fig = go.Figure()
                    x = [b["label"] for b in bars]
                    y = [b["value"] for b in bars]
                    fig.add_trace(go.Bar(x=x, y=y))
                    fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write(res.get("summary","No PVM data."))

            elif module == "PDF KPIs (PDF)":
                st.info("PDF KPIs module returns a narrative summary (no chart).")

            else:
                st.write("No renderer for this module yet.")


# --------------------------- MEMO (placeholder) ------------------------------
with tab_memo:
    st.subheader("Investor memo (demo placeholder)")
    st.caption("Approved cells would be assembled into memo sections here.")
    if REPORTLAB_OK:
        st.write("Use **Run → Export memo PDF (approved only)** to preview.")
    else:
        st.info("Install `reportlab` to enable PDF export.")

