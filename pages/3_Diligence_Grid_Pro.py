# pages/3_Diligence_Grid_Pro.py
# Transform AI — Diligence Grid (Pro)
# Adds: PMV Bridge module, Approvals, Evidence drawer, Run budget, Sidebar what-ifs
# Also: Auto data-quality checks & cleaning on CSV upload (dedupe, neg revenue (toggle), missing core fields)
# (Structure preserved; only PMV + data checks + sidebar toggle added)

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

# Plotly (primary)
try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots  # noqa
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# Altair (fallback)
try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False


# ---------------------------------------------------------------------------
# Page & CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Transform AI — Diligence Grid (Pro)", layout="wide")
st.markdown(
    """
<style>
.block-container {max-width: 1700px !important; padding-top: 0.5rem;}
/* Fix header clipping + spacing */
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


# ---------------------------------------------------------------------------
# Cleaning options (toggle default)
# ---------------------------------------------------------------------------
CLEANING_DROP_NONPOS_REVENUE = True  # default; can be overridden by sidebar toggle


# ---------------------------------------------------------------------------
# Helpers / State
# ---------------------------------------------------------------------------
def ensure_state():
    SS.setdefault("csv_files", {})             # {name: df}  (CLEANED)
    SS.setdefault("pdf_files", {})             # {name: bytes}
    SS.setdefault("schema", {})                # {csv_name: {canonical: source_col or None}}
    SS.setdefault("data_checks", {})           # {csv_name: report dict}

    SS.setdefault("rows", [])                  # [{id, alias, row_type ('table'|'pdf'), source}]
    SS.setdefault("columns", [])               # [{id, label, module}]
    SS.setdefault("matrix", {})                # {row_id: set([module,...])}

    SS.setdefault("results", {})               # {(row_id, col_id): {...}}
    SS.setdefault("cache_key", {})             # {(row_id, col_id): str}
    SS.setdefault("approved", set())           # set of "rid|cid"

    SS.setdefault("jobs", [])
    SS.setdefault("force_rerun", False)

    # What-if inputs (also surfaced in sidebar)
    SS.setdefault("whatif_gm", 0.62)
    SS.setdefault("whatif_cac", 42.0)

    # Cleaning toggle remembered across runs
    SS.setdefault("clean_drop_nonpos", CLEANING_DROP_NONPOS_REVENUE)

    # Budget (cents) and accounting
    SS.setdefault("run_budget_cents", 800)     # default budget for a run
    SS.setdefault("spent_cents", 0)

    # undo/redo snapshots
    SS.setdefault("undo", [])
    SS.setdefault("redo", [])

ensure_state()

def uid(p="row"): return f"{p}_{uuid.uuid4().hex[:8]}"
def now_ts(): return int(time.time())

def keypair(rid: str, cid: str) -> str:
    return f"{rid}|{cid}"

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


# ----------------------------- snapshots -------------------------------------
def snapshot_push():
    SS["undo"].append(json.dumps({
        "rows": SS["rows"],
        "columns": SS["columns"],
        "matrix": {k: list(v) for k, v in SS["matrix"].items()},
        "results": _pack_results(SS["results"]),
        "approved": list(SS["approved"]),
        "spent_cents": SS["spent_cents"],
    }, default=str))
    SS["redo"].clear()

def snapshot_apply(snap: str):
    data = json.loads(snap)
    SS["rows"]    = data.get("rows", [])
    SS["columns"] = data.get("columns", [])
    SS["matrix"]  = {k: set(v) for k, v in data.get("matrix", {}).items()}
    SS["results"] = _unpack_results(data.get("results", {}))
    SS["approved"] = set(data.get("approved", []))
    SS["spent_cents"] = int(data.get("spent_cents", 0))

def undo():
    if not SS["undo"]:
        return
    cur = json.dumps({
        "rows": SS["rows"],
        "columns": SS["columns"],
        "matrix": {k: list(v) for k, v in SS["matrix"].items()},
        "results": _pack_results(SS["results"]),
        "approved": list(SS["approved"]),
        "spent_cents": SS["spent_cents"],
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
        "approved": list(SS["approved"]),
        "spent_cents": SS["spent_cents"],
    }, default=str)
    snap = SS["redo"].pop()
    SS["undo"].append(cur)
    snapshot_apply(snap)
    st.toast("Redone")


# ---------------------------------------------------------------------------
# Schema helpers + cleaner
# ---------------------------------------------------------------------------
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
        "product":     pick("product","sku","item","category"),
    }

def _auto_clean_csv(
    df: pd.DataFrame,
    guess: Dict[str, Optional[str]],
    drop_nonpos_revenue: bool = CLEANING_DROP_NONPOS_REVENUE
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Auto-detect & fix: exact duplicates, negative revenue (optional), missing core fields, bad dates."""
    report: Dict[str, Any] = {}
    before = int(len(df))
    df2 = df.copy()

    # Helper to resolve guessed source columns
    def col(k: str) -> Optional[str]:
        v = guess.get(k)
        return v if (v and v in df2.columns) else None

    # Coerce order_date to datetime
    od = col("order_date")
    if od:
        prev_na = int(df2[od].isna().sum())
        df2[od] = pd.to_datetime(df2[od], errors="coerce")
        report["coerced_bad_dates_to_NaT"] = max(int(df2[od].isna().sum()) - prev_na, 0)

    # Work revenue
    rev = col("revenue"); amt = col("amount"); prc = col("price"); qty = col("quantity")
    if qty and qty in df2.columns:
        missing_q = int(df2[qty].isna().sum())
        df2[qty] = df2[qty].fillna(1)
        report["filled_missing_quantity_to_1"] = missing_q

    if rev and rev in df2.columns:
        wr = pd.to_numeric(df2[rev], errors="coerce")
        report["revenue_source"] = "revenue"
    elif amt and amt in df2.columns:
        wr = pd.to_numeric(df2[amt], errors="coerce")
        report["revenue_source"] = "amount"
    elif prc and qty and prc in df2.columns and qty in df2.columns:
        wr = pd.to_numeric(df2[prc], errors="coerce") * pd.to_numeric(df2[qty], errors="coerce")
        report["revenue_source"] = "price*quantity"
    else:
        wr = None
        report["revenue_source"] = "none"

    # Strong duplicate keys we actually have
    key_cols: List[str] = []
    if col("customer_id"): key_cols.append(col("customer_id"))
    if col("order_date"):  key_cols.append(col("order_date"))
    if amt and amt in df2.columns:
        key_cols.append(amt)
    else:
        if prc and prc in df2.columns: key_cols.append(prc)
        if qty and qty in df2.columns: key_cols.append(qty)
    if col("product"): key_cols.append(col("product"))

    dropped_dupes = 0
    if key_cols:
        dup_mask = df2.duplicated(subset=key_cols, keep="first")
        dropped_dupes = int(dup_mask.sum())
        if dropped_dupes > 0:
            df2 = df2.loc[~dup_mask].copy()
    report["dropped_exact_duplicates_on_keys"] = {"keys": key_cols, "rows": dropped_dupes}

    # Drop rows with missing core fields (customer_id or order_date)
    drop_core = 0
    cid = col("customer_id")
    if cid or od:
        mask_core = pd.Series(False, index=df2.index)
        if cid and cid in df2.columns:
            mask_core |= df2[cid].isna() | (df2[cid].astype(str).str.strip()=="")
        if od and od in df2.columns:
            mask_core |= df2[od].isna()
        drop_core = int(mask_core.sum())
        if drop_core > 0:
            df2 = df2.loc[~mask_core].copy()
    report["dropped_missing_core_fields"] = drop_core

    # Optionally drop rows with non-positive revenue (refunds/voids)
    dropped_nonpos_rev = 0
    if drop_nonpos_revenue and wr is not None:
        # recompute wr aligned to df2
        if rev and rev in df2.columns:
            wr2 = pd.to_numeric(df2[rev], errors="coerce")
        elif amt and amt in df2.columns:
            wr2 = pd.to_numeric(df2[amt], errors="coerce")
        elif prc and qty and prc in df2.columns and qty in df2.columns:
            wr2 = pd.to_numeric(df2[prc], errors="coerce") * pd.to_numeric(df2[qty], errors="coerce")
        else:
            wr2 = None
        if wr2 is not None:
            mask = (wr2 <= 0) | wr2.isna()
            dropped_nonpos_rev = int(mask.sum())
            if dropped_nonpos_rev > 0:
                df2 = df2.loc[~mask].copy()
    report["dropped_non_positive_revenue_rows"] = dropped_nonpos_rev if drop_nonpos_revenue else "skipped (toggle off)"

    report["rows_before"] = before
    report["rows_after"]  = int(len(df2))
    report["rows_removed_total"] = before - int(len(df2))

    return df2, report


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
    # product fallback
    if "product" not in df.columns:
        df["product"] = "all"
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
    "PMV Bridge (CSV)",   # <— ensure present in Matrix
]

QOE_TEMPLATE = [
    ("PDF KPIs",        "PDF KPIs (PDF)"),
    ("Unit Economics",  "Unit Economics (CSV)"),
    ("NRR/GRR",         "NRR/GRR (CSV)"),
    ("Pricing Power",   "Pricing Power (CSV)"),
    ("Cohort Retention","Cohort Retention (CSV)"),
    ("PMV Bridge",      "PMV Bridge (CSV)"),
]

def add_rows_from_csvs():
    snapshot_push()
    for name in SS["csv_files"].keys():
        if not any(r["source"] == name for r in SS["rows"]):
            rid = uid("row")
            SS["rows"].append({"id": rid, "alias": name.replace(".csv",""), "row_type":"table", "source": name})
            # default-map ALL CSV modules, including PMV
            SS["matrix"].setdefault(rid, set([m for m in MODULES if "(CSV)" in m]))

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
    SS["approved"] = set(k for k in SS["approved"] if not k.startswith(tuple(row_ids)))

def delete_cols(col_ids: List[str]):
    if not col_ids: return
    snapshot_push()
    SS["columns"] = [c for c in SS["columns"] if c["id"] not in col_ids]
    SS["results"] = {k:v for k,v in SS["results"].items() if k[1] not in col_ids}
    SS["approved"] = set(k for k in SS["approved"] if not k.endswith(tuple(col_ids)))


# ---------------------------------------------------------------------------
# Costs / Budget
# ---------------------------------------------------------------------------
MODULE_COST_CENTS = {
    "PDF KPIs (PDF)": 5,
    "Cohort Retention (CSV)": 10,
    "Pricing Power (CSV)": 6,
    "NRR/GRR (CSV)": 8,
    "Unit Economics (CSV)": 3,
    "PMV Bridge (CSV)": 8,
}

def module_cost(mod: str) -> int:
    return int(MODULE_COST_CENTS.get(mod, 5))


# ---------------------------------------------------------------------------
# Engines (calculations) + Evidence helpers
# ---------------------------------------------------------------------------
def _csv_evidence(df: pd.DataFrame, n: int = 6) -> Dict[str, Any]:
    head = df.head(n).copy()
    for c in head.columns:
        if pd.api.types.is_datetime64_any_dtype(head[c]):
            head[c] = head[c].astype(str)
    return {"type":"csv_rows","preview": head.to_dict(orient="records"), "rows": int(len(df))}

def _pdf_kpis(_raw: bytes) -> Dict[str, Any]:
    return dict(
        summary="Revenue ≈ $12.5M; EBITDA ≈ $1.3M; GM ≈ 62%; Churn ≈ 4%",
        evidence={"type":"pdf_quotes","pages":[3, 7],
                  "quotes":[
                      "p.3: 'FY Rev $12.5m, GM 62%'",
                      "p.7: 'EBITDA margin 10–12%'"
                  ]}
    )

# --- Cohort (real math) ---
def _cohort(df: pd.DataFrame, min_cohort_size: int = 10, horizon: int = 12) -> Dict[str, Any]:
    try:
        if not {"customer_id", "order_date"}.issubset(df.columns):
            raise ValueError("customer_id/order_date missing")
        d = df[["customer_id", "order_date"]].copy()
        d["order_date"] = pd.to_datetime(d["order_date"], errors="coerce")
        d = d.dropna(subset=["customer_id", "order_date"])
        d["month"] = d["order_date"].dt.to_period("M")
        first = d.groupby("customer_id")["month"].min().rename("cohort")
        d = d.join(first, on="customer_id")
        d["k"] = (d["month"].astype("int64") - d["cohort"].astype("int64"))
        d = d[(d["k"] >= 0) & (d["k"] < horizon)]
        present = (
            d.drop_duplicates(["cohort", "customer_id", "k"])
             .groupby(["cohort", "k"])
             .size()
             .unstack(fill_value=0)
        )
        cohort_sizes = d.groupby("cohort")["customer_id"].nunique().rename("size")
        keep_mask = cohort_sizes >= int(min_cohort_size)
        cohort_sizes = cohort_sizes[keep_mask]
        if cohort_sizes.empty:
            raise ValueError("No cohorts meet min_cohort_size")
        cohorts = cohort_sizes.index.sort_values()
        k_cols = list(range(horizon))
        present = present.reindex(index=cohorts, columns=k_cols, fill_value=0)
        retention = present.div(cohort_sizes, axis=0).fillna(0)
        weights = cohort_sizes.loc[cohorts].astype(float).values
        denom = weights.sum()
        if denom <= 0:
            raise ValueError("Invalid weights")
        curve = (retention.mul(weights, axis=0).sum(axis=0) / denom).tolist()
        m3_col = 3 if horizon > 3 else None
        ev = pd.DataFrame({
            "cohort": cohorts.astype(str).tolist(),
            "size": cohort_sizes.loc[cohorts].astype(int).tolist(),
            "M0": [1.0]*len(cohorts),
            "M3": retention.loc[cohorts, 3].round(4).tolist() if m3_col is not None else [np.nan]*len(cohorts)
        })
        m3_avg = curve[3] if len(curve) > 3 else None
        return dict(
            value=m3_avg,
            curve=[float(x) for x in curve],
            summary=f"Average retention at M3 ≈ {m3_avg:.0%} across {len(cohorts)} cohorts (min size {min_cohort_size})." if m3_avg is not None else
                    f"{len(cohorts)} cohorts retained (min size {min_cohort_size}).",
            heat=retention.values.tolist(),
            cohorts=[str(c) for c in cohorts.tolist()],
            sizes=[int(s) for s in cohort_sizes.loc[cohorts].tolist()],
            citations=[{"source":"csv","selector":"first-purchase cohorts"}],
            evidence={"type":"csv_rows",
                      "preview": ev.head(12).to_dict(orient="records"),
                      "rows": int(ev.shape[0])}
        )
    except Exception:
        curve = [1.0, 0.9, 0.75, 0.6, 0.5, 0.42] + [0.38]*(horizon-6)
        m3 = curve[3]
        return dict(
            value=m3,
            curve=curve[:horizon],
            summary=f"Retention stabilizes ~M3 at {m3:.0%} (fallback).",
            citations=[{"source":"csv","selector":"fallback"}]
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
        return dict(
            value=e, summary=f"Own-price elasticity ≈ {e} → {verdict}.",
            scatter=dict(x=x.tolist(), y=y.tolist(), fit=fit_y.tolist()),
            citations=[{"source":"csv","selector":"price,quantity"}],
            evidence=_csv_evidence(d)
        )
    except Exception:
        return dict(
            value=-1.21, summary="Own-price elasticity ≈ -1.21 (demo).",
            citations=[{"source":"csv","selector":"demo"}]
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
        return dict(
            value=latest["nrr"],
            summary=f"Latest ({latest['month']}): GRR {latest['grr']:.0%}, NRR {latest['nrr']:.0%}.",
            series=series,
            citations=[{"source":"csv","selector":"customer_id×month revenue"}],
            evidence=_csv_evidence(m)
        )
    except Exception:
        return dict(
            value=0.97,
            summary="Latest (demo): GRR 89%, NRR 97%.",
            series=[dict(month="demo", grr=0.89, nrr=0.97)],
            citations=[{"source":"csv","selector":"demo"}]
        )

def _unit_econ(df: pd.DataFrame, gm: float = 0.62, cac: float = 42.0) -> Dict[str, Any]:
    try:
        aov = float(df["amount"].mean()) if "amount" in df.columns else float(df.select_dtypes(np.number).sum(axis=1).mean())
        cm = round(gm*aov - cac, 2)
        return dict(
            value=cm, summary=f"AOV ${aov:.2f}, GM {gm:.0%}, CAC ${cac:.0f} → CM ${cm:.2f}.",
            aov=aov, gm=gm, cac=cac, cm=cm,
            citations=[{"source":"csv","selector":"amount"}],
            evidence=_csv_evidence(df)
        )
    except Exception:
        return dict(
            value=32.0,
            summary="AOV $120.00, GM 60%, CAC $40 → CM $32.00 (demo).",
            aov=120.0, gm=0.6, cac=40.0, cm=32.0,
            citations=[{"source":"csv","selector":"demo"}]
        )

def _pmv_bridge(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Price-Volume-Mix bridge between two periods (earliest vs latest month).
    Requires: month (or order_date), price, quantity, product (fallback 'all').
    """
    try:
        d = df.copy()
        if "month" not in d.columns and "order_date" in d.columns:
            d["order_date"] = pd.to_datetime(d["order_date"], errors="coerce")
            d["month"] = d["order_date"].dt.to_period("M").astype(str)
        d = d.dropna(subset=["price","quantity"])
        # Aggregate by product & month
        g = d.groupby(["product","month"]).agg(
            qty=("quantity","sum"),
            rev=("revenue","sum") if "revenue" in d.columns else ("quantity","sum")
        ).reset_index()
        # Derive avg price
        g["price"] = np.where(g["qty"]>0, g["rev"]/g["qty"], np.nan)

        months = sorted(g["month"].unique())
        if len(months) < 2:
            months = ["P0","P1"]
        a, b = months[0], months[-1]
        A = g[g["month"]==a].set_index("product")
        B = g[g["month"]==b].set_index("product")
        products = sorted(set(A.index) | set(B.index))

        # Fill missing
        for p in products:
            if p not in A.index: A.loc[p] = dict(qty=0, rev=0, price=np.nan, month=a)
            if p not in B.index: B.loc[p] = dict(qty=0, rev=0, price=np.nan, month=b)
        A = A.fillna(0); B = B.fillna(0)

        # Base revenue at A mix/price/qty
        base_rev = float((A["price"] * A["qty"]).sum())
        price_effect = float(((B["price"] - A["price"]) * A["qty"]).sum())
        volume_effect = float((A["price"] * (B["qty"] - A["qty"])).sum())
        # Mix effect as residual
        actual_delta = float((B["price"]*B["qty"]).sum() - base_rev)
        mix_effect = float(actual_delta - price_effect - volume_effect)

        bridge = [
            {"component":"Base ({} total)".format(a), "value": round(base_rev,2)},
            {"component":"Price", "value": round(price_effect,2)},
            {"component":"Volume", "value": round(volume_effect,2)},
            {"component":"Mix", "value": round(mix_effect,2)},
            {"component":"Total Δ", "value": round(actual_delta,2)},
        ]
        summary = f"{a}→{b} ΔRev {actual_delta:+.0f} = Price {price_effect:+.0f} + Volume {volume_effect:+.0f} + Mix {mix_effect:+.0f}."
        return dict(
            value=actual_delta,
            summary=summary,
            bridge=bridge,
            periods={"from": a, "to": b},
            citations=[{"source":"csv","selector":"product×month price,quantity"}],
            evidence=_csv_evidence(d)
        )
    except Exception:
        bridge = [
            {"component":"Base (P0 total)","value":1000.0},
            {"component":"Price","value":120.0},
            {"component":"Volume","value":-80.0},
            {"component":"Mix","value":30.0},
            {"component":"Total Δ","value":70.0},
        ]
        return dict(
            value=70.0,
            summary="P0→P1 ΔRev +70 = Price +120 + Volume -80 + Mix +30 (demo).",
            bridge=bridge,
            periods={"from":"P0","to":"P1"},
            citations=[{"source":"csv","selector":"demo"}]
        )


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
        return {"status":"done","value":None,"summary":k["summary"],"last_run": now_ts(), **{k: v for k,v in k.items() if k!='summary'}}

    if mod == "Cohort Retention (CSV)":
        df = materialize_df(row["source"])
        k = _cohort(df)
        out = {"status":"done","value":k["value"],"summary":k["summary"],"last_run": now_ts()}
        if "curve" in k: out["curve"] = k["curve"]
        if "citations" in k: out["citations"]=k["citations"]
        if "evidence" in k: out["evidence"]=k["evidence"]
        if "heat" in k: out["heat"]=k["heat"]
        if "cohorts" in k: out["cohorts"]=k["cohorts"]
        if "sizes" in k: out["sizes"]=k["sizes"]
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

    if mod == "PMV Bridge (CSV)":
        df = materialize_df(row["source"])
        k = _pmv_bridge(df)
        return {"status":"done","value":k["value"],"summary":k["summary"],"last_run": now_ts(), **k}

    return {"status":"error","value":None,"summary":f"Unknown module: {mod}","last_run": now_ts()}

def enqueue_pairs(pairs: List[Tuple[str,str]], respect_cache=True):
    """Enqueue with budget enforcement & cache checks."""
    by_r = {r["id"]: r for r in SS["rows"]}
    by_c = {c["id"]: c for c in SS["columns"]}

    # Pre-calc required cost
    required = 0
    calc_list = []
    for rid, cid in pairs:
        row = by_r.get(rid); col = by_c.get(cid)
        if not row or not col: continue
        key = (rid, cid)
        ck = cache_key_for(row, col)
        SS["cache_key"][key] = ck
        hit = SS["results"].get(key)
        if respect_cache and (hit and hit.get("status") in {"done","cached"}) and SS["cache_key"].get(key)==ck and not SS["force_rerun"]:
            calc_list.append((rid,cid,"cached",0))
        else:
            c = module_cost(col["module"])
            required += c
            calc_list.append((rid,cid,"queue",c))

    if SS["spent_cents"] + required > SS["run_budget_cents"]:
        st.warning(f"Budget exceeded: need {required}¢, have {SS['run_budget_cents']-SS['spent_cents']}¢ remaining. "
                   f"Lower selection or raise the budget.")
        # Still enqueue cached ones for UX
        for rid,cid,typ,c in calc_list:
            if typ=="cached":
                SS["results"][(rid,cid)] = {**SS["results"][(rid,cid)], "status":"cached"}
                SS["jobs"].append({"rid":rid,"cid":cid,"status":"cached","cost_cents":0,"started":now_ts(),"ended":now_ts(),"note":"cache"})
        return

    # Enqueue
    for rid,cid,typ,c in calc_list:
        key = (rid,cid)
        if typ=="cached":
            SS["results"][key] = {**SS["results"][key], "status":"cached"}
            SS["jobs"].append({"rid":rid,"cid":cid,"status":"cached","cost_cents":0,"started":now_ts(),"ended":now_ts(),"note":"cache"})
        else:
            SS["results"][key] = {"status":"queued","value":None,"summary":None}
            SS["jobs"].append({"rid":rid,"cid":cid,"status":"queued","cost_cents":c,"started":None,"ended":None,"note":""})

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
            SS["spent_cents"] += int(j.get("cost_cents") or 0)
        except Exception as e:
            SS["results"][(rid,cid)]={"status":"error","value":None,"summary":str(e),"last_run": now_ts()}
            j["status"]="error"; j["note"]=str(e); j["ended"]=now_ts()

def retry_cell(rid: str, cid: str):
    SS["jobs"].insert(0, {"rid":rid,"cid":cid,"status":"retry","cost_cents":module_cost(
        next((c["module"] for c in SS["columns"] if c["id"]==cid), "Unknown")
    ),"started":None,"ended":None,"note":"manual retry"})


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
def export_results_csv(only_approved: bool = True) -> bytes:
    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}
    out = []
    for (rid,cid), res in SS["results"].items():
        if only_approved and keypair(rid,cid) not in SS["approved"]:
            continue
        r = rows_by_id.get(rid); c = cols_by_id.get(cid)
        if not r or not c: continue
        out.append(dict(
            row=r["alias"], row_type=r["row_type"], source=r["source"],
            column=c["label"], module=c["module"],
            status=res.get("status"), value=res.get("value"), summary=res.get("summary"), last_run=res.get("last_run")
        ))
    return pd.DataFrame(out).to_csv(index=False).encode("utf-8")

def export_results_pdf(only_approved: bool = True) -> bytes:
    if not REPORTLAB_OK: raise RuntimeError("reportlab not installed")
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    w,h = LETTER
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, h-72, "TransformAI — Investor Memo (Demo)")
    y = h-100; c.setFont("Helvetica", 10)
    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}
    items = list(SS["results"].items())
    for (rid,cid),res in items:
        if only_approved and keypair(rid,cid) not in SS["approved"]:
            continue
        r = rows_by_id.get(rid); cdef = cols_by_id.get(cid)
        if not r or not cdef: continue
        line = f"{r['alias']} → {cdef['label']}: {res.get('summary')}"
        for chunk in [line[i:i+95] for i in range(0,len(line),95)]:
            if y<72: c.showPage(); y=h-72; c.setFont("Helvetica",10)
            c.drawString(72,y,chunk); y-=14
    c.showPage(); c.save(); buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
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

def plot_retention_heatmap_from_curve(curve: List[float]):
    """Heatmap placeholder derived from curve (kept for compatibility)."""
    curve = [float(x) for x in (curve or [])]
    if not curve:
        st.info("No cohort heatmap available.")
        return
    z = np.tile(curve, (5,1))
    if PLOTLY_OK:
        fig = go.Figure(data=go.Heatmap(z=z, colorscale="Blues"))
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
    elif ALTAIR_OK:
        df = pd.DataFrame(z)
        df = df.reset_index().melt("index", var_name="month", value_name="value")
        df = df.rename(columns={"index": "cohort"})
        ch = (
            alt.Chart(df)
            .mark_rect()
            .encode(x="month:O", y="cohort:O", color="value:Q")
            .properties(height=320)
        )
        st.altair_chart(ch, use_container_width=True)
    else:
        st.write(pd.DataFrame(z))

def plot_retention_heatmap_matrix(matrix: List[List[float]]):
    """True cohort heatmap if matrix is available (rows=cohorts, cols=k)."""
    if not matrix:
        st.info("No cohort heatmap available.")
        return
    z = np.array(matrix, dtype=float)
    if PLOTLY_OK:
        fig = go.Figure(data=go.Heatmap(z=z, colorscale="Blues"))
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
    elif ALTAIR_OK:
        df = pd.DataFrame(z)
        df = df.reset_index().melt("index", var_name="k", value_name="value")
        df = df.rename(columns={"index":"cohort_idx"})
        ch = (
            alt.Chart(df)
            .mark_rect()
            .encode(x="k:O", y="cohort_idx:O", color="value:Q")
            .properties(height=320)
        )
        st.altair_chart(ch, use_container_width=True)
    else:
        st.write(pd.DataFrame(z))

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

def plot_pmv(bridge: List[Dict[str, Any]]):
    if not bridge:
        st.info("No PMV bridge available.")
        return
    df = pd.DataFrame(bridge)
    if PLOTLY_OK:
        base = df.iloc[0]["value"]
        labels = [df.iloc[0]["component"]] + [x for x in df["component"].tolist()[1:]]
        vals = [base] + [float(v) for v in df["value"].tolist()[1:]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=vals))
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
    elif ALTAIR_OK:
        ch = (
            alt.Chart(df)
            .mark_bar()
            .encode(x="component:N", y="value:Q", tooltip=["component","value"])
            .properties(height=320)
        )
        st.altair_chart(ch, use_container_width=True)
    else:
        st.bar_chart(df.set_index("component")["value"])


# ---------------------------------------------------------------------------
# Sidebar — What-ifs, Cleaning & Budget
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("What-ifs, Cleaning & Budget")
    # Toggle for cleaning behavior (visible control)
    SS["clean_drop_nonpos"] = st.toggle(
        "Drop non-positive revenue rows",
        value=SS.get("clean_drop_nonpos", CLEANING_DROP_NONPOS_REVENUE),
        help="When on, auto-cleaning removes rows where revenue ≤ 0 (or NaN)."
    )
    SS["whatif_gm"] = st.number_input("Gross Margin (0–1)", min_value=0.0, max_value=1.0, step=0.01, value=float(SS.get("whatif_gm",0.62)))
    SS["whatif_cac"] = st.number_input("CAC ($)", min_value=0.0, step=1.0, value=float(SS.get("whatif_cac",42.0)))
    SS["run_budget_cents"] = st.number_input("Run budget (¢)", min_value=0, step=50, value=int(SS.get("run_budget_cents",800)))
    st.caption(f"Spent this session: **{SS.get('spent_cents',0)}¢**")

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
            loaded = 0
            for f in csvs:
                try:
                    raw_df = pd.read_csv(f)
                except Exception:
                    raw_df = pd.read_csv(io.BytesIO(f.getvalue()))
                guess = _auto_guess_schema(raw_df)
                cleaned_df, report = _auto_clean_csv(
                    raw_df,
                    guess,
                    drop_nonpos_revenue=SS.get("clean_drop_nonpos", CLEANING_DROP_NONPOS_REVENUE),
                )
                SS["csv_files"][f.name] = cleaned_df
                SS["schema"][f.name] = guess
                SS["data_checks"][f.name] = report
                loaded += 1
            st.success(f"Loaded {loaded} CSV file(s) (auto-clean applied).")

    with c2:
        pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if pdfs:
            for f in pdfs:
                SS["pdf_files"][f.name] = f.getvalue()
            st.success(f"Loaded {len(pdfs)} PDF file(s).")

    # Show per-file cleaning reports
    if SS["data_checks"]:
        with st.expander("Data Quality Reports (per CSV)", expanded=False):
            for name, rep in SS["data_checks"].items():
                st.markdown(f"**{name}**")
                st.json(rep)
                st.divider()

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
            pick("Product", "product")
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
        default_mod = SS.get("new_col_mod","NRR/GRR (CSV)")
        new_mod = st.selectbox("Module", MODULES, index=MODULES.index(default_mod) if default_mod in MODULES else 2)
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
                "PMV Bridge (CSV)": "PMV Bridge (CSV)" in sel,   # <— visible checkbox
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
                "PMV Bridge (CSV)": st.column_config.CheckboxColumn(),
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
                # PDF guard
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

    # One-click QoE
    with st.expander("One-click QoE", expanded=True):
        st.caption("Adds QoE columns (if missing), selects mapped pairs from Matrix, runs all within budget.")
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
            st.success(f"Attempted to run {len(pairs)} cell(s). Check Jobs below for cache/budget statuses.")

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
        st.dataframe(pd.DataFrame(SS["jobs"]), use_container_width=True, height=200)
    if SS["results"]:
        pass  # retained behavior
    if SS["results"]:
        c1, c2 = st.columns(2)
        with c1:
            if st.download_button("Export APPROVED results CSV", data=export_results_csv(only_approved=True), file_name="transformai_results.csv"):
                pass
        with c2:
            if REPORTLAB_OK and st.download_button("Export APPROVED memo PDF (demo)", data=export_results_pdf(only_approved=True), file_name="TransformAI_Memo_demo.pdf"):
                pass


# --------------------------- SHEET (Agentic Spreadsheet) ---------------------
with tab_sheet:
    st.subheader("Agentic Spreadsheet (status by cell)")
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
            if keypair(r["id"],c["id"]) in SS["approved"]:
                mark = "✅ " + (mark or "")
            row_vals.append(mark)
        table.append(row_vals)

    df_sheet = pd.DataFrame(table, columns=header)
    st.dataframe(df_sheet, use_container_width=True, height=min(440, 140 + 28*len(df_sheet)))


# --------------------------- REVIEW (focused viz by cell) --------------------
with tab_review:
    st.subheader("Review a single cell — charts & evidence for your selection")

    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}

    row_opt = [(r["id"], r["alias"]) for r in SS["rows"]]
    col_opt = [(c["id"], f"{c['label']}  ·  {c['module']}") for c in SS["columns"]]

    csel1, csel2, csel3, csel4 = st.columns([2,2,1,1])
    with csel1:
        rid = st.selectbox("Row", row_opt, format_func=lambda t: t[1]) if row_opt else None
    with csel2:
        cid = st.selectbox("Column", col_opt, format_func=lambda t: t[1]) if col_opt else None
    with csel3:
        if rid and cid and st.button("Retry"):
            retry_cell(rid[0], cid[0]); run_queued_jobs()
    with csel4:
        if rid and cid:
            kp = keypair(rid[0], cid[0])
            approved = kp in SS["approved"]
            if st.button("Approve" if not approved else "Unapprove"):
                if approved:
                    SS["approved"].discard(kp)
                else:
                    SS["approved"].add(kp)

    if not (rid and cid):
        st.info("Choose a Row and a Column above.")
    else:
        rid, cid = rid[0], cid[0]
        res = SS["results"].get((rid, cid))
        r = rows_by_id.get(rid); c = cols_by_id.get(cid)

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

            # Render only charts for this module (no blending)
            if module == "Cohort Retention (CSV)" or "curve" in res:
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**Retention curve**")
                    plot_retention(res.get("curve", []))
                with colB:
                    st.markdown("**Cohort heatmap**")
                    # Prefer true matrix if available; fallback to curve-tiling
                    if "heat" in res and res.get("heat"):
                        plot_retention_heatmap_matrix(res.get("heat"))
                    else:
                        plot_retention_heatmap_from_curve(res.get("curve", []))

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

            elif module == "PDF KPIs (PDF)":
                st.info("PDF KPIs module returns a narrative summary (no chart).")

            elif module == "PMV Bridge (CSV)":
                st.markdown("**Price-Volume-Mix Bridge**")
                plot_pmv(res.get("bridge", []))
                periods = res.get("periods", {})
                if periods:
                    st.caption(f"Periods: {periods.get('from','?')} → {periods.get('to','?')}")

            # Evidence & Sources — visible and expanded by default
            with st.expander("Evidence & Sources", expanded=True):
                cits = res.get("citations", [])
                if cits:
                    st.write("**Citations**:", cits)
                ev = res.get("evidence")
                if ev:
                    if ev.get("type") == "csv_rows":
                        st.write("Row preview used in calculation:")
                        st.dataframe(pd.DataFrame(ev.get("preview", [])), use_container_width=True, height=180)
                        st.caption(f"Total rows: {ev.get('rows')}")
                    elif ev.get("type") == "pdf_quotes":
                        st.write("Quoted passages:")
                        for q in ev.get("quotes", []):
                            st.write(f"- {q}")

# --------------------------- MEMO (placeholder) ------------------------------
with tab_memo:
    st.subheader("Investor memo (demo placeholder)")
    st.caption("Only **approved** cells are included in exports.")
    if REPORTLAB_OK:
        st.write("Use **Run → Export APPROVED memo PDF (demo)** to preview.")
    else:
        st.info("Install `reportlab` to enable PDF export.")

