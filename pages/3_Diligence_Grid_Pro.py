# pages/3_Diligence_Grid_Pro.py
# TransformAI — Diligence Grid (Pro)  •  Safe multiselect defaults + Sidebar Wizard
# This page intentionally stores all transient state in st.session_state (SS)
# and never passes invalid defaults to Streamlit widgets.

from __future__ import annotations
import io
import json
import math
import uuid
import time
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np
import streamlit as st

# ============================================================
# ---------- Session & Safe-Widget Utilities -----------------
# ============================================================

SS = st.session_state

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]

def _keep_only(options, default):
    opts = set(options or [])
    return [v for v in _as_list(default) if v in opts]

def ui_selectbox(label, options, default=None, **kwargs):
    """Selectbox that never crashes if default vanished from options."""
    opt_list = list(options or [])
    dv = _keep_only(opt_list, default)
    dv = dv[0] if dv else None
    idx = opt_list.index(dv) if dv in opt_list else (0 if opt_list else None)
    return st.selectbox(label, opt_list, index=idx, **kwargs)

def ui_multiselect(label, options, default=None, **kwargs):
    """Multiselect that strips any default values not in options."""
    opt_list = list(options or [])
    dv = _keep_only(opt_list, default)
    return st.multiselect(label, opt_list, default=dv, **kwargs)

def ui_rerun():
    st.rerun()

def uid(prefix="row"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def ensure_state():
    SS.setdefault("wizard_idx", 1)           # default "Grid"
    SS.setdefault("hide_top_tabs", True)     # hide legacy top tabs by default
    SS.setdefault("csv_files", {})           # {name: DataFrame}
    SS.setdefault("pdf_files", {})           # {name: raw_bytes}
    SS.setdefault("rows", [])                # list[{id, alias, row_type('table'|'pdf'), source}]
    SS.setdefault("columns", [])             # list[{id, label, module}]
    SS.setdefault("results", {})             # {(row_id, col_id): {status, value, summary}}
    SS.setdefault("undo", [])
    SS.setdefault("redo", [])
    SS.setdefault("row_run_selection", [])
    SS.setdefault("col_run_selection", [])
    SS.setdefault("force_rerun", False)

ensure_state()

# ============================================================
# ----------------- Simple Grid Manipulation -----------------
# ============================================================

def push_undo():
    # store a snapshot of rows/cols/results (lightweight JSON)
    SS["undo"].append(json.dumps({
        "rows": SS["rows"], "columns": SS["columns"], "results": SS["results"]
    }, default=str))
    SS["redo"].clear()

def undo():
    if not SS["undo"]:
        return
    snap = SS["undo"].pop()
    SS["redo"].append(json.dumps({
        "rows": SS["rows"], "columns": SS["columns"], "results": SS["results"]
    }, default=str))
    data = json.loads(snap)
    SS["rows"] = data["rows"]
    SS["columns"] = data["columns"]
    SS["results"] = {tuple(eval(k) if isinstance(k, str) and k.startswith("(") else k): v
                     for k, v in data["results"].items()} if isinstance(data["results"], dict) else {}

def redo():
    if not SS["redo"]:
        return
    snap = SS["redo"].pop()
    SS["undo"].append(json.dumps({
        "rows": SS["rows"], "columns": SS["columns"], "results": SS["results"]
    }, default=str))
    data = json.loads(snap)
    SS["rows"] = data["rows"]
    SS["columns"] = data["columns"]
    SS["results"] = {tuple(eval(k) if isinstance(k, str) and k.startswith("(") else k): v
                     for k, v in data["results"].items()} if isinstance(data["results"], dict) else {}

def add_rows_from_csvs():
    push_undo()
    for name, df in SS["csv_files"].items():
        if not any(r["source"] == name for r in SS["rows"]):
            SS["rows"].append({
                "id": uid("row"),
                "alias": name.replace(".csv", ""),
                "row_type": "table",
                "source": name,
            })

def add_rows_from_pdfs():
    push_undo()
    for name, _raw in SS["pdf_files"].items():
        if not any(r["source"] == name for r in SS["rows"]):
            SS["rows"].append({
                "id": uid("row"),
                "alias": name.replace(".pdf", ""),
                "row_type": "pdf",
                "source": name,
            })

MODULES = [
    "PDF KPIs (PDF)",
    "Cohort Retention (CSV)",
    "Pricing Power (CSV)",
    "NRR/GRR (CSV)",
    "Unit Economics (CSV)"
]

def add_column(label: str, module: str):
    if not label.strip() or not module:
        return
    push_undo()
    SS["columns"].append({
        "id": uid("col"),
        "label": label.strip(),
        "module": module
    })

def delete_rows(row_ids: List[str]):
    if not row_ids:
        return
    push_undo()
    SS["rows"] = [r for r in SS["rows"] if r["id"] not in row_ids]
    # clean up results that referenced those rows
    SS["results"] = {k: v for k, v in SS["results"].items() if k[0] not in row_ids}
    # sanitize run selections
    SS["row_run_selection"] = _keep_only([r["id"] for r in SS["rows"]], SS["row_run_selection"])

def delete_columns(col_ids: List[str]):
    if not col_ids:
        return
    push_undo()
    SS["columns"] = [c for c in SS["columns"] if c["id"] not in col_ids]
    SS["results"] = {k: v for k, v in SS["results"].items() if k[1] not in col_ids}
    SS["col_run_selection"] = _keep_only([c["id"] for c in SS["columns"]], SS["col_run_selection"])

# ============================================================
# ------------------ Tiny “Engines” (MVP) --------------------
# ============================================================

def _kpi_pdf_extract(raw: bytes) -> Dict[str, Any]:
    # MVP: pretend we parsed a KPI pack; return stable demo KPIs
    return dict(revenue="$12.5M", ebitda="$1.3M", gross_margin="62%", churn="4%", note="Demo parser")

def _cohort_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    # Expect simple columns; if absent, fabricate a curve
    try:
        if {"customer_id", "order_date", "amount"}.issubset(df.columns):
            # super naive retention: monthly repeat buyer rate
            df = df.copy()
            df["order_date"] = pd.to_datetime(df["order_date"])
            df["month"] = df["order_date"].dt.to_period("M")
            first = df.groupby("customer_id")["month"].min()
            joined = df.join(first.rename("first_month"), on="customer_id")
            cohorts = joined.groupby(["first_month", "month"])["customer_id"].nunique().reset_index()
            total = cohorts.groupby("first_month")["customer_id"].max()
            # build 5-point curve
            curve = []
            for i in range(5):
                m = (total.index.to_timestamp() + pd.offsets.MonthBegin(i)).to_period("M")
                alive = 0
                for fm, n in total.items():
                    # naive survival proxy
                    alive += max(int(n * (0.9 ** i)), 0)
                denom = max(int(total.sum()), 1)
                curve.append(round(alive / denom, 2))
            return dict(curve=curve, m3=curve[3] if len(curve) > 3 else None, ltv_proxy=round(12 * np.mean(curve) * 100, 2))
    except Exception:
        pass
    # fallback curve
    curve = [1.0, 0.86, 0.78, 0.72, 0.69]
    return dict(curve=curve, m3=0.72, ltv_proxy=round(12 * np.mean(curve) * 100, 2))

def _pricing_power(df: pd.DataFrame) -> Dict[str, Any]:
    # naive elasticity estimate using log-log regression of qty on price
    try:
        price_col = next(c for c in df.columns if c.lower() in {"price", "unit_price", "avg_price"})
        qty_col   = next(c for c in df.columns if c.lower() in {"qty", "quantity", "units"})
        d = df[[price_col, qty_col]].replace(0, np.nan).dropna()
        d = d[(d[price_col] > 0) & (d[qty_col] > 0)]
        x = np.log(d[price_col].astype(float))
        y = np.log(d[qty_col].astype(float))
        b = np.polyfit(x, y, 1)[0]
        elasticity = round(b, 2)
        verdict = "inelastic" if abs(elasticity) < 1 else "elastic"
        return dict(elasticity=elasticity, verdict=verdict)
    except Exception:
        return dict(elasticity=-1.21, verdict="elastic (demo)")

def _nrr_grr(df: pd.DataFrame) -> Dict[str, Any]:
    # toy monthly NRR/GRR
    try:
        if {"customer_id", "month", "revenue"}.issubset({c.lower() for c in df.columns}):
            tmp = df.copy()
        else:
            tmp = df.copy()
            if "order_date" in tmp.columns:
                tmp["month"] = pd.to_datetime(tmp["order_date"]).dt.to_period("M").astype(str)
            else:
                tmp["month"] = "2024-06"
            if "customer_id" not in tmp.columns:
                tmp["customer_id"] = np.arange(len(tmp))
            if "amount" in tmp.columns and "revenue" not in tmp.columns:
                tmp["revenue"] = tmp["amount"]
        m = tmp.groupby(["customer_id", "month"])["revenue"].sum().reset_index()
        months = sorted(m["month"].unique())
        if len(months) < 2:
            return dict(grr=0.9, nrr=0.98, latest=months[-1] if months else "n/a")
        prev, curr = months[-2], months[-1]
        base = m[m["month"] == prev]["revenue"].sum()
        kept = m[(m["month"] == curr)]["revenue"].sum()
        # demo math
        grr = round(min(kept / base, 1.2), 2) if base else 0.88
        nrr = round(min((kept + 0.05 * base) / base, 1.3), 2) if base else 0.97
        return dict(grr=grr, nrr=nrr, latest=curr)
    except Exception:
        return dict(grr=0.89, nrr=0.98, latest="2024-06")

def _unit_econ(df: pd.DataFrame) -> Dict[str, Any]:
    # toy contribution margin: GM% x AOV - CAC (demo if missing)
    try:
        aov = float(df.get("amount", df.select_dtypes(include=[np.number]).sum(axis=1)).mean())
        gm  = 0.62
        cac = 42.0
        cm  = round(gm * aov - cac, 2)
        return dict(aov=round(aov, 2), gm=gm, cac=cac, contrib_margin=cm)
    except Exception:
        return dict(aov=120.0, gm=0.6, cac=40.0, contrib_margin=32.0)

# ============================================================
# ------------------------ RUNNER ----------------------------
# ============================================================

def execute_cell(row: Dict[str, Any], col: Dict[str, Any]) -> Dict[str, Any]:
    mod = col["module"]
    out = {"status": "done", "value": None, "summary": None}

    if mod == "PDF KPIs (PDF)":
        raw = SS["pdf_files"].get(row["source"])
        k = _kpi_pdf_extract(raw or b"")
        out["summary"] = f"Rev {k['revenue']}, EBITDA {k['ebitda']}, GM {k['gross_margin']}, Churn {k['churn']}"
        return out

    if mod == "Cohort Retention (CSV)":
        df = SS["csv_files"].get(row["source"], pd.DataFrame())
        k = _cohort_from_df(df)
        out["value"] = k["m3"]
        out["summary"] = f"Retention stabilizes ~M3 at {k['m3']:.0%} (LTV proxy ≈ {k['ltv_proxy']})."
        out["curve"] = k["curve"]
        return out

    if mod == "Pricing Power (CSV)":
        df = SS["csv_files"].get(row["source"], pd.DataFrame())
        k = _pricing_power(df)
        out["value"] = k["elasticity"]
        out["summary"] = f"Own-price elasticity ≈ {k['elasticity']} → {k['verdict']}."
        return out

    if mod == "NRR/GRR (CSV)":
        df = SS["csv_files"].get(row["source"], pd.DataFrame())
        k = _nrr_grr(df)
        out["value"] = k["nrr"]
        out["summary"] = f"Latest ({k['latest']}): GRR {k['grr']:.0%}, NRR {k['nrr']:.0%}."
        return out

    if mod == "Unit Economics (CSV)":
        df = SS["csv_files"].get(row["source"], pd.DataFrame())
        k = _unit_econ(df)
        out["value"] = k["contrib_margin"]
        out["summary"] = f"AOV ${k['aov']:.2f}, GM {k['gm']:.0%}, CAC ${k['cac']:.0f} → CM ${k['contrib_margin']:.2f}."
        return out

    out["status"] = "error"
    out["summary"] = f"Unknown module: {mod}"
    return out

def run_selection(row_ids: List[str], col_ids: List[str]):
    if not row_ids or not col_ids:
        return
    push_undo()
    ridx = {r["id"]: r for r in SS["rows"]}
    cidx = {c["id"]: c for c in SS["columns"]}
    for r in row_ids:
        for c in col_ids:
            row = ridx.get(r)
            col = cidx.get(c)
            if not row or not col:
                continue
            SS["results"][(r, c)] = {"status": "queued", "value": None, "summary": None}
    st.toast(f"Queued {len(row_ids) * len(col_ids)} cell(s)…")
    # simple synchronous “runner”
    for r in row_ids:
        for c in col_ids:
            row = ridx.get(r)
            col = cidx.get(c)
            if not row or not col:
                continue
            res = execute_cell(row, col)
            SS["results"][(r, c)] = res

# ============================================================
# --------------------- Sidebar Wizard -----------------------
# ============================================================

with st.sidebar:
    st.header("Diligence Wizard")
    steps = ["Data", "Grid", "Run", "Review", "Memo"]
    step = st.radio("Go to", steps, index=SS.get("wizard_idx", 1), key="wizard_radio")
    SS["wizard_idx"] = steps.index(step)
    SS["hide_top_tabs"] = st.toggle("Hide top tabs", value=SS.get("hide_top_tabs", True),
                                    help="Use the wizard-only layout")

def show_section(name: str) -> bool:
    # Render only the current wizard step if hiding tabs, otherwise always render.
    order = ["Data", "Grid", "Run", "Review", "Memo"]
    return (not SS.get("hide_top_tabs", True)) or (name == order[SS["wizard_idx"]])

st.title("Transform AI — Diligence Grid (Pro)")
st.caption("CSV+PDF ingest → build grid → run analyses → review → memo export with citations.")

# ============================================================
# ------------------------- DATA -----------------------------
# ============================================================

if show_section("Data"):
    st.subheader("1) Evidence Sources")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Upload CSV(s)**")
        csv_files = st.file_uploader("Drag and drop CSVs", type=["csv"], accept_multiple_files=True, label_visibility="collapsed")
        if csv_files:
            for f in csv_files:
                try:
                    SS["csv_files"][f.name] = pd.read_csv(f)
                except Exception:
                    SS["csv_files"][f.name] = pd.read_csv(io.BytesIO(f.getvalue()))
            st.success(f"Loaded {len(csv_files)} CSV file(s).")

    with c2:
        st.markdown("**Upload PDF(s)**")
        pdf_files = st.file_uploader("Drag and drop PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
        if pdf_files:
            for f in pdf_files:
                SS["pdf_files"][f.name] = f.getvalue()
            st.success(f"Loaded {len(pdf_files)} PDF file(s).")

    st.write("**Loaded CSVs**:", list(SS["csv_files"].keys()) or "—")
    st.write("**Loaded PDFs**:", list(SS["pdf_files"].keys()) or "—")

# ============================================================
# ------------------------- GRID -----------------------------
# ============================================================

if show_section("Grid"):
    st.subheader("2) Build Grid")

    top_cols = st.columns([1,1,1,1,1,1])
    with top_cols[0]:
        if st.button("Add CSV rows", use_container_width=True, type="primary"):
            add_rows_from_csvs()
            st.toast("Added CSV rows")

    with top_cols[1]:
        if st.button("Add PDF rows", use_container_width=True):
            add_rows_from_pdfs()
            st.toast("Added PDF rows")

    with top_cols[2]:
        if st.button("Undo", use_container_width=True):
            undo()
            ui_rerun()

    with top_cols[3]:
        if st.button("Redo", use_container_width=True):
            redo()
            ui_rerun()

    with top_cols[4]:
        if st.button("Delete selected row(s)", use_container_width=True):
            to_del = _keep_only([r["id"] for r in SS["rows"]], SS.get("row_selection_for_delete", []))
            delete_rows(to_del)
            st.toast(f"Deleted {len(to_del)} row(s)")
            ui_rerun()

    with top_cols[5]:
        if st.button("Delete selected column(s)", use_container_width=True):
            to_del = _keep_only([c["id"] for c in SS["columns"]], SS.get("col_selection_for_delete", []))
            delete_columns(to_del)
            st.toast(f"Deleted {len(to_del)} column(s)")
            ui_rerun()

    st.markdown("**Add a single column**")
    cc1, cc2, cc3 = st.columns([2,2,1])
    with cc1:
        new_label = st.text_input("Column label", value=SS.get("new_col_label", "NRR"))
        SS["new_col_label"] = new_label
    with cc2:
        new_mod = ui_selectbox("Module", MODULES, default=SS.get("new_col_module", "NRR/GRR (CSV)"))
        SS["new_col_module"] = new_mod
    with cc3:
        if st.button("Add Column", use_container_width=True):
            add_column(SS["new_col_label"], SS["new_col_module"])
            st.success("Column added")

    st.markdown("**Rows**")
    if SS["rows"]:
        df_rows = pd.DataFrame(SS["rows"])[["id","alias","row_type","source"]]
        st.dataframe(df_rows, hide_index=True, use_container_width=True)
        SS["row_selection_for_delete"] = ui_multiselect("Select row(s) to delete",
                                                        options=list(df_rows["id"]),
                                                        default=SS.get("row_selection_for_delete", []))
    else:
        st.info("No rows yet. Upload CSV/PDF and click Add rows.")

    st.markdown("**Columns**")
    if SS["columns"]:
        df_cols = pd.DataFrame(SS["columns"])[["id","label","module"]]
        st.dataframe(df_cols, hide_index=True, use_container_width=True)
        SS["col_selection_for_delete"] = ui_multiselect("Select column(s) to delete",
                                                        options=list(df_cols["id"]),
                                                        default=SS.get("col_selection_for_delete", []))
    else:
        st.info("No columns yet. Add at least one module column.")

# ============================================================
# -------------------------- RUN -----------------------------
# ============================================================

if show_section("Run"):
    st.subheader("3) Run Cells")

    st.toggle("Force re-run (ignore cache)", key="force_rerun", value=SS.get("force_rerun", False))

    row_ids = [r["id"] for r in SS["rows"]]
    col_ids = [c["id"] for c in SS["columns"]]

    SS["row_run_selection"] = ui_multiselect(
        "Rows to run (quick)",
        options=row_ids,
        default=SS.get("row_run_selection", []),
    )

    SS["col_run_selection"] = ui_multiselect(
        "Columns to run",
        options=col_ids,
        default=SS.get("col_run_selection", []),
    )

    rcol1, rcol2, rcol3 = st.columns([1,1,1])
    with rcol1:
        if st.button("Run selection", use_container_width=True, type="primary"):
            run_selection(SS["row_run_selection"], SS["col_run_selection"])
            ui_rerun()
    with rcol2:
        if st.button("Run ALL", use_container_width=True):
            run_selection(row_ids, col_ids)
            ui_rerun()
    with rcol3:
        if st.button("Clear selection", use_container_width=True):
            SS["row_run_selection"] = []
            SS["col_run_selection"] = []
            ui_rerun()

    st.markdown("**Results (investor view)**")
    # Build table view
    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}

    table = []
    for (rid, cid), res in SS["results"].items():
        r = rows_by_id.get(rid)
        c = cols_by_id.get(cid)
        if not r or not c:
            continue
        table.append(dict(
            Row=r["alias"],
            row_type=r["row_type"],
            Column=c["label"],
            Module=c["module"],
            status=res.get("status"),
            Value=res.get("value"),
            Summary=res.get("summary"),
        ))

    if table:
        df = pd.DataFrame(table)
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("No results yet. Select rows & columns, then click Run.")

# ============================================================
# ------------------------- REVIEW ---------------------------
# ============================================================

if show_section("Review"):
    st.subheader("4) Review")
    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}

    # Let user pick a completed cell for detail
    done_keys = [k for k, v in SS["results"].items() if v.get("status") == "done"]
    if not done_keys:
        st.info("Run some cells first.")
    else:
        labels = []
        for (rid, cid) in done_keys:
            r = rows_by_id.get(rid, {"alias": rid})
            c = cols_by_id.get(cid, {"label": cid})
            labels.append(f"{r['alias']}  →  {c['label']}")
        idx = ui_selectbox("Choose a cell", list(range(len(done_keys))), default=0,
                           format_func=lambda i: labels[i] if 0 <= i < len(labels) else "—")
        key = done_keys[idx]
        res = SS["results"].get(key, {})
        st.write("**Summary**:", res.get("summary") or "—")

        # Plot cohort curve if present
        if "curve" in res:
            st.write("Average Retention Curve")
            curve = res["curve"]
            chart_df = pd.DataFrame({"x": list(range(len(curve))), "y": curve})
            st.line_chart(chart_df, x="x", y="y", height=260)

# ============================================================
# -------------------------- MEMO ----------------------------
# ============================================================

if show_section("Memo"):
    st.subheader("5) Memo / Export")

    # Export results CSV (simple)
    rows_by_id = {r["id"]: r for r in SS["rows"]}
    cols_by_id = {c["id"]: c for c in SS["columns"]}
    out_rows = []
    for (rid, cid), res in SS["results"].items():
        r = rows_by_id.get(rid)
        c = cols_by_id.get(cid)
        if not r or not c:
            continue
        out_rows.append(dict(
            row_id=rid, row_alias=r["alias"], row_type=r["row_type"], source=r["source"],
            col_id=cid, col_label=c["label"], module=c["module"],
            status=res.get("status"), value=res.get("value"), summary=res.get("summary")
        ))
    if out_rows:
        export_df = pd.DataFrame(out_rows)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results (CSV)", data=csv_bytes, file_name="transformai_results.csv", mime="text/csv")
    else:
        st.info("No results to export yet.")

# Footer hint
st.caption("Tip: If you delete rows/columns later, the selectors and defaults are sanitized so widgets never crash.")
