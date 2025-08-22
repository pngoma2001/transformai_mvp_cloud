# pages/0_Demo_Data_Seeder.py
import io
import textwrap
import zipfile
from datetime import datetime, timedelta
from math import isnan

import numpy as np
import pandas as pd
import streamlit as st

# PDFs
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

st.set_page_config(page_title="Demo Data Seeder (Pro)", layout="wide")
st.title("Demo Data Seeder — Pro Pack")
st.caption("Generates CSVs and PDFs used by the **Diligence Grid (Pro)** page: transactions, QoE P&L, NRR/GRR, customers, AR aging, inventory, and two PDFs (KPI Pack + QoE Summary).")

# -----------------------------
# Helpers: PDF primitives
# -----------------------------
def _pdf_paragraph(c, text, x, y, width_chars=95, size=11, leading=15, bold=False):
    """Draw a wrapped paragraph, return the ending y."""
    font = "Helvetica-Bold" if bold else "Helvetica"
    c.setFont(font, size)
    for line in textwrap.wrap(text, width=width_chars):
        c.drawString(x, y, line)
        y -= leading
    return y

def _tbl(c, x, y, rows, col_widths, size=10, leading=14, header=True):
    """Very simple table printer; returns new y."""
    c.setFont("Helvetica-Bold" if header else "Helvetica", size)
    for i, row in enumerate(rows):
        if i == 1 and header:  # switch to body font after header row
            c.setFont("Helvetica", size)
        yy = y - i * leading
        for j, cell in enumerate(row):
            c.drawString(x + sum(col_widths[:j]), yy, str(cell))
    return y - len(rows) * leading - 6

# -----------------------------
# 1) Existing KPI PDF (kept)
# -----------------------------
def make_kpi_pdf_bytes(company="Nimbus Retail, Inc.", period_label="FY2024"):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    W, H = LETTER
    M = 0.85 * inch
    y = H - M

    c.setTitle("Demo KPI Pack")

    c.setFont("Helvetica-Bold", 18)
    c.drawString(M, y, f"Demo KPI Pack — {company}")
    y -= 22
    c.setFont("Helvetica", 11)
    c.drawString(M, y, f"Period: {period_label}  |  Generated: {datetime.now():%b %d, %Y}")
    y -= 24

    y = _pdf_paragraph(
        c, "Income Statement Summary", M, y, size=16, leading=20, bold=True
    )
    y = _pdf_paragraph(
        c,
        "FY2024 total revenue was $12.5 million with strong seasonality in Q4. "
        "On an adjusted basis, EBITDA reached $3.2 million driven by improved merchandising mix.",
        M,
        y,
    )
    y = _pdf_paragraph(
        c,
        "Gross margin expanded to 64% due to lower freight costs and higher private-label penetration.",
        M,
        y,
    )

    c.showPage()
    y = H - M
    y = _pdf_paragraph(c, "Customer Health", M, y, size=16, leading=20, bold=True)
    y = _pdf_paragraph(
        c,
        "Monthly customer churn improved to ~4% on average in the second half of FY2024, reflecting stronger retention programs.",
        M,
        y,
    )
    y = _pdf_paragraph(
        c,
        "Net revenue per active customer increased as cross-sell attachments rose, while acquisition spend held flat.",
        M,
        y,
    )

    c.showPage()
    y = H - M
    y = _pdf_paragraph(c, "Notes & Outlook", M, y, size=16, leading=20, bold=True)
    y = _pdf_paragraph(
        c,
        "The company plans to expand marketplace integrations and optimize fulfillment. "
        "Management is assessing selective price increases where elasticity is favorable.",
        M,
        y,
    )
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()

# -----------------------------
# 2) NEW: QoE PDF
# -----------------------------
def make_qoe_pdf_bytes(qoe_df: pd.DataFrame, company="Nimbus Retail, Inc."):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    W, H = LETTER
    M = 0.85 * inch
    y = H - M
    c.setTitle("QoE Summary")

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(M, y, f"Quality of Earnings — {company}")
    y -= 22
    c.setFont("Helvetica", 11)
    c.drawString(M, y, f"Generated: {datetime.now():%b %d, %Y}")
    y -= 24

    # Narrative
    y = _pdf_paragraph(
        c,
        "This QoE summary shows monthly revenue, COGS, gross margin, operating expenses and EBITDA. "
        "EBITDA margin trends help assess sustainability of earnings and sensitivity to gross margin shifts.",
        M,
        y,
    )

    # Table header + first page table (up to ~10 rows)
    header = ["Month", "Revenue", "COGS", "GM %", "Opex", "EBITDA", "EBITDA %"]
    def fmt_pct(v):
        try:
            return f"{float(v):.1%}"
        except Exception:
            return str(v)

    rows = [header]
    for _, r in qoe_df.iterrows():
        rows.append([
            r["month"],
            f"${r['revenue']:,.0f}",
            f"${r['cogs']:,.0f}",
            fmt_pct(r['gross_margin_pct']),
            f"${r['opex']:,.0f}",
            f"${r['ebitda']:,.0f}",
            fmt_pct(r['ebitda_margin_pct']),
        ])

    # Paginate 12 rows per page
    page_rows = 12
    start = 0
    while start < len(rows):
        chunk = rows[start : start + page_rows]
        y = _tbl(c, M, y, chunk, col_widths=[90, 90, 90, 60, 90, 90, 70], header=True)
        start += page_rows
        if start < len(rows):
            c.showPage()
            y = H - M

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()

# -----------------------------
# 3) CSV Generators
# -----------------------------
def make_transactions_df() -> pd.DataFrame:
    """
    Order-level transactions (CSV) with columns:
      customer_id, order_date, price, quantity, net_revenue
    NOTE: net_revenue is price * quantity; this feeds pricing power & cohort modules.
    """
    rows = [
        ["O000001","C0001","2024-01-05",21.50,3,64.5],
        ["O000002","C0001","2024-02-11",23.00,2,46.0],
        ["O000003","C0001","2024-03-18",24.75,2,49.5],
        ["O000004","C0001","2024-05-02",26.10,2,52.2],
        ["O000005","C0002","2024-01-09",18.00,4,72.0],
        ["O000006","C0002","2024-02-20",19.20,3,57.6],
        ["O000007","C0002","2024-04-03",20.10,3,60.3],
        ["O000008","C0002","2024-06-12",21.40,2,42.8],
        ["O000009","C0003","2024-01-15",35.00,1,35.0],
        ["O000010","C0003","2024-02-19",36.50,1,36.5],
        ["O000011","C0003","2024-03-25",38.20,1,38.2],
        ["O000012","C0003","2024-04-28",39.80,1,39.8],
        ["O000013","C0004","2024-01-07",12.00,6,72.0],
        ["O000014","C0004","2024-02-10",12.60,5,63.0],
        ["O000015","C0004","2024-03-14",13.30,5,66.5],
        ["O000016","C0004","2024-05-01",14.00,4,56.0],
        ["O000017","C0005","2024-01-22",27.00,2,54.0],
        ["O000018","C0005","2024-02-22",28.10,2,56.2],
        ["O000019","C0005","2024-03-21",29.20,2,58.4],
        ["O000020","C0005","2024-05-15",30.00,2,60.0],
        ["O000021","C0006","2024-01-03",50.00,1,50.0],
        ["O000022","C0006","2024-02-07",51.00,1,51.0],
        ["O000023","C0006","2024-03-09",52.50,1,52.5],
        ["O000024","C0006","2024-04-13",53.20,1,53.2],
        ["O000025","C0007","2024-01-28",16.00,5,80.0],
        ["O000026","C0007","2024-02-26",16.80,4,67.2],
        ["O000027","C0007","2024-03-27",17.50,4,70.0],
        ["O000028","C0007","2024-05-05",18.20,3,54.6],
        ["O000029","C0008","2024-01-11",40.00,2,80.0],
        ["O000030","C0008","2024-02-14",41.20,2,82.4],
        ["O000031","C0008","2024-03-17",42.00,2,84.0],
        ["O000032","C0008","2024-04-22",43.50,1,43.5],
        ["O000033","C0009","2024-01-19",22.00,3,66.0],
        ["O000034","C0009","2024-02-21",23.10,3,69.3],
        ["O000035","C0009","2024-03-23",24.00,2,48.0],
        ["O000036","C0009","2024-04-30",25.50,2,51.0],
        ["O000037","C0010","2024-01-06",31.00,2,62.0],
        ["O000038","C0010","2024-02-12",32.40,2,64.8],
        ["O000039","C0010","2024-03-19",33.10,2,66.2],
        ["O000040","C0010","2024-04-27",34.20,1,34.2],
        ["O000041","C0001","2024-06-15",27.10,2,54.2],
        ["O000042","C0003","2024-06-10",41.00,1,41.0],
        ["O000043","C0004","2024-06-20",14.60,4,58.4],
        ["O000044","C0007","2024-06-18",18.90,3,56.7],
        ["O000045","C0008","2024-06-22",44.20,1,44.2],
        ["O000046","C0002","2024-05-28",21.90,2,43.8],
        ["O000047","C0005","2024-06-05",30.80,1,30.8],
        ["O000048","C0006","2024-06-08",54.10,1,54.1],
        ["O000049","C0009","2024-06-03",26.00,2,52.0],
        ["O000050","C0010","2024-06-12",35.00,1,35.0],
    ]
    df = pd.DataFrame(rows, columns=["order_id","customer_id","order_date","price","quantity","net_revenue"])
    return df[["customer_id","order_date","price","quantity","net_revenue"]]

def make_customers_df(n=200, start="2023-07-01", end="2024-06-30"):
    rng = pd.date_range(start, end, freq="D")
    np.random.seed(7)
    first_dates = np.random.choice(rng, size=n, replace=True)
    cust_ids = [f"C{str(i).zfill(4)}" for i in range(1, n + 1)]
    df = pd.DataFrame({"customer_id": cust_ids, "first_order_date": pd.to_datetime(first_dates)})
    df["cohort_month"] = df["first_order_date"].dt.to_period("M").astype(str)
    return df

def make_qoe_monthly_df(start="2024-01-01", months=12):
    # Revenue trend + stable GM ~64%, Opex ~35% of revenue drifting down slightly
    rng = pd.date_range(start, periods=months, freq="MS")
    base = 900_000.0
    rev = []
    for i in range(months):
        season = 1.00 + 0.05*np.sin(i/12 * 2*np.pi)  # tiny seasonality
        rev.append(base * (1 + 0.02*i) * season)
    rev = np.array(rev)

    gm_pct = 0.64 + 0.01*np.sin(np.linspace(0, 2*np.pi, months))  # 63–65%
    cogs = rev * (1 - gm_pct)

    opex_pct = 0.34 - 0.002*np.arange(months)  # trending slightly down
    opex = rev * opex_pct

    ebitda = rev - cogs - opex
    ebitda_margin = ebitda / rev

    df = pd.DataFrame({
        "month": rng.strftime("%Y-%m"),
        "revenue": rev.round(0),
        "cogs": cogs.round(0),
        "gross_profit": (rev - cogs).round(0),
        "gross_margin_pct": gm_pct,
        "opex": opex.round(0),
        "ebitda": ebitda.round(0),
        "ebitda_margin_pct": ebitda_margin
    })
    return df

def make_nrr_grr_df(start="2024-01-01", months=12, start_mrr=1_000_000):
    rng = pd.date_range(start, periods=months, freq="MS")
    np.random.seed(11)
    data = []
    mrr_begin = float(start_mrr)
    for i, d in enumerate(rng):
        new = 40_000 + np.random.randint(-5_000, 5_000)
        expansion = 22_000 + np.random.randint(-4_000, 4_000)
        contraction = 10_000 + np.random.randint(-3_000, 3_000)
        churn = 60_000 + np.random.randint(-10_000, 10_000)
        mrr_end = mrr_begin + new + expansion - contraction - churn
        grr = (mrr_begin - churn - contraction) / mrr_begin if mrr_begin else np.nan
        nrr = (mrr_begin - churn - contraction + expansion + new) / mrr_begin if mrr_begin else np.nan
        data.append([d.strftime("%Y-%m"), round(mrr_begin, 0), new, expansion, contraction, churn, round(mrr_end, 0), grr, nrr])
        mrr_begin = mrr_end
    df = pd.DataFrame(data, columns=["month","mrr_begin","new_mrr","expansion_mrr","contraction_mrr","churn_mrr","mrr_end","grr","nrr"])
    return df

def make_ar_aging_df(as_of=None):
    if as_of is None:
        as_of = datetime(2024, 6, 30)
    custs = [f"C{str(i).zfill(4)}" for i in range(1, 41)]
    np.random.seed(3)
    rows = []
    for cid in custs:
        cur = max(0, np.random.normal(3500, 1200))
        d30 = max(0, np.random.normal(2100, 900))
        d60 = max(0, np.random.normal(1000, 600))
        d90 = max(0, np.random.normal(700, 400))
        rows.append([cid, as_of.strftime("%Y-%m-%d"), round(cur, 0), round(d30, 0), round(d60, 0), round(d90, 0)])
    return pd.DataFrame(rows, columns=["customer_id","as_of","current","30d","60d","90d"])

def make_inventory_df(as_of=None):
    if as_of is None:
        as_of = datetime(2024, 6, 30)
    skus = [f"SKU-{i:03d}" for i in range(1, 31)]
    np.random.seed(5)
    rows = []
    for s in skus:
        unit_cost = np.random.uniform(6, 40)
        price = unit_cost * np.random.uniform(1.4, 2.6)
        on_hand = np.random.randint(50, 800)
        turns = np.random.uniform(3.0, 9.5)
        rows.append([s, as_of.strftime("%Y-%m-%d"), round(unit_cost,2), round(price,2), on_hand, round(turns,1)])
    return pd.DataFrame(rows, columns=["sku","as_of","unit_cost","unit_price","on_hand_qty","annual_turns"])

# -----------------------------
# Build all datasets now
# -----------------------------
company = "Nimbus Retail, Inc."
period_label = "FY2024"

tx_df   = make_transactions_df()
cust_df = make_customers_df()
qoe_df  = make_qoe_monthly_df()
nrr_df  = make_nrr_grr_df()
ar_df   = make_ar_aging_df()
inv_df  = make_inventory_df()

kpi_pdf_bytes = make_kpi_pdf_bytes(company=company, period_label=period_label)
qoe_pdf_bytes = make_qoe_pdf_bytes(qoe_df, company=company)

# -----------------------------
# UI: Previews + Downloads
# -----------------------------
colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Core CSVs")
    st.markdown("These align with the Pro grid mapping: **Transactions** → Pricing/Cohorts, **NRR/GRR**, **Customers** (cohorts), **AR Aging**, **Inventory**.")

    st.write("**sample_transactions.csv**")
    st.dataframe(tx_df.head(10), use_container_width=True, height=230)
    st.download_button("⬇️ Download sample_transactions.csv",
                       data=tx_df.to_csv(index=False).encode("utf-8"),
                       file_name="sample_transactions.csv",
                       mime="text/csv")

    st.write("**sample_customers.csv**")
    st.dataframe(cust_df.head(10), use_container_width=True, height=230)
    st.download_button("⬇️ Download sample_customers.csv",
                       data=cust_df.to_csv(index=False).encode("utf-8"),
                       file_name="sample_customers.csv",
                       mime="text/csv")

    st.write("**sample_nrr_grr.csv**")
    st.dataframe(nrr_df.head(12), use_container_width=True, height=260)
    st.download_button("⬇️ Download sample_nrr_grr.csv",
                       data=nrr_df.to_csv(index=False).encode("utf-8"),
                       file_name="sample_nrr_grr.csv",
                       mime="text/csv")

with colB:
    st.subheader("QoE & Ops CSVs")
    st.write("**sample_qoe_pnl_monthly.csv**")
    st.dataframe(qoe_df, use_container_width=True, height=260)
    st.download_button("⬇️ Download sample_qoe_pnl_monthly.csv",
                       data=qoe_df.to_csv(index=False).encode("utf-8"),
                       file_name="sample_qoe_pnl_monthly.csv",
                       mime="text/csv")

    st.write("**sample_ar_aging.csv**")
    st.dataframe(ar_df.head(12), use_container_width=True, height=230)
    st.download_button("⬇️ Download sample_ar_aging.csv",
                       data=ar_df.to_csv(index=False).encode("utf-8"),
                       file_name="sample_ar_aging.csv",
                       mime="text/csv")

    st.write("**sample_inventory.csv**")
    st.dataframe(inv_df.head(12), use_container_width=True, height=230)
    st.download_button("⬇️ Download sample_inventory.csv",
                       data=inv_df.to_csv(index=False).encode("utf-8"),
                       file_name="sample_inventory.csv",
                       mime="text/csv")

st.divider()
st.subheader("PDFs")
c1, c2 = st.columns(2)
with c1:
    st.write("**Sample_KPI_Pack.pdf**")
    st.download_button("⬇️ Download KPI Pack PDF",
                       data=kpi_pdf_bytes,
                       file_name="Sample_KPI_Pack.pdf",
                       mime="application/pdf")
with c2:
    st.write("**QoE_Summary.pdf**")
    st.download_button("⬇️ Download QoE Summary PDF",
                       data=qoe_pdf_bytes,
                       file_name="QoE_Summary.pdf",
                       mime="application/pdf")

# -----------------------------
# ZIP with everything (one click)
# -----------------------------
st.divider()
st.subheader("Bundle")
with io.BytesIO() as mem:
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
         z.writestr("sample_transactions.csv", tx_df.to_csv(index=False))
         z.writestr("sample_customers.csv",   cust_df.to_csv(index=False))
         z.writestr("sample_nrr_grr.csv",     nrr_df.to_csv(index=False))
         z.writestr("sample_qoe_pnl_monthly.csv", qoe_df.to_csv(index=False))
         z.writestr("sample_ar_aging.csv",    ar_df.to_csv(index=False))
         z.writestr("sample_inventory.csv",   inv_df.to_csv(index=False))
         z.writestr("Sample_KPI_Pack.pdf",    kpi_pdf_bytes)
         z.writestr("QoE_Summary.pdf",        qoe_pdf_bytes)
    mem.seek(0)
    st.download_button("⬇️ Download All (ZIP)", data=mem.read(),
                       file_name="TransformAI_Demo_Data.zip",
                       mime="application/zip")

st.info(
    "Next: open **Diligence Grid (Pro)** → upload the CSVs/PDFs → Map schema → "
    "Add columns (Cohort Retention, Pricing Power, NRR/GRR, PDF KPIs, Unit Economics) → "
    "Run selection → Approve → Export memo with citations."
)

