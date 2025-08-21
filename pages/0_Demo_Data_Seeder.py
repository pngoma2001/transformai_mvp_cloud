import io, textwrap
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

st.set_page_config(page_title="Demo Data Seeder", layout="centered")
st.title("Demo Data Seeder")
st.caption("Generate a sample CSV and a KPI PDF for the Diligence Grid (Pro) page.")

def make_pdf_bytes():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    W,H = LETTER; M = 0.85*inch; y = H - M
    def p(txt, size=12, leading=16, bold=False):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        for line in textwrap.wrap(txt, width=95):
            if y < M: c.showPage(); y = H - M; c.setFont("Helvetica", size)
            c.drawString(M, y, line); y -= leading
        y -= 6
    p("Demo KPI Pack — Nimbus Retail, Inc.", 18, 22, bold=True)
    c.setFont("Helvetica", 11); c.drawString(M, y, f"Period: FY2024  |  Generated: {datetime.now():%b %d, %Y}"); y -= 24
    p("Income Statement Summary", 16, 20, bold=True)
    p("FY2024 total revenue was $12.5 million with strong seasonality in Q4. On an adjusted basis, EBITDA reached $3.2 million driven by improved merchandising mix.")
    p("Gross margin expanded to 64% due to lower freight costs and higher private-label penetration.")
    c.showPage(); y = H - M
    p("Customer Health", 16, 20, bold=True)
    p("Monthly customer churn improved to 3.8% on average in the second half of FY2024, reflecting stronger retention programs.")
    p("Net revenue per active customer increased as cross-sell attachments rose, while acquisition spend held flat.")
    c.showPage(); y = H - M
    p("Notes & Outlook", 16, 20, bold=True)
    p("The company plans to expand its marketplace integrations and optimize fulfillment. Management is assessing selective price increases where elasticity is favorable. Capital needs are modest and expected to be funded by cash from operations.")
    c.showPage(); c.save(); buf.seek(0); return buf.getvalue()

def make_csv_df():
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
    return pd.DataFrame(rows, columns=["order_id","customer_id","order_date","price","quantity","net_revenue"])

csv_df = make_csv_df()
csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
pdf_bytes = make_pdf_bytes()

st.success("Demo data ready.")
st.download_button("⬇️ Download sample_transactions.csv", data=csv_bytes, file_name="sample_transactions.csv", mime="text/csv")
st.download_button("⬇️ Download Sample_KPI_Pack.pdf", data=pdf_bytes, file_name="Sample_KPI_Pack.pdf", mime="application/pdf")

st.divider()
st.write("Next: go to **Diligence Grid (Pro)**, upload the CSV & PDF, map the schema, add columns, run, approve, and export the memo.")
