
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from datetime import datetime

def export_summary_pdf(out_path, company, kpis: dict, decisions: dict, evidence: dict = None):
    c = canvas.Canvas(out_path, pagesize=LETTER)
    w, h = LETTER

    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(colors.HexColor("#4F46E5"))
    c.drawString(1*inch, h-1*inch, f"TransformAI — {company}")
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)
    c.drawString(1*inch, h-1.2*inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(1*inch, h-1.6*inch, "Summary KPIs")
    c.setFont("Helvetica", 10)
    y = h-1.8*inch
    lines = [
        f"Revenue: ${kpis.get('revenue',0):,.0f}",
        f"EBITDA: ${kpis.get('ebitda',0):,.0f}",
        f"Gross margin: {kpis.get('gross_margin',0)*100:.1f}%",
        f"Revenue YoY: {kpis.get('revenue_yoy',0)*100:.1f}%" if kpis.get('revenue_yoy') is not None else "Revenue YoY: n/a",
    ]
    for ln in lines:
        c.drawString(1*inch, y, ln); y -= 0.18*inch

    c.setFont("Helvetica-Bold", 12)
    c.drawString(1*inch, y-0.2*inch, "Decisions")
    y -= 0.45*inch
    c.setFont("Helvetica", 10)
    if not decisions:
        c.drawString(1*inch, y, "No decisions yet.")
    else:
        for pid, d in decisions.items():
            txt = f"- {pid}: {d.get('status','pending')}"
            if d.get("rationale"): txt += f" — {d['rationale']}"
            c.drawString(1*inch, y, txt)
            y -= 0.18*inch
            if y < 1*inch:
                c.showPage(); y = h-1*inch

    if evidence:
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1*inch, h-1*inch, "Appendix — Evidence Summary")
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1*inch, h-1.4*inch, "Levels")
        c.setFont("Helvetica", 10)
        y = h-1.6*inch
        lv = evidence.get("levels", {})
        for k in ["pricing_power","churn_risk","supply_stress","utilization_gap"]:
            c.drawString(1*inch, y, f"- {k.replace('_',' ').title()}: {lv.get(k,'medium')}")
            y -= 0.18*inch

        c.setFont("Helvetica-Bold", 12)
        c.drawString(1*inch, y-0.2*inch, "Details")
        c.setFont("Helvetica", 10); y -= 0.45*inch
        details = evidence.get("details", {})
        for bucket, rows in details.items():
            c.setFont("Helvetica-Bold", 10)
            c.drawString(1*inch, y, bucket.replace('_',' ').title()); y -= 0.18*inch
            c.setFont("Helvetica", 10)
            for r in rows:
                c.drawString(1.2*inch, y, f"- {r}"); y -= 0.18*inch
                if y < 1*inch:
                    c.showPage(); y = h-1*inch

    c.save()
