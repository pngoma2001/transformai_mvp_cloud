
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def export_summary_pdf(path, company, kpis, decisions):
    c = canvas.Canvas(path, pagesize=LETTER)
    width, height = LETTER

    # Header
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1*inch, height-1*inch, f"TransformAI Summary — {company}")
    c.setFont("Helvetica", 10)
    c.drawString(1*inch, height-1.2*inch, f"Period: {kpis.get('period','')}")

    # KPIs
    y = height - 1.6*inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1*inch, y, "KPIs")
    y -= 0.2*inch
    c.setFont("Helvetica", 10)
    kpi_lines = [
        f"Revenue: ${kpis.get('revenue',0):,.0f}",
        f"EBITDA: ${kpis.get('ebitda',0):,.0f}",
        f"Gross Margin: {kpis.get('gross_margin',0)*100:.1f}%",
    ]
    for line in kpi_lines:
        c.drawString(1*inch, y, line); y -= 0.18*inch

    # Decisions
    y -= 0.2*inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1*inch, y, "Decisions")
    y -= 0.2*inch
    c.setFont("Helvetica", 10)
    if not decisions:
        c.drawString(1*inch, y, "No decisions saved."); y -= 0.18*inch
    else:
        for pid, d in decisions.items():
            text = f"{pid}: {d.get('status','')} — {d.get('rationale','')}"
            c.drawString(1*inch, y, text[:95]); y -= 0.18*inch
            if y < 1*inch:
                c.showPage(); y = height - 1*inch

    c.showPage()
    c.save()
