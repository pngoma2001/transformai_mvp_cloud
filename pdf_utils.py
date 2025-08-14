
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def export_summary_pdf(path, company, kpis, decisions, evidence=None, logo_path="assets/logo.png"):
    c = canvas.Canvas(path, pagesize=LETTER)
    width, height = LETTER

    # Header
    try:
        c.drawImage(logo_path, 1*inch, height-0.9*inch, width=1.8*inch, preserveAspectRatio=True, mask="auto")
    except Exception:
        pass
    c.setFillColor(colors.HexColor("#4F46E5"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1*inch, height-1.05*inch, f"TransformAI Summary — {company}")
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

    # Evidence appendix
    try:
        _draw_evidence_appendix(c, evidence, width, height)
    except Exception:
        pass
    c.save()


def _draw_evidence_appendix(c, evidence, width, height):
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1*inch, height-1*inch, "Evidence Appendix")
    y = height - 1.3*inch
    c.setFont("Helvetica", 10)
    if not evidence:
        c.drawString(1*inch, y, "No additional evidence supplied."); return
    order = [("Pricing power","pricing_power","recommended_uplift"),
             ("Churn risk","churn_risk","recommended_reduction"),
             ("Supply stress","supply_stress","recommended_delta"),
             ("Utilization gap","utilization_gap","recommended_delta")]
    for title, key, rec in order:
        block = evidence.get(key, {})
        level = (block.get("level","") or "").title()
        c.drawString(1*inch, y, f"{title}: {level}"); y -= 0.18*inch
        rec_val = block.get(rec)
        if rec_val is not None:
            c.drawString(1*inch, y, f"  Recommendation: {rec_val:.2%}"); y -= 0.18*inch
        for n in block.get("notes", []):
            c.drawString(1*inch, y, f"  - {n}"); y -= 0.18*inch
        y -= 0.1*inch
        if y < 1*inch:
            c.showPage(); y = height - 1*inch
    c.showPage()
