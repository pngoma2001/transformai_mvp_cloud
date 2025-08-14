
import pandas as pd
from typing import Dict, Any, List
import numpy as np

USD = lambda x: float(np.round(x,2))

def summarize_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    def pct(a,b): 
        try:
            return float(np.round((a-b)/b,4))
        except Exception:
            return 0.0
    return {
        "company": str(last.get("company","Company")),
        "period": str(last.get("period","")),
        "revenue": float(last["revenue"]),
        "revenue_yoy": pct(last["revenue"], prev["revenue"]),
        "ebitda": float(last["ebitda"]),
        "ebitda_margin": float(last["ebitda"]/last["revenue"]) if last["revenue"] else 0.0,
        "gross_margin": float(last.get("gross_margin", np.nan)),
        "churn_rate": float(last.get("churn_rate", np.nan)) if not pd.isna(last.get("churn_rate", np.nan)) else None,
        "inventory_turns": float(last.get("inventory_turns", np.nan)) if "inventory_turns" in df.columns else None,
        "utilization": float(last.get("utilization", np.nan)) if "utilization" in df.columns else None
    }

def pricing_play(kpis: Dict[str, Any]) -> Dict[str, Any]:
    rev = kpis["revenue"]
    gm = kpis.get("gross_margin", 0.35) or 0.35
    adoption = 0.6
    price_lift = 0.06
    uplift = rev * price_lift * gm * adoption
    return {
        "id":"pricing1",
        "type":"pricing",
        "title":"Pricing Optimization",
        "hypothesis":"Increase prices 5–7% on top 30% SKUs with low elasticity.",
        "uplift_usd": USD(uplift),
        "uplift_pct": float(np.round(uplift/max(rev,1),4)),
        "confidence":"high",
        "complexity":"medium",
        "assumptions":[f"Gross margin ~{int(gm*100)}%", "Competitive response limited", "No major churn impact"],
        "risks":["Customer backlash on sensitive SKUs","Competitor undercutting"],
        "plan":[
            {"week":1, "step":"Price test cohorts by SKU decile", "owner_role":"RevOps"},
            {"week":2, "step":"Roll to 30% of traffic; monitor conversion", "owner_role":"Ecomm"},
            {"week":3, "step":"Expand to 70%; adjust per elasticity", "owner_role":"GM"},
            {"week":4, "step":"Full rollout; set guardrails", "owner_role":"Finance"},
        ]
    }

def retention_play(kpis: Dict[str, Any]) -> Dict[str, Any]:
    rev = kpis["revenue"]
    gm = kpis.get("gross_margin", 0.35) or 0.35
    churn = kpis.get("churn_rate", 0.2) or 0.2
    reduce_pp = 0.03
    uplift = rev * reduce_pp * gm * 0.5
    return {
        "id":"retention1",
        "type":"retention",
        "title":"Loyalty & Personalization",
        "hypothesis":"Reduce churn by 2–4 pp via targeted lifecycle offers.",
        "uplift_usd": USD(uplift),
        "uplift_pct": float(np.round(uplift/max(rev,1),4)),
        "confidence":"medium",
        "complexity":"medium",
        "assumptions":[f"Baseline churn {int(churn*100)}%", "Offer cost netted in GM"],
        "risks":["Offer cannibalization","Data quality issues"],
        "plan":[
            {"week":1, "step":"Segment cohorts by LTV/RFM", "owner_role":"CRM"},
            {"week":2, "step":"Launch win-back & VIP offers", "owner_role":"Marketing"},
            {"week":3, "step":"Automate journeys in Salesforce", "owner_role":"CRM"},
            {"week":4, "step":"Readout & scale", "owner_role":"Ops"},
        ]
    }

def inventory_play(kpis: Dict[str, Any]) -> Dict[str, Any]:
    rev = kpis["revenue"]
    gm = kpis.get("gross_margin", 0.35) or 0.35
    uplift = rev * 0.01 * gm
    return {
        "id":"supply1",
        "type":"supply",
        "title":"Auto-Replenishment & Stockout Control",
        "hypothesis":"Reduce stockouts on top SKUs and auto-replenish based on turns.",
        "uplift_usd": USD(uplift),
        "uplift_pct": float(np.round(uplift/max(rev,1),4)),
        "confidence":"medium",
        "complexity":"low",
        "assumptions":["Backorders fulfilled", "Supplier lead times stable"],
        "risks":["Supplier delays","Overstock risk"],
        "plan":[
            {"week":1, "step":"Flag SKUs with frequent stockouts", "owner_role":"Supply Chain"},
            {"week":2, "step":"Set reorder points & buffers", "owner_role":"Ops"},
            {"week":3, "step":"Pilot auto-PO flow", "owner_role":"Procurement"},
            {"week":4, "step":"Scale across categories", "owner_role":"COO"},
        ]
    }

def utilization_play(kpis: Dict[str, Any]) -> Dict[str, Any]:
    rev = kpis["revenue"]
    gm = kpis.get("gross_margin", 0.33) or 0.33
    uplift = rev * 0.02 * gm
    return {
        "id":"util1",
        "type":"utilization",
        "title":"Scheduling Utilization Boost",
        "hypothesis":"Improve provider fill-rate by 8–12% via overbooking & reminders.",
        "uplift_usd": USD(uplift),
        "uplift_pct": float(np.round(uplift/max(rev,1),4)),
        "confidence":"medium",
        "complexity":"medium",
        "assumptions":["No increase in no-show penalties"],
        "risks":["Provider burnout","Patient dissatisfaction"],
        "plan":[
            {"week":1, "step":"Gap analysis of schedule grid", "owner_role":"Ops"},
            {"week":2, "step":"Enable overbooking rules + SMS", "owner_role":"IT"},
            {"week":3, "step":"Optimize templates per location", "owner_role":"Ops"},
            {"week":4, "step":"Measure & lock SOP", "owner_role":"COO"},
        ]
    }

def analyze(df: pd.DataFrame) -> Dict[str, Any]:
    kpis = summarize_kpis(df)
    company = (df.iloc[-1].get("company","")).lower()
    if "retail" in company:
        plays = [pricing_play(kpis), retention_play(kpis), inventory_play(kpis)]
    elif "health" in company:
        plays = [utilization_play(kpis), retention_play(kpis), inventory_play(kpis)]
    else:
        plays = [pricing_play(kpis), retention_play(kpis), inventory_play(kpis)]
    return {"kpis": kpis, "plays": plays}
