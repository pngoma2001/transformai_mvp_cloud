
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

def _level(score: float) -> str:
    if score is None: 
        return "unknown"
    if score < 0.33: return "low"
    if score < 0.66: return "medium"
    return "high"

def _safe_mean(series):
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        return float(s.mean()) if len(s) else None
    except Exception:
        return None

def pricing_power_index(survey: Optional[pd.DataFrame], market: Optional[pd.DataFrame], macro: Optional[pd.DataFrame]) -> Dict[str, Any]:
    # Survey tolerance: wider acceptable range -> higher power
    tol_score = None
    if survey is not None and {"price_sensitivity_low","price_sensitivity_high"} <= set(survey.columns):
        low = _safe_mean(survey["price_sensitivity_low"])
        high = _safe_mean(survey["price_sensitivity_high"])
        if low is not None and high is not None and high > 0:
            width = max(high - low, 0)
            tol_score = min(width / max(high*0.8, 1e-6), 1.0)  # normalized tolerance
    # Market gap: if our_price < comp_price, room to lift
    gap_score = None
    if market is not None and {"our_price","comp_price"} <= set(market.columns):
        diffs = (pd.to_numeric(market["comp_price"], errors="coerce") - pd.to_numeric(market["our_price"], errors="coerce")) / pd.to_numeric(market["our_price"], errors="coerce")
        diffs = diffs.replace([np.inf,-np.inf], np.nan).dropna()
        if len(diffs):
            gap_score = float(np.clip(diffs.mean(), 0, 1))  # only positive gaps help
    # Macro inflation boost (very rough): higher CPI -> more price acceptance
    macro_boost = None
    if macro is not None and "value" in macro.columns:
        v = _safe_mean(macro["value"])
        if v is not None:
            macro_boost = float(np.clip(v/100.0, 0, 0.2))  # cap small boost

    # Weighted combination
    parts = []
    weights = []
    if tol_score is not None: parts.append(tol_score); weights.append(0.5)
    if gap_score is not None: parts.append(gap_score); weights.append(0.35)
    if macro_boost is not None: parts.append(macro_boost); weights.append(0.15)
    score = None
    if parts:
        score = float(np.clip(np.average(parts, weights=weights), 0, 1))
    level = _level(score if score is not None else 0.5)

    # Recommended price uplift default
    reco = 0.06
    if level == "low": reco = 0.03
    elif level == "medium": reco = 0.06
    else: reco = 0.09

    notes = []
    if tol_score is not None: notes.append(f"Survey tolerance score ~ {tol_score:.2f}")
    if gap_score is not None: notes.append(f"Avg competitor gap ~ {gap_score*100:.1f}%")
    if macro_boost is not None: notes.append(f"Macro CPI boost ~ {macro_boost*100:.1f} pp")
    return {"score": score, "level": level, "recommended_uplift": reco, "notes": notes}

def churn_risk_index(survey: Optional[pd.DataFrame], kpi_churn: Optional[float]) -> Dict[str, Any]:
    # Use detractor share + baseline churn
    detr_share = None
    if survey is not None and "nps" in survey.columns:
        s = pd.to_numeric(survey["nps"], errors="coerce").dropna()
        if len(s):
            detr_share = float((s < 7).mean())
    churn_norm = None
    if kpi_churn is not None:
        churn_norm = float(np.clip(kpi_churn, 0, 0.4)) / 0.4  # normalize 0..0.4 to 0..1

    parts = []; weights=[]
    if detr_share is not None: parts.append(detr_share); weights.append(0.6)
    if churn_norm is not None: parts.append(churn_norm); weights.append(0.4)
    score=None
    if parts:
        score = float(np.clip(np.average(parts, weights=weights), 0, 1))
    level = _level(score if score is not None else 0.5)

    reco = 0.03
    if level == "low": reco = 0.01
    elif level == "medium": reco = 0.03
    else: reco = 0.05

    notes = []
    if detr_share is not None: notes.append(f"NPS detractors ~ {detr_share*100:.1f}%")
    if kpi_churn is not None: notes.append(f"Baseline churn ~ {kpi_churn*100:.1f}%")
    return {"score": score, "level": level, "recommended_reduction": reco, "notes": notes}

def supply_stress_index(stockouts: Optional[pd.DataFrame]) -> Dict[str, Any]:
    rate=None; vol=None
    if stockouts is not None and "stockout_flag" in stockouts.columns:
        s = pd.to_numeric(stockouts["stockout_flag"], errors="coerce").dropna()
        if len(s): rate = float(s.mean())
    if stockouts is not None and "lead_time_days" in stockouts.columns:
        lt = pd.to_numeric(stockouts["lead_time_days"], errors="coerce").dropna()
        if len(lt): vol = float(np.clip(lt.std()/max(lt.mean(),1e-6), 0, 2)/2)  # 0..1
    parts=[]; weights=[]
    if rate is not None: parts.append(rate); weights.append(0.7)
    if vol is not None: parts.append(vol); weights.append(0.3)
    score=None
    if parts:
        score = float(np.clip(np.average(parts, weights=weights), 0, 1))
    level = _level(score if score is not None else 0.5)

    reco = 0.02
    if level == "low": reco = 0.01
    elif level == "medium": reco = 0.02
    else: reco = 0.03

    notes=[]
    if rate is not None: notes.append(f"Stockout rate ~ {rate*100:.1f}%")
    if vol is not None: notes.append(f"Lead time volatility score ~ {vol:.2f}")
    return {"score": score, "level": level, "recommended_delta": reco, "notes": notes}

def utilization_gap_index(util: Optional[pd.DataFrame]) -> Dict[str, Any]:
    gap=None
    if util is not None and {"capacity","filled"} <= set(util.columns):
        cap = pd.to_numeric(util["capacity"], errors="coerce")
        filled = pd.to_numeric(util["filled"], errors="coerce")
        util_rate = (filled / cap).replace([np.inf,-np.inf], np.nan).dropna()
        if len(util_rate):
            gap = float(np.clip(1 - util_rate.mean(), 0, 1))
    ns=None
    if util is not None and "no_show_rate" in util.columns:
        ns = float(pd.to_numeric(util["no_show_rate"], errors="coerce").dropna().mean()) if len(util) else None

    parts=[]; weights=[]
    if gap is not None: parts.append(gap); weights.append(0.7)
    if ns is not None: parts.append(ns); weights.append(0.3)
    score=None
    if parts:
        score = float(np.clip(np.average(parts, weights=weights), 0, 1))
    level = _level(score if score is not None else 0.5)

    reco = 0.02
    if level == "low": reco = 0.01
    elif level == "medium": reco = 0.02
    else: reco = 0.03

    notes=[]
    if gap is not None: notes.append(f"Average unused capacity ~ {gap*100:.1f}%")
    if ns is not None: notes.append(f"No‑show rate ~ {ns*100:.1f}%")
    return {"score": score, "level": level, "recommended_delta": reco, "notes": notes}

def compute_evidence(signals_input: Dict[str, Optional[pd.DataFrame]], kpi_churn: Optional[float]) -> Dict[str, Any]:
    survey = signals_input.get("survey")
    market = signals_input.get("market")
    macro = signals_input.get("macro")
    stock = signals_input.get("stockouts")
    util = signals_input.get("utilization")

    pricing = pricing_power_index(survey, market, macro)
    churn = churn_risk_index(survey, kpi_churn)
    supply = supply_stress_index(stock)
    utilg = utilization_gap_index(util)

    return {
        "pricing_power": pricing,
        "churn_risk": churn,
        "supply_stress": supply,
        "utilization_gap": utilg
    }

def reorder_plays(plays: list, signals: Dict[str, Any]) -> list:
    # Score plays by relevant signal
    weights = {
        "pricing": signals.get("pricing_power",{}).get("score",0.5) or 0.5,
        "retention": signals.get("churn_risk",{}).get("score",0.5) or 0.5,
        "supply": signals.get("supply_stress",{}).get("score",0.5) or 0.5,
        "utilization": signals.get("utilization_gap",{}).get("score",0.5) or 0.5,
        "claims": 0.5,
        "referrals": 0.5
    }
    return sorted(plays, key=lambda p: weights.get(p.get("type",""),0.5), reverse=True)
