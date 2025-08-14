
from __future__ import annotations
from pathlib import Path
import pandas as pd

def _maybe_read_csv(handle_or_path):
    if handle_or_path is None:
        return None
    try:
        if hasattr(handle_or_path, "read"):
            return pd.read_csv(handle_or_path)
        p = Path(handle_or_path)
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        return None
    return None

def load_sample_evidence(use_sample_ev: bool = True,
                         survey_file=None, market_file=None,
                         macro_file=None, stockouts_file=None, util_file=None) -> dict:
    ev = {}
    ev["survey"]     = _maybe_read_csv(survey_file)    or (_maybe_read_csv("data/sample_customer_survey.csv") if use_sample_ev else None)
    ev["market"]     = _maybe_read_csv(market_file)    or (_maybe_read_csv("data/sample_market_prices.csv") if use_sample_ev else None)
    ev["macro"]      = _maybe_read_csv(macro_file)     or (_maybe_read_csv("data/sample_macro_cpi.csv") if use_sample_ev else None)
    ev["stockouts"]  = _maybe_read_csv(stockouts_file) or (_maybe_read_csv("data/sample_stockouts.csv") if use_sample_ev else None)
    ev["utilization"]= _maybe_read_csv(util_file)      or (_maybe_read_csv("data/sample_utilization.csv") if use_sample_ev else None)
    return ev

def _level_from_value(v: float, thresholds: tuple[float,float], reverse: bool=False) -> str:
    if v is None:
        return "medium"
    lo, hi = thresholds
    if reverse:
        if v <= lo: return "high"
        if v >= hi: return "low"
    else:
        if v <= lo: return "low"
        if v >= hi: return "high"
    return "medium"

def compute_evidence_signals(ev: dict, df: pd.DataFrame) -> dict:
    details = {}

    gap = None
    if ev.get("market") is not None and {"our_price","competitor_price"} <= set(ev["market"].columns):
        m = ev["market"].copy()
        m["gap_pct"] = (m["our_price"] - m["competitor_price"]) / m["competitor_price"]
        gap = float(m["gap_pct"].median())
        details.setdefault("market_prices", []).append(f"Median price gap vs competitor: {gap*100:.1f}%")

    tol = None
    if ev.get("survey") is not None and "price_tolerance_pct" in ev["survey"].columns:
        tol = float(ev["survey"]["price_tolerance_pct"].median())/100.0
        details.setdefault("survey", []).append(f"Median tolerated uplift from survey: {tol*100:.1f}%")

    pricing_power_level = _level_from_value((tol if tol is not None else (0.0 if gap is None else -gap)),
                                            thresholds=(0.03, 0.07), reverse=False)

    churn = None
    if ev.get("survey") is not None:
        if "intent_to_churn_pct" in ev["survey"].columns:
            churn = float(ev["survey"]["intent_to_churn_pct"].mean())/100.0
            details.setdefault("survey", []).append(f"Avg intent to churn: {churn*100:.1f}%")
        elif "nps" in ev["survey"].columns:
            detractors = (ev["survey"]["nps"] <= 6).mean()
            churn = float(detractors)
            details.setdefault("survey", []).append(f"NPS detractor share: {detractors*100:.1f}%")
    churn_risk_level = _level_from_value(churn if churn is not None else 0.15, thresholds=(0.10,0.25), reverse=False)

    stockout_rate = None
    if ev.get("stockouts") is not None and "stockout_rate" in ev["stockouts"].columns:
        stockout_rate = float(ev["stockouts"]["stockout_rate"].mean())
        details.setdefault("stockouts", []).append(f"Avg stockout rate: {stockout_rate*100:.1f}%")
    supply_stress_level = _level_from_value(stockout_rate if stockout_rate is not None else 0.07,
                                            thresholds=(0.05,0.15), reverse=False)

    util_gap = None
    if ev.get("utilization") is not None and {"utilization","target"}.issubset(ev["utilization"].columns):
        u = ev["utilization"]
        util_gap = float((u["target"] - u["utilization"]).median())
        details.setdefault("utilization", []).append(f"Median gap to target utilization: {util_gap:.1f} pp")
    utilization_gap_level = _level_from_value(util_gap if util_gap is not None else 5.0,
                                              thresholds=(2.0,8.0), reverse=False)

    defaults = {
        "pricing_uplift":    0.03 if pricing_power_level=="low" else (0.06 if pricing_power_level=="medium" else 0.09),
        "pricing_adoption":  0.4  if pricing_power_level=="low" else (0.6  if pricing_power_level=="medium" else 0.7),
        "retention_reduce_pp": 0.01 if churn_risk_level=="low" else (0.03 if churn_risk_level=="medium" else 0.05),
        "ops_delta":           0.01 if (supply_stress_level=="low" and utilization_gap_level=="low") else (0.02 if "medium" in (supply_stress_level,utilization_gap_level) else 0.03)
    }

    warnings = {}
    if pricing_power_level == "low":
        warnings["pricing"] = "Price sensitivity appears high → cap initial uplift to 3% and A/B on a subset."
    if churn_risk_level == "high":
        warnings["retention"] = "Churn risk elevated → prioritize retention play before broad price changes."

    rank = {
        "pricing":      2 if pricing_power_level=="low" else (3 if pricing_power_level=="medium" else 4),
        "retention":    2 if churn_risk_level=="low" else (3 if churn_risk_level=="medium" else 4),
        "supply":       2 if supply_stress_level=="low" else (3 if supply_stress_level=="medium" else 4),
        "utilization":  2 if utilization_gap_level=="low" else (3 if utilization_gap_level=="medium" else 4),
    }

    return {
        "levels": {
            "pricing_power": pricing_power_level,
            "churn_risk": churn_risk_level,
            "supply_stress": supply_stress_level,
            "utilization_gap": utilization_gap_level,
        },
        "defaults": defaults,
        "warnings": warnings,
        "rank": rank,
        "details": details,
    }

def rank_plays_by_evidence(plays: list, signals: dict) -> list:
    rank = signals.get("rank", {})
    def key(p):
        return (-rank.get(p.get("type",""), 1), p.get("title",""))
    return sorted(plays, key=key)

def chips_from_signals(signals: dict) -> list:
    levels = signals.get("levels", {})
    def chip(label, level):
        color = {"low":"#F59E0B","medium":"#6366F1","high":"#10B981"}.get(level, "#6B7280")
        return f"<span style='background:{color}22;border:1px solid {color};padding:4px 8px;border-radius:999px;font-size:12px;margin-right:6px;'>{label}: {level.title()}</span>"
    chips = [
        chip("Pricing power", levels.get("pricing_power","medium")),
        chip("Churn risk", levels.get("churn_risk","medium")),
        chip("Supply stress", levels.get("supply_stress","medium")),
        chip("Utilization gap", levels.get("utilization_gap","medium")),
    ]
    return chips
