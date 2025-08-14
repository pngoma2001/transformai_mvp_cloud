
import json
from typing import Dict, Any, List, Optional

# Lazy imports so the app runs without these packages when AI mode is off
def _get_openai_client():
    try:
        from openai import OpenAI
        return OpenAI()
    except Exception:
        return None

def _get_anthropic_client():
    try:
        import anthropic
        return anthropic.Anthropic()
    except Exception:
        return None

SYSTEM_PROMPT = (
    "You are a PE Portfolio Ops strategist. Given KPI JSON, produce exactly 3 'plays' as JSON. "
    "Each play must include: id, type, title, hypothesis, uplift_usd, uplift_pct, confidence, "
    "complexity, assumptions[], risks[], plan[{week, step, owner_role}]. "
    "Only output JSON. Do not include commentary."
)

# Minimal schema-like contract for prompt (we still validate in app/backend)
def build_user_prompt(kpis: Dict[str, Any]) -> str:
    return json.dumps({
        "kpis": kpis,
        "instructions": {
            "types_allowed": ["pricing","retention","supply","utilization","claims","referrals"],
            "confidence_allowed": ["low","medium","high"],
            "complexity_allowed": ["low","medium","high"]
        }
    }, indent=2)

def call_openai(model: str, kpis: Dict[str, Any], temperature: float = 0.2) -> Optional[List[Dict[str, Any]]]:
    client = _get_openai_client()
    if client is None:
        return None
    try:
        content = [
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": build_user_prompt(kpis)}
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=content,
            temperature=temperature,
            response_format={"type":"json_object"}
        )
        txt = resp.choices[0].message.content
        data = json.loads(txt)
        plays = data.get("plays") or data  # tolerate {plays:[...]} or [...]
        return plays
    except Exception:
        return None

def call_anthropic(model: str, kpis: Dict[str, Any], temperature: float = 0.2) -> Optional[List[Dict[str, Any]]]:
    client = _get_anthropic_client()
    if client is None:
        return None
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=temperature,
            system=SYSTEM_PROMPT,
            messages=[
                {"role":"user","content": build_user_prompt(kpis)}
            ]
        )
        # Extract JSON from text block
        txt = ""
        for block in msg.content:
            if hasattr(block, "text"):
                txt += block.text
            elif isinstance(block, dict) and block.get("type")=="text":
                txt += block.get("text","")
        # Find first JSON object/array
        match = re.search(r"\{.*\}|\[.*\]", txt, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group(0))
        plays = data.get("plays") or data
        return plays
    except Exception:
        return None
