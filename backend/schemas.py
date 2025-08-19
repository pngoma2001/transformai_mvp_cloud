from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class DecisionCreate(BaseModel):
    play_id: str
    play_title: str
    status: str
    rationale: Optional[str] = ""
    actor: Optional[str] = "user"


class DecisionOut(DecisionCreate):
    ts: str

    class Config:
        orm_mode = True


class ActivityOut(BaseModel):
    ts: str
    action: str
    play_title: str
    target: str
    status: str

    class Config:
        orm_mode = True


class AnalyzeResult(BaseModel):
    kpis: Dict[str, Any]
    plays: List[Dict[str, Any]]
