# backend/schemas.py
# Pydantic models used by FastAPI endpoints

from typing import Optional
from pydantic import BaseModel, ConfigDict

# Keep compatibility with .from_orm(...) on Pydantic v2
class _ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_orm(cls, obj):
        return cls.model_validate(obj)

# ---- Decision schemas ----
class DecisionCreate(BaseModel):
    play_id: Optional[str] = None
    play_title: Optional[str] = None
    status: Optional[str] = None
    rationale: Optional[str] = None
    actor: Optional[str] = None

class DecisionOut(_ORMModel):
    id: Optional[int] = None
    play_id: Optional[str] = None
    play_title: Optional[str] = None
    status: Optional[str] = None
    rationale: Optional[str] = None
    actor: Optional[str] = None
    ts: Optional[str] = None

# ---- Activity schemas ----
class ActivityOut(_ORMModel):
    id: Optional[int] = None
    ts: Optional[str] = None
    action: Optional[str] = None
    play_title: Optional[str] = None
    target: Optional[str] = None
    status: Optional[str] = None

