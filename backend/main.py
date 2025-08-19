import os, json, time
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Depends, Header, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jsonschema import validate, ValidationError
from sqlalchemy.orm import Session

from .db import SessionLocal, engine, Base
from . import models, schemas
from .engine_logic import run_analysis

# ------------------------------------------------------------------------------
# Boot & Config
# ------------------------------------------------------------------------------

# Create tables for existing models
Base.metadata.create_all(bind=engine)

# Auth key (set in env for prod)
API_KEY = os.getenv("API_KEY", "changeme")

# --- Feature flags (env-driven; OFF by default) ---
def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")

FF_GRID_RUNTIME = _env_bool("FF_GRID_RUNTIME", False)
FF_MODULES = _env_bool("FF_MODULES", False)
FF_MEMO = _env_bool("FF_MEMO", False)
FF_UI_GRID = _env_bool("FF_UI_GRID", False)

# Simple per-API-key rate limit (requests per minute)
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "120"))

# ------------------------------------------------------------------------------
# App & Middleware
# ------------------------------------------------------------------------------

app = FastAPI(title="TransformAI Backend", version="0.3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# DB session helper
# ------------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ------------------------------------------------------------------------------
# Auth + Rate limiting + Feature gating
# ------------------------------------------------------------------------------

# in-memory token buckets (per-process; fine for MVP)
from collections import deque
import threading
_hits_lock = threading.Lock()
_hits: Dict[str, deque] = {}

def _rate_limit(key: str, rpm: int) -> None:
    now = time.monotonic()
    with _hits_lock:
        q = _hits.setdefault(key, deque())
        # drop events older than 60s
        while q and (now - q[0]) > 60.0:
            q.popleft()
        if len(q) >= rpm:
            raise HTTPException(status_code=429, detail="Rate limit exceeded for this API key.")
        q.append(now)

def require_api_key(x_api_key: str = Header(...)) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    _rate_limit(x_api_key, RATE_LIMIT_RPM)

def feature_flag(enabled: bool, name: str):
    def _check():
        if not enabled:
            raise HTTPException(
                status_code=501,
                detail=f"{name} is disabled by feature flag."
            )
    return _check

# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "flags": {
            "FF_GRID_RUNTIME": FF_GRID_RUNTIME,
            "FF_MODULES": FF_MODULES,
            "FF_MEMO": FF_MEMO,
            "FF_UI_GRID": FF_UI_GRID,
        }
    }

# ------------------------------------------------------------------------------
# Existing endpoints (kept from your file)
# ------------------------------------------------------------------------------

@app.post("/analyze", dependencies=[Depends(require_api_key)])
async def analyze_endpoint(
    sample: Optional[str] = None,
    file: Optional[UploadFile] = File(None)
) -> Dict[str, Any]:
    if file:
        df = pd.read_csv(file.file)
    else:
        data_dir = Path(__file__).resolve().parents[1] / "data"
        if (sample or "").lower() == "healthco":
            df = pd.read_csv(data_dir / "sample_healthco.csv")
        else:
            df = pd.read_csv(data_dir / "sample_retailco.csv")

    result = run_analysis(df)

    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "analysis.schema.json"
    analysis_schema = json.loads(open(schema_path).read())
    try:
        validate(instance=result, schema=analysis_schema)
    except ValidationError as e:
        raise HTTPException(status_code=500, detail=f"Schema validation failed: {e.message}")
    return {"ok": True, "result": result}

@app.post("/decision", dependencies=[Depends(require_api_key)])
def save_decision(payload: schemas.DecisionCreate, db: Session = Depends(get_db)) -> Dict[str, bool]:
    dec = models.Decision(
        play_id=payload.play_id,
        play_title=payload.play_title,
        status=payload.status,
        rationale=payload.rationale,
        actor=payload.actor,
        ts=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    db.add(dec); db.commit()
    return {"ok": True}

@app.get("/decisions", dependencies=[Depends(require_api_key)])
def list_decisions(db: Session = Depends(get_db)) -> Dict[str, Any]:
    items = db.query(models.Decision).order_by(models.Decision.id.desc()).all()
    return {"ok": True, "decisions": [schemas.DecisionOut.from_orm(i) for i in items]}

@app.post("/integrations/push", dependencies=[Depends(require_api_key)])
def push_mock(payload: Dict[str, Any] = Body(...), db: Session = Depends(get_db)) -> Dict[str, Any]:
    act = models.Activity(
        ts=time.strftime("%Y-%m-%d %H:%M:%S"),
        action="push",
        play_title=payload.get("play_title", ""),
        target=payload.get("target", "salesforce"),
        status="success",
    )
    db.add(act); db.commit()
    return {"ok": True, "job_id": f"job_{int(time.time())}", "status": "success"}

@app.get("/activity", dependencies=[Depends(require_api_key)])
def activity_log(db: Session = Depends(get_db)) -> Dict[str, Any]:
    items = db.query(models.Activity).order_by(models.Activity.id.desc()).all()
    return {"ok": True, "activity": [schemas.ActivityOut.from_orm(i) for i in items]}

@app.post("/export", dependencies=[Depends(require_api_key)])
def export_placeholder() -> Dict[str, str]:
    return {"ok": True, "note": "export not yet implemented"}

@app.post("/import", dependencies=[Depends(require_api_key)])
def import_placeholder() -> Dict[str, str]:
    return {"ok": True, "note": "import not yet implemented"}

# ------------------------------------------------------------------------------
# NEW: Minimal in-memory Grid API (feature-gated) to unblock UI wiring
# ------------------------------------------------------------------------------

class ColumnSpec(BaseModel):
    name: str
    kind: str  # "metric" | "question"
    tool: Optional[str] = None
    params: Optional[Dict[str, Any]] = None

class RowRef(BaseModel):
    row_ref: str

class GridCreate(BaseModel):
    project_id: str
    name: str
    columns: List[ColumnSpec]
    rows: List[RowRef]

class CellOut(BaseModel):
    id: str
    row_ref: str
    col_name: str
    status: str

class GridOut(BaseModel):
    id: str
    project_id: str
    name: str

import uuid
_GRIDS: Dict[str, GridOut] = {}
_CELLS: Dict[str, List[CellOut]] = {}

grid_guard = [
    Depends(require_api_key),
    Depends(feature_flag(FF_GRID_RUNTIME, "GRID_RUNTIME"))
]

@app.get("/_feature_probe/grid", dependencies=grid_guard)
def probe_grid_feature():
    return {"status": "GRID_RUNTIME enabled"}

@app.post("/grid", dependencies=grid_guard, response_model=GridOut)
def create_grid(payload: GridCreate):
    gid = str(uuid.uuid4())
    _GRIDS[gid] = GridOut(id=gid, project_id=payload.project_id, name=payload.name)
    cells: List[CellOut] = []
    for r in payload.rows:
        for c in payload.columns:
            cells.append(CellOut(id=str(uuid.uuid4()), row_ref=r.row_ref, col_name=c.name, status="queued"))
    _CELLS[gid] = cells
    return _GRIDS[gid]

@app.get("/grid/{grid_id}", dependencies=grid_guard, response_model=GridOut)
def get_grid(grid_id: str):
    if grid_id not in _GRIDS:
        raise HTTPException(status_code=404, detail="grid not found")
    return _GRIDS[grid_id]

@app.get("/cells", dependencies=grid_guard, response_model=List[CellOut])
def list_cells(grid_id: str = Query(...)):
    return _CELLS.get(grid_id, [])

