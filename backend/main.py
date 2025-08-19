import os, json, time
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Depends, Header, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from jsonschema import validate, ValidationError
from sqlalchemy.orm import Session

from .db import SessionLocal, engine, Base
from . import models, schemas
from .engine_logic import run_analysis

# Create tables
Base.metadata.create_all(bind=engine)
API_KEY = os.getenv("API_KEY", "changeme")

app = FastAPI(title="TransformAI Backend", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def require_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}

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
