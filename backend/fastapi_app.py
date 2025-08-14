
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
import pandas as pd
import sqlite3, time, json, os
from pathlib import Path
from jsonschema import validate, ValidationError

from engine import analyze

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "analysis.schema.json"
analysis_schema = json.loads(open(SCHEMA_PATH).read())

DB_PATH = Path(__file__).resolve().parents[1] / "transformai.db"

def ensure_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        play_id TEXT,
        play_title TEXT,
        status TEXT,
        rationale TEXT,
        actor TEXT,
        ts TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS activity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        action TEXT,
        play_title TEXT,
        target TEXT,
        status TEXT
    )""")
    con.commit()
    con.close()

ensure_db()
app = FastAPI(title="TransformAI Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_endpoint(sample: Optional[str] = None, file: Optional[UploadFile] = File(None)) -> Dict[str, Any]:
    if file:
        df = pd.read_csv(file.file)
    else:
        # Use sample CSVs relative to project root
        data_dir = Path(__file__).resolve().parents[1] / "data"
        if (sample or "").lower() == "healthco":
            df = pd.read_csv(data_dir / "sample_healthco.csv")
        else:
            df = pd.read_csv(data_dir / "sample_retailco.csv")
    result = analyze(df)
    # Validate against JSON schema
    try:
        validate(instance=result, schema=analysis_schema)
    except ValidationError as e:
        return {"ok": False, "error": f"Schema validation failed: {e.message}"}
    return {"ok": True, "result": result}

@app.post("/decision")
async def decision_endpoint(payload: Dict[str, Any] = Body(...)):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""INSERT INTO decisions (play_id, play_title, status, rationale, actor, ts)
                   VALUES (?,?,?,?,?,?)""",
                (payload.get("play_id"), payload.get("play_title"), payload.get("status"),
                 payload.get("rationale",""), payload.get("actor","user"),
                 time.strftime("%Y-%m-%d %H:%M:%S")))
    con.commit(); con.close()
    return {"ok": True}

@app.get("/decisions")
async def list_decisions():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    rows = cur.execute("SELECT play_id, play_title, status, rationale, actor, ts FROM decisions ORDER BY id DESC").fetchall()
    con.close()
    keys = ["play_id","play_title","status","rationale","actor","ts"]
    return {"ok": True, "decisions": [dict(zip(keys,r)) for r in rows]}

@app.post("/integrations/push")
async def push_mock(payload: Dict[str, Any] = Body(...)):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""INSERT INTO activity (ts, action, play_title, target, status)
                   VALUES (?,?,?,?,?)""",
                (time.strftime("%Y-%m-%d %H:%M:%S"), "push", payload.get("play_title",""), payload.get("target","salesforce"), "success"))
    con.commit(); con.close()
    return {"ok": True, "job_id": f"job_{int(time.time())}", "status": "success"}

@app.get("/activity")
async def activity_log():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    rows = cur.execute("SELECT ts, action, play_title, target, status FROM activity ORDER BY id DESC").fetchall()
    con.close()
    keys = ["ts","action","play_title","target","status"]
    return {"ok": True, "activity": [dict(zip(keys,r)) for r in rows]}
