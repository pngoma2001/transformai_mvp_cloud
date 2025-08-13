# TransformAI Backend (FastAPI)
Run locally:
```bash
pip install -r requirements.txt
uvicorn fastapi_app:app --reload --port 8000
```
Endpoints:
- POST /analyze (form-data: sample=RetailCo|HealthCo or file=csv)
- POST /decision { play_id, play_title, status, rationale, actor }
- GET /decisions
- POST /integrations/push { play_title, target }
- GET /activity
