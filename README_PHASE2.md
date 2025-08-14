# TransformAI — Phase 2 Enhancements

This package adds:
- **FastAPI backend** with JSON Schema validation and SQLite persistence.
- **Optional backend usage** in Streamlit (enter backend URL in sidebar).
- **KPI trend chart** on the summary screen.
- **SQLite** storage for decisions and activity when using the backend.

## Running the Backend (local)
```bash
cd backend
pip install -r requirements.txt
uvicorn fastapi_app:app --reload --port 8000
```
The first run creates `transformai.db` in the project root.

## Using the Backend from Streamlit
- In the Streamlit sidebar, set **Backend URL** to `http://localhost:8000` (or your hosted URL).
- Analyze a company, save decisions, and push — these are persisted to SQLite via the backend.
- You can retrieve records via:
  - `GET /decisions`
  - `GET /activity`

## Deploy
- **Streamlit Cloud:** You can still deploy the Streamlit app as before. (The backend is optional and usually run locally or on a small VM.)
- **Backend hosting:** If you want the Streamlit Cloud app to use the backend, host the FastAPI server (e.g., Render, Fly, Railway) and paste its URL into the Streamlit sidebar.

