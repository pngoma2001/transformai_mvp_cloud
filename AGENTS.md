# AGENTS.md

## Mission
Implement a *due-diligence operator* (grid runtime, quant modules, memo composer) with audit-grade provenance.

## Tech Constraints
- Python 3.11
- FastAPI + SQLAlchemy (SQLite for dev, Postgres-ready)
- Streamlit UI (ok to add small React later)
- Enforce `x-api-key` on all routes

## Run Commands
- Backend: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`
- Frontend: `streamlit run app.py`
- Tests: `pytest -q`

## Implementation Order (PRs)
0. Feature flags + rate limits
1. Grid schema & CRUD
2. Ingest (CSV/PDF)
3. Run engine (queue + statuses)
4. Module: `cohort_retention`
5. Approvals + provenance
6. Memo composer + export
7. UI grid; then metrics/guardrails

## Definitions of Done
- Each PR adds tests; CI must pass.
- No secrets in code. Use env vars (`API_KEY`, `DATABASE_URL`).
- Only **approved** cells appear in memo/export.
- Every numeric claim has a `citations[]` pointer.

## Style & Tools
- Format: black + isort; lint with ruff.
- Type hints (mypy tolerant).
- Commit messages: feat/fix/chore + short scope.

## Do Not
- Don’t commit generated PDFs, datasets, or keys.
- Don’t bypass auth middleware.
