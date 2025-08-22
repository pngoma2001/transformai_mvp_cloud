# TransformAI Roadmap

## North Star
A diligence grid where rows are companies/docs/tables, columns are investor questions, and each cell is an auditable agent run that:
- pulls/structures data,
- runs real math (cohorts, elasticity, unit economics, etc.),
- outputs prose + numbers with citations.

Depth like Keye, provenance like Hebbia, throughput like Paradigm.

---

## Phase 1 — Throughput & Trust (Now)
**Goal:** Feels instant and professional; every number traceable.

- Streaming cell status, error-only view, approve/retry workflow
- Template grids (CDD/QofE), saved views
- FastAPI: `/grid`, `/grid/run`, `/cells`, `/modules/:run`, `/ingest`
- Worker queue + Postgres + Alembic, artifact store
- SSE/WebSocket for live updates
- Decision log tied to evidence snapshots

**Acceptance:**
- 500+ cells run reliably; <1% error
- All memo claims have at least one citation

---

## Phase 2 — Evidence Graph & Connectors
**Goal:** Turn messy rooms into structured, cited evidence.

- Connectors: Drive/Dropbox/Box/SharePoint
- Auto-classify docs; OCR fallback; table detection
- Entity table (customers/SKUs/plans) + ID resolution
- Evidence graph (doc → extraction → entity → cell lineage)
- “Show lineage” in UI

**Acceptance:**
- KPI PDF + transactions CSV → starter columns complete with citations in minutes

---

## Phase 3 — Quant Modules v2
**Goal:** Investor-grade analyses beyond summaries.

- Unit economics (CM1/CM2), LTV/CAC, payback
- Stockouts/lead-time, utilization/turns
- Cohort segmentation (plan/region/ACV)
- Anomaly guardrails; “Explain this number” (formula + rows + citations)
- Elasticity simulator (+5% price impact by segment)

**Acceptance:**
- Each module returns KPIs + 1–2 charts + narrative + citations; passes fixtures

---

## Phase 4 — Memo Composer v2
**Goal:** Board-ready memo in one click.

- Section templates (Exec, Market, Customer, Pricing, Ops, Risks)
- Red-flag box auto-filled from cross-checks
- Comparative memos (A vs B) with small multiples
- Appendix builder; export to branded PDF/DOCX

**Acceptance:**
- Partners can verify any figure via citations/appendix quickly

---

## Phase 5 — Enterprise polish
**Goal:** Secure, scalable, deployable.

- API keys per project, rate-limits, PII scrubbing
- SSO (Google/MS) and project roles
- Horizontal workers, cost telemetry, budgets
- Packaging tiers: Starter, Pro, Enterprise

---

## KPIs
- Cells/min, time-to-first-insight, first-pass approval rate
- % columns with citations, % of checklist auto-answered
- # red flags surfaced pre-review

## Non-goals (for now)
- Giant “chat the entire room” detours
- Exotic connectors; keep to 1–2 critical ones first
- Over-engineered auth/infra before pilots

