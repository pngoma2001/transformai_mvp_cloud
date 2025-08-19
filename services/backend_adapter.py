# services/backend_adapter.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time, uuid, requests

# ---------- Contract (keep these method names stable) ----------
class BaseBackend:
    def health(self) -> Dict[str, Any]: ...
    def create_grid(self, project_id: str, name: str,
                    columns: List[Dict[str, Any]],
                    rows: List[Dict[str, Any]]) -> Dict[str, Any]: ...
    def list_cells(self, grid_id: str) -> List[Dict[str, Any]]: ...
    def run_cells(self, grid_id: str) -> Dict[str, Any]: ...
    def approve_cell(self, cell_id: str, note: str = "") -> Dict[str, Any]: ...
    def memo(self, project_id: str) -> Dict[str, Any]: ...
    def export_pdf(self, project_id: str, template: str = "ic_default_v1") -> Dict[str, Any]: ...

# ---------- Mock implementation (no backend required) ----------
@dataclass
class MockBackend(BaseBackend):
    latency_ms: int = 150
    _grids: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _cells: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    _approved: set = field(default_factory=set)

    def _sleep(self): 
        time.sleep(self.latency_ms/1000.0)

    def health(self) -> Dict[str, Any]:
        self._sleep()
        return {"ok": True, "flags": {"FF_GRID_RUNTIME": True, "FF_MEMO": True}, "mode": "mock"}

    def create_grid(self, project_id: str, name: str, columns, rows):
        self._sleep()
        gid = str(uuid.uuid4())
        self._grids[gid] = {"id": gid, "project_id": project_id, "name": name}
        cells = []
        for r in rows:
            for c in columns:
                cells.append({
                    "id": str(uuid.uuid4()),
                    "row_ref": r["row_ref"],
                    "col_name": c["name"],
                    "status": "queued",
                    "output_text": None,
                    "numeric_value": None,
                    "units": None,
                    "citations": [],
                })
        self._cells[gid] = cells
        return self._grids[gid]

    def list_cells(self, grid_id: str) -> List[Dict[str, Any]]:
        self._sleep()
        return list(self._cells.get(grid_id, []))

    def run_cells(self, grid_id: str) -> Dict[str, Any]:
        self._sleep()
        changed = 0
        for c in self._cells.get(grid_id, []):
            if c["status"] in ("queued","error"):
                c["status"] = "done"
                c["output_text"] = f"{c['col_name']} looks healthy (mock)."
                c["numeric_value"] = 0.71
                c["units"] = "ratio"
                changed += 1
        return {"ok": True, "updated": changed}

    def approve_cell(self, cell_id: str, note: str = "") -> Dict[str, Any]:
        self._sleep()
        self._approved.add(cell_id)
        return {"ok": True, "cell_id": cell_id, "note": note}

    def memo(self, project_id: str) -> Dict[str, Any]:
        self._sleep()
        # trivial memo from approved cells
        bullets = []
        for cells in self._cells.values():
            for c in cells:
                if c["id"] in self._approved:
                    bullets.append(f"• {c['col_name']}: {c.get('output_text') or 'n/a'}")
        return {"sections": [
            {"title": "Executive Summary", "content": "Mock memo for demo."},
            {"title": "Evidence", "content": "\n".join(bullets) or "No approved cells yet."}
        ]}

    def export_pdf(self, project_id: str, template: str = "ic_default_v1") -> Dict[str, Any]:
        self._sleep()
        # In mock, just return a fake URL
        return {"ok": True, "url": f"mock://export/{project_id}/{template}.pdf"}

# ---------- HTTP implementation (safe fallback if endpoint missing) ----------
@dataclass
class HttpBackend(BaseBackend):
    base_url: str
    api_key: Optional[str] = None
    timeout_s: float = 20.0

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def _get(self, path: str, **kw):
        return requests.get(self.base_url+path, headers=self._headers(), timeout=self.timeout_s, **kw)

    def _post(self, path: str, json=None, **kw):
        return requests.post(self.base_url+path, json=json, headers=self._headers(), timeout=self.timeout_s, **kw)

    def health(self):
        r = self._get("/health")
        r.raise_for_status()
        out = r.json()
        out["mode"] = "http"
        return out

    def create_grid(self, project_id, name, columns, rows):
        r = self._post("/grid", json={"project_id": project_id, "name": name, "columns": columns, "rows": rows})
        if r.status_code == 501:  # feature off
            raise RuntimeError("GRID_RUNTIME is disabled on backend.")
        r.raise_for_status()
        return r.json()

    def list_cells(self, grid_id):
        r = self._get("/cells", params={"grid_id": grid_id})
        r.raise_for_status()
        return r.json()

    def run_cells(self, grid_id):
        # If backend has `/grid/{id}/run`, use it; otherwise simulate with client-side assumption.
        r = self._post(f"/grid/{grid_id}/run", json={})
        if r.status_code in (404,405):  # endpoint not implemented yet
            return {"ok": True, "updated": 0, "note": "run endpoint not implemented"}
        r.raise_for_status()
        return r.json()

    def approve_cell(self, cell_id: str, note: str = ""):
        # Try real endpoint, else pretend success
        r = self._post(f"/cells/{cell_id}/approve", json={"note": note})
        if r.status_code in (404,405):
            return {"ok": True, "cell_id": cell_id, "note": note, "simulated": True}
        r.raise_for_status()
        return r.json()

    def memo(self, project_id: str):
        r = self._get("/memo", params={"project_id": project_id})
        if r.status_code in (404,405):
            return {"sections":[{"title":"Memo","content":"Backend memo not implemented yet."}]}
        r.raise_for_status()
        return r.json()

    def export_pdf(self, project_id: str, template: str = "ic_default_v1"):
        r = self._post("/export", json={"project_id": project_id, "template": template, "include_appendix": True})
        if r.status_code in (404,405):
            return {"ok": True, "url": "mock://export.pdf", "simulated": True}
        r.raise_for_status()
        return r.json()

# ---------- Factory ----------
def make_backend(backend_url: Optional[str], api_key: Optional[str]) -> BaseBackend:
    if backend_url and backend_url.strip():
        return HttpBackend(base_url=backend_url.rstrip("/"), api_key=api_key or None)
    return MockBackend()
