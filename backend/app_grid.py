import pandas as pd
import streamlit as st
from services.backend_adapter import make_backend

st.set_page_config(page_title="Transform AI — Grid", layout="wide")
st.title("Transform AI — Grid")

# --- Sidebar: choose backend or mock ---
st.sidebar.header("Backend")
backend_url = st.sidebar.text_input("Backend URL (leave blank for mock)", value="")
use_auth = st.sidebar.checkbox("Send API Key", value=bool(backend_url))
api_key = st.sidebar.text_input("API Key", value="transformai-dev-001" if use_auth else "", type="password")

backend = make_backend(backend_url, api_key if use_auth else None)

# --- Health ---
colh1, colh2 = st.columns([1,3])
with colh1:
    if st.button("Check Health"):
        try:
            health = backend.health()
            st.success(f"OK — mode: {health.get('mode')}")
            st.json(health)
        except Exception as e:
            st.error(f"Health failed: {e}")

# --- keep some state ---
if "grid_id" not in st.session_state: st.session_state["grid_id"] = ""
if "project_id" not in st.session_state: st.session_state["project_id"] = "proj_123"

st.divider()

# --- Create grid ---
st.subheader("1) Create grid")
with st.form("create_grid_form"):
    c1,c2,c3 = st.columns([1,1,1])
    with c1: project_id = st.text_input("Project ID", value=st.session_state["project_id"])
    with c2: grid_name   = st.text_input("Grid Name", value="Alpha CDD")
    with c3: metric_name = st.text_input("Metric (column)", value="Cohort retention")

    r1 = st.text_input("Row ref (doc/entity)", value="doc:transactions.csv")
    if st.form_submit_button("Create Grid"):
        try:
            res = backend.create_grid(project_id, grid_name,
                                      columns=[{"name": metric_name, "kind":"metric", "tool":"cohort_retention"}],
                                      rows=[{"row_ref": r1}])
            st.session_state["grid_id"] = res["id"]
            st.session_state["project_id"] = project_id
            st.success(f"Created grid: {res['id']}")
            st.json(res)
        except Exception as e:
            st.error(f"Create grid failed: {e}")

st.divider()

# --- List + run cells ---
st.subheader("2) Cells")
grid_id = st.text_input("Grid ID", value=st.session_state.get("grid_id",""))
bcol1, bcol2, bcol3 = st.columns([1,1,6])

with bcol1:
    if st.button("Refresh Cells"):
        try:
            cells = backend.list_cells(grid_id) if grid_id else []
            st.session_state["cells_df"] = pd.DataFrame(cells) if cells else pd.DataFrame()
        except Exception as e:
            st.error(f"List cells failed: {e}")

with bcol2:
    if st.button("Run Cells"):
        try:
            res = backend.run_cells(grid_id)
            st.toast(f"Ran: {res}")
            cells = backend.list_cells(grid_id)
            st.session_state["cells_df"] = pd.DataFrame(cells)
        except Exception as e:
            st.error(f"Run failed: {e}")

df = st.session_state.get("cells_df", pd.DataFrame())
if not df.empty:
    st.dataframe(df, use_container_width=True, height=320)
else:
    st.info("No cells to show. Create a grid, then Refresh.")

st.divider()

# --- Approve & Memo (works in mock; HTTP version calls backend if available) ---
st.subheader("3) Approvals & Memo")
sel_cell_id = st.text_input("Cell ID to approve", value=df["id"].iloc[0] if not df.empty else "")
acols = st.columns([1,1,2,6])
with acols[0]:
    if st.button("Approve Cell"):
        try:
            out = backend.approve_cell(sel_cell_id, note="Looks good.")
            st.success(out)
        except Exception as e:
            st.error(f"Approve failed: {e}")
with acols[1]:
    if st.button("Compose Memo"):
        try:
            memo = backend.memo(st.session_state["project_id"])
            st.json(memo)
        except Exception as e:
            st.error(f"Memo failed: {e}")
with acols[2]:
    if st.button("Export PDF"):
        try:
            out = backend.export_pdf(st.session_state["project_id"])
            st.success(out)
        except Exception as e:
            st.error(f"Export failed: {e}")

st.caption("Tip: leave Backend URL empty to use the built-in mock backend. Paste your public FastAPI URL here later to go live.")
