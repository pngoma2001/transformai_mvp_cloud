import pandas as pd
import streamlit as st
from services.backend_adapter import make_backend

st.set_page_config(page_title="Transform AI — Diligence Grid", layout="wide")
st.title("Diligence Grid")

# Sidebar controls
st.sidebar.header("Backend")
backend_url = st.sidebar.text_input("Backend URL (leave blank for mock)", value="")
use_auth = st.sidebar.checkbox("Send API Key", value=bool(backend_url))
api_key = st.sidebar.text_input("API Key", value="transformai-dev-001" if use_auth else "", type="password")

backend = make_backend(backend_url, api_key if use_auth else None)

# Health
if st.button("Check Health"):
    try:
        st.json(backend.health())
    except Exception as e:
        st.error(f"Health failed: {e}")

st.divider()

# Keep minimal state
if "grid_id" not in st.session_state: st.session_state["grid_id"] = ""
if "project_id" not in st.session_state: st.session_state["project_id"] = "proj_123"

# Create Grid
st.subheader("1) Create Grid")
with st.form("create_grid"):
    c1,c2,c3 = st.columns(3)
    with c1: project_id = st.text_input("Project ID", value=st.session_state["project_id"])
    with c2: grid_name   = st.text_input("Grid Name", value="Alpha CDD")
    with c3: metric_name = st.text_input("Metric (column)", value="Cohort retention")
    row_ref = st.text_input("Row ref (doc/entity)", value="doc:transactions.csv")
    if st.form_submit_button("Create"):
        try:
            res = backend.create_grid(project_id, grid_name,
                                      columns=[{"name":metric_name,"kind":"metric","tool":"cohort_retention"}],
                                      rows=[{"row_ref":row_ref}])
            st.session_state["grid_id"] = res["id"]
            st.session_state["project_id"] = project_id
            st.success(f"Grid: {res['id']}")
            st.json(res)
        except Exception as e:
            st.error(f"Create failed: {e}")

st.divider()

# Cells
st.subheader("2) Cells")
grid_id = st.text_input("Grid ID", value=st.session_state["grid_id"])
b1,b2,_ = st.columns([1,1,8])
with b1:
    if st.button("Refresh Cells"):
        try:
            cells = backend.list_cells(grid_id) if grid_id else []
            st.session_state["cells_df"] = pd.DataFrame(cells) if cells else pd.DataFrame()
        except Exception as e:
            st.error(f"List failed: {e}")
with b2:
    if st.button("Run Cells"):
        try:
            res = backend.run_cells(grid_id)
            st.toast(str(res))
            cells = backend.list_cells(grid_id)
            st.session_state["cells_df"] = pd.DataFrame(cells)
        except Exception as e:
            st.error(f"Run failed: {e}")

df = st.session_state.get("cells_df", pd.DataFrame())
if not df.empty:
    st.dataframe(df, use_container_width=True, height=320)
else:
    st.info("No cells yet. Create a grid above, then Refresh.")

st.divider()

# Approvals & Memo
st.subheader("3) Approve & Memo")
sel_cell_id = st.text_input("Cell ID to approve", value=(df["id"].iloc[0] if not df.empty else ""))
c1,c2,c3 = st.columns([1,1,2])
with c1:
    if st.button("Approve Cell"):
        try:
            st.success(backend.approve_cell(sel_cell_id, note="Looks good."))
        except Exception as e:
            st.error(f"Approve failed: {e}")
with c2:
    if st.button("Compose Memo"):
        try:
            st.json(backend.memo(st.session_state["project_id"]))
        except Exception as e:
            st.error(f"Memo failed: {e}")
with c3:
    if st.button("Export PDF"):
        try:
            st.success(backend.export_pdf(st.session_state["project_id"]))
        except Exception as e:
            st.error(f"Export failed: {e}")

st.caption("Leave Backend URL empty to use the built-in mock backend. Paste your FastAPI URL later to go live.")
