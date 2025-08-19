import time
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Transform AI — Grid (beta)", layout="wide")
st.title("Transform AI — Grid (beta)")

# --- Sidebar: backend config ---
st.sidebar.header("Backend")
backend_url = st.sidebar.text_input("Backend URL", value="http://127.0.0.1:8000")
use_auth = st.sidebar.checkbox("Send API Key header", value=False)
api_key = st.sidebar.text_input("API Key (optional)", value="dev-key-123" if use_auth else "", type="password")

def _headers():
    if use_auth and api_key:
        return {"x-api-key": api_key, "Content-Type": "application/json"}
    return {"Content-Type": "application/json"}

# --- Health check ---
health_col1, health_col2 = st.columns([1, 3])
with health_col1:
    if st.button("Check Health"):
        try:
            r = requests.get(f"{backend_url}/health", timeout=8)
            r.raise_for_status()
            st.success("Backend OK")
            st.json(r.json())
        except Exception as e:
            st.error(f"Health check failed: {e}")

# --- Session state for grid id ---
if "grid_id" not in st.session_state:
    st.session_state["grid_id"] = None

st.divider()

# --- Create Sample Grid ---
st.subheader("1) Create a sample grid")

with st.form("create_grid_form"):
    project_id = st.text_input("Project ID", value="proj_123")
    grid_name = st.text_input("Grid Name", value="Alpha CDD")
    # For MVP we use one metric column and one row referencing a CSV/doc
    col_name = st.text_input("Column (metric name)", value="Cohort retention")
    row_ref = st.text_input("Row ref (doc/entity)", value="doc:transactions.csv")

    submitted = st.form_submit_button("Create Grid")
    if submitted:
        try:
            payload = {
                "project_id": project_id,
                "name": grid_name,
                "columns": [
                    {"name": col_name, "kind": "metric", "tool": "cohort_retention", "params": {"window": "monthly"}}
                ],
                "rows": [{"row_ref": row_ref}],
            }
            r = requests.post(f"{backend_url}/grid", json=payload, headers=_headers(), timeout=15)
            if r.status_code == 501:
                st.error("Grid feature is disabled on the backend. Set FF_GRID_RUNTIME=true and restart the server.")
            r.raise_for_status()
            grid = r.json()
            st.session_state["grid_id"] = grid["id"]
            st.success(f"Grid created ✅  (grid_id: {grid['id']})")
            st.json(grid)
        except Exception as e:
            st.error(f"Create Grid failed: {e}")

st.divider()

# --- List Cells ---
st.subheader("2) View cells for the grid")
grid_id_input = st.text_input("Grid ID", value=st.session_state.get("grid_id") or "")
col_a, col_b = st.columns([1, 3])
with col_a:
    if st.button("Refresh Cells"):
        if not grid_id_input:
            st.warning("Enter a grid_id first (create one above).")
        else:
            try:
                r = requests.get(f"{backend_url}/cells", params={"grid_id": grid_id_input}, headers=_headers(), timeout=15)
                r.raise_for_status()
                cells = r.json()
                if not isinstance(cells, list):
                    st.error("Unexpected response for /cells.")
                else:
                    df = pd.DataFrame(cells)
                    if df.empty:
                        st.info("No cells yet.")
                    else:
                        st.success(f"Loaded {len(df)} cells")
                        st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"List Cells failed: {e}")

# --- Friendly tips ---
st.caption(
    "Tips: If you get 501 on /grid, the backend flag FF_GRID_RUNTIME is off. "
    "If you turned off auth (DISABLE_AUTH=true), uncheck 'Send API Key header'."
)
