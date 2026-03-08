"""
Streamlit frontend for the Databricks Notebook Generator.

Communicates with the FastAPI backend (default: http://localhost:8000).
Provides:
  - API key + backend URL configuration in the sidebar
  - Requirements text area for user input
  - Real-time polling of job status with a live progress log
  - Expandable sections for the Execution Plan, Generated Code, Architect Feedback
  - One-click .ipynb download
  - Raw notebook JSON preview

Run with:
    streamlit run frontend/app.py
"""

import time
import json

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Databricks Notebook Generator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar – configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    api_base_url = st.text_input(
        "Backend API URL",
        value="http://localhost:8000",
        help="Base URL of the running FastAPI backend.",
    )

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help=(
            "Your OpenAI API key. This is sent to the backend as a header "
            "so the LangGraph pipeline can call the LLM. "
            "The key is NEVER stored on disk by this app."
        ),
    )

    openai_model = st.selectbox(
        "Model",
        options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0,
        help="OpenAI model to use for all agents.",
    )

    poll_interval = st.slider(
        "Poll interval (seconds)",
        min_value=2,
        max_value=15,
        value=4,
        help="How often to refresh job status while the pipeline runs.",
    )

    st.divider()
    st.markdown(
        """
**Workflow**
1. Enter requirements below
2. Click **Generate Notebook**
3. Watch agent progress in real time
4. Download the `.ipynb` file

**Agents in pipeline**
- 🗂 Product Manager
- 👨‍💻 Data Engineer
- 🏛 Senior Architect (review loop ≤ 3×)
- 🚀 DevOps Assembler
        """
    )

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("⚡ Databricks Notebook Generator")
st.caption("Powered by LangGraph · FastAPI · Streamlit · GPT-4o")
st.divider()

# -- Requirements input area
st.subheader("📋 Requirements")
st.markdown(
    "Enter plain-English ETL requirements **or** paste legacy Hadoop / Java Spark code to modernise."
)

example_requirements = (
    "Create a daily ETL pipeline that:\n"
    "1. Reads customer records (JSON) from S3 bucket s3://my-company/customers/raw/\n"
    "2. Cleans the email column (strip whitespace, lowercase, remove nulls)\n"
    "3. Adds an audit column 'ingestion_timestamp' using current_timestamp()\n"
    "4. Upserts the cleaned data into a Delta table at dbfs:/delta/customers/\n"
    "   using customer_id as the merge key\n"
    "5. Sends an alert via a webhook secret if the row count drops below 1000\n"
    "Secrets are managed in the 'prod-scope' Databricks secret scope."
)

requirements_input = st.text_area(
    label="Requirements",
    value=example_requirements,
    height=220,
    label_visibility="collapsed",
    placeholder="Describe the Databricks ETL job or paste legacy Spark logic here...",
)

col_btn, col_clear = st.columns([2, 8])
with col_btn:
    generate_clicked = st.button(
        "🚀 Generate Notebook",
        type="primary",
        use_container_width=True,
        disabled=not requirements_input.strip(),
    )
with col_clear:
    if st.button("🗑 Clear", use_container_width=False):
        st.session_state.pop("job_id", None)
        st.session_state.pop("job_done", None)
        st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Helper: call the API
# ---------------------------------------------------------------------------

def _headers() -> dict:
    """Build request headers, including the OpenAI key for the backend."""
    h = {"Content-Type": "application/json"}
    if openai_api_key:
        # The backend reads this header to set OPENAI_API_KEY env-var for the request.
        # See the note in main.py – for security, prefer setting the env-var server-side.
        h["X-OpenAI-Api-Key"] = openai_api_key
    return h


def _submit_job(requirements: str) -> str | None:
    """POST /generate and return the job_id, or None on error."""
    try:
        payload = {
            "requirements": requirements,
            "openai_api_key": openai_api_key,  # sent securely in JSON body over HTTPS
        }
        resp = requests.post(
            f"{api_base_url}/generate",
            json=payload,
            headers=_headers(),
            timeout=30,
        )
        if resp.status_code == 202:
            return resp.json()["job_id"]
        st.error(f"API error {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ Could not connect to the backend at **{api_base_url}**. "
            "Make sure the FastAPI server is running:\n\n"
            "```bash\nuvicorn backend.main:app --reload --port 8000\n```"
        )
    return None


def _poll_status(job_id: str) -> dict | None:
    """GET /job/{job_id}/status and return the JSON response dict, or None on error."""
    try:
        resp = requests.get(
            f"{api_base_url}/job/{job_id}/status",
            headers=_headers(),
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Status poll error {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError:
        st.warning("⏳ Waiting for backend response...")
    return None


def _get_notebook_content(job_id: str) -> str | None:
    """Fetch the raw .ipynb JSON text for preview."""
    try:
        resp = requests.get(
            f"{api_base_url}/job/{job_id}/notebook_content",
            headers=_headers(),
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("notebook_json", "")
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------
if generate_clicked:
    if not openai_api_key:
        st.warning("⚠️ Please enter your OpenAI API Key in the sidebar before generating.")
        st.stop()

    with st.spinner("Submitting job to the backend..."):
        job_id = _submit_job(requirements_input)

    if job_id:
        st.session_state["job_id"] = job_id
        st.session_state["job_done"] = False
        st.success(f"✅ Job submitted! **Job ID:** `{job_id}`")

# ---------------------------------------------------------------------------
# Status polling & result display
# ---------------------------------------------------------------------------
if "job_id" in st.session_state:
    job_id = st.session_state["job_id"]

    if not st.session_state.get("job_done", False):
        # -- Live progress section -------------------------------------------
        st.subheader("📡 Pipeline Progress")
        progress_placeholder = st.empty()
        log_placeholder = st.empty()

        with st.spinner("Multi-agent pipeline is running... This may take 1–3 minutes."):
            while True:
                status_data = _poll_status(job_id)

                if status_data is None:
                    time.sleep(poll_interval)
                    continue

                status = status_data.get("status", "running")

                # Render live log
                log_entries = status_data.get("status_log", [])
                if log_entries:
                    log_text = "\n".join(f"• {entry}" for entry in log_entries)
                    log_placeholder.code(log_text, language="")

                if status == "running":
                    progress_placeholder.info("🔄 Pipeline running – agents working...")
                    time.sleep(poll_interval)
                    continue

                elif status == "error":
                    progress_placeholder.error(
                        f"❌ Pipeline failed: {status_data.get('error', 'Unknown error')}"
                    )
                    st.stop()

                elif status == "done":
                    progress_placeholder.success("✅ Pipeline complete!")
                    st.session_state["job_done"] = True
                    st.session_state["status_data"] = status_data
                    break

    # -- Results display -------------------------------------------------------
    if st.session_state.get("job_done"):
        status_data = st.session_state.get("status_data", {})

        # Summary metrics
        st.subheader("📊 Pipeline Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Revision Cycles", status_data.get("revision_count", 0))
        col2.metric(
            "Architect Verdict",
            "✅ Approved" if status_data.get("is_approved") else "⚠️ Fail-safe",
        )

        notebook_path = status_data.get("final_notebook_path", "")
        col3.metric(
            "Notebook Generated",
            "Yes" if notebook_path else "No",
        )
        col4.metric("Status", status_data.get("status", "").capitalize())

        st.divider()

        # -- Execution Plan
        with st.expander("🗂 Execution Plan (Product Manager)", expanded=True):
            plan = status_data.get("plan", "")
            if plan:
                st.markdown(plan)
            else:
                st.info("Plan not available.")

        # -- Generated Code
        with st.expander("👨‍💻 Generated PySpark Code (Data Engineer)", expanded=False):
            draft_code = status_data.get("draft_code", "")
            if draft_code:
                st.code(draft_code, language="python")
            else:
                st.info("Code not available.")

        # -- Architect Feedback (shown only if code was flagged)
        review_feedback = status_data.get("review_feedback", "")
        if review_feedback:
            with st.expander("🏛 Last Architect Review Feedback", expanded=False):
                st.warning(review_feedback)

        # -- Activity Log
        with st.expander("📋 Full Activity Log", expanded=False):
            log_entries = status_data.get("status_log", [])
            for entry in log_entries:
                st.text(f"• {entry}")

        st.divider()

        # -- Download / Preview
        if notebook_path:
            st.subheader("💾 Download Notebook")

            dl_col, preview_col = st.columns([1, 3])

            with dl_col:
                # Fetch the raw .ipynb bytes for the download button
                try:
                    dl_resp = requests.get(
                        f"{api_base_url}/job/{job_id}/download",
                        headers=_headers(),
                        timeout=30,
                    )
                    if dl_resp.status_code == 200:
                        from pathlib import Path as _Path
                        nb_filename = _Path(notebook_path).name
                        st.download_button(
                            label="⬇️ Download .ipynb",
                            data=dl_resp.content,
                            file_name=nb_filename,
                            mime="application/octet-stream",
                            use_container_width=True,
                        )
                    else:
                        st.error("Could not fetch notebook for download.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Download error: {e}")

            with preview_col:
                st.caption(f"Saved on server at: `{notebook_path}`")

            # -- Raw JSON preview
            with st.expander("🔍 Raw Notebook JSON Preview", expanded=False):
                nb_content = _get_notebook_content(job_id)
                if nb_content:
                    try:
                        nb_dict = json.loads(nb_content)
                        # Pretty-print a compact view of the cells
                        cell_count = len(nb_dict.get("cells", []))
                        st.caption(f"Notebook contains **{cell_count} cells**")
                        st.code(
                            json.dumps(nb_dict, indent=2)[:8000] + (
                                "\n... (truncated for display)" if len(nb_content) > 8000 else ""
                            ),
                            language="json",
                        )
                    except json.JSONDecodeError:
                        st.code(nb_content[:5000])
                else:
                    st.info("Preview not available.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Databricks Notebook Generator · Multi-Agent LangGraph System · "
    "Built with LangChain, LangGraph, FastAPI & Streamlit"
)
