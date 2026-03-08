"""
Streamlit frontend for the Databricks Notebook Generator.

Communicates with the FastAPI backend (default: http://localhost:8000).
Enhanced UI with:
  - Agent progress cards showing which agent is running
  - Real-time token usage tracking
  - Detailed agent communication logs
  - Model information display
  - Live pipeline progress bar

Run with:
    streamlit run frontend/app.py
"""

import time
import json
import os

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
# Custom CSS for enhanced UI
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .agent-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        background-color: #f8f9fa;
        text-align: center;
    }
    .agent-card-active {
        border: 2px solid #ff6b35;
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        background-color: #fff3e0;
        text-align: center;
    }
    .agent-card-done {
        border: 2px solid #4caf50;
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        background-color: #e8f5e9;
        text-align: center;
    }
    .token-badge {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.15rem;
    }
    .time-badge {
        display: inline-block;
        background-color: #fce4ec;
        color: #c62828;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.15rem;
    }
    .model-badge {
        display: inline-block;
        background-color: #f3e5f5;
        color: #6a1b9a;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.15rem;
    }
    .log-line {
        font-family: 'Courier New', monospace;
        font-size: 0.78rem;
        padding: 0.2rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Agent pipeline definition for progress display
# ---------------------------------------------------------------------------
AGENT_PIPELINE = [
    {"key": "product_manager", "name": "Product Manager (Planner)", "icon": "📋", "match": "Product Manager"},
    {"key": "data_engineer", "name": "Data Engineer (Coder)", "icon": "👨‍💻", "match": "Data Engineer"},
    {"key": "senior_architect", "name": "Senior Architect (Reviewer)", "icon": "🏗️", "match": "Senior Architect"},
    {"key": "devops", "name": "DevOps Assembler", "icon": "📦", "match": "DevOps"},
]


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
        help="Your OpenAI API key. Sent to backend for LLM calls. Never stored.",
    )

    openai_model = st.selectbox(
        "Model",
        options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0,
        help="OpenAI model to use for all agents.",
    )
    st.markdown(f"<span class='model-badge'>🤖 {openai_model}</span>", unsafe_allow_html=True)

    poll_interval = st.slider(
        "Poll interval (seconds)",
        min_value=2,
        max_value=15,
        value=4,
        help="How often to refresh job status while the pipeline runs.",
    )

    st.divider()
    st.markdown("### 🏗️ Agent Pipeline")
    st.markdown("""
1. **📋 Product Manager** — Plans steps  
2. **👨‍💻 Data Engineer** — Writes code  
3. **🏗️ Senior Architect** — Reviews (≤3×)  
4. **📦 DevOps Assembler** — Packages .ipynb  
    """)

    st.divider()
    st.markdown("### 📊 Session Stats")
    if "total_tokens" in st.session_state and st.session_state["total_tokens"]:
        tokens = st.session_state["total_tokens"]
        st.metric("Total Tokens Used", f"{tokens.get('total_tokens', 0):,}")
        st.metric("Prompt Tokens", f"{tokens.get('prompt_tokens', 0):,}")
        st.metric("Completion Tokens", f"{tokens.get('completion_tokens', 0):,}")
    else:
        st.info("Run a pipeline to see stats.")

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("⚡ Databricks Notebook Generator")
st.caption("Powered by LangGraph · FastAPI · Streamlit · Multi-Agent AI Pipeline")
st.divider()

# -- Requirements input area
st.subheader("📝 Enter Your ETL Requirement")
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
        h["X-OpenAI-Api-Key"] = openai_api_key
    return h


def _submit_job(requirements: str) -> str | None:
    """POST /generate and return the job_id, or None on error."""
    try:
        payload = {
            "requirements": requirements,
            "openai_api_key": openai_api_key,
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
            f"Could not connect to the backend at **{api_base_url}**. "
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
        st.warning("Waiting for backend response...")
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


def render_agent_progress(agent_logs: list, current_step: str = ""):
    """Render visual agent progress cards based on completed agent_logs."""
    cols = st.columns(4)
    for i, agent_def in enumerate(AGENT_PIPELINE):
        with cols[i]:
            match_key = agent_def["match"]
            # Find completed logs for this agent
            agent_entries = [l for l in agent_logs if match_key in l.get("agent", "")]
            is_done = len(agent_entries) > 0
            is_active = match_key.lower() in current_step.lower() if current_step else False

            if is_done:
                latest = agent_entries[-1]
                tokens = latest.get("token_usage", {})
                elapsed = latest.get("elapsed_seconds", 0)
                st.markdown(f"""
                <div class='agent-card-done'>
                    <div style='font-size:1.5rem;'>{agent_def['icon']}</div>
                    <div style='font-weight:600; font-size:0.85rem;'>Done</div>
                    <div style='font-size:0.75rem; color:#666;'>{agent_def['name'].split('(')[0].strip()}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(
                    f"<span class='token-badge'>{tokens.get('total_tokens', 0):,} tk</span>"
                    f"<span class='time-badge'>{elapsed}s</span>",
                    unsafe_allow_html=True,
                )
            elif is_active:
                st.markdown(f"""
                <div class='agent-card-active'>
                    <div style='font-size:1.5rem;'>{agent_def['icon']}</div>
                    <div style='font-weight:600; font-size:0.85rem; color:#e65100;'>Running...</div>
                    <div style='font-size:0.75rem; color:#666;'>{agent_def['name'].split('(')[0].strip()}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='agent-card'>
                    <div style='font-size:1.5rem; opacity:0.4;'>{agent_def['icon']}</div>
                    <div style='font-weight:600; font-size:0.85rem; color:#999;'>Waiting</div>
                    <div style='font-size:0.75rem; color:#bbb;'>{agent_def['name'].split('(')[0].strip()}</div>
                </div>
                """, unsafe_allow_html=True)


def render_agent_communication_logs(agent_logs: list):
    """Render detailed agent communication log entries in expandable sections."""
    if not agent_logs:
        st.info("No agent logs available yet.")
        return

    for i, log_entry in enumerate(agent_logs):
        agent = log_entry.get("agent", "Unknown")
        model = log_entry.get("model", "?")
        tokens = log_entry.get("token_usage", {})
        elapsed = log_entry.get("elapsed_seconds", 0)
        detail = log_entry.get("detail", "")
        timestamp = log_entry.get("timestamp", "")

        with st.expander(f"#{i+1}  {agent} — {detail}", expanded=(i == len(agent_logs) - 1)):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<span class='model-badge'>🤖 {model}</span>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<span class='token-badge'>{tokens.get('total_tokens', 0):,} tokens</span>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<span class='time-badge'>{elapsed}s</span>", unsafe_allow_html=True)

            st.markdown(f"**Timestamp:** `{timestamp}`  |  **Status:** `{log_entry.get('status', '')}`")
            if tokens:
                st.markdown(
                    f"Prompt: **{tokens.get('prompt_tokens', 0):,}** | "
                    f"Completion: **{tokens.get('completion_tokens', 0):,}** | "
                    f"Total: **{tokens.get('total_tokens', 0):,}**"
                )


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------
if generate_clicked:
    if not openai_api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar before generating.")
        st.stop()

    with st.spinner("Submitting job to the backend..."):
        job_id = _submit_job(requirements_input)

    if job_id:
        st.session_state["job_id"] = job_id
        st.session_state["job_done"] = False
        st.success(f"Job submitted! **Job ID:** `{job_id}`")

# ---------------------------------------------------------------------------
# Status polling & result display
# ---------------------------------------------------------------------------
if "job_id" in st.session_state:
    job_id = st.session_state["job_id"]

    if not st.session_state.get("job_done", False):
        # -- Live progress section -------------------------------------------
        st.divider()
        st.subheader("🔄 Pipeline Progress")

        # Agent progress cards
        progress_cards_placeholder = st.empty()
        # Status message
        status_msg_placeholder = st.empty()
        # Progress bar
        progress_bar_placeholder = st.empty()
        # Live log
        log_placeholder = st.empty()

        status_msg_placeholder.info("🚀 Multi-agent pipeline is running...")
        progress_bar_placeholder.progress(5, text="Initializing agents...")

        with st.spinner(""):
            poll_count = 0
            while True:
                status_data = _poll_status(job_id)

                if status_data is None:
                    time.sleep(poll_interval)
                    continue

                status = status_data.get("status", "running")
                agent_logs = status_data.get("agent_logs", [])
                log_entries = status_data.get("status_log", [])

                # Update progress cards
                if agent_logs:
                    current_step = ""
                    if log_entries:
                        last_log = log_entries[-1]
                        if "Product_Manager" in last_log or "Supervisor -> Product_Manager" in last_log:
                            current_step = "Product Manager"
                        elif "Data_Engineer" in last_log or "Supervisor -> Data_Engineer" in last_log:
                            current_step = "Data Engineer"
                        elif "Senior_Architect" in last_log or "Supervisor -> Senior_Architect" in last_log:
                            current_step = "Senior Architect"
                        elif "DevOps" in last_log or "Supervisor -> DevOps" in last_log:
                            current_step = "DevOps"

                    with progress_cards_placeholder.container():
                        render_agent_progress(agent_logs, current_step)

                # Update progress bar
                completed_count = len(agent_logs)
                progress_pct = min(5 + (completed_count * 22), 95) if status == "running" else 100
                if status == "running":
                    progress_bar_placeholder.progress(progress_pct, text=f"Agents completed: {completed_count} / ~4+")
                
                # Render live log
                if log_entries:
                    log_text = "\n".join(f"• {entry}" for entry in log_entries)
                    log_placeholder.code(log_text, language="")

                if status == "running":
                    time.sleep(poll_interval)
                    poll_count += 1
                    continue

                elif status == "error":
                    progress_bar_placeholder.progress(100, text="Pipeline failed!")
                    status_msg_placeholder.error(
                        f"Pipeline failed: {status_data.get('error', 'Unknown error')}"
                    )
                    st.stop()

                elif status == "done":
                    progress_bar_placeholder.progress(100, text="Pipeline complete!")
                    total_tokens = status_data.get("total_tokens", {})
                    st.session_state["total_tokens"] = total_tokens
                    status_msg_placeholder.success(
                        f"Pipeline completed! | "
                        f"Total tokens: {total_tokens.get('total_tokens', 0):,} | "
                        f"Revisions: {status_data.get('revision_count', 0)}"
                    )
                    # Final render of agent cards
                    with progress_cards_placeholder.container():
                        render_agent_progress(status_data.get("agent_logs", []))
                    st.session_state["job_done"] = True
                    st.session_state["status_data"] = status_data
                    break

    # -- Results display -------------------------------------------------------
    if st.session_state.get("job_done"):
        status_data = st.session_state.get("status_data", {})
        agent_logs = status_data.get("agent_logs", [])
        total_tokens = status_data.get("total_tokens", {})

        # Summary metrics
        st.divider()
        st.subheader("📊 Pipeline Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Revision Cycles", status_data.get("revision_count", 0))
        col2.metric(
            "Architect Verdict",
            "Approved" if status_data.get("is_approved") else "Fail-safe",
        )
        notebook_path = status_data.get("final_notebook_path", "")
        col3.metric("Notebook", "Generated" if notebook_path else "None")
        col4.metric("Total Tokens", f"{total_tokens.get('total_tokens', 0):,}")
        col5.metric("Status", status_data.get("status", "").capitalize())

        st.divider()

        # -- Tabbed results
        tab_plan, tab_code, tab_review, tab_agents, tab_download, tab_raw = st.tabs([
            "📋 Execution Plan",
            "👨‍💻 Generated Code",
            "🏗️ Review Result", 
            "🔍 Agent Logs & Tokens",
            "💾 Download Notebook",
            "📜 Activity Log",
        ])

        with tab_plan:
            plan = status_data.get("plan", "")
            if plan:
                st.markdown(plan)
            else:
                st.info("Plan not available.")

        with tab_code:
            draft_code = status_data.get("draft_code", "")
            if draft_code:
                st.code(draft_code, language="python", line_numbers=True)
            else:
                st.info("Code not available.")

        with tab_review:
            review_feedback = status_data.get("review_feedback", "")
            if status_data.get("is_approved"):
                st.success("Code was APPROVED by the Senior Architect!")
            if review_feedback:
                st.warning("Last review feedback:")
                st.markdown(review_feedback)
            elif not status_data.get("is_approved"):
                st.info("No review feedback available (fail-safe packaging).")

        with tab_agents:
            st.subheader("🔍 Agent Communication Log")
            render_agent_communication_logs(agent_logs)

            st.divider()
            st.subheader("📊 Token Usage Summary")
            tc1, tc2, tc3, tc4 = st.columns(4)
            with tc1:
                st.metric("Total Tokens", f"{total_tokens.get('total_tokens', 0):,}")
            with tc2:
                st.metric("Prompt Tokens", f"{total_tokens.get('prompt_tokens', 0):,}")
            with tc3:
                st.metric("Completion Tokens", f"{total_tokens.get('completion_tokens', 0):,}")
            with tc4:
                st.metric("Agent Invocations", len(agent_logs))

            # Per-agent breakdown table
            if agent_logs:
                st.divider()
                st.subheader("📈 Per-Agent Breakdown")
                for log_entry in agent_logs:
                    tokens = log_entry.get("token_usage", {})
                    st.markdown(
                        f"**{log_entry.get('agent', '')}** — "
                        f"{log_entry.get('detail', '')} | "
                        f"<span class='model-badge'>🤖 {log_entry.get('model', '')}</span> "
                        f"<span class='token-badge'>{tokens.get('total_tokens', 0):,} tokens</span> "
                        f"<span class='time-badge'>{log_entry.get('elapsed_seconds', 0)}s</span>",
                        unsafe_allow_html=True,
                    )

        with tab_download:
            if notebook_path:
                st.success(f"Notebook saved at: `{notebook_path}`")
                dl_col, preview_col = st.columns([1, 3])
                with dl_col:
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
                    except Exception as e:
                        st.error(f"Download error: {e}")

                # Raw JSON preview
                with st.expander("🔍 Raw Notebook JSON Preview", expanded=False):
                    nb_content = _get_notebook_content(job_id)
                    if nb_content:
                        try:
                            nb_dict = json.loads(nb_content)
                            cell_count = len(nb_dict.get("cells", []))
                            st.caption(f"Notebook contains **{cell_count} cells**")
                            st.code(
                                json.dumps(nb_dict, indent=2)[:8000] + (
                                    "\n... (truncated)" if len(nb_content) > 8000 else ""
                                ),
                                language="json",
                            )
                        except json.JSONDecodeError:
                            st.code(nb_content[:5000])
                    else:
                        st.info("Preview not available.")
            else:
                st.warning("No notebook was generated.")

        with tab_raw:
            st.subheader("📜 Full Activity Log")
            log_entries = status_data.get("status_log", [])
            if log_entries:
                for entry in log_entries:
                    st.text(f"• {entry}")
            else:
                st.info("No activity log available.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Databricks Notebook Generator · Multi-Agent LangGraph System · "
    "Built with LangChain, LangGraph, FastAPI & Streamlit"
)
