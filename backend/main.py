"""
FastAPI backend for the Databricks Notebook Generator.

Endpoints:
  POST /generate          – Run the full multi-agent pipeline; returns final state.
  GET  /download/{job_id} – Stream the generated .ipynb file to the caller.
  GET  /health            – Simple liveness probe.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

import os
import uuid
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# The compiled LangGraph workflow
from .graph import run_pipeline

logger = logging.getLogger("AgentPipeline")

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Databricks Notebook Generator",
    description=(
        "Multi-agent LangGraph system that converts plain-text requirements "
        "or legacy Hadoop/Spark code into fully-formed Databricks .ipynb notebooks."
    ),
    version="1.0.0",
)

# Allow the Streamlit frontend (running on localhost) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory job store
# Keyed by job_id (UUID str).  Each value is a dict with:
#   status : "running" | "done" | "error"
#   result : final state dict (populated when done)
#   error  : error message string (populated on exception)
# ---------------------------------------------------------------------------
_job_store: Dict[str, Dict[str, Any]] = {}
_job_store_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""
    requirements: str = Field(
        ...,
        min_length=10,
        description=(
            "Plain-text requirements for the Databricks ETL notebook to generate, "
            "or paste legacy Hadoop/Java Spark logic to be modernised."
        ),
        example=(
            "Create a daily ETL that reads customer JSON from s3://my-bucket/customers/, "
            "cleans the email column, and upserts into a Delta table at dbfs:/delta/customers."
        ),
    )
    openai_api_key: str = Field(
        default="",
        description=(
            "Optional OpenAI API key. If omitted, the server falls back to the "
            "OPENAI_API_KEY environment variable."
        ),
    )


class JobSubmittedResponse(BaseModel):
    """Returned immediately after a job is queued."""
    job_id: str
    message: str


class JobStatusResponse(BaseModel):
    """Returned by /job/{job_id}/status."""
    job_id: str
    status: str                       # "running" | "done" | "error"
    status_log: list = []
    plan: str = ""
    draft_code: str = ""
    review_feedback: str = ""
    is_approved: bool = False
    revision_count: int = 0
    final_notebook_path: str = ""
    error: str = ""
    agent_logs: list = []             # structured log entries per agent
    total_tokens: dict = {}           # cumulative token usage


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_job(job_id: str, requirements: str, openai_api_key: str = "") -> None:
    """
    Execute the LangGraph pipeline in a background thread.
    Updates _job_store with the result or error on completion.
    """
    logger.info(f"Job {job_id}: Starting pipeline...")
    try:
        final_state = run_pipeline(requirements, openai_api_key=openai_api_key)
        with _job_store_lock:
            _job_store[job_id]["status"] = "done"
            _job_store[job_id]["result"] = final_state
        logger.info(f"Job {job_id}: Pipeline completed successfully.")
    except Exception as exc:  # noqa: BLE001
        with _job_store_lock:
            _job_store[job_id]["status"] = "error"
            _job_store[job_id]["error"] = str(exc)
        logger.error(f"Job {job_id}: Pipeline FAILED — {exc}", exc_info=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health_check():
    """Liveness probe."""
    return {"status": "ok", "service": "Databricks Notebook Generator"}


@app.post(
    "/generate",
    response_model=JobSubmittedResponse,
    status_code=202,
    tags=["Pipeline"],
    summary="Submit a notebook-generation job",
)
def generate_notebook(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
) -> JobSubmittedResponse:
    """
    Accept user requirements, spin up a background thread that runs the
    full multi-agent LangGraph pipeline, and return a job_id for polling.

    The pipeline typically takes 60-180 seconds depending on model latency
    and the number of architect review cycles needed.
    """
    # Resolve the API key: prefer the one sent in the request body,
    # then fall back to the server-side OPENAI_API_KEY env var.
    resolved_key = request.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    if not resolved_key:
        raise HTTPException(
            status_code=400,
            detail=(
                "An OpenAI API key is required. Provide it in the request body "
                "(openai_api_key) or set OPENAI_API_KEY on the server."
            ),
        )

    job_id = str(uuid.uuid4())

    with _job_store_lock:
        _job_store[job_id] = {"status": "running", "result": None, "error": ""}

    # Run the pipeline in a background thread so FastAPI stays responsive
    background_tasks.add_task(_run_job, job_id, request.requirements, resolved_key)

    return JobSubmittedResponse(
        job_id=job_id,
        message="Job submitted. Poll /job/{job_id}/status for progress.",
    )


@app.get(
    "/job/{job_id}/status",
    response_model=JobStatusResponse,
    tags=["Pipeline"],
    summary="Poll job status and intermediate results",
)
def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Poll the status of a notebook-generation job.

    Returns:
      - status="running"  while the pipeline is executing
      - status="done"     when the notebook has been saved
      - status="error"    if the pipeline raised an unhandled exception
    """
    with _job_store_lock:
        job = _job_store.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job["status"] == "error":
        return JobStatusResponse(job_id=job_id, status="error", error=job["error"])

    if job["status"] == "running":
        return JobStatusResponse(job_id=job_id, status="running")

    # status == "done"
    result = job["result"]
    return JobStatusResponse(
        job_id=job_id,
        status="done",
        status_log=result.get("status_log", []),
        plan=result.get("plan", ""),
        draft_code=result.get("draft_code", ""),
        review_feedback=result.get("review_feedback", ""),
        is_approved=result.get("is_approved", False),
        revision_count=result.get("revision_count", 0),
        final_notebook_path=result.get("final_notebook_path", ""),
        agent_logs=result.get("agent_logs", []),
        total_tokens=result.get("total_tokens", {}),
    )


@app.get(
    "/job/{job_id}/download",
    tags=["Pipeline"],
    summary="Download the generated .ipynb notebook",
)
def download_notebook(job_id: str) -> FileResponse:
    """
    Stream the generated Jupyter notebook (.ipynb) as a file download.
    Only available after the job status is 'done'.
    """
    with _job_store_lock:
        job = _job_store.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job["status"] != "done":
        raise HTTPException(
            status_code=400,
            detail="Notebook is not ready yet. Wait until job status is 'done'.",
        )

    result = job["result"]
    notebook_path = result.get("final_notebook_path", "")

    if not notebook_path or not Path(notebook_path).exists():
        raise HTTPException(
            status_code=404,
            detail="Notebook file not found on disk.",
        )

    return FileResponse(
        path=notebook_path,
        media_type="application/octet-stream",
        filename=Path(notebook_path).name,
    )


@app.get(
    "/job/{job_id}/notebook_content",
    tags=["Pipeline"],
    summary="Return raw notebook JSON content as text",
)
def get_notebook_content(job_id: str) -> JSONResponse:
    """
    Return the raw content of the generated .ipynb file as JSON.
    Useful for previewing the notebook in the Streamlit UI without downloading.
    """
    with _job_store_lock:
        job = _job_store.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job["status"] != "done":
        raise HTTPException(
            status_code=400,
            detail="Notebook is not ready yet.",
        )

    result = job["result"]
    notebook_path = result.get("final_notebook_path", "")

    if not notebook_path or not Path(notebook_path).exists():
        raise HTTPException(
            status_code=404,
            detail="Notebook file not found on disk.",
        )

    with open(notebook_path, "r", encoding="utf-8") as f:
        content = f.read()

    return JSONResponse(content={"notebook_json": content})
