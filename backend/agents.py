"""
Agent node implementations for the Databricks Notebook Generator LangGraph workflow.

Each function follows the LangGraph node contract:
    Input  -> NotebookGeneratorState
    Output -> partial NotebookGeneratorState (dict with only updated keys)

Nodes: product_manager_node, data_engineer_node,
       senior_architect_node, devops_node, supervisor_node
"""

import os
import re
import time
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .state import NotebookGeneratorState
from .prompts import (
    PRODUCT_MANAGER_PROMPT,
    DATA_ENGINEER_PROMPT,
    SENIOR_ARCHITECT_PROMPT,
    DEVOPS_PROMPT,
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AgentPipeline")

# ---------------------------------------------------------------------------
# Helper: build an LLM instance
# The API key is read from the OPENAI_API_KEY environment variable.
# Model can be overridden via the OPENAI_MODEL env var (defaults to gpt-4o).
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")
logger.info(f"Configured LLM model: {MODEL_NAME}")


def _get_llm(api_key: str | None = None, temperature: float = 0.2) -> ChatOpenAI:
    """
    Return a configured ChatOpenAI instance.
    Priority for API key: explicit arg -> OPENAI_API_KEY env var.
    """
    resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not resolved_key:
        raise ValueError(
            "OpenAI API key is required. Pass it via the Streamlit sidebar or "
            "set the OPENAI_API_KEY environment variable."
        )
    return ChatOpenAI(model=MODEL_NAME, temperature=temperature, api_key=resolved_key)


# ---------------------------------------------------------------------------
# Token usage extraction helpers
# ---------------------------------------------------------------------------
def _extract_token_usage(response) -> dict:
    """Extract token usage from LLM response metadata."""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if hasattr(response, "response_metadata"):
        meta = response.response_metadata
        if "token_usage" in meta:
            usage = {
                "prompt_tokens": meta["token_usage"].get("prompt_tokens", 0),
                "completion_tokens": meta["token_usage"].get("completion_tokens", 0),
                "total_tokens": meta["token_usage"].get("total_tokens", 0),
            }
    return usage


def _log_agent_start(agent_name: str):
    logger.info("=" * 60)
    logger.info(f"AGENT START: {agent_name}")
    logger.info(f"   Model: {MODEL_NAME}")
    logger.info("=" * 60)


def _log_agent_end(agent_name: str, token_usage: dict, elapsed: float):
    logger.info("-" * 60)
    logger.info(f"AGENT COMPLETE: {agent_name}")
    logger.info(f"   Elapsed:           {elapsed:.2f}s")
    logger.info(f"   Prompt tokens:     {token_usage.get('prompt_tokens', 0)}")
    logger.info(f"   Completion tokens: {token_usage.get('completion_tokens', 0)}")
    logger.info(f"   Total tokens:      {token_usage.get('total_tokens', 0)}")
    logger.info("-" * 60)


def _append_agent_log(state: NotebookGeneratorState, agent_name: str, status: str,
                      token_usage: dict, elapsed: float, detail: str = "") -> list:
    """Append a structured log entry to the state's agent_logs list."""
    logs = list(state.get("agent_logs") or [])
    logs.append({
        "agent": agent_name,
        "status": status,
        "model": MODEL_NAME,
        "token_usage": token_usage,
        "elapsed_seconds": round(elapsed, 2),
        "detail": detail,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    return logs


def _update_total_tokens(state: NotebookGeneratorState, token_usage: dict) -> dict:
    """Accumulate total token counts across the pipeline."""
    totals = dict(state.get("total_tokens") or {})
    if not totals:
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    totals["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
    totals["completion_tokens"] += token_usage.get("completion_tokens", 0)
    totals["total_tokens"] += token_usage.get("total_tokens", 0)
    return totals


def _invoke_llm(system_prompt: str, user_message: str, api_key: str | None = None) -> Tuple[str, dict]:
    """
    Send a system + user message pair to the LLM.
    Returns (response_text, token_usage_dict).
    """
    llm = _get_llm(api_key=api_key)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    token_usage = _extract_token_usage(response)
    return response.content.strip(), token_usage



# ---------------------------------------------------------------------------
# Supervisor Node
# Inspects the current state and decides which agent to run next.
# Returns {"next_agent": "<node_name>"} - consumed by conditional_edges.
# ---------------------------------------------------------------------------
def supervisor_node(state: NotebookGeneratorState) -> Dict[str, Any]:
    """
    Hub of the hub-and-spoke architecture.
    Evaluates state fields and returns the name of the next spoke to activate.
    """
    log = list(state.get("status_log", []))

    plan = state.get("plan", "")
    draft_code = state.get("draft_code", "")
    is_approved = state.get("is_approved", False)
    revision_count = state.get("revision_count", 0)
    review_feedback = state.get("review_feedback", "")
    final_notebook_path = state.get("final_notebook_path", "")

    logger.info("=" * 60)
    logger.info("SUPERVISOR — Decision Point")
    logger.info(f"   plan={bool(plan)}, code={bool(draft_code)}, "
                f"approved={is_approved}, revisions={revision_count}, "
                f"feedback={bool(review_feedback)}, notebook={bool(final_notebook_path)}")

    # If the notebook has already been generated, we are done
    if final_notebook_path:
        next_agent = "END"
        log.append("Supervisor -> END (notebook already packaged)")
        logger.info("   -> Routing to: END (notebook already packaged)")

    # Rule 1: No plan yet -> Product_Manager to create the execution plan
    elif not plan:
        next_agent = "Product_Manager"
        log.append("Supervisor -> Product_Manager (plan not yet created)")
        logger.info("   -> Routing to: Product_Manager (plan not yet created)")

    # Rule 2: Plan exists but no code yet -> Data_Engineer for first code generation
    elif not draft_code:
        next_agent = "Data_Engineer"
        log.append("Supervisor -> Data_Engineer (code generation required, no draft yet)")
        logger.info("   -> Routing to: Data_Engineer (initial code generation)")

    # Rule 4-fail-safe: Revision cap reached -> package whatever we have
    elif not is_approved and revision_count >= 3:
        next_agent = "DevOps"
        log.append(
            f"Supervisor -> DevOps (FAIL-SAFE: revision cap {revision_count}/3 reached, packaging best available code)"
        )
        logger.info(f"   -> Routing to: DevOps (FAIL-SAFE: revision cap {revision_count}/3 reached)")

    # Rule 5: Code passed review -> package it
    elif is_approved:
        next_agent = "DevOps"
        log.append("Supervisor -> DevOps (code approved, ready for packaging)")
        logger.info("   -> Routing to: DevOps (code APPROVED)")

    # Rule 3a: Architect has returned feedback
    elif not is_approved and review_feedback:
        next_agent = "Data_Engineer"
        log.append(
            f"Supervisor -> Data_Engineer (revision required; feedback from architect, "
            f"revision_count={revision_count})"
        )
        logger.info(f"   -> Routing to: Data_Engineer (revision #{revision_count + 1} needed)")

    # Rule 3b: Draft code exists, no feedback yet -> review it
    else:
        next_agent = "Senior_Architect"
        log.append(
            f"Supervisor -> Senior_Architect (review pass, revision_count={revision_count})"
        )
        logger.info(f"   -> Routing to: Senior_Architect (review pass #{revision_count})")

    logger.info("=" * 60)

    return {"next_agent": next_agent, "status_log": log}


# ---------------------------------------------------------------------------
# Product Manager Node
# ---------------------------------------------------------------------------
def product_manager_node(state: NotebookGeneratorState) -> Dict[str, Any]:
    """
    Reads the raw requirements and produces a structured, step-by-step execution plan.
    """
    agent_name = "Product Manager (Planner)"
    _log_agent_start(agent_name)
    start = time.time()

    log = list(state.get("status_log", []))
    log.append("Product_Manager: Analysing requirements and creating execution plan...")

    requirements = state["requirements"]
    api_key = state.get("openai_api_key", "")
    user_message = f"USER REQUIREMENTS:\n{requirements}"

    logger.info(f"   Input: User requirement ({len(requirements)} chars)")
    logger.info(f"   Sending to LLM: 'Create structured execution plan'")

    plan, token_usage = _invoke_llm(PRODUCT_MANAGER_PROMPT, user_message, api_key=api_key)

    elapsed = time.time() - start

    logger.info(f"   Response received ({len(plan)} chars)")
    logger.info(f"   Plan preview: {plan[:200]}...")
    _log_agent_end(agent_name, token_usage, elapsed)

    log.append("Product_Manager: Execution plan created.")
    agent_logs = _append_agent_log(state, agent_name, "completed", token_usage, elapsed,
                                   f"Generated plan ({len(plan)} chars)")
    total_tokens = _update_total_tokens(state, token_usage)

    return {
        "plan": plan,
        "status_log": log,
        "next_agent": "Supervisor",
        "agent_logs": agent_logs,
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Data Engineer Node
# ---------------------------------------------------------------------------
def data_engineer_node(state: NotebookGeneratorState) -> Dict[str, Any]:
    """
    Translates the execution plan into production-ready PySpark / Databricks SQL code.
    If review_feedback is present, this is a revision cycle.
    """
    agent_name = "Data Engineer (Coder)"
    _log_agent_start(agent_name)
    start = time.time()

    log = list(state.get("status_log", []))
    revision_count = state.get("revision_count", 0)
    review_feedback = state.get("review_feedback", "")

    api_key = state.get("openai_api_key", "")

    if review_feedback:
        log.append(
            f"Data_Engineer: Revision cycle {revision_count + 1}/3 – addressing architect feedback..."
        )
        user_message = (
            f"EXECUTION PLAN:\n{state['plan']}\n\n"
            f"PREVIOUS CODE (requires revision):\n{state['draft_code']}\n\n"
            f"REVIEW FEEDBACK (you MUST fix all items below):\n{review_feedback}"
        )
        logger.info(f"   Revision #{revision_count + 1} — Addressing review feedback")
        logger.info(f"   Feedback preview: {review_feedback[:200]}...")
    else:
        log.append("Data_Engineer: Generating initial PySpark code from execution plan...")
        user_message = f"EXECUTION PLAN:\n{state['plan']}"
        logger.info(f"   First pass — Generating code from execution plan")

    draft_code, token_usage = _invoke_llm(DATA_ENGINEER_PROMPT, user_message, api_key=api_key)

    elapsed = time.time() - start

    # Strip any accidental markdown fences the LLM may add despite instructions
    draft_code = re.sub(r"^```(?:python)?\n?", "", draft_code, flags=re.MULTILINE)
    draft_code = re.sub(r"\n?```$", "", draft_code, flags=re.MULTILINE)
    draft_code = draft_code.strip()

    logger.info(f"   Code generated ({len(draft_code)} chars, {draft_code.count(chr(10))} lines)")
    _log_agent_end(agent_name, token_usage, elapsed)

    detail = f"Revision #{revision_count + 1}" if review_feedback else "Initial code generation"
    log.append("Data_Engineer: Code generation complete.")
    agent_logs = _append_agent_log(state, agent_name, "completed", token_usage, elapsed, detail)
    total_tokens = _update_total_tokens(state, token_usage)

    return {
        "draft_code": draft_code,
        "revision_count": revision_count + 1,
        "is_approved": False,
        "review_feedback": "",
        "status_log": log,
        "next_agent": "Supervisor",
        "agent_logs": agent_logs,
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Senior Architect Node
# ---------------------------------------------------------------------------
def senior_architect_node(state: NotebookGeneratorState) -> Dict[str, Any]:
    """
    Reviews draft_code against Databricks best practices.
    Sets is_approved=True and clears feedback if the code is perfect,
    otherwise populates review_feedback with actionable bullet points.
    """
    agent_name = "Senior Architect (Reviewer)"
    _log_agent_start(agent_name)
    start = time.time()

    log = list(state.get("status_log", []))
    log.append("Senior_Architect: Reviewing code for best practices and security...")

    logger.info(f"   Reviewing code ({len(state['draft_code'])} chars)")

    api_key = state.get("openai_api_key", "")
    user_message = (
        f"EXECUTION PLAN (context only):\n{state['plan']}\n\n"
        f"CODE TO REVIEW:\n{state['draft_code']}"
    )

    verdict, token_usage = _invoke_llm(SENIOR_ARCHITECT_PROMPT, user_message, api_key=api_key)

    elapsed = time.time() - start

    # The architect must reply with "APPROVED" (case-insensitive) or bullets
    if verdict.strip().upper() == "APPROVED":
        logger.info("   VERDICT: Code APPROVED")
        _log_agent_end(agent_name, token_usage, elapsed)

        log.append("Senior_Architect: Code APPROVED. No issues found.")
        agent_logs = _append_agent_log(state, agent_name, "completed", token_usage, elapsed, "APPROVED")
        total_tokens = _update_total_tokens(state, token_usage)

        return {
            "is_approved": True,
            "review_feedback": "",
            "status_log": log,
            "next_agent": "Supervisor",
            "agent_logs": agent_logs,
            "total_tokens": total_tokens,
        }
    else:
        logger.info("   VERDICT: Revisions required")
        logger.info(f"   Feedback preview: {verdict[:300]}...")
        _log_agent_end(agent_name, token_usage, elapsed)

        log.append("Senior_Architect: Code REJECTED – feedback returned to Data Engineer.")
        agent_logs = _append_agent_log(state, agent_name, "completed", token_usage, elapsed, "Revisions requested")
        total_tokens = _update_total_tokens(state, token_usage)

        return {
            "is_approved": False,
            "review_feedback": verdict,
            "status_log": log,
            "next_agent": "Supervisor",
            "agent_logs": agent_logs,
            "total_tokens": total_tokens,
        }


# ---------------------------------------------------------------------------
# DevOps (Notebook Assembler) Node
# ---------------------------------------------------------------------------
def devops_node(state: NotebookGeneratorState) -> Dict[str, Any]:
    """
    Packages the approved (or fail-safe) code into a Jupyter .ipynb notebook.
    """
    agent_name = "DevOps Assembler"
    _log_agent_start(agent_name)
    start = time.time()

    log = list(state.get("status_log", []))
    log.append("DevOps: Assembling final Databricks .ipynb notebook...")

    plan = state.get("plan", "")
    draft_code = state.get("draft_code", "")
    is_approved = state.get("is_approved", False)

    logger.info(f"   Assembling notebook from {'APPROVED' if is_approved else 'FAIL-SAFE'} code ({len(draft_code)} chars)")

    # ------------------------------------------------------------------
    # 1. Build notebook metadata
    # ------------------------------------------------------------------
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.10.0"}
    nb.metadata["databricks"] = {"notebook_source": True}

    # ------------------------------------------------------------------
    # 2. First cell: Databricks notebook source header + plan as markdown
    # ------------------------------------------------------------------
    approval_note = (
        "**Status:** Architect Approved"
        if is_approved
        else "**Status:** Packaged under fail-safe (max revisions reached)"
    )
    header_md = (
        f"# Databricks Notebook — Auto-Generated\n\n"
        f"{approval_note}\n\n"
        f"---\n\n"
        f"## Execution Plan\n\n{plan}\n\n"
        f"---\n\n"
        f"*Generated by Databricks Notebook Generator · {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*"
    )
    nb.cells.append(new_markdown_cell(header_md))

    # ------------------------------------------------------------------
    # 3. Split draft_code on Databricks command delimiter into code cells
    # ------------------------------------------------------------------
    delimiter = "# COMMAND ----------"
    raw_blocks = draft_code.split(delimiter)

    for block in raw_blocks:
        stripped = block.strip()
        if not stripped:
            continue

        if stripped.startswith("# MAGIC %md"):
            md_content = re.sub(r"^# MAGIC %md\s*", "", stripped, flags=re.MULTILINE)
            nb.cells.append(new_markdown_cell(md_content.strip()))
        else:
            nb.cells.append(new_code_cell(stripped))

    # ------------------------------------------------------------------
    # 4. Persist to generated_notebooks/
    # ------------------------------------------------------------------
    output_dir = Path(__file__).parent.parent / "generated_notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    notebook_path = output_dir / f"databricks_notebook_{timestamp}.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    elapsed = time.time() - start

    logger.info(f"   Notebook saved -> {notebook_path}")
    logger.info(f"   Notebook cells: {len(nb.cells)}")

    # DevOps doesn't call the LLM, so token_usage is zero
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    _log_agent_end(agent_name, token_usage, elapsed)

    log.append(f"DevOps: Notebook saved -> {notebook_path}")
    agent_logs = _append_agent_log(state, agent_name, "completed", token_usage, elapsed,
                                   f"Notebook saved ({len(nb.cells)} cells)")
    total_tokens = _update_total_tokens(state, token_usage)

    return {
        "final_notebook_path": str(notebook_path),
        "status_log": log,
        "next_agent": "END",
        "agent_logs": agent_logs,
        "total_tokens": total_tokens,
    }
