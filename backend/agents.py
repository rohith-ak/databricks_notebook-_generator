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
import datetime
from pathlib import Path
from typing import Dict, Any

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
# Helper: build an LLM instance
# The API key is read from the OPENAI_API_KEY environment variable.
# Model can be overridden via the OPENAI_MODEL env var (defaults to gpt-4o).
# ---------------------------------------------------------------------------
def _get_llm(api_key: str | None = None, temperature: float = 0.2) -> ChatOpenAI:
    """
    Return a configured ChatOpenAI instance.
    Priority for API key: explicit arg -> OPENAI_API_KEY env var.
    """
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not resolved_key:
        raise ValueError(
            "OpenAI API key is required. Pass it via the Streamlit sidebar or "
            "set the OPENAI_API_KEY environment variable."
        )
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=resolved_key)


def _invoke_llm(system_prompt: str, user_message: str, api_key: str | None = None) -> str:
    """Send a system + user message pair to the LLM and return the reply text."""
    llm = _get_llm(api_key=api_key)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    return response.content.strip()



# ---------------------------------------------------------------------------
# Supervisor Node
# Inspects the current state and decides which agent to run next.
# Returns {"next_agent": "<node_name>"} - consumed by conditional_edges.
# ---------------------------------------------------------------------------
def supervisor_node(state: NotebookGeneratorState) -> Dict[str, Any]:
    """
    Hub of the hub-and-spoke architecture.
    Evaluates state fields and returns the name of the next spoke to activate.

    Routing rules (in priority order):
      1. plan is empty            -> Product_Manager
      2. draft_code is empty      -> Data_Engineer
      3. not approved + revisions < 3 -> Senior_Architect (first pass)
         but if draft already reviewed (feedback exists) -> route back based on is_approved
      4. not approved + revisions >= 3 -> DevOps (fail-safe: ship best available)
      5. is_approved is True      -> DevOps
      6. After DevOps             -> END  (handled in graph.py via edge, not here)
    """
    log = list(state.get("status_log", []))

    plan = state.get("plan", "")
    draft_code = state.get("draft_code", "")
    is_approved = state.get("is_approved", False)
    revision_count = state.get("revision_count", 0)
    review_feedback = state.get("review_feedback", "")
    final_notebook_path = state.get("final_notebook_path", "")

    # If the notebook has already been generated, we are done
    if final_notebook_path:
        next_agent = "END"
        log.append("Supervisor -> END (notebook already packaged)")

    # Rule 1: No plan yet -> Product_Manager to create the execution plan
    elif not plan:
        next_agent = "Product_Manager"
        log.append("Supervisor -> Product_Manager (plan not yet created)")

    # Rule 2: Plan exists but no code yet -> Data_Engineer for first code generation
    elif not draft_code:
        next_agent = "Data_Engineer"
        log.append("Supervisor -> Data_Engineer (code generation required, no draft yet)")

    # Rule 4-fail-safe: Revision cap reached -> package whatever we have
    elif not is_approved and revision_count >= 3:
        next_agent = "DevOps"
        log.append(
            f"Supervisor -> DevOps (FAIL-SAFE: revision cap {revision_count}/3 reached, packaging best available code)"
        )

    # Rule 5: Code passed review -> package it
    elif is_approved:
        next_agent = "DevOps"
        log.append("Supervisor -> DevOps (code approved, ready for packaging)")

    # Rule 3a: Architect has returned feedback (is_approved=False, review_feedback set)
    #          -> Send back to Data_Engineer for revision
    elif not is_approved and review_feedback:
        next_agent = "Data_Engineer"
        log.append(
            f"Supervisor -> Data_Engineer (revision required; feedback from architect, "
            f"revision_count={revision_count})"
        )

    # Rule 3b: Draft code exists, no feedback yet (first or subsequent clean pass)
    #          -> Send to Senior_Architect for review
    else:
        next_agent = "Senior_Architect"
        log.append(
            f"Supervisor -> Senior_Architect (review pass, revision_count={revision_count})"
        )

    return {"next_agent": next_agent, "status_log": log}


# ---------------------------------------------------------------------------
# Product Manager Node
# ---------------------------------------------------------------------------
def product_manager_node(state: NotebookGeneratorState) -> Dict[str, Any]:
    """
    Reads the raw requirements and produces a structured, step-by-step execution plan.
    Each step corresponds to one notebook cell (CODE or MARKDOWN).
    """
    log = list(state.get("status_log", []))
    log.append("Product_Manager: Analysing requirements and creating execution plan...")

    requirements = state["requirements"]
    api_key = state.get("openai_api_key", "")
    user_message = f"USER REQUIREMENTS:\n{requirements}"

    plan = _invoke_llm(PRODUCT_MANAGER_PROMPT, user_message, api_key=api_key)

    log.append("Product_Manager: Execution plan created.")
    return {
        "plan": plan,
        "status_log": log,
        "next_agent": "Supervisor",  # always return to Supervisor
    }


# ---------------------------------------------------------------------------
# Data Engineer Node
# ---------------------------------------------------------------------------
def data_engineer_node(state: NotebookGeneratorState) -> Dict[str, Any]:
    """
    Translates the execution plan into production-ready PySpark / Databricks SQL code.
    If review_feedback is present, this is a revision cycle – all feedback is addressed.
    Increments revision_count on every invocation.
    """
    log = list(state.get("status_log", []))
    revision_count = state.get("revision_count", 0)
    review_feedback = state.get("review_feedback", "")

    # Build context message
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
    else:
        log.append("Data_Engineer: Generating initial PySpark code from execution plan...")
        user_message = f"EXECUTION PLAN:\n{state['plan']}"

    draft_code = _invoke_llm(DATA_ENGINEER_PROMPT, user_message, api_key=api_key)

    # Strip any accidental markdown fences the LLM may add despite instructions
    draft_code = re.sub(r"^```(?:python)?\n?", "", draft_code, flags=re.MULTILINE)
    draft_code = re.sub(r"\n?```$", "", draft_code, flags=re.MULTILINE)
    draft_code = draft_code.strip()

    log.append("Data_Engineer: Code generation complete.")
    return {
        "draft_code": draft_code,
        "revision_count": revision_count + 1,
        "is_approved": False,          # reset approval on every new draft
        "review_feedback": "",          # clear old feedback
        "status_log": log,
        "next_agent": "Supervisor",
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
    log = list(state.get("status_log", []))
    log.append("Senior_Architect: Reviewing code for best practices and security...")

    api_key = state.get("openai_api_key", "")
    user_message = (
        f"EXECUTION PLAN (context only):\n{state['plan']}\n\n"
        f"CODE TO REVIEW:\n{state['draft_code']}"
    )

    verdict = _invoke_llm(SENIOR_ARCHITECT_PROMPT, user_message, api_key=api_key)

    # The architect must reply with "APPROVED" (case-insensitive) or bullets
    if verdict.strip().upper() == "APPROVED":
        log.append("Senior_Architect: Code APPROVED. No issues found.")
        return {
            "is_approved": True,
            "review_feedback": "",
            "status_log": log,
            "next_agent": "Supervisor",
        }
    else:
        log.append("Senior_Architect: Code REJECTED – feedback returned to Data Engineer.")
        return {
            "is_approved": False,
            "review_feedback": verdict,
            "status_log": log,
            "next_agent": "Supervisor",
        }


# ---------------------------------------------------------------------------
# DevOps (Notebook Assembler) Node
# ---------------------------------------------------------------------------
def devops_node(state: NotebookGeneratorState) -> Dict[str, Any]:
    """
    Packages the approved (or fail-safe) code into a Jupyter .ipynb notebook.

    Notebook structure:
      - Cell 0  : Markdown title + plan summary
      - Cell 1+ : One code cell per # COMMAND ---------- delimiter

    The generated file is saved to generated_notebooks/<timestamp>.ipynb
    """
    log = list(state.get("status_log", []))
    log.append("DevOps: Assembling final Databricks .ipynb notebook...")

    plan = state.get("plan", "")
    draft_code = state.get("draft_code", "")
    is_approved = state.get("is_approved", False)

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
        "✅ **Status:** Architect Approved"
        if is_approved
        else "⚠️ **Status:** Packaged under fail-safe (max revisions reached)"
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
            continue  # skip empty blocks from leading/trailing delimiters

        # Detect MAGIC %md blocks and convert to markdown cells
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

    log.append(f"DevOps: Notebook saved → {notebook_path}")
    return {
        "final_notebook_path": str(notebook_path),
        "status_log": log,
        "next_agent": "END",
    }
