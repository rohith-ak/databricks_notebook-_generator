"""
State definition for the Databricks Notebook Generator multi-agent system.
Uses TypedDict so LangGraph can track the full workflow state across all nodes.
"""

from typing import TypedDict, List, Dict, Any


class NotebookGeneratorState(TypedDict):
    """
    Central state object passed between all agent nodes in the LangGraph workflow.

    Fields:
        requirements      : Raw user input describing the ETL/notebook task.
        plan              : Structured execution plan output by the Product Manager agent.
        draft_code        : PySpark / Databricks SQL code produced by the Data Engineer.
        review_feedback   : Actionable critique returned by the Senior Architect when code is rejected.
        is_approved       : True once the Senior Architect approves the draft code.
        revision_count    : Number of Data Engineer revision cycles (capped at 3 to prevent loops).
        final_notebook_path: Absolute path to the generated .ipynb file written by the DevOps agent.
        next_agent        : Token used by the Supervisor to declare which spoke to invoke next.
        status_log        : Human-readable list of status messages for UI display / debugging.
        agent_logs        : Detailed structured log entries per agent invocation (for UI display).
        total_tokens      : Cumulative token usage across all agent LLM calls.
    """

    requirements: str
    plan: str
    draft_code: str
    review_feedback: str
    is_approved: bool
    revision_count: int
    final_notebook_path: str
    next_agent: str
    status_log: List[str]
    openai_api_key: str    # passed per-request; never logged or stored on disk
    agent_logs: List[Dict[str, Any]]     # structured log entries for UI
    total_tokens: Dict[str, int]         # cumulative {prompt_tokens, completion_tokens, total_tokens}
