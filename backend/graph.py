"""
LangGraph workflow for the Databricks Notebook Generator.

Architecture: Hub-and-Spoke (Supervisor Pattern)
  - Every spoke node (Product_Manager, Data_Engineer,
    Senior_Architect, DevOps) sets state["next_agent"]
    before returning to the Supervisor.
  - The Supervisor evaluates state and routes to the
    appropriate spoke via conditional edges.

Graph execution flow:
  START
    └─> Supervisor
          ├─> Product_Manager ──> Supervisor
          ├─> Data_Engineer   ──> Supervisor
          ├─> Senior_Architect──> Supervisor
          └─> DevOps          ──> END
"""

from langgraph.graph import StateGraph, START, END

import logging

from .state import NotebookGeneratorState
from .agents import (
    supervisor_node,
    product_manager_node,
    data_engineer_node,
    senior_architect_node,
    devops_node,
)

logger = logging.getLogger("AgentPipeline")

# ---------------------------------------------------------------------------
# Node names – kept as constants to avoid typos in edge definitions
# ---------------------------------------------------------------------------
SUPERVISOR = "Supervisor"
PRODUCT_MANAGER = "Product_Manager"
DATA_ENGINEER = "Data_Engineer"
SENIOR_ARCHITECT = "Senior_Architect"
DEVOPS = "DevOps"


# ---------------------------------------------------------------------------
# Routing function
# Called by conditional_edges coming OUT of the Supervisor node.
# Maps state["next_agent"] -> actual node name (or END sentinel).
# ---------------------------------------------------------------------------
def route_from_supervisor(state: NotebookGeneratorState) -> str:
    """
    Read the next_agent token written by supervisor_node and return the
    name of the spoke node to activate, or END to terminate the graph.
    """
    destination = state.get("next_agent", PRODUCT_MANAGER)
    if destination == "END":
        return END
    return destination


# ---------------------------------------------------------------------------
# Build and compile the LangGraph StateGraph
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    """
    Construct, wire, and compile the multi-agent workflow graph.

    Returns a compiled Runnable that accepts an initial state dict and
    streams / invokes the full pipeline to completion.
    """
    logger.info("Building agent workflow graph...")

    graph = StateGraph(NotebookGeneratorState)

    # -- Register all nodes --------------------------------------------------
    graph.add_node(SUPERVISOR, supervisor_node)
    graph.add_node(PRODUCT_MANAGER, product_manager_node)
    graph.add_node(DATA_ENGINEER, data_engineer_node)
    graph.add_node(SENIOR_ARCHITECT, senior_architect_node)
    graph.add_node(DEVOPS, devops_node)

    # -- Entry point: START always goes to the Supervisor --------------------
    graph.add_edge(START, SUPERVISOR)

    # -- Conditional edges from Supervisor -----------------------------------
    # The supervisor_node sets state["next_agent"]; route_from_supervisor maps
    # that value to the correct destination node (or END).
    graph.add_conditional_edges(
        SUPERVISOR,
        route_from_supervisor,
        {
            PRODUCT_MANAGER: PRODUCT_MANAGER,
            DATA_ENGINEER: DATA_ENGINEER,
            SENIOR_ARCHITECT: SENIOR_ARCHITECT,
            DEVOPS: DEVOPS,
            END: END,
        },
    )

    # -- Every spoke returns unconditionally to the Supervisor ---------------
    # Exception: DevOps routes to END (the graph terminates after packaging).
    graph.add_edge(PRODUCT_MANAGER, SUPERVISOR)
    graph.add_edge(DATA_ENGINEER, SUPERVISOR)
    graph.add_edge(SENIOR_ARCHITECT, SUPERVISOR)
    graph.add_edge(DEVOPS, END)  # pipeline complete after notebook is written

    logger.info("Workflow graph built successfully")
    logger.info("   Flow: START -> Supervisor -> [PM | DE | SA | DevOps] -> ... -> END")

    return graph.compile()


# ---------------------------------------------------------------------------
# Module-level compiled graph (imported by main.py)
# ---------------------------------------------------------------------------
workflow = build_graph()


# ---------------------------------------------------------------------------
# Convenience helper used by FastAPI
# ---------------------------------------------------------------------------
def run_pipeline(requirements: str, openai_api_key: str = "") -> NotebookGeneratorState:
    """
    Execute the full multi-agent pipeline for the given requirements string.

    Args:
        requirements:   Plain-text user requirements / legacy code to migrate.
        openai_api_key: OpenAI API key (optional; falls back to OPENAI_API_KEY env var).

    Returns:
        The final NotebookGeneratorState after the graph reaches END.
    """
    import time as _time

    logger.info("=" * 70)
    logger.info("PIPELINE START")
    logger.info(f"   Requirement: {requirements[:120]}...")
    logger.info("=" * 70)

    pipeline_start = _time.time()

    initial_state: NotebookGeneratorState = {
        "requirements": requirements,
        "plan": "",
        "draft_code": "",
        "review_feedback": "",
        "is_approved": False,
        "revision_count": 0,
        "final_notebook_path": "",
        "next_agent": "",
        "status_log": ["Pipeline started."],
        "openai_api_key": openai_api_key,
        "agent_logs": [],
        "total_tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    final_state = workflow.invoke(initial_state)

    pipeline_elapsed = _time.time() - pipeline_start
    total_tokens = final_state.get("total_tokens", {})

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"   Total time:            {pipeline_elapsed:.2f}s")
    logger.info(f"   Total prompt tokens:    {total_tokens.get('prompt_tokens', 0)}")
    logger.info(f"   Total completion tokens:{total_tokens.get('completion_tokens', 0)}")
    logger.info(f"   Total tokens used:      {total_tokens.get('total_tokens', 0)}")
    logger.info(f"   Revisions performed:    {final_state.get('revision_count', 0)}")
    logger.info(f"   Notebook path:          {final_state.get('final_notebook_path', 'N/A')}")
    logger.info("=" * 70)

    return final_state
