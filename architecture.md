# Databricks Notebook Generator - Architecture Documentation

## 🎯 Architecture Pattern

This system implements a **Sequential Pipeline Architecture with Conditional Feedback Loops** (also known as a **Linear Chain with Retry Pattern**). This is NOT a Hub-and-Spoke architecture.

### Architecture Type Classification

- **Primary Pattern**: Sequential Pipeline (Linear Chain)
- **Secondary Pattern**: Conditional Branching with Feedback Loop
- **Orchestration**: State Machine with Directed Acyclic Graph (DAG)

### Why Not Hub-and-Spoke?

In a Hub-and-Spoke architecture, a central coordinator (hub) would dispatch tasks to multiple agents (spokes) that work independently and report back. However, this system uses a **sequential handoff pattern** where each agent completes its work before passing control to the next agent in a predefined order.

## 🏗️ Detailed Architecture Overview

### System Flow Diagram

```
┌─────────────────┐
│  User Input     │
│  (Requirements) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestrator                       │
│                    (backend/graph.py)                           │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      State Manager                              │
│                    (backend/state.py)                           │
│  Tracks: requirements, specifications, code, feedback, output   │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌────────────────────────────────────────────────────────┐
    │                   Agent Pipeline                       │
    └────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  1. Product Manager │  (Planner Agent)
│     - Analyzes requirements
│     - Creates specifications
│     - Defines data sources
└──────────┬──────────┘
           │
           │ specifications
           ▼
┌─────────────────────┐
│  2. Data Engineer   │  (Coder Agent)
│     - Writes PySpark code
│     - Implements Delta Lake ops
│     - Follows best practices
└──────────┬──────────┘
           │
           │ code
           ▼
┌─────────────────────┐
│ 3. Senior Architect │  (Reviewer Agent)
│     - Reviews code quality
│     - Checks performance
│     - Validates security
└──────────┬──────────┘
           │
           │ decision point
           ▼
      ┌─────────┐
      │ APPROVED?│
      └─────────┘
       │        │
       │ NO     │ YES
       │        │
       ▼        ▼
   ┌────────┐  ┌─────────────────────┐
   │feedback│  │  4. DevOps Assembler│  (Assembler Agent)
   │  loop  │  │     - Creates .ipynb format
   └───┬────┘  │     - Adds metadata
       │       │     - Structures cells
       │       └──────────┬──────────┘
       │                  │
       │                  │ notebook
       │                  ▼
       │          ┌──────────────────┐
       │          │  Generated       │
       │          │  Notebook Output │
       │          └──────────────────┘
       │
       └─────► Back to Data Engineer (retry with feedback)
```

## 🔄 Agent-to-Agent Communication Flow

### 1. **Product Manager → Data Engineer**

**Direction**: Unidirectional (one-way)
**Data Transferred**: Technical specifications
**State Update**: `specifications` field in state

```python
# Product Manager Output
{
    "specifications": "Detailed technical plan with data sources, transformations, and outputs"
}

# Data Engineer Input
# Receives specifications from state
# Uses it to generate PySpark code
```

**Communication Method**:
- State-based handoff via [`backend/state.py`](backend/state.py)
- Product Manager writes to state
- Data Engineer reads from state
- No direct API calls between agents

---

### 2. **Data Engineer → Senior Architect**

**Direction**: Unidirectional (one-way, initially)
**Data Transferred**: PySpark code implementation
**State Update**: `code` field in state

```python
# Data Engineer Output
{
    "code": "Complete PySpark implementation with Delta Lake operations"
}

# Senior Architect Input
# Receives code from state
# Reviews for quality, performance, security
```

**Communication Method**:
- State-based handoff
- Data Engineer writes code to state
- Senior Architect reads code from state
- Review feedback stored in state

---

### 3. **Senior Architect → Data Engineer (Feedback Loop)**

**Direction**: Conditional Bidirectional (only if issues found)
**Data Transferred**: Review feedback or approval
**State Update**: `feedback` field in state

```python
# Senior Architect Decision Logic
if code_has_issues:
    return {
        "feedback": "- Issue 1: Performance bottleneck in join operation\n- Issue 2: Missing partition handling",
        "next_agent": "data_engineer"  # Retry
    }
else:
    return {
        "feedback": "APPROVED",
        "next_agent": "devops_assembler"  # Proceed
    }

# Data Engineer (Retry)
# Receives feedback from state
# Revises code based on feedback
# Resubmits to Senior Architect
```

**Communication Method**:
- Conditional routing via [`backend/graph.py`](backend/graph.py)
- Uses LangGraph conditional edges
- State stores feedback history
- Loop repeats until "APPROVED" status

**Loop Termination**:
- Senior Architect returns exactly "APPROVED"
- Graph routes to DevOps Assembler
- No maximum retry limit (iterates until perfect)

---

### 4. **Senior Architect → DevOps Assembler**

**Direction**: Unidirectional (one-way)
**Data Transferred**: Approved PySpark code
**State Update**: No new state field, reads existing `code`

```python
# Senior Architect Output (on approval)
{
    "feedback": "APPROVED",
    "code": "Validated and approved PySpark code"
}

# DevOps Assembler Input
# Receives approved code from state
# Packages into .ipynb format
```

**Communication Method**:
- State-based handoff
- DevOps Assembler reads approved code
- Only triggered after "APPROVED" feedback
- No direct communication with Data Engineer

---

### 5. **DevOps Assembler → Output**

**Direction**: Unidirectional (one-way)
**Data Transferred**: Final Jupyter notebook (.ipynb)
**State Update**: `notebook` field in state

```python
# DevOps Assembler Output
{
    "notebook": {
        "cells": [...],
        "metadata": {...},
        "nbformat": 4,
        "nbformat_minor": 0
    }
}

# File System Output
# Writes to generated_notebooks/databricks_notebook_YYYYMMDD_HHMMSS.ipynb
```

**Communication Method**:
- State-based output
- File system write operation
- Terminal node in graph (no further agents)

---

## 🔐 State Management Architecture

### State Schema (from [`backend/state.py`](backend/state.py))

The entire system relies on a shared state object that flows through agents:

```python
class State(TypedDict):
    requirements: str          # User input
    specifications: str        # Product Manager output
    code: str                 # Data Engineer output
    feedback: str             # Senior Architect output
    notebook: dict            # DevOps Assembler output
    conversation_history: list # Full audit trail
```

### State Flow Pattern

```
Initial State
    ↓
[Product Manager] → State + specifications
    ↓
[Data Engineer] → State + code
    ↓
[Senior Architect] → State + feedback
    ↓
    ├─ if feedback != "APPROVED" → [Data Engineer] (loop back)
    └─ if feedback == "APPROVED" → [DevOps Assembler]
                                        ↓
                                    State + notebook
```

## 🎛️ Orchestration Layer

### LangGraph Implementation (from [`backend/graph.py`](backend/graph.py))

The graph orchestrator manages:

1. **Node Definition**: Each agent is a graph node
2. **Edge Routing**: Defines agent sequence
3. **Conditional Edges**: Implements feedback loop
4. **State Persistence**: Maintains state across transitions

### Graph Structure

```python
# Simplified graph structure
graph = StateGraph(State)

# Add nodes (agents)
graph.add_node("product_manager", product_manager_node)
graph.add_node("data_engineer", data_engineer_node)
graph.add_node("senior_architect", senior_architect_node)
graph.add_node("devops_assembler", devops_assembler_node)

# Define edges (flow)
graph.set_entry_point("product_manager")
graph.add_edge("product_manager", "data_engineer")
graph.add_edge("data_engineer", "senior_architect")

# Conditional edge (feedback loop)
graph.add_conditional_edges(
    "senior_architect",
    should_continue,  # Decision function
    {
        "data_engineer": "data_engineer",    # If issues found
        "devops_assembler": "devops_assembler"  # If approved
    }
)

graph.add_edge("devops_assembler", END)
```

## 🔍 Agent Communication Patterns

### Pattern 1: Direct Sequential Handoff
- **Used By**: Product Manager → Data Engineer → Senior Architect → DevOps
- **Characteristics**: 
  - Each agent completes its task fully
  - Next agent starts only after previous completes
  - No parallelism
  - State-mediated communication

### Pattern 2: Conditional Feedback Loop
- **Used By**: Senior Architect ↔ Data Engineer
- **Characteristics**:
  - Iterative refinement
  - Conditional routing based on review outcome
  - Stateful retry mechanism
  - No maximum iteration limit

### Pattern 3: One-Way Terminal Output
- **Used By**: DevOps Assembler → File System
- **Characteristics**:
  - Final output generation
  - No further agent processing
  - State to file persistence
  - End of workflow

## 📊 Data Flow Summary

| Source Agent        | Target Agent        | Data Type            | Communication Method | Bidirectional? |
|---------------------|---------------------|----------------------|----------------------|----------------|
| User                | Product Manager     | Requirements (text)  | Direct input         | No             |
| Product Manager     | Data Engineer       | Specifications (text)| State handoff        | No             |
| Data Engineer       | Senior Architect    | Code (text)          | State handoff        | No             |
| Senior Architect    | Data Engineer       | Feedback (text)      | State handoff        | Yes (loop)     |
| Senior Architect    | DevOps Assembler    | Approval + Code      | State handoff        | No             |
| DevOps Assembler    | File System         | Notebook (JSON)      | File write           | No             |

## 🔧 Technical Implementation Details

### Agent Definitions (from [`backend/agents.py`](backend/agents.py))

Each agent is implemented as:
- LangChain LLM chain
- Custom system prompt from [`prompts/`](prompts/) directory
- State input/output handling
- No direct agent-to-agent imports

### Prompt Management (from [`backend/prompts.py`](backend/prompts.py))

- Prompts loaded from text files
- Each agent has dedicated prompt template
- Prompts define agent behavior and output format
- No hardcoded prompts in code

### Frontend Integration (from [`frontend/app.py`](frontend/app.py))

- User interface layer
- Calls backend graph execution
- Displays agent conversation history
- Downloads generated notebooks

## 🎯 Key Architectural Decisions

### Why Sequential Pipeline?

1. **Clear Separation of Concerns**: Each agent has one responsibility
2. **Predictable Flow**: Easy to debug and trace execution
3. **Quality Gates**: Review step ensures code quality before packaging
4. **Stateful Progress**: State machine tracks entire workflow

### Why Feedback Loop?

1. **Quality Assurance**: Ensures code meets standards before deployment
2. **Iterative Improvement**: Data Engineer learns from feedback
3. **No Manual Intervention**: Automated quality control
4. **Audit Trail**: All iterations stored in state

### Why State Machine?

1. **Centralized Data**: Single source of truth for workflow data
2. **Agent Independence**: Agents don't need to know about each other
3. **Easy Testing**: Can inject state at any point for testing
4. **History Tracking**: Full conversation history maintained

## 🚀 Execution Flow Example

```
User: "Create a notebook to read CSV from S3, transform with PySpark, and write to Delta Lake"
    ↓
Product Manager: Creates detailed specifications
    ↓
Data Engineer: Writes PySpark code implementation
    ↓
Senior Architect: Reviews code → Finds issue (hardcoded password)
    ↓
Data Engineer: Revises code (uses dbutils.secrets.get())
    ↓
Senior Architect: Reviews again → "APPROVED"
    ↓
DevOps Assembler: Packages code into databricks_notebook_20260308_181216.ipynb
    ↓
Output: Notebook saved to generated_notebooks/
```

## 📈 Scalability Considerations

### Current Limitations
- Sequential processing (no parallelism)
- Unbounded retry loop (could infinite loop)
- Single notebook per execution
- No batch processing

### Potential Enhancements
1. Add max retry limit for Senior Architect feedback loop
2. Implement parallel agent execution for independent tasks
3. Add caching layer for repeated specifications
4. Support batch notebook generation

## 🔐 Security Architecture

- API keys stored in `.env` file (not committed)
- Secrets management enforced by Senior Architect review
- No hardcoded credentials in generated code
- Databricks secrets integration required in output

---

## Summary

This system uses a **Sequential Pipeline Architecture with Conditional Feedback Loop**, NOT a Hub-and-Spoke model. Communication is entirely **state-mediated** with no direct agent-to-agent API calls. The Senior Architect acts as a quality gate with the ability to send work back to the Data Engineer for revision, creating an iterative refinement loop until code quality standards are met.