# databricks_notebook-_generator
# databricks_notebook-_generator
# Databricks Notebook Generator

An AI-powered agentic system that automatically generates production-ready Databricks notebooks using a multi-agent architecture powered by LangGraph.

## 🏗️ Architecture Overview

This project implements a multi-agent workflow where specialized AI agents collaborate to plan, code, review, and assemble Databricks notebooks from natural language requirements.

### Agent Workflow

```
User Requirement → Product Manager → Data Engineer → Senior Architect → DevOps Assembler → Databricks Notebook
```

### Multi-Agent System

The system consists of four specialized agents that work in sequence:

1. **Product Manager (Planner)** - [`prompts/Product_Manager_Planner.txt`](prompts/Product_Manager_Planner.txt)
   - Analyzes user requirements
   - Creates detailed technical specifications
   - Defines data sources, transformations, and outputs

2. **Data Engineer (Coder)** - [`prompts/Data_Engineer_Coder.txt`](prompts/Data_Engineer_Coder.txt)
   - Writes PySpark code based on specifications
   - Implements Delta Lake operations
   - Follows Databricks best practices

3. **Senior Architect (Reviewer)** - [`prompts/Senior_Architect_Reviewer.txt`](prompts/Senior_Architect_Reviewer.txt)
   - Reviews code for performance bottlenecks
   - Validates Delta Lake usage patterns
   - Ensures security compliance (secrets management)
   - Returns feedback loop to Data Engineer if issues found

4. **DevOps Assembler** - [`prompts/DevOps_Assembler.txt`](prompts/DevOps_Assembler.txt)
   - Packages code into Databricks notebook format (.ipynb)
   - Adds metadata and cell structure
   - Prepares notebook for deployment

## 📁 Project Structure

```
.
├── backend/                          # Core application logic
│   ├── agents.py                     # Agent definitions and implementations
│   ├── graph.py                      # LangGraph workflow orchestration
│   ├── main.py                       # Backend API/entry point
│   ├── prompts.py                    # Prompt management
│   └── state.py                      # State management for agents
│
├── frontend/                         # User interface
│   └── app.py                        # Streamlit/Gradio frontend
│
├── prompts/                          # Agent system prompts
│   ├── Product_Manager_Planner.txt
│   ├── Data_Engineer_Coder.txt
│   ├── Senior_Architect_Reviewer.txt
│   └── DevOps_Assembler.txt
│
├── generated_notebooks/              # Output directory for generated notebooks
│   └── databricks_notebook_*.ipynb
│
├── .env.example                      # Environment variables template
├── requirements.txt                  # Python dependencies
└── startup.py                        # Application startup script
```

## 🔄 Workflow Details

### State Management
The system uses a state machine ([`backend/state.py`](backend/state.py)) to track:
- Current agent in workflow
- User requirements
- Generated specifications
- Code iterations
- Review feedback
- Final notebook output

### Graph Architecture
The LangGraph implementation ([`backend/graph.py`](backend/graph.py)) creates a directed graph where:
- Each node represents an agent
- Edges define the workflow transitions
- Conditional edges handle review feedback loops
- State is passed between nodes

### Feedback Loop
If the Senior Architect finds issues:
```
Data Engineer → Senior Architect → [ISSUES FOUND] → Data Engineer → Senior Architect → [APPROVED] → DevOps
```

## 🔧 Key Features

- **Multi-Agent Collaboration**: Specialized agents work together with defined roles
- **Iterative Code Review**: Automatic feedback loop until code meets quality standards
- **Databricks Best Practices**: Built-in checks for performance, security, and Delta Lake patterns
- **Production-Ready Output**: Generated notebooks are ready for Databricks deployment
- **Conversation History**: Tracks all agent interactions and iterations

## 🛠️ Technology Stack

- **Framework**: LangGraph (for agent orchestration)
- **Backend**: Python
- **Frontend**: Streamlit/Gradio (in [`frontend/app.py`](frontend/app.py))
- **AI/LLM**: OpenAI/Anthropic Claude (configurable)
- **Output Format**: Jupyter Notebook (.ipynb) for Databricks

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key or Anthropic API key
- Databricks workspace (for deploying generated notebooks)

## ⚙️ Configuration

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Configure your `.env` file with:
   ```env
   OPENAI_API_KEY=your_api_key_here
   # or
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rohith-ak/databricks_notebook-_generator.git
   cd databricks_notebook-_generator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 How to Run

### Option 1: Using Startup Script
```bash
python startup.py
```

### Option 2: Run Frontend Directly
```bash
python frontend/app.py
```

### Option 3: Run Backend API
```bash
python backend/main.py
```

The application will start and you can:
1. Enter your Databricks notebook requirements in natural language
2. Watch as agents collaborate to generate the notebook
3. Find the generated notebook in the [`generated_notebooks/`](generated_notebooks/) directory
4. Upload the notebook to your Databricks workspace

---

**Generated notebooks** are saved with timestamps: `databricks_notebook_YYYYMMDD_HHMMSS.ipynb`