"""
startup.py  –  Helper script to launch both the FastAPI backend and
               the Streamlit frontend in separate processes.

Usage:
    python startup.py

This script:
  1. Reads .env for OPENAI_API_KEY (optional if you set it in the OS environment).
  2. Starts uvicorn serving backend.main:app on port 8000.
  3. Starts streamlit run frontend/app.py on port 8501.
  4. Waits; Ctrl-C kills both processes cleanly.
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from the project root (if it exists)
load_dotenv(Path(__file__).parent / ".env")

# Validate API key presence
if not os.getenv("OPENAI_API_KEY"):
    print(
        "[startup] WARNING: OPENAI_API_KEY is not set in environment or .env file.\n"
        "          Requests to /generate will fail until you set it.\n"
        "          Either set it in your OS environment or create a .env file:\n"
        "          OPENAI_API_KEY=sk-...\n"
    )

# Determine Python executable path (handles venv)
python_exe = sys.executable

# Command definitions
backend_cmd = [
    python_exe, "-m", "uvicorn",
    "backend.main:app",
    "--reload",
    "--host", "0.0.0.0",
    "--port", "8000",
]

frontend_cmd = [
    python_exe, "-m", "streamlit",
    "run", "frontend/app.py",
    "--server.port", "8501",
]

print("[startup] Starting FastAPI backend on http://localhost:8000 ...")
backend_proc = subprocess.Popen(backend_cmd)

# Give the backend a moment to initialise before the frontend tries to connect
time.sleep(2)

print("[startup] Starting Streamlit frontend on http://localhost:8501 ...")
frontend_proc = subprocess.Popen(frontend_cmd)

print("[startup] Both services running. Press Ctrl-C to stop.")


def _shutdown(sig, frame):  # noqa: ANN001
    """Gracefully terminate both child processes on Ctrl-C."""
    print("\n[startup] Shutting down...")
    backend_proc.terminate()
    frontend_proc.terminate()
    backend_proc.wait()
    frontend_proc.wait()
    print("[startup] All services stopped.")
    sys.exit(0)


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

# Keep the script alive until interrupted
backend_proc.wait()
frontend_proc.wait()
