#!/usr/bin/env python3
"""Unified project launcher for RAG-FOR-RAGS.

Usage:
    python run.py cli              # Launch interactive terminal chat
    python run.py api              # Launch FastAPI web server
    python run.py ingest           # Wipe and reingest documents into ChromaDB
    python run.py eval             # Run automated evaluation benchmark
    python run.py test             # Run test suite
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def run_cli():
    """Run interactive CLI chat."""
    import src.main as cli_main
    cli_main.main()

def run_api():
    """Start Uvicorn FastAPI server."""
    import uvicorn
    from src.config import settings
    print(f"Starting RAG API server on http://localhost:8000 (docs at http://localhost:8000/docs)")
    print(f"API Key: {settings.api_key}")
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)

def run_ingest():
    """Rebuild database and BM25 index."""
    script = PROJECT_ROOT / "scripts" / "reingest.py"
    cmd = [sys.executable, str(script)]
    subprocess.run(cmd, check=True)

def run_eval():
    """Run evaluation benchmark."""
    script = PROJECT_ROOT / "eval" / "run_eval.py"
    cmd = [sys.executable, str(script)]
    subprocess.run(cmd, check=True)

def run_tests():
    """Run pytest suite."""
    cmd = [sys.executable, "-m", "pytest", "-v"]
    subprocess.run(cmd, check=True)

def print_help():
    print(__doc__)

def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    action = sys.argv[1].lower()
    if action in ("cli", "chat"):
        run_cli()
    elif action in ("api", "server"):
        run_api()
    elif action in ("ingest", "reingest"):
        run_ingest()
    elif action in ("eval", "benchmark"):
        run_eval()
    elif action in ("test", "tests"):
        run_tests()
    else:
        print(f"Unknown command: '{action}'\n")
        print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
