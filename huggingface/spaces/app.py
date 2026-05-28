"""Wrapper that re-uses src/fabsentinel/ui/app.py for Hugging Face Spaces deploy."""
import sys
from pathlib import Path

# When deployed to a Space, the repo is the root; src/ is on the path via pyproject.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Import to execute the Streamlit script.
from fabsentinel.ui import app  # noqa: F401
