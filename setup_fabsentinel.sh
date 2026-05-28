#!/usr/bin/env bash
# Mac-bound bootstrap for FabSentinel.  Run from the unzipped repo root:
#   ./setup_fabsentinel.sh
#
# What it does:
#   1. Verifies Python 3.12 (installs via uv if missing).
#   2. Installs uv if not already on PATH.
#   3. Creates .venv with uv.
#   4. Installs FabSentinel + dev dependencies.
#   5. Runs the full pipeline + pytest.
#   6. Tells you how to launch the UI.

set -euo pipefail

YELLOW='\033[1;33m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
say() { echo -e "${YELLOW}=>${NC} $*"; }
ok()  { echo -e "${GREEN}OK${NC} $*"; }
die() { echo -e "${RED}ERROR${NC} $*" >&2; exit 1; }

say "FabSentinel bootstrap — Mac / Apple Silicon edition"

# Verify we are in a FabSentinel repo
[[ -f pyproject.toml ]] || die "Run this from the unzipped fabsentinel/ directory (no pyproject.toml here)."
grep -q '"fabsentinel"' pyproject.toml || die "pyproject.toml does not look like FabSentinel's. Aborting."

# 1. uv installer (idempotent)
if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv (https://astral.sh/uv)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Source the appropriate shell init
  if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
  fi
  export PATH="$HOME/.local/bin:$PATH"
  ok "uv installed."
else
  ok "uv already installed ($(uv --version))"
fi

# 2. Python 3.12 via uv (handles install if missing)
say "Creating .venv with Python 3.12..."
uv venv --python 3.12

# 3. Install in editable mode with dev extras
say "Installing fabsentinel and dev dependencies (this takes ~1-2 minutes)..."
uv pip install -e ".[dev]"
ok "Install complete."

# 4. Activate and run the pipeline
# shellcheck disable=SC1091
source .venv/bin/activate

say "Running test suite..."
pytest -q -m "not slow" || die "Tests failed. See output above."
ok "All tests passed."

say "Running end-to-end pipeline (ingest -> quality -> SPC -> VM -> RCA -> scorecard)..."
make ingest quality univariate multivariate vm rca scorecard
ok "Pipeline complete. Artifacts in artifacts/"

# 5. Optional Quarto check
if ! command -v quarto >/dev/null 2>&1; then
  echo ""
  echo -e "${YELLOW}Note:${NC} Quarto is not installed. Reports rendered as HTML only."
  echo "  To enable PDF rendering of the scorecard and 8D memo:"
  echo "     brew install --cask quarto"
fi

cat << "EOF"

  =======================================================
  FabSentinel is ready.

  Next steps:
    source .venv/bin/activate
    make ui                       # launch Streamlit on http://localhost:8501

  To generate an 8D memo for a specific lot with LLM narrative:
    export OPENAI_API_KEY=sk-...
    make memo                     # auto-picks the worst-T² lot
    # or:
    python -m fabsentinel.cli memo --lot-idx 942 --use-llm

  Artifacts:
    artifacts/scorecards/*.html   # weekly SPC scorecard
    artifacts/memos/*.html        # 8D excursion memos
    mlruns/                       # MLflow tracking
  =======================================================
EOF
