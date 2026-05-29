.PHONY: install env test lint format clean demo ingest quality univariate multivariate vm rca scorecard memo ui all-artifacts space-build deploy-hf

PYTHON := python
UV := uv

# === Setup ===

install:
	@command -v uv >/dev/null 2>&1 || { echo "Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	$(UV) venv --python 3.12
	$(UV) pip install -e ".[dev]"
	@echo ""
	@echo "  Done. Activate the venv with:  source .venv/bin/activate"
	@echo ""

env:
	@echo "PYTHON_VERSION=$$($(PYTHON) --version)"
	@echo "UV_VERSION=$$($(UV) --version 2>/dev/null || echo 'not installed')"
	@$(PYTHON) -c "import fabsentinel; print('fabsentinel:', fabsentinel.__version__)" 2>/dev/null || echo "fabsentinel not yet installed (run: make install)"

# === Quality gates ===

test:
	pytest -m "not slow" -v

test-all:
	pytest -v

lint:
	ruff check src tests

format:
	ruff format src tests

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf artifacts/scorecards/*.pdf artifacts/memos/*.pdf

# === Pipeline stages (each is idempotent and resumable) ===

ingest:
	$(PYTHON) -m fabsentinel.cli ingest

quality:
	$(PYTHON) -m fabsentinel.cli quality

univariate:
	$(PYTHON) -m fabsentinel.cli univariate

multivariate:
	$(PYTHON) -m fabsentinel.cli multivariate

vm:
	$(PYTHON) -m fabsentinel.cli train-vm

rca:
	$(PYTHON) -m fabsentinel.cli rca

scorecard:
	$(PYTHON) -m fabsentinel.cli scorecard

memo:
	$(PYTHON) -m fabsentinel.cli memo --week current

# === Full end-to-end demo ===

demo: install ingest quality univariate multivariate vm rca scorecard
	@echo ""
	@echo "  DEMO COMPLETE. Artifacts in:"
	@echo "    - artifacts/scorecards/   (weekly SPC scorecard PDF)"
	@echo "    - artifacts/memos/        (8D excursion memos)"
	@echo "    - mlruns/                 (MLflow tracking)"
	@echo ""
	@echo "  Launch the UI with:  make ui"
	@echo ""

ui:
	streamlit run src/fabsentinel/ui/app.py

# === Everything ===

all-artifacts: ingest quality univariate multivariate vm rca scorecard memo
	@echo "All artifacts regenerated."

# === Hugging Face Space ===

space-build:
	$(PYTHON) scripts/deploy_space.py --build

deploy-hf:
	$(PYTHON) scripts/deploy_space.py --deploy --repo-id kbd0011/fabsentinel
