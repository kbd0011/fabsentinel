"""Assemble a Hugging Face Space build tree and optionally deploy it.

The Space is a focused subset of the repo: the package, its config, and the
data/model artifacts the Streamlit app reads at runtime — plus the three HF
files (app.py, requirements.txt, README.md with frontmatter) from
``huggingface/spaces/``. We upload via ``huggingface_hub`` so large binaries
are handled over HTTP/LFS automatically; ``create_repo(exist_ok=True)`` makes
re-deploys idempotent.

Usage::

    python scripts/deploy_space.py --build                       # assemble space_build/ only
    python scripts/deploy_space.py --deploy --repo-id kbd0011/fabsentinel
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "space_build"
SPACE_SRC = REPO_ROOT / "huggingface" / "spaces"

# Paths (relative to repo root) the running app needs. Dirs copied recursively.
INCLUDE = [
    "src/fabsentinel",
    "configs/config.yaml",
    "data/processed.parquet",
    "data/cleaned.parquet",  # quality-filtered (440 cols) — what every model was trained on
    "data/secom.data",
    "data/secom_labels.data",
    "data/metadata.json",
    "artifacts/vm_model.joblib",
    "artifacts/mspc_model.joblib",
    "artifacts/multivariate_stats.csv",
    "artifacts/shap_global.csv",
    "artifacts/univariate_ooc.csv",
    "artifacts/quality_report.json",
]


def build() -> Path:
    """Assemble ``space_build/`` from the HF files + the runtime manifest."""
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)

    for name in ("app.py", "requirements.txt", "README.md", "Dockerfile"):
        shutil.copy2(SPACE_SRC / name, BUILD_DIR / name)

    for rel in INCLUDE:
        src = REPO_ROOT / rel
        dst = BUILD_DIR / rel
        if not src.exists():
            raise FileNotFoundError(f"manifest entry missing: {rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)

    print(f"built {BUILD_DIR} ({sum(1 for _ in BUILD_DIR.rglob('*') if _.is_file())} files)")
    return BUILD_DIR


def deploy(repo_id: str) -> str:
    """Build, create the Space if needed, and upload the build tree."""
    from huggingface_hub import HfApi, create_repo

    build()
    create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api = HfApi()
    api.upload_folder(
        folder_path=str(BUILD_DIR),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy FabSentinel demo",
    )
    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"deployed -> {url}")
    return url


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="assemble space_build/ only")
    ap.add_argument("--deploy", action="store_true", help="build and upload to the Space")
    ap.add_argument("--repo-id", default="kbdev0011/fabsentinel")
    args = ap.parse_args()
    if args.deploy:
        deploy(args.repo_id)
    else:
        build()


if __name__ == "__main__":
    main()
