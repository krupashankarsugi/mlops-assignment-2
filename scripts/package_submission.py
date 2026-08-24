#!/usr/bin/env python3
"""Build the submission zip (Deliverable 1).

Bundles all source code, configuration (DVC, CI/CD, Docker, K8s manifests),
the trained model artifacts and the reports -- while excluding the dataset,
the virtualenv, and the raw MLflow store, which are far too large to submit.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directories copied wholesale.
INCLUDE_DIRS = [
    "src", "api", "tests", "scripts", "k8s", "monitoring",
    ".github", "notebooks", "reports",
]

# Individual files at the repo root.
INCLUDE_FILES = [
    "README.md", "SUBMISSION.md", "Makefile", "Dockerfile", ".dockerignore",
    "docker-compose.yml", "requirements.txt", "requirements-train.txt",
    "requirements-dev.txt", "params.yaml", "dvc.yaml", "dvc.lock",
    "pytest.ini", "ruff.toml", ".gitignore", ".dvc/config",
    "models/model.pt", "models/model_metadata.json",
    "data/processed/dataset_stats.json",
]

# Skipped anywhere in the tree.
EXCLUDE_PARTS = {
    "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", ".git",
    "mlruns", "assignment_1", "node_modules", ".ipynb_checkpoints",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".mov", ".mp4", ".zip"}


def is_excluded(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    return path.name == ".DS_Store"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package the assignment submission")
    parser.add_argument(
        "--name", default="MLOps_Assignment02_KrupaShankar", help="zip base name"
    )
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT))
    args = parser.parse_args(argv)

    out_path = Path(args.out_dir) / f"{args.name}.zip"
    root_name = args.name
    added, missing = 0, []

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in INCLUDE_FILES:
            src = PROJECT_ROOT / rel
            if not src.exists():
                missing.append(rel)
                continue
            zf.write(src, f"{root_name}/{rel}")
            added += 1

        for dirname in INCLUDE_DIRS:
            base = PROJECT_ROOT / dirname
            if not base.is_dir():
                missing.append(f"{dirname}/")
                continue
            for src in sorted(base.rglob("*")):
                if src.is_dir() or is_excluded(src.relative_to(PROJECT_ROOT)):
                    continue
                zf.write(src, f"{root_name}/{src.relative_to(PROJECT_ROOT)}")
                added += 1

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path}")
    print(f"  files : {added}")
    print(f"  size  : {size_mb:.2f} MB")
    if missing:
        print(f"  note  : {len(missing)} expected path(s) not present: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
