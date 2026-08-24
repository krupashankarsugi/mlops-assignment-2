#!/usr/bin/env python3
"""Offline prediction CLI -- classify images without starting the API.

Usage:
    python scripts/predict_cli.py path/to/image.jpg
    python scripts/predict_cli.py data/processed/test/dog --limit 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.predict import Predictor  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify images as cat or dog")
    parser.add_argument("target", help="image file or directory of images")
    parser.add_argument("--limit", type=int, default=10, help="max images when given a directory")
    args = parser.parse_args(argv)

    target = Path(args.target)
    paths = (
        sorted(target.rglob("*.jpg"))[: args.limit] if target.is_dir() else [target]
    )
    if not paths:
        raise SystemExit(f"no images found at {target}")

    predictor = Predictor.from_config()
    for path in paths:
        result = predictor.predict_path(path)
        probs = "  ".join(f"{k}={v:.4f}" for k, v in result.probabilities.items())
        print(f"{path.name:<28} -> {result.label:<4} ({result.confidence:.4f})   {probs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
