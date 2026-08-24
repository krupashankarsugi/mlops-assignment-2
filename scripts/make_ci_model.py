#!/usr/bin/env python3
"""Produce a valid model.pt for CI when the DVC-tracked checkpoint is absent.

The trained weights live in DVC, not Git, so a fresh CI checkout has no
model file. Rather than skip the image build, CI writes a correctly-shaped
untrained checkpoint here: the container then starts, answers /health and
serves /predict, which is exactly what the build-verification step asserts.
Real weights come from `dvc pull` whenever a DVC remote is configured.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import NORM_MEAN, NORM_STD, load_config  # noqa: E402
from src.models.cnn import build_model, count_parameters  # noqa: E402


def main() -> int:
    cfg = load_config()
    class_names = list(cfg.serving.class_names)
    model = build_model(
        cfg.train.model_name, len(class_names), cfg.train.dropout, pretrained=False
    ).eval()

    model_path = cfg.path(cfg.serving.model_path)
    meta_path = cfg.path(cfg.serving.metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": cfg.train.model_name,
            "num_classes": len(class_names),
            "dropout": cfg.train.dropout,
            "class_names": class_names,
            "image_size": cfg.data.image_size,
            "norm_mean": NORM_MEAN,
            "norm_std": NORM_STD,
        },
        model_path,
    )
    meta_path.write_text(
        json.dumps(
            {
                "model_name": cfg.train.model_name,
                "framework": f"pytorch-{torch.__version__}",
                "class_names": class_names,
                "image_size": cfg.data.image_size,
                "parameters_total": count_parameters(model, trainable_only=False),
                "parameters_trainable": count_parameters(model),
                "test_metrics": {},
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "note": "CI placeholder checkpoint (untrained) - not for production use",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote CI placeholder checkpoint -> {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
