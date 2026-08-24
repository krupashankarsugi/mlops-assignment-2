"""Typed access to params.yaml.

Every script reads its settings from here so that the DVC pipeline, the training
job and the inference service all agree on image size, class order and paths.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = Path(os.getenv("PARAMS_PATH", PROJECT_ROOT / "params.yaml"))

# ImageNet statistics -- used for normalisation at train and at inference time.
# Kept in code (not params.yaml) because changing them silently invalidates a
# serialized model, so they are treated as part of the model contract.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    image_size: int = 224
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    max_images_per_class: int = 2000


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "simple_cnn"
    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    num_workers: int = 2
    seed: int = 42
    early_stopping_patience: int = 3


@dataclass(frozen=True)
class AugmentConfig:
    horizontal_flip: float = 0.5
    rotation_degrees: int = 15
    color_jitter: float = 0.2
    random_resized_crop_scale: tuple[float, float] = (0.8, 1.0)


@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "cats-vs-dogs"
    registered_model_name: str = "cats-vs-dogs-cnn"


@dataclass(frozen=True)
class ServingConfig:
    model_path: str = "models/model.pt"
    metadata_path: str = "models/model_metadata.json"
    class_names: list[str] = field(default_factory=lambda: ["cat", "dog"])


@dataclass(frozen=True)
class Config:
    data: DataConfig
    train: TrainConfig
    augment: AugmentConfig
    mlflow: MlflowConfig
    serving: ServingConfig

    def path(self, relative: str) -> Path:
        """Resolve a repo-relative path against the project root."""
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    raw = _load_yaml(Path(path) if path else PARAMS_PATH)
    augment_raw = dict(raw.get("augment", {}))
    scale = augment_raw.get("random_resized_crop_scale")
    if scale is not None:
        augment_raw["random_resized_crop_scale"] = tuple(scale)
    return Config(
        data=DataConfig(**raw.get("data", {})),
        train=TrainConfig(**raw.get("train", {})),
        augment=AugmentConfig(**augment_raw),
        mlflow=MlflowConfig(**raw.get("mlflow", {})),
        serving=ServingConfig(**raw.get("serving", {})),
    )
