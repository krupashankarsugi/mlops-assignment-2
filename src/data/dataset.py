"""Torch datasets / dataloaders with augmentation for the processed images."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.config import NORM_MEAN, NORM_STD, AugmentConfig, Config


def build_train_transform(image_size: int, aug: AugmentConfig) -> transforms.Compose:
    """Augmented pipeline used for the training split only."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=tuple(aug.random_resized_crop_scale), antialias=True
            ),
            transforms.RandomHorizontalFlip(p=aug.horizontal_flip),
            transforms.RandomRotation(aug.rotation_degrees),
            transforms.ColorJitter(
                brightness=aug.color_jitter,
                contrast=aug.color_jitter,
                saturation=aug.color_jitter,
            ),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    """Deterministic pipeline for val/test and for production inference.

    The API reuses this exact function so that serving-time preprocessing can
    never drift away from evaluation-time preprocessing.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


def build_dataloaders(
    cfg: Config, splits: tuple[str, ...] = ("train", "val", "test")
) -> tuple[dict[str, DataLoader], list[str]]:
    """Return ``{split: DataLoader}`` plus the class-name list in label order."""
    processed = cfg.path(cfg.data.processed_dir)
    train_tf = build_train_transform(cfg.data.image_size, cfg.augment)
    eval_tf = build_eval_transform(cfg.data.image_size)

    loaders: dict[str, DataLoader] = {}
    class_names: list[str] = []
    generator = torch.Generator().manual_seed(cfg.train.seed)

    for split in splits:
        root = processed / split
        if not root.is_dir():
            raise FileNotFoundError(
                f"missing {root}. Run `python -m src.data.preprocess` first."
            )
        dataset = ImageFolder(root, transform=train_tf if split == "train" else eval_tf)
        if not class_names:
            class_names = list(dataset.classes)
        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.train.num_workers,
            pin_memory=False,
            generator=generator if split == "train" else None,
            drop_last=False,
        )
    return loaders, class_names


def dataset_sizes(processed_dir: Path) -> dict[str, int]:
    """Count images per split, for logging and for the DVC metrics file."""
    return {
        split.name: sum(1 for _ in split.rglob("*.jpg"))
        for split in sorted(processed_dir.iterdir())
        if split.is_dir()
    }
