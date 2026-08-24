"""Stage 2 of the DVC pipeline: clean, resize and split the raw images.

Produces ``data/processed/{train,val,test}/{cat,dog}/*.jpg`` where every image
is a 224x224 RGB JPEG, ready to be consumed by a standard CNN.

The public helpers (:func:`is_valid_image`, :func:`resize_image`,
:func:`split_indices`) are pure and are covered by the unit tests in
``tests/test_preprocess.py``.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.config import load_config

LOGGER = logging.getLogger("preprocess")

# Raw folder name -> canonical class label used everywhere downstream.
CLASS_MAP = {"Cat": "cat", "Dog": "dog"}
SPLITS = ("train", "val", "test")

# Pillow refuses to decode very large files by default; the corpus is clean
# enough that we would rather skip a bad file than raise.
Image.MAX_IMAGE_PIXELS = None


def is_valid_image(path: Path) -> bool:
    """Return True when *path* is a decodable, non-empty image file.

    The Cats vs Dogs corpus ships a handful of zero-byte and truncated JPEGs
    that crash training loaders, so they are filtered out here.
    """
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with Image.open(path) as img:
            img.verify()  # cheap header/structure check
        # verify() invalidates the handle, so re-open to force a full decode.
        with Image.open(path) as img:
            img.convert("RGB").load()
        return True
    except (OSError, UnidentifiedImageError, ValueError):
        return False


def resize_image(img: Image.Image, size: int) -> Image.Image:
    """Convert to RGB and resize to a square ``size`` x ``size`` image."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    return img.convert("RGB").resize((size, size), Image.BILINEAR)


def split_indices(
    n: int,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Deterministically partition ``range(n)`` into train/val/test index lists.

    Ratios must sum to 1.0. Every index lands in exactly one split, so the three
    returned lists are disjoint and together cover ``range(n)``.
    """
    total = train_split + val_split + test_split
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"splits must sum to 1.0, got {total}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_train = int(n * train_split)
    n_val = int(n * val_split)
    # Any rounding remainder goes to the test split so nothing is dropped.
    return (
        indices[:n_train],
        indices[n_train : n_train + n_val],
        indices[n_train + n_val :],
    )


def _collect_source_images(raw_dir: Path, raw_class: str, limit: int) -> list[Path]:
    src = raw_dir / "PetImages" / raw_class
    if not src.is_dir():
        raise FileNotFoundError(
            f"missing raw class directory {src}. Run `python -m src.data.download` first."
        )
    # Sorted for determinism; the shuffle in split_indices provides randomness.
    paths = sorted(src.glob("*.jpg"), key=lambda p: p.name)
    valid: list[Path] = []
    skipped = 0
    for p in paths:
        if limit and len(valid) >= limit:
            break
        if is_valid_image(p):
            valid.append(p)
        else:
            skipped += 1
    LOGGER.info("%s: %d usable images (%d corrupt/skipped)", raw_class, len(valid), skipped)
    return valid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess Cats vs Dogs images")
    parser.add_argument("--clean", action="store_true", help="wipe the processed dir first")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    dcfg = cfg.data
    raw_dir = cfg.path(dcfg.raw_dir)
    out_dir = cfg.path(dcfg.processed_dir)

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    for split in SPLITS:
        for label in CLASS_MAP.values():
            (out_dir / split / label).mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
    for raw_class, label in CLASS_MAP.items():
        sources = _collect_source_images(raw_dir, raw_class, dcfg.max_images_per_class)
        tr, va, te = split_indices(
            len(sources), dcfg.train_split, dcfg.val_split, dcfg.test_split, dcfg.seed
        )
        for split, idxs in zip(SPLITS, (tr, va, te)):
            dest = out_dir / split / label
            for i in idxs:
                src_path = sources[i]
                try:
                    with Image.open(src_path) as img:
                        resize_image(img, dcfg.image_size).save(
                            dest / f"{label}_{src_path.stem}.jpg", "JPEG", quality=90
                        )
                except (OSError, UnidentifiedImageError):
                    LOGGER.warning("failed late on %s, skipping", src_path)
                    continue
            summary[split][label] = len(list(dest.glob("*.jpg")))
            LOGGER.info("%s/%s -> %d images", split, label, summary[split][label])

    stats = {
        "image_size": dcfg.image_size,
        "splits": summary,
        "total": sum(v for s in summary.values() for v in s.values()),
        "seed": dcfg.seed,
        "class_names": sorted(CLASS_MAP.values()),
    }
    stats_path = out_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s (total %d images)", stats_path, stats["total"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
