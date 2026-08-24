"""Stage 1 of the DVC pipeline: fetch the Cats vs Dogs dataset.

Resolution order:
  1. An already-extracted ``data/raw/PetImages`` directory (nothing to do).
  2. An already-downloaded archive in ``data/raw`` (extract only).
  3. The Kaggle CLI, when credentials are configured.
  4. The Microsoft research mirror of the same corpus (no credentials needed).

The Kaggle "Cats and Dogs" dataset and the Microsoft "Dogs vs. Cats" download
are the same 25k-image corpus, so either source yields an identical pipeline.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from src.config import load_config

LOGGER = logging.getLogger("download")

MIRROR_URL = (
    "https://download.microsoft.com/download/3/E/1/"
    "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
)
ARCHIVE_NAME = "kagglecatsanddogs_5340.zip"
KAGGLE_DATASET = "shaunthesheep/microsoft-catsvsdogs-dataset"
CLASS_DIRS = ("Cat", "Dog")


def _already_extracted(raw_dir: Path) -> bool:
    pet_images = raw_dir / "PetImages"
    return all((pet_images / c).is_dir() for c in CLASS_DIRS)


def _download_via_kaggle(raw_dir: Path) -> bool:
    if shutil.which("kaggle") is None:
        LOGGER.info("kaggle CLI not installed; skipping Kaggle source")
        return False
    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        LOGGER.info("no ~/.kaggle/kaggle.json credentials; skipping Kaggle source")
        return False
    LOGGER.info("downloading via Kaggle CLI: %s", KAGGLE_DATASET)
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(raw_dir)],
        check=False,
    )
    return result.returncode == 0


def _download_via_mirror(archive: Path) -> None:
    LOGGER.info("downloading dataset from mirror -> %s", archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive.with_suffix(".part")
    with urllib.request.urlopen(MIRROR_URL) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.rename(archive)


def _extract(archive: Path, raw_dir: Path) -> None:
    LOGGER.info("extracting %s", archive.name)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)


def _count_images(raw_dir: Path) -> dict[str, int]:
    counts = {}
    for cls in CLASS_DIRS:
        d = raw_dir / "PetImages" / cls
        counts[cls] = sum(1 for _ in d.glob("*.jpg")) if d.is_dir() else 0
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the Cats vs Dogs dataset")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    raw_dir = cfg.path(cfg.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if _already_extracted(raw_dir) and not args.force:
        LOGGER.info("dataset already extracted at %s", raw_dir / "PetImages")
        LOGGER.info("image counts: %s", _count_images(raw_dir))
        return 0

    archive = raw_dir / ARCHIVE_NAME
    if not archive.exists() or args.force:
        if not _download_via_kaggle(raw_dir):
            _download_via_mirror(archive)

    # The Kaggle CLI may name the archive differently; take whatever zip landed.
    if not archive.exists():
        zips = sorted(raw_dir.glob("*.zip"))
        if not zips:
            LOGGER.error("no archive found in %s after download", raw_dir)
            return 1
        archive = zips[0]

    _extract(archive, raw_dir)
    counts = _count_images(raw_dir)
    LOGGER.info("image counts: %s", counts)
    if min(counts.values()) == 0:
        LOGGER.error("extraction did not produce both class folders")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
