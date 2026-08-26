"""Stage 1 of the DVC pipeline: fetch the Cats vs Dogs dataset.

Fetches the Kaggle dataset named in the assignment brief:
``bhavikjikadara/dog-and-cat-classification-dataset`` -- 24,998 JPEGs, 12,499
per class, laid out as ``PetImages/Cat`` and ``PetImages/Dog``.

Resolution order:
  1. An already-extracted ``data/raw/PetImages`` directory (nothing to do).
  2. An already-downloaded archive in ``data/raw`` (extract only).
  3. The Kaggle CLI, when ``~/.kaggle/kaggle.json`` credentials are configured.
  4. Kaggle's public dataset endpoint, which serves this dataset without
     credentials.
  5. Microsoft's release of the same corpus, as a last-resort fallback.

Step 5 exists only for resilience. Microsoft's archive is the same corpus plus
``Cat/666.jpg`` and ``Dog/11702.jpg``, two corrupt files the Kaggle upload
removed; ``preprocess.is_valid_image`` rejects them either way, so both sources
yield byte-identical processed data.
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

# The dataset named in the assignment brief.
KAGGLE_DATASET = "bhavikjikadara/dog-and-cat-classification-dataset"
KAGGLE_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"

# Same corpus, used only if Kaggle is unreachable. See the module docstring.
MIRROR_URL = (
    "https://download.microsoft.com/download/3/E/1/"
    "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
)
ARCHIVE_NAME = "cats-vs-dogs.zip"
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


def _stream_to(url: str, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive.with_suffix(".part")
    # Kaggle redirects to a signed storage URL; urlopen follows it.
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.rename(archive)


def _download_via_kaggle_public(archive: Path) -> bool:
    """Kaggle serves this dataset without credentials; prefer it over the mirror."""
    LOGGER.info("downloading from Kaggle: %s", KAGGLE_DATASET)
    try:
        _stream_to(KAGGLE_URL, archive)
        return True
    except OSError as exc:
        LOGGER.warning("Kaggle download failed (%s); falling back to mirror", exc)
        return False


def _download_via_mirror(archive: Path) -> None:
    LOGGER.info("downloading dataset from mirror -> %s", archive)
    _stream_to(MIRROR_URL, archive)


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
        if not _download_via_kaggle(raw_dir) and not _download_via_kaggle_public(archive):
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
