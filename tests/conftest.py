"""Shared fixtures. Builds a tiny synthetic model + images so the whole test
suite runs in CI without the 800 MB dataset or a trained checkpoint."""
from __future__ import annotations

import io

import pytest
import torch
from PIL import Image

from src.models.cnn import build_model
from src.models.predict import Predictor

CLASS_NAMES = ["cat", "dog"]


def make_image_bytes(size: tuple[int, int] = (300, 240), color=(120, 90, 60), fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def image_bytes() -> bytes:
    return make_image_bytes()


@pytest.fixture
def sample_image() -> Image.Image:
    return Image.new("RGB", (300, 240), (10, 200, 130))


@pytest.fixture
def predictor() -> Predictor:
    """A Predictor around an untrained model -- exercises the plumbing, not accuracy."""
    torch.manual_seed(0)
    model = build_model("simple_cnn", num_classes=len(CLASS_NAMES), dropout=0.0)
    return Predictor(model, CLASS_NAMES, image_size=224)


@pytest.fixture
def api_client(predictor, monkeypatch):
    """TestClient with the model dependency swapped for the synthetic predictor."""
    from fastapi.testclient import TestClient

    import api.main as main

    monkeypatch.setattr(main, "_predictor", predictor)
    monkeypatch.setattr(main, "_load_error", None)
    # Startup would otherwise try to load the real checkpoint and overwrite it.
    monkeypatch.setattr(main, "load_model", lambda: None)
    with TestClient(main.app) as client:
        monkeypatch.setattr(main, "_predictor", predictor)
        yield client
