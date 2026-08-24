"""Model loading and single-image inference.

Shared by the FastAPI service, the unit tests and the batch evaluation script so
that there is exactly one definition of "how a prediction is made".
"""
from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from src.config import Config, load_config
from src.data.dataset import build_eval_transform
from src.models.cnn import build_model

LOGGER = logging.getLogger("predict")

DEFAULT_CLASS_NAMES = ["cat", "dog"]


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes are not a decodable image."""


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 6),
            "probabilities": {k: round(v, 6) for k, v in self.probabilities.items()},
        }


class Predictor:
    """Holds the loaded model plus the preprocessing pipeline it was trained with."""

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: list[str],
        image_size: int,
        device: torch.device | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()
        self.class_names = class_names
        self.image_size = image_size
        self.metadata = metadata or {}
        self.transform = build_eval_transform(image_size)

    @classmethod
    def from_checkpoint(
        cls, model_path: str | Path, metadata_path: str | Path | None = None
    ) -> Predictor:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"model checkpoint not found at {model_path}. "
                "Train one with `python -m src.models.train`."
            )
        # weights_only=False: the checkpoint is a trusted artifact we produced,
        # and it carries the class names / image size alongside the weights.
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        class_names = ckpt.get("class_names", DEFAULT_CLASS_NAMES)
        # pretrained=False: the checkpoint's state_dict is authoritative, and
        # this keeps startup offline and free of any cache writes.
        model = build_model(
            ckpt.get("model_name", "simple_cnn"),
            ckpt.get("num_classes", len(class_names)),
            ckpt.get("dropout", 0.3),
            pretrained=False,
        )
        model.load_state_dict(ckpt["state_dict"])

        metadata = {}
        if metadata_path and Path(metadata_path).exists():
            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        LOGGER.info("loaded model from %s (classes=%s)", model_path, class_names)
        return cls(model, class_names, ckpt.get("image_size", 224), metadata=metadata)

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> Predictor:
        cfg = cfg or load_config()
        return cls.from_checkpoint(
            cfg.path(cfg.serving.model_path), cfg.path(cfg.serving.metadata_path)
        )

    def preprocess_bytes(self, payload: bytes) -> torch.Tensor:
        """Decode raw upload bytes into a normalised (1, 3, H, W) batch tensor."""
        if not payload:
            raise InvalidImageError("empty request body")
        try:
            with Image.open(io.BytesIO(payload)) as img:
                rgb = img.convert("RGB")
                return self.transform(rgb).unsqueeze(0)
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError(f"could not decode image: {exc}") from exc

    @torch.no_grad()
    def predict_tensor(self, batch: torch.Tensor) -> list[Prediction]:
        probs = torch.softmax(self.model(batch.to(self.device)), dim=1).cpu()
        results = []
        for row in probs:
            idx = int(row.argmax())
            results.append(
                Prediction(
                    label=self.class_names[idx],
                    confidence=float(row[idx]),
                    probabilities={
                        name: float(row[i]) for i, name in enumerate(self.class_names)
                    },
                )
            )
        return results

    def predict_bytes(self, payload: bytes) -> Prediction:
        return self.predict_tensor(self.preprocess_bytes(payload))[0]

    def predict_path(self, path: str | Path) -> Prediction:
        return self.predict_bytes(Path(path).read_bytes())


_PREDICTOR: Predictor | None = None


def get_predictor(cfg: Config | None = None) -> Predictor:
    """Process-wide singleton so the model is deserialised only once."""
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = Predictor.from_config(cfg)
    return _PREDICTOR


def reset_predictor() -> None:
    """Drop the cached instance (used by tests)."""
    global _PREDICTOR
    _PREDICTOR = None
