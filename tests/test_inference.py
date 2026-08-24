"""Unit tests for the model utility / inference functions (M3 requirement)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.cnn import MODEL_REGISTRY, build_model, count_parameters
from src.models.predict import InvalidImageError, Prediction, Predictor
from src.models.train import compute_metrics
from tests.conftest import CLASS_NAMES, make_image_bytes


class TestModelFactory:
    def test_builds_the_baseline_cnn(self):
        model = build_model("simple_cnn", num_classes=2, dropout=0.3)
        assert count_parameters(model) > 0

    def test_rejects_unknown_architecture(self):
        with pytest.raises(ValueError, match="unknown model_name"):
            build_model("resnet999")

    def test_registry_exposes_both_architectures(self):
        assert set(MODEL_REGISTRY) == {"simple_cnn", "resnet18_transfer"}

    def test_transfer_model_freezes_its_backbone(self):
        # Only the replaced classification head should be trainable, otherwise
        # the "transfer" run would be a full fine-tune on CPU.
        model = build_model("resnet18_transfer", num_classes=2)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert trainable, "the new head must remain trainable"
        assert all("fc" in n for n in trainable)

    def test_forward_pass_returns_two_logits_per_image(self):
        model = build_model("simple_cnn", num_classes=2).eval()
        with torch.no_grad():
            out = model(torch.randn(4, 3, 224, 224))
        assert out.shape == (4, 2)
        assert torch.isfinite(out).all()

    def test_is_resolution_agnostic_via_global_pooling(self):
        model = build_model("simple_cnn", num_classes=2).eval()
        with torch.no_grad():
            assert model(torch.randn(1, 3, 128, 128)).shape == (1, 2)


class TestPredictorPreprocessing:
    def test_produces_a_normalised_batch_tensor(self, predictor, image_bytes):
        tensor = predictor.preprocess_bytes(image_bytes)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_normalisation_shifts_data_off_the_zero_one_range(self, predictor):
        # A pure-white image maps to strictly positive normalised values.
        tensor = predictor.preprocess_bytes(make_image_bytes(color=(255, 255, 255)))
        assert tensor.min() > 1.0

    def test_accepts_png_as_well_as_jpeg(self, predictor):
        tensor = predictor.preprocess_bytes(make_image_bytes(fmt="PNG"))
        assert tensor.shape == (1, 3, 224, 224)

    def test_rejects_empty_payload(self, predictor):
        with pytest.raises(InvalidImageError, match="empty"):
            predictor.preprocess_bytes(b"")

    def test_rejects_non_image_payload(self, predictor):
        with pytest.raises(InvalidImageError):
            predictor.preprocess_bytes(b"not an image at all")


class TestPredictorInference:
    def test_returns_a_known_class_with_valid_probabilities(self, predictor, image_bytes):
        result = predictor.predict_bytes(image_bytes)
        assert isinstance(result, Prediction)
        assert result.label in CLASS_NAMES
        assert 0.0 <= result.confidence <= 1.0
        assert pytest.approx(sum(result.probabilities.values()), abs=1e-5) == 1.0

    def test_confidence_matches_the_winning_class(self, predictor, image_bytes):
        result = predictor.predict_bytes(image_bytes)
        assert result.confidence == max(result.probabilities.values())
        assert result.probabilities[result.label] == result.confidence

    def test_is_deterministic_in_eval_mode(self, predictor, image_bytes):
        first = predictor.predict_bytes(image_bytes)
        second = predictor.predict_bytes(image_bytes)
        assert first.label == second.label
        assert first.confidence == pytest.approx(second.confidence)

    def test_batches_are_handled_per_row(self, predictor, image_bytes):
        batch = torch.cat([predictor.preprocess_bytes(image_bytes) for _ in range(3)])
        results = predictor.predict_tensor(batch)
        assert len(results) == 3
        assert all(r.label in CLASS_NAMES for r in results)

    def test_predict_path_reads_from_disk(self, predictor, tmp_path, sample_image):
        path = tmp_path / "pet.jpg"
        sample_image.save(path, "JPEG")
        assert predictor.predict_path(path).label in CLASS_NAMES

    def test_as_dict_is_json_serialisable(self, predictor, image_bytes):
        payload = predictor.predict_bytes(image_bytes).as_dict()
        assert set(payload) == {"label", "confidence", "probabilities"}

    def test_missing_checkpoint_raises_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="model checkpoint not found"):
            Predictor.from_checkpoint(tmp_path / "absent.pt")

    def test_loading_a_checkpoint_never_fetches_pretrained_weights(self, tmp_path, monkeypatch):
        """Regression: a read-only container must be able to load a transfer model.

        Reconstructing the architecture with pretrained=True made torchvision
        write ImageNet weights to ~/.cache/torch, which fails under
        `readOnlyRootFilesystem: true` and needs network access at startup.
        """
        import torchvision.models as tv_models

        model = build_model("resnet18_transfer", num_classes=2, pretrained=False)
        ckpt = tmp_path / "transfer.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_name": "resnet18_transfer",
                "num_classes": 2,
                "dropout": 0.3,
                "class_names": CLASS_NAMES,
                "image_size": 224,
            },
            ckpt,
        )

        def explode(*args, **kwargs):
            raise AssertionError("resnet18 must not download weights when loading a checkpoint")

        monkeypatch.setattr(tv_models.ResNet18_Weights.IMAGENET1K_V1, "get_state_dict", explode)
        loaded = Predictor.from_checkpoint(ckpt)
        assert loaded.class_names == CLASS_NAMES


class TestComputeMetrics:
    def test_perfect_predictions_score_one(self):
        y_true = np.array([0, 1, 0, 1])
        metrics = compute_metrics(y_true, y_true, np.array([0.1, 0.9, 0.2, 0.8]))
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_accuracy_reflects_partial_correctness(self):
        metrics = compute_metrics(
            np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]), np.array([0.2, 0.9, 0.6, 0.8])
        )
        assert metrics["accuracy"] == pytest.approx(0.75)

    def test_roc_auc_omitted_for_single_class_ground_truth(self):
        metrics = compute_metrics(np.zeros(4), np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]))
        assert "roc_auc" not in metrics
