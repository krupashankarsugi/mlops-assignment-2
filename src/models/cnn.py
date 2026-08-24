"""Baseline CNN for Cats vs Dogs binary classification.

Four Conv-BN-ReLU-MaxPool blocks (3 -> 32 -> 64 -> 128 -> 256) followed by
global average pooling and a 2-way linear head. Global pooling keeps the head
independent of the input resolution and keeps the parameter count small enough
to train on CPU within the assignment's time budget.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """Baseline convolutional classifier. Emits raw logits of shape (N, 2)."""

    def __init__(
        self, num_classes: int = 2, dropout: float = 0.3, pretrained: bool = False
    ) -> None:
        # `pretrained` is accepted for a uniform registry signature; this model
        # is always trained from scratch, so it is ignored.
        super().__init__()
        self.features = nn.Sequential(
            _block(3, 32),
            _block(32, 64),
            _block(64, 128),
            _block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


class ResNet18Transfer(nn.Module):
    """Transfer-learning comparison model.

    ImageNet-pretrained ResNet-18 with the backbone frozen and a fresh 2-way
    head. Included so the MLflow experiment holds more than one run to compare:
    it shows how much a pretrained feature extractor buys over the from-scratch
    baseline on the same data and the same budget.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.fc.in_features
        # Only this head is trainable, which keeps CPU training tractable.
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
    "resnet18_transfer": ResNet18Transfer,
}


def build_model(
    name: str = "simple_cnn",
    num_classes: int = 2,
    dropout: float = 0.3,
    pretrained: bool = True,
) -> nn.Module:
    """Factory so the model architecture is selectable from params.yaml.

    Pass ``pretrained=False`` when the weights are about to be overwritten by a
    checkpoint. Otherwise torchvision fetches the ImageNet weights into
    ``~/.cache/torch`` first -- pointless work that also makes the inference
    container require network access and a writable HOME at startup.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model_name {name!r}; expected one of {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](
        num_classes=num_classes, dropout=dropout, pretrained=pretrained
    )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Transfer models freeze most of their weights, so the trainable count and the
    total count differ by orders of magnitude -- report whichever the caller
    actually means rather than silently conflating them.
    """
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
