"""Pydantic request/response models for the inference API."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    version: str
    uptime_seconds: float


class PredictionResponse(BaseModel):
    label: str = Field(..., description="Predicted class", examples=["dog"])
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(..., description="Probability per class")
    inference_ms: float = Field(..., description="Server-side inference latency")
    request_id: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    framework: str
    class_names: list[str]
    image_size: int
    parameters_total: int | None = None
    parameters_trainable: int | None = None
    test_metrics: dict[str, float] = {}
    trained_at: str | None = None


class StatsResponse(BaseModel):
    """Lightweight in-app counters, mirrored by the Prometheus /metrics endpoint."""

    total_requests: int
    predictions: int
    errors: int
    predictions_by_class: dict[str, int]
    avg_latency_ms: float
    p95_latency_ms: float
    uptime_seconds: float


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
