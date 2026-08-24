"""Request/response logging and metrics for the inference service (M5).

Provides three things:
  * a structured JSON access log (no image bytes -- only shape and outcome),
  * in-process counters exposed at ``/stats``,
  * Prometheus counters/histograms exposed at ``/metrics``.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import Counter, deque

from prometheus_client import CollectorRegistry, Histogram
from prometheus_client import Counter as PromCounter

SERVICE_NAME = "cats-vs-dogs-api"


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line so logs are greppable in Docker/Kubernetes."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "service": SERVICE_NAME,
            "message": record.getMessage(),
        }
        # Anything attached via logger.info(..., extra={"event": {...}}) is merged
        # in. Callers only ever pass metadata here -- never raw image content.
        if hasattr(record, "event"):
            payload.update(record.event)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # uvicorn's own access log would duplicate our structured entries.
    logging.getLogger("uvicorn.access").disabled = True
    for noisy in ("uvicorn.error", "uvicorn"):
        logging.getLogger(noisy).handlers = [handler]
    return logging.getLogger(SERVICE_NAME)


# --- Prometheus metrics -----------------------------------------------------
# A dedicated registry keeps repeated imports (e.g. in tests) from raising
# duplicate-timeseries errors against the global default registry.
REGISTRY = CollectorRegistry()

REQUEST_COUNT = PromCounter(
    "inference_requests_total",
    "Total HTTP requests handled by the inference service",
    ["endpoint", "method", "status"],
    registry=REGISTRY,
)
PREDICTION_COUNT = PromCounter(
    "predictions_total",
    "Total predictions served, labelled by predicted class",
    ["predicted_class"],
    registry=REGISTRY,
)
ERROR_COUNT = PromCounter(
    "inference_errors_total",
    "Total failed requests",
    ["endpoint", "reason"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "End-to-end request latency in seconds",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)


class InAppMetrics:
    """Thread-safe counters backing the ``/stats`` endpoint."""

    def __init__(self, window: int = 1000) -> None:
        self._lock = threading.Lock()
        self._started = time.time()
        self.total_requests = 0
        self.predictions = 0
        self.errors = 0
        self.by_class: Counter[str] = Counter()
        self._latencies: deque[float] = deque(maxlen=window)

    def record_request(self, latency_ms: float) -> None:
        with self._lock:
            self.total_requests += 1
            self._latencies.append(latency_ms)

    def record_prediction(self, label: str) -> None:
        with self._lock:
            self.predictions += 1
            self.by_class[label] += 1

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def snapshot(self) -> dict:
        with self._lock:
            latencies = sorted(self._latencies)
            avg = sum(latencies) / len(latencies) if latencies else 0.0
            p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)] if latencies else 0.0
            return {
                "total_requests": self.total_requests,
                "predictions": self.predictions,
                "errors": self.errors,
                "predictions_by_class": dict(self.by_class),
                "avg_latency_ms": round(avg, 3),
                "p95_latency_ms": round(p95, 3),
                "uptime_seconds": round(time.time() - self._started, 3),
            }


METRICS = InAppMetrics()
