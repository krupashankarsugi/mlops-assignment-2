"""FastAPI inference service for the Cats vs Dogs classifier.

Endpoints
---------
GET  /health      liveness + whether the model deserialised successfully
GET  /ready       readiness probe (503 until the model is loaded)
POST /predict     multipart image upload -> class label + probabilities
POST /predict/batch  several images in one call
GET  /model-info  metadata about the served checkpoint
GET  /stats       in-app request/latency counters
GET  /metrics     Prometheus exposition format
"""
from __future__ import annotations

import os
import time
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.monitoring import (
    ERROR_COUNT,
    METRICS,
    PREDICTION_COUNT,
    REGISTRY,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    configure_logging,
)
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    StatsResponse,
)
from src.config import load_config
from src.models.predict import InvalidImageError, get_predictor

SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))  # 10 MB
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 16))

LOGGER = configure_logging(os.getenv("LOG_LEVEL", "INFO"))
START_TIME = time.time()

app = FastAPI(
    title="Cats vs Dogs Inference API",
    description="Binary image classifier served for a pet adoption platform.",
    version=SERVICE_VERSION,
)

# Populated on startup; stays None if the checkpoint is missing so that /health
# can report the failure instead of the container crash-looping.
_predictor = None
_load_error: str | None = None


@app.on_event("startup")
def load_model() -> None:
    global _predictor, _load_error
    try:
        _predictor = get_predictor(load_config())
        _load_error = None
        LOGGER.info(
            "model loaded",
            extra={"event": {"event": "startup", "classes": _predictor.class_names}},
        )
    except Exception as exc:  # noqa: BLE001 - surfaced through /health
        _predictor, _load_error = None, str(exc)
        LOGGER.error("model load failed", extra={"event": {"event": "startup_failed", "error": str(exc)}})


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Assign a request id, time the call, and emit one structured access log."""
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request.state.request_id = request_id
    started = time.perf_counter()

    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        latency_ms = (time.perf_counter() - started) * 1000
        METRICS.record_request(latency_ms)
        METRICS.record_error()
        ERROR_COUNT.labels(endpoint=request.url.path, reason="unhandled").inc()
        LOGGER.exception(
            "unhandled error",
            extra={"event": {"request_id": request_id, "path": request.url.path}},
        )
        raise

    latency_ms = (time.perf_counter() - started) * 1000
    METRICS.record_request(latency_ms)
    REQUEST_COUNT.labels(
        endpoint=request.url.path, method=request.method, status=str(status)
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency_ms / 1000)
    if status >= 400:
        METRICS.record_error()

    response.headers["x-request-id"] = request_id
    response.headers["x-process-time-ms"] = f"{latency_ms:.2f}"

    # Metrics scrapes are frequent and uninteresting; keep them out of the log.
    if request.url.path != "/metrics":
        LOGGER.info(
            "request handled",
            extra={
                "event": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "latency_ms": round(latency_ms, 2),
                    "client": request.client.host if request.client else None,
                }
            },
        )
    return response


def _require_model():
    if _predictor is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_load_error}")
    return _predictor


async def _read_upload(file: UploadFile, request_id: str) -> bytes:
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        ERROR_COUNT.labels(endpoint="/predict", reason="payload_too_large").inc()
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {MAX_UPLOAD_BYTES} bytes",
        )
    if not payload:
        ERROR_COUNT.labels(endpoint="/predict", reason="empty_file").inc()
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    return payload


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe. Returns 200 whenever the process is serving."""
    return HealthResponse(
        status="ok" if _predictor is not None else "degraded",
        model_loaded=_predictor is not None,
        version=SERVICE_VERSION,
        uptime_seconds=round(time.time() - START_TIME, 3),
    )


@app.get("/ready", tags=["ops"])
def ready():
    """Readiness probe. 503 until the model is usable, so K8s holds traffic back."""
    if _predictor is None:
        return JSONResponse(
            status_code=503, content={"status": "not_ready", "detail": _load_error}
        )
    return {"status": "ready"}


@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    predictor = _require_model()
    meta = predictor.metadata or {}
    return ModelInfoResponse(
        model_name=meta.get("model_name", "simple_cnn"),
        framework=meta.get("framework", "pytorch"),
        class_names=predictor.class_names,
        image_size=predictor.image_size,
        parameters_total=meta.get("parameters_total"),
        parameters_trainable=meta.get("parameters_trainable"),
        test_metrics=meta.get("test_metrics", {}),
        trained_at=meta.get("trained_at"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["model"])
async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    """Classify a single uploaded image as cat or dog."""
    predictor = _require_model()
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    payload = await _read_upload(file, request_id)

    started = time.perf_counter()
    try:
        prediction = predictor.predict_bytes(payload)
    except InvalidImageError as exc:
        ERROR_COUNT.labels(endpoint="/predict", reason="invalid_image").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    inference_ms = (time.perf_counter() - started) * 1000

    METRICS.record_prediction(prediction.label)
    PREDICTION_COUNT.labels(predicted_class=prediction.label).inc()
    # Log the outcome and the payload size only -- never the image bytes.
    LOGGER.info(
        "prediction served",
        extra={
            "event": {
                "request_id": request_id,
                "event": "prediction",
                "filename": os.path.basename(file.filename or ""),
                "bytes": len(payload),
                "label": prediction.label,
                "confidence": round(prediction.confidence, 4),
                "inference_ms": round(inference_ms, 2),
            }
        },
    )
    return PredictionResponse(
        **prediction.as_dict(),
        inference_ms=round(inference_ms, 3),
        request_id=request_id,
    )


@app.post("/predict/batch", tags=["model"])
async def predict_batch(request: Request, files: list[UploadFile] = File(...)):
    """Classify up to MAX_BATCH_SIZE images in a single call."""
    predictor = _require_model()
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=413, detail=f"at most {MAX_BATCH_SIZE} files per batch")

    started = time.perf_counter()
    results = []
    for upload in files:
        payload = await upload.read()
        try:
            prediction = predictor.predict_bytes(payload)
        except InvalidImageError as exc:
            ERROR_COUNT.labels(endpoint="/predict/batch", reason="invalid_image").inc()
            results.append({"filename": upload.filename, "error": str(exc)})
            continue
        METRICS.record_prediction(prediction.label)
        PREDICTION_COUNT.labels(predicted_class=prediction.label).inc()
        results.append({"filename": upload.filename, **prediction.as_dict()})

    return {
        "request_id": request_id,
        "count": len(results),
        "inference_ms": round((time.perf_counter() - started) * 1000, 3),
        "results": results,
    }


@app.get("/stats", response_model=StatsResponse, tags=["ops"])
def stats() -> StatsResponse:
    """In-app counters: request count, prediction mix and latency percentiles."""
    return StatsResponse(**METRICS.snapshot())


@app.get("/metrics", tags=["ops"])
def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/", tags=["ops"])
def root():
    return {
        "service": "cats-vs-dogs-inference",
        "version": SERVICE_VERSION,
        "docs": "/docs",
        "endpoints": ["/health", "/ready", "/predict", "/predict/batch", "/model-info", "/stats", "/metrics"],
    }
