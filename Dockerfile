# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Multi-stage build for the Cats vs Dogs inference service.
# Stage 1 resolves the pinned dependencies into a virtualenv; stage 2 copies
# only that venv plus the application code, so build toolchains never ship.
# ---------------------------------------------------------------------------

FROM python:3.10-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# CPU-only torch wheels: ~4x smaller than the default CUDA build, and the
# service is deployed to CPU nodes.
RUN pip install --upgrade pip==24.0 && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2 torchvision==0.17.2 && \
    pip install -r requirements.txt


FROM python:3.10-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="cats-vs-dogs-api" \
      org.opencontainers.image.description="Binary cat/dog image classifier inference service" \
      org.opencontainers.image.source="https://github.com/krupashankarsugi/mlops-assignment-2"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    LOG_LEVEL=INFO \
    SERVICE_VERSION=1.0.0

# libgomp1 is required by the torch CPU runtime; curl backs the HEALTHCHECK.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser params.yaml ./params.yaml
COPY --chown=appuser:appuser models/ ./models/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
