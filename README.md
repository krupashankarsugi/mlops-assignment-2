# Cats vs Dogs — End-to-End MLOps Pipeline

**MLOps (S1-25_AIMLCZG523) — Assignment 2**

Binary image classification (cat vs dog) for a pet adoption platform, built as a
complete MLOps pipeline: data/model versioning, experiment tracking, a
containerized inference service, CI/CD to a Kubernetes cluster, and
post-deployment monitoring.

---

## Module Map

| Module | Requirement | Where it lives |
|---|---|---|
| **M1** | Git + DVC versioning | [`dvc.yaml`](dvc.yaml), [`params.yaml`](params.yaml), [`.dvc/`](.dvc/) |
| **M1** | Baseline model + serialization | [`src/models/cnn.py`](src/models/cnn.py), [`src/models/train.py`](src/models/train.py) → `models/model.pt` |
| **M1** | Experiment tracking | MLflow — `mlruns/`, logged params/metrics/artifacts |
| **M2** | Inference service | [`api/main.py`](api/main.py) — FastAPI |
| **M2** | Environment spec | [`requirements.txt`](requirements.txt) (pinned) |
| **M2** | Containerization | [`Dockerfile`](Dockerfile) — multi-stage, CPU-only torch |
| **M3** | Automated testing | [`tests/`](tests/) — 56 pytest tests |
| **M3** | CI pipeline | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| **M3** | Artifact publishing | GitHub Container Registry (GHCR) |
| **M4** | Deployment target | [`k8s/`](k8s/) + [`docker-compose.yml`](docker-compose.yml) |
| **M4** | CD / GitOps flow | [`.github/workflows/cd.yml`](.github/workflows/cd.yml) |
| **M4** | Smoke tests | [`scripts/smoke_test.py`](scripts/smoke_test.py) |
| **M5** | Logging & metrics | [`api/monitoring.py`](api/monitoring.py) — JSON logs, `/stats`, `/metrics` |
| **M5** | Post-deploy tracking | [`scripts/monitor_performance.py`](scripts/monitor_performance.py) |

---

## Architecture

```
                   ┌─────────────────────────────────────────────┐
   Kaggle /        │  M1  Data & Model Development               │
   MS mirror  ────►│  download → preprocess (224×224, 80/10/10)  │
                   │           → train (CNN) → MLflow            │
                   │  versioned by DVC ─────────────► model.pt   │
                   └───────────────────┬─────────────────────────┘
                                       ▼
                   ┌─────────────────────────────────────────────┐
                   │  M2  Packaging                              │
                   │  FastAPI service  +  multi-stage Docker     │
                   └───────────────────┬─────────────────────────┘
                                       ▼
   git push ──────►┌─────────────────────────────────────────────┐
                   │  M3  CI (GitHub Actions)                    │
                   │  lint → pytest → build image → push GHCR    │
                   └───────────────────┬─────────────────────────┘
                                       ▼
                   ┌─────────────────────────────────────────────┐
                   │  M4  CD                                     │
                   │  pull image → kubectl apply → rollout       │
                   │  → smoke test (fail ⇒ rollback)             │
                   └───────────────────┬─────────────────────────┘
                                       ▼
                   ┌─────────────────────────────────────────────┐
                   │  M5  Monitoring                             │
                   │  JSON access logs · /stats · /metrics       │
                   │  Prometheus · live accuracy replay          │
                   └─────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Environment (Python 3.10)
make setup

# 2. Data: download ~800 MB corpus, then resize + split
make data

# 3. Train the baseline CNN (logs to MLflow, writes models/model.pt)
make train

# 4. Tests
make test

# 5. Serve locally
make serve            # http://localhost:8000/docs

# 6. Container
make docker-build && make docker-run
make smoke            # post-deploy smoke test

# 7. Kubernetes
make k8s-deploy
```

Run `make help` to list every target.

---

## M1 — Model Development & Experiment Tracking

### Data versioning (DVC)

The pipeline is declared in [`dvc.yaml`](dvc.yaml) as three stages:

```
download ──► preprocess ──► train
```

`dvc repro` re-runs only the stages whose dependencies or `params.yaml` values
changed. `data/processed/` and `models/model.pt` are DVC outputs (kept out of
Git via [`.gitignore`](.gitignore)); source code and configuration are versioned
in Git.

```bash
dvc repro                    # reproduce the whole pipeline
dvc metrics show             # test metrics from reports/metrics.json
dvc dag                      # visualise stage dependencies
```

### Preprocessing

[`src/data/preprocess.py`](src/data/preprocess.py) converts every image to
**224×224 RGB JPEG** and splits **80 / 10 / 10 → train / val / test** with a
fixed seed. The corpus ships a handful of zero-byte and truncated JPEGs, so
`is_valid_image()` filters them out before the split — otherwise they crash the
training loader.

Augmentation (training split only, [`src/data/dataset.py`](src/data/dataset.py)):
random-resized crop, horizontal flip, ±15° rotation, colour jitter. Validation
and test use a deterministic resize + normalise pipeline — **the same function
the API calls at serving time**, so preprocessing cannot drift between training
and production.

### Models

| Model | Description |
|---|---|
| `simple_cnn` | Baseline: 4 × (Conv-BN-ReLU-MaxPool) → global avg pool → FC head, trained from scratch |
| `resnet18_transfer` | Comparison: ImageNet-pretrained ResNet-18, frozen backbone, fresh 2-way head |

The checkpoint (`models/model.pt`) stores the weights **plus** the class names,
image size and normalisation constants, so the serving side never has to guess
how the model was trained.

### Experiment tracking (MLflow)

Every run logs parameters (architecture, LR, batch size, augmentation, seed,
dataset sizes), per-epoch metrics (train/val loss and accuracy, learning rate),
test metrics (accuracy, precision, recall, F1, ROC-AUC), and artifacts — the
confusion matrix, the loss/accuracy curves, and the serialized model.

```bash
make mlflow-ui        # http://localhost:5000
```

### Promoting a run to production

Tracking is only useful if the winning run can actually ship.
[`scripts/promote_model.py`](scripts/promote_model.py) ranks the runs in the
MLflow store, pulls the best one's weights and rewrites `models/model.pt` plus
the metadata sidecar the API reads:

```bash
python scripts/promote_model.py --best-by test_accuracy
make docker-build     # ship the promoted weights
```

---

## M2 — Packaging & Containerization

### API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + whether the model deserialised |
| `GET` | `/ready` | Readiness — 503 until the model is usable |
| `POST` | `/predict` | Single image → label + class probabilities |
| `POST` | `/predict/batch` | Up to 16 images per call |
| `GET` | `/model-info` | Served checkpoint metadata + test metrics |
| `GET` | `/stats` | In-app counters and latency percentiles |
| `GET` | `/metrics` | Prometheus exposition format |

```bash
curl -F "file=@data/processed/test/dog/dog_101.jpg" http://localhost:8000/predict
```

```json
{
  "label": "dog",
  "confidence": 0.8571,
  "probabilities": {"cat": 0.1429, "dog": 0.8571},
  "inference_ms": 24.7,
  "request_id": "a3f9c1d20b84"
}
```

### Environment

All key ML libraries are pinned to exact versions:

- [`requirements.txt`](requirements.txt) — runtime (what ships in the image)
- [`requirements-train.txt`](requirements-train.txt) — adds MLflow, DVC, scikit-learn
- [`requirements-dev.txt`](requirements-dev.txt) — adds pytest, ruff

### Container

Multi-stage build: stage 1 resolves dependencies into a venv, stage 2 copies
only that venv plus application code. Notable choices:

- **CPU-only torch wheels** (`download.pytorch.org/whl/cpu`) — roughly 4× smaller than the default CUDA build
- Runs as **non-root** (uid 10001), read-only root filesystem, all capabilities dropped
- `HEALTHCHECK` wired to `/health`

---

## M3 — CI Pipeline

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and PR to `main`:

1. **Checkout** the repository
2. **Install** pinned dependencies (pip cache enabled)
3. **Lint** with ruff
4. **Run unit tests** with coverage → uploaded as a build artifact
5. **Build** the Docker image (Buildx + GitHub Actions layer cache)
6. **Push** to `ghcr.io/<owner>/<repo>` — tagged `latest`, branch name, and `sha-<short>`
7. **Verify** the built image actually serves a prediction before it is trusted

PRs build the image but do not push it.

### Tests

56 tests across three files:

| File | Covers |
|---|---|
| [`tests/test_preprocess.py`](tests/test_preprocess.py) | `resize_image`, `split_indices`, `is_valid_image` — ratios, determinism, disjointness, corrupt-file rejection |
| [`tests/test_inference.py`](tests/test_inference.py) | Model factory, preprocessing, prediction shape/probabilities, metric computation |
| [`tests/test_api.py`](tests/test_api.py) | Health, readiness, predict, batch, error paths, `/stats`, `/metrics` |

Tests build a synthetic model and synthetic images in fixtures, so CI never
needs the 800 MB dataset or a trained checkpoint.

---

## M4 — CD Pipeline & Deployment

[`.github/workflows/cd.yml`](.github/workflows/cd.yml) triggers when CI succeeds on `main`:

1. Spin up a **k3d Kubernetes cluster** on the runner
2. **Pull** the new image from GHCR and import it into the cluster
3. **Apply** the manifests and `kubectl set image` to the new tag
4. **Wait** for the rollout (`kubectl rollout status`)
5. **Smoke test** — `scripts/smoke_test.py` calls `/health`, `/ready` and one real `/predict`
6. **Roll back** automatically (`kubectl rollout undo`) if the smoke test fails

The smoke test exits non-zero on any failure, which fails the pipeline.

### Manifests

- [`k8s/deployment.yaml`](k8s/deployment.yaml) — 2 replicas, rolling update with `maxUnavailable: 0` (zero-downtime), startup/liveness/readiness probes, resource requests and limits, hardened `securityContext`
- [`k8s/service.yaml`](k8s/service.yaml) — NodePort `30080`
- [`k8s/configmap.yaml`](k8s/configmap.yaml) — runtime configuration
- [`k8s/hpa.yaml`](k8s/hpa.yaml) — optional CPU-based autoscaling

The `startupProbe` matters here: torch takes several seconds to import and
deserialise the checkpoint, and without it the liveness probe would kill the pod
mid-cold-start.

`readOnlyRootFilesystem: true` caught a real bug during this build. Rebuilding
the transfer architecture to load a checkpoint was calling
`torchvision.resnet18(weights=IMAGENET1K_V1)`, which downloads ImageNet weights
into `~/.cache/torch` — weights that are immediately overwritten by the
checkpoint's own `state_dict`. Under Docker Compose this merely wasted a
download on every start; under a read-only root filesystem the pod never became
ready. `build_model(..., pretrained=False)` on the load path fixes it, and
`tests/test_inference.py` guards the regression.

**Docker Compose** is provided as an alternative target — it brings up the API
plus Prometheus in one command (`make compose-up`).

---

## M5 — Monitoring & Logging

### Request/response logging

Structured **one-JSON-object-per-line** logs (`api/monitoring.py`), so they are
greppable in Docker and Kubernetes:

```json
{"ts":"2026-08-24T14:02:11Z","level":"INFO","service":"cats-vs-dogs-api",
 "message":"prediction served","request_id":"a3f9c1d20b84","event":"prediction",
 "filename":"dog_101.jpg","bytes":18422,"label":"dog","confidence":0.8571,
 "inference_ms":24.7}
```

**No image bytes are ever logged** — only the filename, payload size and
outcome. Every request gets an `x-request-id` (honoured from the inbound header
when present) echoed back in the response for tracing.

### Metrics

Two complementary surfaces:

- **`/stats`** — in-app counters: total requests, predictions, errors, prediction mix per class, mean and p95 latency
- **`/metrics`** — Prometheus: `inference_requests_total`, `predictions_total`, `inference_errors_total`, `inference_request_latency_seconds` (histogram)

Prometheus scrape config in [`monitoring/prometheus.yml`](monitoring/prometheus.yml);
the K8s pods carry `prometheus.io/scrape` annotations.

### Post-deployment performance tracking

[`scripts/monitor_performance.py`](scripts/monitor_performance.py) replays a
batch of held-out test images — whose true labels are known from the directory
name — against the **deployed** service and compares served predictions with
ground truth:

```bash
make monitor          # or: python scripts/monitor_performance.py --samples 100
```

This measures the model *as deployed* (container + preprocessing + weights),
not as trained — the two can diverge, and this is what catches it. Writes
`reports/monitoring/performance_report.json`, a `predictions.jsonl` audit trail,
and a live confusion matrix.

---

## Project Layout

```
.
├── api/                    FastAPI inference service + monitoring
├── src/
│   ├── config.py           typed access to params.yaml
│   ├── data/               download, preprocess, datasets/augmentation
│   └── models/             CNN definition, training, prediction
├── tests/                  56 pytest tests
├── scripts/                smoke test, perf monitor, prediction CLI
├── k8s/                    Deployment, Service, ConfigMap, HPA
├── monitoring/             Prometheus configuration
├── .github/workflows/      ci.yml, cd.yml
├── reports/                metrics.json, figures, monitoring reports
├── dvc.yaml                data → model pipeline
├── params.yaml             all tunable parameters
├── Dockerfile              multi-stage inference image
├── docker-compose.yml      API + Prometheus stack
└── Makefile                task shortcuts
```

---

## Reproducing from Scratch

```bash
git clone <repo> && cd <repo>
make setup
dvc repro                 # download → preprocess → train
make test
make docker-build
make compose-up
make smoke
make monitor
```

---

## Troubleshooting

**`docker compose` reports "unknown command"** — the Compose CLI plugin is not
linked. With the Homebrew standalone binary:

```bash
mkdir -p ~/.docker/cli-plugins
ln -sf "$(which docker-compose)" ~/.docker/cli-plugins/docker-compose
```

**Prometheus fails with `operation not permitted` on the config bind-mount
(macOS + Colima)** — the Colima VM cannot read files under `~/Documents`
because of macOS privacy protection (TCC). Either grant Full Disk Access to
Colima/your terminal in *System Settings → Privacy & Security*, or keep the
project outside `~/Documents`. `$HOME` itself is mounted normally, and Docker
Desktop is unaffected.

**A rebuilt image does not reach minikube pods** — `minikube image load`
silently keeps the image already on the node when the tag matches, so a
rebuilt `:latest` never lands. Either remove it first
(`minikube image rm <image>:<tag>`) or push a unique tag per build;
`scripts/deploy_local.sh` does the former and then verifies the node's image ID
matches the local one.

**DVC fails with `cannot import name '_DIR_MARK'`** — `pathspec` 1.x removed a
private API that DVC 3.51 imports, and DVC's own `pathspec>=0.10.3` constraint
is too loose to prevent it. `requirements-train.txt` pins `pathspec==0.12.1`.

**MLflow fails with `No module named 'pkg_resources'`** — MLflow 2.14 still
imports `pkg_resources`, which recent virtualenvs omit. `requirements-train.txt`
pins `setuptools==69.5.1`.
