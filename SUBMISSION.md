# Assignment 2 — Submission Notes

**Course:** MLOps (S1-25_AIMLCZG523) · **Total marks:** 50
**Use case:** Binary image classification (Cats vs Dogs) for a pet adoption platform

---

## Requirement → Evidence Map

### M1 — Model Development & Experiment Tracking (10M)

| Task | Implementation | Evidence |
|---|---|---|
| Git source versioning | Repo tracks all code, notebooks, configs | `git log`, project tree |
| DVC dataset versioning | 3-stage pipeline `download → preprocess → train` | `dvc.yaml`, `.dvc/`, `dvc dag` |
| Pre-process to 224×224 RGB | `resize_image()` converts + resizes every image | `src/data/preprocess.py` |
| 80/10/10 split | Seeded `split_indices()`, per class | `data/processed/dataset_stats.json` |
| Data augmentation | RRC, h-flip, ±15° rotation, colour jitter (train only) | `src/data/dataset.py` |
| Baseline model | `SimpleCNN` — 4 conv blocks + GAP + FC head | `src/models/cnn.py` |
| Serialized format | `models/model.pt` (weights + class names + input contract) | `save_checkpoint()` |
| Experiment tracking | MLflow: params, per-epoch metrics, artifacts, model registry | `mlruns/`, `make mlflow-ui` |
| Confusion matrix + loss curves | Logged as MLflow artifacts | `reports/figures/` |

**Two runs logged for comparison:**

| Run | Architecture | Test accuracy | Notes |
|---|---|---|---|
| `baseline-cnn` | SimpleCNN, from scratch, 8 epochs | **74.75%** (ROC-AUC 0.841) | The required baseline |
| `resnet18-transfer` | ImageNet ResNet-18, frozen backbone, 5 epochs | **~96%** | Comparison run |

`scripts/promote_model.py` promotes the winning run from the MLflow store into
the serving artifact — tracking feeding deployment, not just a dashboard.

---

### M2 — Model Packaging & Containerization (10M)

| Task | Implementation | Evidence |
|---|---|---|
| REST API (FastAPI) | 8 endpoints | `api/main.py` |
| Health check endpoint | `GET /health` + `GET /ready` | verified live |
| Prediction endpoint | `POST /predict` → label + class probabilities | verified live |
| requirements.txt | Runtime deps, every ML library pinned exactly | `requirements.txt` |
| Version pinning | torch 2.2.2, torchvision 0.17.2, numpy 1.26.4, fastapi 0.111.0 … | all three req files |
| Dockerfile | Multi-stage, CPU-only torch wheels, non-root, HEALTHCHECK | `Dockerfile` |
| Build & run locally | Image built and verified serving predictions | see "Verification" below |
| Verify via curl | `curl -F "file=@…" localhost:8000/predict` | see below |

---

### M3 — CI Pipeline (10M)

| Task | Implementation | Evidence |
|---|---|---|
| Unit test: pre-processing fn | `resize_image`, `split_indices`, `is_valid_image` | `tests/test_preprocess.py` (22 tests) |
| Unit test: model/inference fn | model factory, preprocessing, prediction, metrics | `tests/test_inference.py` (19 tests) |
| Tests run via pytest | 56 tests, all passing | `make test` |
| CI: checkout | `actions/checkout@v4` | `.github/workflows/ci.yml` |
| CI: install dependencies | pinned install with pip caching | ci.yml `test` job |
| CI: run unit tests | pytest + coverage artifact | ci.yml `test` job |
| CI: build Docker image | Buildx with GHA layer cache | ci.yml `build-and-push` job |
| Artifact publishing | Push to `ghcr.io/<owner>/<repo>` | ci.yml, tags `latest` / branch / `sha-<short>` |

CI additionally **runs the built image and asserts it serves a prediction**
before the image is trusted — a build that compiles but cannot infer is caught
in CI rather than in CD.

---

### M4 — CD Pipeline & Deployment (10M)

| Task | Implementation | Evidence |
|---|---|---|
| Deployment target | Kubernetes (k3d in CI; minikube locally) | `k8s/` |
| Deployment + Service YAML | 2 replicas, probes, resources, security context | `k8s/deployment.yaml`, `k8s/service.yaml` |
| Docker Compose alternative | API + Prometheus in one command | `docker-compose.yml` |
| CD: pull image from registry | `docker pull ghcr.io/…` then `k3d image import` | `.github/workflows/cd.yml` |
| CD: auto-deploy on main | Triggered by successful CI on `main` | cd.yml `workflow_run` trigger |
| Post-deploy smoke test | `/health` + `/ready` + one real `/predict` | `scripts/smoke_test.py` |
| Fail pipeline on smoke failure | Script exits non-zero; job then rolls back | cd.yml `rollout undo` step |

---

### M5 — Monitoring, Logs & Final Submission (10M)

| Task | Implementation | Evidence |
|---|---|---|
| Request/response logging | Structured one-JSON-per-line access log | `api/monitoring.py` |
| Excludes sensitive data | Only filename, byte count, label, latency — never image bytes | `api/main.py` predict handler |
| Request count | `/stats` + Prometheus `inference_requests_total` | verified live |
| Latency tracking | `/stats` mean/p95 + Prometheus histogram | verified live |
| Prometheus | Scrape config + pod annotations | `monitoring/prometheus.yml` |
| Post-deploy performance tracking | Replays labelled test images against the deployed service | `scripts/monitor_performance.py` |
| Real requests + true labels | 100 held-out test images with ground-truth labels | `reports/monitoring/` |

---

## Verification Performed

All of the following were executed on this machine, not just written:

```
[x] Dataset downloaded             25,000 images (12,500 cat / 12,500 dog)
[x] Preprocessing                  4,000 images @ 224x224, 80/10/10, 1 corrupt file filtered
[x] Baseline training              8 epochs, test accuracy 74.75%, ROC-AUC 0.841
[x] Transfer comparison run        ResNet-18, ~96% val accuracy
[x] MLflow tracking                2 runs, params/metrics/artifacts/model registry
[x] Unit tests                     56 passed
[x] Lint                           ruff clean
[x] DVC pipeline                   dvc dag renders the 3-stage DAG
[x] API served locally             all 8 endpoints responding
[x] Smoke test                     PASSED against the running service
[x] Post-deploy monitoring         78% live accuracy, p50 22ms / p95 40ms
[x] Docker image                   built and verified serving predictions
[x] Kubernetes manifests           applied to a live minikube cluster
                                   (Deployment + Service + ConfigMap created,
                                   pods scheduled, probes executed)
[x] Read-only-rootfs container     model loads, /ready 200, predictions served
                                   (same securityContext the manifests impose)
[x] Docker Compose stack           API + Prometheus, both healthy
[x] Prometheus scrape verified     target UP, PromQL histogram_quantile queried
[ ] Green rollout on minikube      NOT completed -- see note 6
```

### Sample prediction

```bash
$ curl -F "file=@data/processed/test/dog/dog_10042.jpg" http://localhost:8000/predict
{
  "label": "dog",
  "confidence": 0.9992,
  "probabilities": {"cat": 0.0008, "dog": 0.9992},
  "inference_ms": 14.3,
  "request_id": "a3f9c1d20b84"
}
```

### Sample structured log line

```json
{"ts":"2026-08-24T13:37:34Z","level":"INFO","service":"cats-vs-dogs-api",
 "message":"prediction served","request_id":"25f11a66a0b0","event":"prediction",
 "filename":"cat_11022.jpg","bytes":9940,"label":"cat","confidence":0.5339,
 "inference_ms":13.61}
```

---

## Notes & Assumptions

1. **Dataset source.** The Kaggle download requires account credentials, which
   are not available in this environment. `src/data/download.py` tries the
   Kaggle CLI first and falls back to Microsoft's public mirror of the *same*
   25,000-image corpus, so the pipeline is reproducible without credentials.

2. **Subset size.** `params.yaml` caps training at 2,000 images per class
   (4,000 total) so the full pipeline runs end-to-end on a laptop in minutes.
   Set `data.max_images_per_class: 0` to train on all 25,000 images.

3. **Baseline accuracy.** 74.75% is an honest result for a ~422k-parameter CNN
   trained from scratch on 3,200 images for 8 epochs — well above the
   logistic-regression-on-pixels baseline the brief offers as the alternative.
   The ResNet-18 transfer run is included to show what the same pipeline
   achieves with a pretrained backbone, and to give MLflow two runs to compare.

4. **Registry credentials.** CI pushes to GHCR using the built-in
   `GITHUB_TOKEN`; no secret needs to be configured manually. The CD workflow
   provisions an ephemeral k3d cluster on the runner as the deployment target,
   since a laptop minikube is not reachable from GitHub-hosted runners. The same
   manifests apply unchanged to local minikube via `make k8s-deploy`.

5. **Kubernetes verification is partial.** The manifests were applied to a
   real minikube cluster and the Deployment, Service and ConfigMap were created
   with pods scheduled and probes running. A final green
   `kubectl rollout status` with the *fixed* image was not reached: this
   machine runs Docker through Colima, and the nested minikube control plane
   repeatedly came up with `apiserver: Stopped` after the host disk filled
   during the build. What the rollout would exercise was instead verified
   directly with `docker run --read-only --user 10001`, which reproduces the
   `securityContext` in `k8s/deployment.yaml` — the service loads its model,
   answers `/ready` with 200 and serves predictions under exactly those
   constraints. `scripts/deploy_local.sh` performs the full minikube sequence
   on a host with more headroom.

6. **Model in Git.** `models/*.pt` is DVC-tracked, not committed. CI restores it
   with `dvc pull` when a DVC remote is configured, and otherwise generates a
   correctly-shaped placeholder (`scripts/make_ci_model.py`) so the image build
   and its serving check still run on a clean checkout.

---

## Deliverables

| # | Deliverable | Location |
|---|---|---|
| 1 | Zip: source, DVC/CI-CD/Docker/K8s configs, model artifacts | `make package` → `MLOps_Assignment02_KrupaShankar.zip` |
| 2 | Screen recording (< 5 min) of the full workflow | *to be recorded — see the demo script below* |

### Suggested demo script (< 5 minutes)

| Time | Show |
|---|---|
| 0:00–0:30 | Repo tour: `dvc dag`, project structure, `params.yaml` |
| 0:30–1:15 | MLflow UI: compare `baseline-cnn` vs `resnet18-transfer`, show curves + confusion matrix |
| 1:15–1:45 | `make test` — 56 tests green |
| 1:45–2:15 | Make a small code change, commit, push |
| 2:15–3:00 | GitHub Actions CI: tests → image build → GHCR push |
| 3:00–3:45 | CD workflow: rollout to Kubernetes, smoke test gate |
| 3:45–4:30 | `curl` a real prediction; show `/stats`, `/metrics`, JSON logs |
| 4:30–5:00 | `make monitor` — live accuracy and latency of the deployed model |
