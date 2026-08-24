# Convenience targets for the Cats vs Dogs MLOps pipeline.
.PHONY: help setup data train test lint serve docker-build docker-run compose-up compose-down \
        smoke monitor k8s-deploy k8s-delete dvc-repro clean package report

PYTHON      ?= .venv/bin/python
IMAGE       ?= cats-vs-dogs-api
TAG         ?= latest
URL         ?= http://localhost:8000

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the virtualenv and install all dependencies
	python3.10 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -r requirements-dev.txt

data:  ## Download and preprocess the dataset
	$(PYTHON) -m src.data.download
	$(PYTHON) -m src.data.preprocess --clean

train:  ## Train the baseline CNN and log the run to MLflow
	$(PYTHON) -m src.models.train

dvc-repro:  ## Reproduce the full DVC pipeline (download -> preprocess -> train)
	$(PYTHON) -m dvc repro

test:  ## Run the unit tests with coverage
	$(PYTHON) -m pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

lint:  ## Lint the source tree
	$(PYTHON) -m ruff check src api tests scripts

serve:  ## Run the API locally with hot reload
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:  ## Open the MLflow tracking UI on :5000
	$(PYTHON) -m mlflow ui --backend-store-uri file:./mlruns --port 5000

docker-build:  ## Build the inference image
	docker build -t $(IMAGE):$(TAG) .

docker-run:  ## Run the image locally on :8000
	docker run -d --rm --name $(IMAGE) -p 8000:8000 $(IMAGE):$(TAG)

docker-stop:  ## Stop the local container
	-docker rm -f $(IMAGE)

compose-up:  ## Start the API + Prometheus stack
	docker compose up -d --build

compose-down:  ## Tear down the compose stack
	docker compose down

smoke:  ## Run the post-deploy smoke test
	$(PYTHON) scripts/smoke_test.py --url $(URL)

monitor:  ## Replay labelled test images against the deployed service
	$(PYTHON) scripts/monitor_performance.py --url $(URL) --samples 100

k8s-deploy:  ## Apply the Kubernetes manifests
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	kubectl rollout status deployment/cats-vs-dogs-api --timeout=300s

k8s-delete:  ## Remove the Kubernetes resources
	-kubectl delete -f k8s/service.yaml
	-kubectl delete -f k8s/deployment.yaml
	-kubectl delete -f k8s/configmap.yaml

report:  ## Render report.md to report.pdf (needs pandoc + weasyprint)
	pandoc report.md -o report.pdf \
	  --pdf-engine=weasyprint --css=.report/report.css \
	  --metadata title="MLOps Assignment 02 — Cats vs Dogs" \
	  --toc --toc-depth=2 --standalone

package:  ## Build the submission zip
	$(PYTHON) scripts/package_submission.py

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
