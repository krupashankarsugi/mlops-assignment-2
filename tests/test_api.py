"""Contract tests for the FastAPI inference service (health + prediction)."""
from __future__ import annotations

from tests.conftest import CLASS_NAMES, make_image_bytes


class TestHealthEndpoints:
    def test_health_reports_ok_when_model_is_loaded(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["uptime_seconds"] >= 0

    def test_ready_returns_200_when_model_is_loaded(self, api_client):
        assert api_client.get("/ready").json() == {"status": "ready"}

    def test_root_lists_available_endpoints(self, api_client):
        endpoints = api_client.get("/").json()["endpoints"]
        assert "/health" in endpoints and "/predict" in endpoints


class TestPredictEndpoint:
    def test_returns_label_and_probabilities(self, api_client):
        response = api_client.post(
            "/predict", files={"file": ("pet.jpg", make_image_bytes(), "image/jpeg")}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["label"] in CLASS_NAMES
        assert 0.0 <= body["confidence"] <= 1.0
        assert set(body["probabilities"]) == set(CLASS_NAMES)
        assert body["inference_ms"] >= 0
        assert body["request_id"]

    def test_response_carries_tracing_headers(self, api_client):
        response = api_client.post(
            "/predict", files={"file": ("pet.jpg", make_image_bytes(), "image/jpeg")}
        )
        assert "x-request-id" in response.headers
        assert "x-process-time-ms" in response.headers

    def test_rejects_a_corrupt_image_with_400(self, api_client):
        response = api_client.post(
            "/predict", files={"file": ("bad.jpg", b"not an image", "image/jpeg")}
        )
        assert response.status_code == 400

    def test_rejects_an_empty_upload_with_400(self, api_client):
        response = api_client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
        assert response.status_code == 400

    def test_requires_the_file_field(self, api_client):
        assert api_client.post("/predict").status_code == 422

    def test_batch_endpoint_scores_every_file(self, api_client):
        files = [("files", (f"pet{i}.jpg", make_image_bytes(), "image/jpeg")) for i in range(3)]
        body = api_client.post("/predict/batch", files=files).json()
        assert body["count"] == 3
        assert all(r["label"] in CLASS_NAMES for r in body["results"])


class TestObservabilityEndpoints:
    def test_model_info_describes_the_served_checkpoint(self, api_client):
        body = api_client.get("/model-info").json()
        assert body["class_names"] == CLASS_NAMES
        assert body["image_size"] == 224

    def test_stats_counts_predictions(self, api_client):
        before = api_client.get("/stats").json()["predictions"]
        api_client.post("/predict", files={"file": ("p.jpg", make_image_bytes(), "image/jpeg")})
        after = api_client.get("/stats").json()
        assert after["predictions"] == before + 1
        assert after["total_requests"] > 0
        assert after["avg_latency_ms"] >= 0

    def test_metrics_endpoint_exposes_prometheus_format(self, api_client):
        response = api_client.get("/metrics")
        assert response.status_code == 200
        assert "inference_requests_total" in response.text
