#!/usr/bin/env python3
"""Post-deploy smoke test (M4).

Calls the health endpoint and makes one real prediction against a deployed
instance. Exits non-zero on any failure so the CD pipeline fails the build.

Usage:
    python scripts/smoke_test.py --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import requests
from PIL import Image

TIMEOUT = 15


def _synthetic_image() -> bytes:
    """A deterministic 224x224 JPEG, so the test needs no dataset on disk."""
    img = Image.new("RGB", (224, 224))
    pixels = img.load()
    for y in range(224):
        for x in range(224):
            pixels[x, y] = ((x * 7) % 256, (y * 5) % 256, ((x + y) * 3) % 256)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _image_payload(sample: str | None) -> tuple[str, bytes]:
    if sample:
        path = Path(sample)
        if path.is_dir():
            candidates = sorted(path.rglob("*.jpg"))
            if not candidates:
                raise SystemExit(f"no .jpg files under {path}")
            path = candidates[0]
        return path.name, path.read_bytes()
    return "smoke.jpg", _synthetic_image()


def wait_for_service(url: str, retries: int, delay: float = 2.0) -> None:
    """Poll /health until the service answers or the retry budget runs out."""
    last_error = "no attempt made"
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(f"{url}/health", timeout=TIMEOUT)
            if response.status_code == 200:
                print(f"[1/4] service reachable after {attempt} attempt(s)")
                return
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        print(f"      waiting for {url} ({attempt}/{retries}): {last_error}")
        time.sleep(delay)
    raise SystemExit(f"FAIL: service never became reachable: {last_error}")


def check_health(url: str) -> None:
    response = requests.get(f"{url}/health", timeout=TIMEOUT)
    if response.status_code != 200:
        raise SystemExit(f"FAIL: /health returned HTTP {response.status_code}")
    body = response.json()
    if body.get("status") != "ok" or not body.get("model_loaded"):
        raise SystemExit(f"FAIL: /health reports an unhealthy service: {body}")
    print(f"[2/4] /health ok  (version={body.get('version')}, "
          f"uptime={body.get('uptime_seconds')}s)")


def check_readiness(url: str) -> None:
    response = requests.get(f"{url}/ready", timeout=TIMEOUT)
    if response.status_code != 200:
        raise SystemExit(f"FAIL: /ready returned HTTP {response.status_code}: {response.text}")
    print("[3/4] /ready ok")


def check_prediction(url: str, sample: str | None) -> None:
    filename, payload = _image_payload(sample)
    response = requests.post(
        f"{url}/predict",
        files={"file": (filename, payload, "image/jpeg")},
        timeout=TIMEOUT,
    )
    if response.status_code != 200:
        raise SystemExit(f"FAIL: /predict returned HTTP {response.status_code}: {response.text}")

    body = response.json()
    for field in ("label", "confidence", "probabilities"):
        if field not in body:
            raise SystemExit(f"FAIL: /predict response missing '{field}': {body}")
    if body["label"] not in ("cat", "dog"):
        raise SystemExit(f"FAIL: unexpected label {body['label']!r}")
    if not 0.0 <= body["confidence"] <= 1.0:
        raise SystemExit(f"FAIL: confidence out of range: {body['confidence']}")
    total = sum(body["probabilities"].values())
    if abs(total - 1.0) > 1e-3:
        raise SystemExit(f"FAIL: probabilities sum to {total}, expected 1.0")

    print(f"[4/4] /predict ok  (label={body['label']}, "
          f"confidence={body['confidence']:.4f}, "
          f"latency={body.get('inference_ms', 0):.1f}ms)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post-deploy smoke test")
    parser.add_argument("--url", default="http://localhost:8000", help="base URL of the service")
    parser.add_argument("--retries", type=int, default=15, help="health-poll attempts")
    parser.add_argument("--sample", default=None, help="image file or directory to predict on")
    args = parser.parse_args(argv)

    url = args.url.rstrip("/")
    print(f"=== Smoke test against {url} ===")
    wait_for_service(url, args.retries)
    check_health(url)
    check_readiness(url)
    check_prediction(url, args.sample)
    print("=== SMOKE TEST PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
