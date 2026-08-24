#!/usr/bin/env bash
# Deploy the inference service to a local minikube cluster and gate on the
# smoke test -- the same sequence the CD workflow runs against k3d.
set -euo pipefail

IMAGE="${IMAGE:-cats-vs-dogs-api}"
TAG="${TAG:-latest}"
NAMESPACE="${NAMESPACE:-default}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

echo "==> 1/6  Ensuring minikube is running"
if ! minikube status >/dev/null 2>&1; then
  minikube start --cpus=4 --memory=4096
fi

echo "==> 2/6  Building ${IMAGE}:${TAG}"
docker build -t "${IMAGE}:${TAG}" .

echo "==> 3/6  Loading the image into minikube"
# minikube silently keeps the existing image when the tag already exists on the
# node, so a rebuilt ":latest" would never reach the pods. Drop it first.
minikube image rm "${IMAGE}:${TAG}" >/dev/null 2>&1 || true
minikube image load "${IMAGE}:${TAG}"

LOCAL_ID="$(docker images -q "${IMAGE}:${TAG}")"
NODE_ID="$(minikube ssh -- "docker images -q ${IMAGE}:${TAG}" 2>/dev/null | tr -d '\r')"
if [ -n "$LOCAL_ID" ] && [ -n "$NODE_ID" ] && [ "${NODE_ID:0:12}" != "${LOCAL_ID:0:12}" ]; then
  echo "ERROR: minikube still holds a stale ${IMAGE}:${TAG} (${NODE_ID:0:12} != ${LOCAL_ID:0:12})" >&2
  exit 1
fi

echo "==> 4/6  Applying manifests"
kubectl apply -n "$NAMESPACE" -f k8s/configmap.yaml
kubectl apply -n "$NAMESPACE" -f k8s/deployment.yaml
kubectl apply -n "$NAMESPACE" -f k8s/service.yaml
# The manifest pins the GHCR tag; point it at the locally built image instead.
kubectl set image -n "$NAMESPACE" deployment/cats-vs-dogs-api "api=${IMAGE}:${TAG}"
kubectl patch -n "$NAMESPACE" deployment/cats-vs-dogs-api \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","imagePullPolicy":"Never"}]}}}}'

echo "==> 5/6  Waiting for the rollout"
kubectl rollout status -n "$NAMESPACE" deployment/cats-vs-dogs-api --timeout=300s

echo "==> 6/6  Smoke testing"
URL="$(minikube service cats-vs-dogs-api -n "$NAMESPACE" --url | head -n1)"
echo "    service URL: $URL"
python scripts/smoke_test.py --url "$URL" --retries 30

echo
echo "Deployed. Try:"
echo "  curl $URL/health"
echo "  curl -F \"file=@data/processed/test/dog/\$(ls data/processed/test/dog | head -1)\" $URL/predict"
echo "  python scripts/monitor_performance.py --url $URL --samples 100"
