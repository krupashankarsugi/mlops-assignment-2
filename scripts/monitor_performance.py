#!/usr/bin/env python3
"""Post-deployment model performance tracking (M5).

Replays a batch of held-out test images -- for which the true label is known
from the directory name -- against the *deployed* service, then compares the
served predictions with the ground truth. This measures the model as deployed
(container + preprocessing + weights), not the model as trained.

Writes reports/monitoring/performance_report.json and a confusion-matrix plot.

Usage:
    python scripts/monitor_performance.py --url http://localhost:8000 --samples 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 30


def collect_labelled_samples(test_dir: Path, n: int, seed: int = 42) -> list[tuple[Path, str]]:
    """Gather (image_path, true_label) pairs, balanced across classes."""
    if not test_dir.is_dir():
        raise SystemExit(
            f"missing {test_dir}. Run `python -m src.data.preprocess` to build the test split."
        )
    class_dirs = sorted(d for d in test_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"no class sub-directories under {test_dir}")

    rng = random.Random(seed)
    per_class = max(1, n // len(class_dirs))
    samples: list[tuple[Path, str]] = []
    for class_dir in class_dirs:
        images = sorted(class_dir.glob("*.jpg"))
        samples.extend((p, class_dir.name) for p in rng.sample(images, min(per_class, len(images))))
    rng.shuffle(samples)
    return samples


def score_sample(url: str, path: Path) -> dict:
    started = time.perf_counter()
    response = requests.post(
        f"{url}/predict",
        files={"file": (path.name, path.read_bytes(), "image/jpeg")},
        timeout=TIMEOUT,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    body = response.json()
    return {
        "predicted": body["label"],
        "confidence": body["confidence"],
        "client_latency_ms": latency_ms,
        "server_inference_ms": body.get("inference_ms", 0.0),
    }


def summarise(records: list[dict], class_names: list[str]) -> dict:
    total = len(records)
    correct = sum(1 for r in records if r["predicted"] == r["true_label"])
    index = {name: i for i, name in enumerate(class_names)}
    matrix = [[0] * len(class_names) for _ in class_names]
    for r in records:
        matrix[index[r["true_label"]]][index[r["predicted"]]] += 1

    per_class = {}
    for name in class_names:
        i = index[name]
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(len(class_names)) if j != i)
        fn = sum(matrix[i][j] for j in range(len(class_names)) if j != i)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "support": tp + fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    latencies = sorted(r["client_latency_ms"] for r in records)
    def pct(p: float) -> float:
        return round(latencies[min(int(len(latencies) * p), len(latencies) - 1)], 2) if latencies else 0.0

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "samples": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "class_names": class_names,
        "confusion_matrix": matrix,
        "per_class": per_class,
        "mean_confidence": round(sum(r["confidence"] for r in records) / total, 4) if total else 0.0,
        "prediction_distribution": dict(Counter(r["predicted"] for r in records)),
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p50": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": round(latencies[-1], 2) if latencies else 0.0,
        },
    }


def plot_confusion(report: dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping the confusion-matrix plot")
        return

    matrix = report["confusion_matrix"]
    names = report["class_names"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(matrix, cmap="Greens")
    ax.set(
        xticks=range(len(names)), yticks=range(len(names)),
        xticklabels=names, yticklabels=names,
        xlabel="predicted", ylabel="true",
        title=f"Deployed model - live accuracy {report['accuracy']:.1%}",
    )
    peak = max(max(row) for row in matrix) or 1
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > peak / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track deployed-model performance")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--test-dir", default=str(PROJECT_ROOT / "data" / "processed" / "test"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "reports" / "monitoring"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    url = args.url.rstrip("/")
    samples = collect_labelled_samples(Path(args.test_dir), args.samples, args.seed)
    print(f"Replaying {len(samples)} labelled requests against {url} ...")

    records, failures = [], 0
    for i, (path, true_label) in enumerate(samples, start=1):
        try:
            result = score_sample(url, path)
        except requests.RequestException as exc:
            failures += 1
            print(f"  request failed for {path.name}: {exc}")
            continue
        records.append({"file": path.name, "true_label": true_label, **result})
        if i % 25 == 0:
            print(f"  {i}/{len(samples)} scored")

    if not records:
        raise SystemExit("FAIL: no successful requests -- is the service running?")

    class_names = sorted({r["true_label"] for r in records} | {r["predicted"] for r in records})
    report = summarise(records, class_names)
    report["failed_requests"] = failures

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "performance_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )
    plot_confusion(report, out_dir / "live_confusion_matrix.png")

    print("\n=== Deployed model performance ===")
    print(f"  samples          : {report['samples']}")
    print(f"  accuracy         : {report['accuracy']:.2%}")
    print(f"  mean confidence  : {report['mean_confidence']:.4f}")
    print(f"  latency p50/p95  : {report['latency_ms']['p50']:.1f}ms / {report['latency_ms']['p95']:.1f}ms")
    for name, stats in report["per_class"].items():
        print(f"  {name:<6} precision={stats['precision']:.3f} "
              f"recall={stats['recall']:.3f} f1={stats['f1']:.3f} (n={stats['support']})")
    print(f"\nWrote {out_dir / 'performance_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
