#!/usr/bin/env python3
"""Promote a tracked MLflow run to the deployable serving artifact (M1 -> M2).

Experiment tracking is only useful if the winning run can actually be shipped.
This script picks a run from the MLflow store -- by run id, or automatically the
best run by a chosen metric -- and rewrites ``models/model.pt`` plus the
metadata sidecar the API reads, so the next image build serves those weights.

Usage:
    python scripts/promote_model.py --best-by test_accuracy
    python scripts/promote_model.py --run-id 9a3f...  --yes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import torch
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import NORM_MEAN, NORM_STD, load_config  # noqa: E402
from src.models.cnn import count_parameters  # noqa: E402


def find_best_run(client: MlflowClient, experiment_name: str, metric: str):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(f"no MLflow experiment named {experiment_name!r}")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"metrics.{metric} > 0",
        order_by=[f"metrics.{metric} DESC"],
        max_results=25,
    )
    if not runs:
        raise SystemExit(f"no runs in {experiment_name!r} logged the metric {metric!r}")
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote an MLflow run to models/model.pt")
    parser.add_argument("--run-id", default=None, help="explicit MLflow run id")
    parser.add_argument("--best-by", default="test_accuracy", help="metric used to rank runs")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args(argv)

    cfg = load_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    client = MlflowClient()

    if args.run_id:
        run = client.get_run(args.run_id)
    else:
        runs = find_best_run(client, cfg.mlflow.experiment_name, args.best_by)
        print(f"Runs in {cfg.mlflow.experiment_name!r} ranked by {args.best_by}:")
        for r in runs:
            print(
                f"  {r.data.tags.get('mlflow.runName', r.info.run_id[:8]):<24}"
                f"  {args.best_by}={r.data.metrics.get(args.best_by, 0):.4f}"
                f"  model={r.data.params.get('model_name', '?')}"
            )
        run = runs[0]

    run_name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
    model_name = run.data.params.get("model_name", cfg.train.model_name)
    print(f"\nPromoting run {run_name!r} (model={model_name}, id={run.info.run_id})")

    if not args.yes:
        if input("Overwrite models/model.pt? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("aborted")
            return 1

    # Load the pytorch flavour logged by src/models/train.py.
    model = mlflow.pytorch.load_model(f"runs:/{run.info.run_id}/pytorch-model").eval()

    class_names = list(cfg.serving.class_names)
    model_path = cfg.path(cfg.serving.model_path)
    meta_path = cfg.path(cfg.serving.metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.to("cpu").state_dict(),
            "model_name": model_name,
            "num_classes": len(class_names),
            "dropout": float(run.data.params.get("dropout", cfg.train.dropout)),
            "class_names": class_names,
            "image_size": int(run.data.params.get("image_size", cfg.data.image_size)),
            "norm_mean": NORM_MEAN,
            "norm_std": NORM_STD,
        },
        model_path,
    )

    test_metrics = {
        k.removeprefix("test_"): round(v, 6)
        for k, v in run.data.metrics.items()
        if k.startswith("test_")
    }
    meta_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "framework": f"pytorch-{torch.__version__}",
                "class_names": class_names,
                "image_size": int(run.data.params.get("image_size", cfg.data.image_size)),
                "parameters_total": count_parameters(model, trainable_only=False),
                "parameters_trainable": count_parameters(model),
                "test_metrics": test_metrics,
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "promoted_from_run": run.info.run_id,
                "promoted_run_name": run_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {model_path}  ({size_mb:.1f} MB)")
    print(f"Wrote {meta_path}")
    print(f"Test metrics: {test_metrics}")
    print("\nRebuild the image to ship these weights:  make docker-build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
