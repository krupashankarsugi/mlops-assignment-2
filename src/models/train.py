"""Stage 3 of the DVC pipeline: train the baseline CNN and log to MLflow.

Logs parameters, per-epoch metrics, the confusion matrix, the loss/accuracy
curves and the serialized model as MLflow artifacts, then writes the deployable
checkpoint to ``models/model.pt`` alongside a metadata sidecar the API reads.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: CI has no display
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn  # noqa: E402

from src.config import NORM_MEAN, NORM_STD, Config, load_config  # noqa: E402
from src.data.dataset import build_dataloaders  # noqa: E402
from src.models.cnn import build_model, count_parameters  # noqa: E402

LOGGER = logging.getLogger("train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon GPU when present; falls back to CPU in CI and in Docker.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one pass. Trains when *optimizer* is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)

    return total_loss / max(seen, 1), correct / max(seen, 1)


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_pred, positive-class probability)."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for images, labels in loader:
        probs = torch.softmax(model(images.to(device)), dim=1).cpu()
        y_true.append(labels.numpy())
        y_pred.append(probs.argmax(1).numpy())
        y_prob.append(probs[:, 1].numpy())
    return (
        np.concatenate(y_true) if y_true else np.array([]),
        np.concatenate(y_pred) if y_pred else np.array([]),
        np.concatenate(y_prob) if y_prob else np.array([]),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    # roc_auc is undefined when a split happens to hold a single class.
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def plot_curves(history: dict[str, list[float]], out_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, history["train_loss"], marker="o", label="train")
    ax_loss.plot(epochs, history["val_loss"], marker="o", label="validation")
    ax_loss.set(xlabel="epoch", ylabel="loss", title="Loss curve")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], marker="o", label="train")
    ax_acc.plot(epochs, history["val_acc"], marker="o", label="validation")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title="Accuracy curve")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="predicted",
        ylabel="true",
        title="Confusion matrix (test split)",
    )
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_checkpoint(
    model: nn.Module,
    cfg: Config,
    class_names: list[str],
    metrics: dict[str, float],
    model_name: str | None = None,
) -> tuple[Path, Path]:
    """Persist the .pt checkpoint plus the metadata sidecar used by the API."""
    model_name = model_name or cfg.train.model_name
    model_path = cfg.path(cfg.serving.model_path)
    meta_path = cfg.path(cfg.serving.metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.to("cpu").state_dict(),
            "model_name": model_name,
            "num_classes": len(class_names),
            "dropout": cfg.train.dropout,
            "class_names": class_names,
            "image_size": cfg.data.image_size,
            "norm_mean": NORM_MEAN,
            "norm_std": NORM_STD,
        },
        model_path,
    )
    meta_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "framework": f"pytorch-{torch.__version__}",
                "class_names": class_names,
                "image_size": cfg.data.image_size,
                "parameters_total": count_parameters(model, trainable_only=False),
                "parameters_trainable": count_parameters(model),
                "test_metrics": metrics,
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_path, meta_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Cats vs Dogs baseline CNN")
    parser.add_argument("--epochs", type=int, default=None, help="override params.yaml epochs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--model-name", default=None, help="override params.yaml train.model_name"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="log the run to MLflow but do not overwrite models/model.pt",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    epochs = args.epochs if args.epochs is not None else cfg.train.epochs
    model_name = args.model_name or cfg.train.model_name
    set_seed(cfg.train.seed)
    device = pick_device()
    LOGGER.info("training on device: %s", device)

    loaders, class_names = build_dataloaders(cfg)
    LOGGER.info("classes: %s", class_names)

    model = build_model(model_name, len(class_names), cfg.train.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    # Transfer models freeze their backbone, so only pass trainable params.
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    figures_dir = cfg.path("reports/figures")

    with mlflow.start_run(run_name=args.run_name or f"{model_name}-{int(time.time())}"):
        mlflow.log_params(
            {
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": cfg.train.batch_size,
                "learning_rate": cfg.train.learning_rate,
                "weight_decay": cfg.train.weight_decay,
                "dropout": cfg.train.dropout,
                "image_size": cfg.data.image_size,
                "seed": cfg.train.seed,
                "optimizer": "adam",
                "trainable_parameters": count_parameters(model),
                "total_parameters": count_parameters(model, trainable_only=False),
                "train_images": len(loaders["train"].dataset),
                "val_images": len(loaders["val"].dataset),
                "test_images": len(loaders["test"].dataset),
                "augmentation": "rrc+hflip+rotation+jitter",
            }
        )

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss, best_state, epochs_without_gain = float("inf"), None, 0

        for epoch in range(1, epochs + 1):
            started = time.time()
            train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
            val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )
            LOGGER.info(
                "epoch %d/%d  train_loss=%.4f train_acc=%.4f  val_loss=%.4f val_acc=%.4f  (%.1fs)",
                epoch, epochs, train_loss, train_acc, val_loss, val_acc, time.time() - started,
            )

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_gain = 0
            else:
                epochs_without_gain += 1
                if epochs_without_gain >= cfg.train.early_stopping_patience:
                    LOGGER.info("early stopping at epoch %d", epoch)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        y_true, y_pred, y_prob = collect_predictions(model, loaders["test"], device)
        test_metrics = compute_metrics(y_true, y_pred, y_prob)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        LOGGER.info("test metrics: %s", test_metrics)

        cm = confusion_matrix(y_true, y_pred)
        suffix = "" if not args.no_save else f"_{model_name}"
        cm_path = figures_dir / f"confusion_matrix{suffix}.png"
        curves_path = figures_dir / f"training_curves{suffix}.png"
        plot_confusion_matrix(cm, class_names, cm_path)
        plot_curves(history, curves_path)
        mlflow.log_artifact(str(cm_path), artifact_path="figures")
        mlflow.log_artifact(str(curves_path), artifact_path="figures")

        if args.no_save:
            LOGGER.info("--no-save: leaving models/model.pt untouched")
        else:
            model_path, meta_path = save_checkpoint(
                model, cfg, class_names, test_metrics, model_name
            )
            mlflow.log_artifact(str(model_path), artifact_path="model")
            mlflow.log_artifact(str(meta_path), artifact_path="model")
            LOGGER.info("saved model -> %s", model_path)
        mlflow.pytorch.log_model(
            model, artifact_path="pytorch-model",
            registered_model_name=cfg.mlflow.registered_model_name,
        )

        metrics_out = cfg.path(
            "reports/metrics.json" if not args.no_save else f"reports/metrics_{model_name}.json"
        )
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "test": test_metrics,
                    "best_val_loss": best_val_loss,
                    "epochs_run": len(history["train_loss"]),
                    "confusion_matrix": cm.tolist(),
                    "class_names": class_names,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        LOGGER.info("saved metrics -> %s", metrics_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
