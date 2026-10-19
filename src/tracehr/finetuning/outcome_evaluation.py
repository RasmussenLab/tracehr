# src/tracehr/finetuning/outcome_evaluation.py
"""Evaluation of an NTPOutcomeModel (pipeline 06): the task's metrics over a loader, and the
task's evaluation plot (ROC/PR, predicted vs true, confusion matrix, or Kaplan-Meier curves
by predicted risk group)."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from ..training.runtime import ProgressLog, batch_to_device
from .survival_core import km_by_risk_panel, split_survival_labels, survival_metrics


@torch.no_grad()
def evaluate_outcome(
    model,
    loader: DataLoader,
    device: torch.device,
    task: str,
    show_progress: bool = False,
) -> dict[str, float]:
    """Task-generic evaluation for NTPOutcomeModel.

    Returns a dict that always contains "loss" plus the task's monitored metric
    (binary->auroc, multiclass->macro_f1, regression->mae, survival->cindex) and
    companion metrics. Collected predictions are: probabilities (binary), class-prob
    matrix (multiclass), raw values (regression), or log relative risk (survival).
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total_weight = 0.0, 0.0

    for batch in ProgressLog(loader, "[evaluate]", disable=not show_progress):
        batch = batch_to_device(batch, device)
        out = model(batch)
        # Only the two classification tasks produce a probability; regression and survival
        # both report the head's raw output.
        key = "probs" if task in ("binary", "multiclass") else "predictions"
        all_preds.append(out[key].cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())
        if "loss" in out:
            # Scale each batch's mean loss back by what it was divided by, so the sum over
            # batches is the mean over the whole split, whatever the batch size: the batch
            # size, except for class-weighted cross-entropy, which PyTorch divides by the
            # summed class weights of the batch's labels.
            weight = (model.class_weights[batch["labels"].long()].sum().item()
                      if task == "multiclass" and model.class_weights is not None
                      else batch["labels"].shape[0])
            total_loss += out["loss"].item() * weight
            total_weight += weight

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    # A sum of class weights can be below 1, so only an empty split is guarded
    metrics = {"loss": total_loss / total_weight if total_weight else 0.0, "n_total": int(len(labels))}

    if task == "survival":
        # Straight from survival_core, the same function the stage-05 arms report through,
        # so "cindex" means exactly the same thing in both stages' metric files. It also
        # supplies n_total, so drop the scalar-label version set above.
        metrics.pop("n_total", None)
        metrics.update(survival_metrics(labels, preds))
    elif task == "regression":
        metrics["mse"] = float(mean_squared_error(labels, preds))
        metrics["mae"] = float(mean_absolute_error(labels, preds))
        metrics["r2"] = float(r2_score(labels, preds)) if len(labels) > 1 else 0.0
    elif task == "binary":
        y = labels.astype(int)
        pred = (preds >= 0.5).astype(int)
        # A one-class split leaves both ranking metrics undefined. NaN, not 0.0, for the
        # reason stage 05's classification_metrics gives: 0.0 reads as a measured failure. json_safe
        # writes it as null.
        both = len(np.unique(y)) > 1
        metrics.update({
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "auroc": float(roc_auc_score(y, preds)) if both else float("nan"),
            "auprc": float(average_precision_score(y, preds)) if both else float("nan"),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
        })
    else:  # multiclass
        y = labels.astype(int)
        pred_class = preds.argmax(axis=-1)
        present = np.unique(y)
        metrics["accuracy"] = float(accuracy_score(y, pred_class))
        metrics["macro_f1"] = float(f1_score(y, pred_class, average="macro", zero_division=0))
        # One-vs-rest AUROC needs every class represented in the labels. With K=2 sklearn reads the
        # labels as binary and wants one score column: the class-1 probability, whose AUROC equals
        # the two-class one-vs-rest macro average.
        scores = preds[:, 1] if preds.shape[1] == 2 else preds
        metrics["ovr_auroc"] = (
            float(roc_auc_score(y, scores, multi_class="ovr", average="macro"))
            if len(present) == preds.shape[1] else float("nan")
        )

    return metrics


def plot_outcome_eval(task: str, labels: np.ndarray, preds: np.ndarray, output_path: str) -> None:
    """Task-dispatched evaluation plot: PR/ROC (binary), scatter (regression),
    confusion matrix (multiclass), Kaplan-Meier by predicted risk (survival)."""
    import matplotlib.pyplot as plt

    if task == "binary":
        plot_curves(labels, preds, output_path)
        return

    if task == "survival":
        # labels: [N, 2] = (duration, event). Same panel the stage-05 report draws, so
        # the two are read the same way: ordered, separated curves mean the risk score
        # discriminates; overlapping ones mean the C-index is noise.
        durations, events = split_survival_labels(labels)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        km_by_risk_panel(ax, durations, events, preds)
        ax.set_title("Observed survival by predicted risk group — test set")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    if task == "regression":
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(labels, preds, alpha=0.4, s=12, color="steelblue")
        lo, hi = float(min(labels.min(), preds.min())), float(max(labels.max(), preds.max()))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.6)
        ax.set_xlabel("True value")
        ax.set_ylabel("Predicted value")
        ax.set_title("Regression: predicted vs true")
        ax.grid(True, alpha=0.3)
    else:  # multiclass confusion matrix
        y = labels.astype(int)
        pred_class = preds.argmax(axis=-1)
        K = preds.shape[1]
        cm = confusion_matrix(y, pred_class, labels=list(range(K)))
        fig, ax = plt.subplots(figsize=(1.2 * K + 2, 1.2 * K + 2))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("Multiclass confusion matrix")
        for r in range(K):
            for c in range(K):
                ax.text(c, r, int(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    output_path: str,
) -> None:
    """Plot ROC and PR curves for a single-output binary classifier."""
    import matplotlib.pyplot as plt

    probs_flat = probs.squeeze(-1) if probs.ndim == 2 else probs
    labels_flat = labels.squeeze(-1) if labels.ndim == 2 else labels

    if len(np.unique(labels_flat)) < 2:
        # The two panels' summary numbers are the metrics evaluate_outcome just reported as
        # undefined; roc_auc_score would raise here and take the rest of the stage with it.
        print("  [plot] one-class test split: ROC/PR curves are undefined, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(labels_flat, probs_flat)
    auroc = roc_auc_score(labels_flat, probs_flat)
    axes[0].plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    precision, recall, _ = precision_recall_curve(labels_flat, probs_flat)
    auprc = average_precision_score(labels_flat, probs_flat)
    axes[1].plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
    axes[1].axhline(y=labels_flat.mean(), color="k", linestyle="--", alpha=0.5, label="Baseline")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
