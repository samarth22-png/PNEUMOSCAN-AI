"""
roc_auc.py
ROC curve and Precision-Recall curve plotting for medical image classification.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_roc_curve(
    labels: List[int],
    probs: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "ROC Curve",
) -> plt.Figure:
    """
    Plot ROC curve with AUC score.

    Args:
        labels:      Ground truth binary labels.
        probs:       Predicted probabilities, shape (N, num_classes) or (N,).
        class_names: Optional class names for legend.
        save_path:   Optional path to save the figure.
        title:       Figure title.

    Returns:
        matplotlib Figure.
    """
    labels = np.array(labels)
    probs = np.array(probs)

    # Extract positive class probabilities
    if probs.ndim == 2:
        pos_probs = probs[:, 1]
    else:
        pos_probs = probs

    fpr, tpr, thresholds = roc_curve(labels, pos_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#070b14")
    ax.set_facecolor("#0d1424")

    # ROC curve
    ax.plot(fpr, tpr, color="#00c9ff", lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")

    # Diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], color="#1e2d4a", lw=1.5, linestyle="--",
            label="Random Classifier (AUC = 0.50)")

    # Optimal threshold point (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    ax.scatter(fpr[best_idx], tpr[best_idx], color="#00ffa3", s=100, zorder=5,
               label=f"Optimal Threshold = {thresholds[best_idx]:.3f}")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", color="white", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", color="white", fontsize=12)
    ax.set_title(title, color="#00c9ff", fontsize=14)
    ax.tick_params(colors="white")
    ax.legend(loc="lower right", facecolor="#0d1424", edgecolor="#1e2d4a",
              labelcolor="white", fontsize=10)

    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d4a")

    # Add AUC annotation
    ax.text(0.6, 0.15, f"AUC = {roc_auc:.4f}", color="#ffc947",
            fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#121c30", edgecolor="#ffc947", alpha=0.8))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved ROC curve → {save_path}")

    return fig


def plot_precision_recall_curve(
    labels: List[int],
    probs: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Precision-Recall Curve",
) -> plt.Figure:
    """
    Plot Precision-Recall curve with Average Precision score.

    Args:
        labels:    Ground truth binary labels.
        probs:     Predicted probabilities, shape (N, 2) or (N,).
        save_path: Optional path to save the figure.
        title:     Figure title.

    Returns:
        matplotlib Figure.
    """
    labels = np.array(labels)
    probs = np.array(probs)

    if probs.ndim == 2:
        pos_probs = probs[:, 1]
    else:
        pos_probs = probs

    precision, recall, thresholds = precision_recall_curve(labels, pos_probs)
    avg_precision = average_precision_score(labels, pos_probs)

    # Baseline: random classifier
    baseline = labels.sum() / len(labels)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#070b14")
    ax.set_facecolor("#0d1424")

    ax.plot(recall, precision, color="#7b61ff", lw=2.5,
            label=f"PR Curve (AP = {avg_precision:.4f})")
    ax.axhline(y=baseline, color="#1e2d4a", lw=1.5, linestyle="--",
               label=f"Baseline (AP = {baseline:.4f})")

    # Find F1-optimal point
    if len(thresholds) > 0:
        f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
        best_idx = np.argmax(f1_scores)
        ax.scatter(recall[best_idx], precision[best_idx], color="#00ffa3", s=100, zorder=5,
                   label=f"Best F1 = {f1_scores[best_idx]:.3f}")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", color="white", fontsize=12)
    ax.set_ylabel("Precision (PPV)", color="white", fontsize=12)
    ax.set_title(title, color="#7b61ff", fontsize=14)
    ax.tick_params(colors="white")
    ax.legend(loc="upper right", facecolor="#0d1424", edgecolor="#1e2d4a",
              labelcolor="white", fontsize=10)

    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d4a")

    ax.text(0.05, 0.1, f"AP = {avg_precision:.4f}", color="#ffc947",
            fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#121c30", edgecolor="#ffc947", alpha=0.8))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved PR curve → {save_path}")

    return fig


def plot_training_curves(
    history: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history:   Training history dict from Trainer.fit().
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#070b14")

    plot_configs = [
        ("Loss", "train_loss", "val_loss", "#ff6b6b", "#00c9ff"),
        ("Accuracy", "train_acc", "val_acc", "#ff6b6b", "#00c9ff"),
        ("F1 Score", "train_f1", "val_f1", "#ff6b6b", "#00c9ff"),
    ]

    for ax, (metric_name, train_key, val_key, train_color, val_color) in zip(axes, plot_configs):
        ax.set_facecolor("#0d1424")
        if train_key in history:
            ax.plot(epochs, history[train_key], color=train_color, lw=2, label="Train")
        if val_key in history:
            ax.plot(epochs, history[val_key], color=val_color, lw=2, label="Val")
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel(metric_name, color="white")
        ax.set_title(f"Training {metric_name}", color="white", fontsize=12)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#0d1424", edgecolor="#1e2d4a", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2d4a")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved training curves → {save_path}")

    return fig
