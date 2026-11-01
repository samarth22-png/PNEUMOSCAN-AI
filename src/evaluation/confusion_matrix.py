"""
confusion_matrix.py
Confusion matrix visualisation and analysis for medical image classification.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    labels: List[int],
    preds: List[int],
    class_names: List[str],
    save_path: Optional[str] = None,
    normalise: bool = True,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Plot a styled confusion matrix.

    Args:
        labels:      Ground truth labels.
        preds:       Predicted labels.
        class_names: List of class names.
        save_path:   Optional path to save the figure.
        normalise:   If True, show percentages instead of counts.
        title:       Figure title.

    Returns:
        matplotlib Figure.
    """
    cm = confusion_matrix(labels, preds)

    if normalise:
        cm_display = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2%"
        vmax = 1.0
    else:
        cm_display = cm
        fmt = "d"
        vmax = cm.max()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#070b14")
    ax.set_facecolor("#0d1424")

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="#1e2d4a",
        vmin=0,
        vmax=vmax,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted Label", color="white", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", color="white", fontsize=12, labelpad=10)
    ax.set_title(title, color="#00c9ff", fontsize=14, pad=15)
    ax.tick_params(colors="white")

    # Style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="white")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved confusion matrix → {save_path}")

    return fig


def print_classification_report(
    labels: List[int],
    preds: List[int],
    class_names: List[str],
) -> None:
    """Print a detailed per-class classification report."""
    cm = confusion_matrix(labels, preds)
    n_classes = len(class_names)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(f"{'Class':<15} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'Sens':>8} {'Spec':>8} {'F1':>8}")
    print("-" * 60)

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        print(
            f"{class_names[i]:<15} {tp:>6} {fp:>6} {fn:>6} {tn:>6} "
            f"{sensitivity:>8.4f} {specificity:>8.4f} {f1:>8.4f}"
        )

    print("=" * 60)
    overall_acc = np.trace(cm) / cm.sum()
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print("=" * 60 + "\n")
