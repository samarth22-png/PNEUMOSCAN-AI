"""
metrics.py
All evaluation metrics for medical image classification.
Includes: Accuracy, Precision, Recall, F1, AUC, Sensitivity, Specificity.
Returns structured dicts loggable to TensorBoard and JSON.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    preds: List[int],
    labels: List[int],
    probs: Optional[np.ndarray] = None,
    num_classes: int = 2,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute all classification metrics.

    Args:
        preds:       Predicted class indices, shape (N,).
        labels:      Ground truth class indices, shape (N,).
        probs:       Predicted probabilities, shape (N, num_classes). Required for AUC.
        num_classes: Number of classes.
        class_names: Optional list of class names for per-class metrics.

    Returns:
        Dict with all metrics, ready for TensorBoard / JSON logging.
    """
    preds = np.array(preds)
    labels = np.array(labels)

    # Basic metrics
    acc = accuracy_score(labels, preds)

    avg = "binary" if num_classes == 2 else "macro"
    precision = precision_score(labels, preds, average=avg, zero_division=0)
    recall = recall_score(labels, preds, average=avg, zero_division=0)
    f1 = f1_score(labels, preds, average=avg, zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)

    # Sensitivity and Specificity (binary)
    if num_classes == 2 and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-8)   # Recall for positive class
        specificity = tn / (tn + fp + 1e-8)   # True negative rate
        ppv = tp / (tp + fp + 1e-8)           # Positive predictive value
        npv = tn / (tn + fn + 1e-8)           # Negative predictive value
    else:
        sensitivity = recall
        specificity = None
        ppv = precision
        npv = None

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(sensitivity),
        "f1": float(f1),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
    }

    if specificity is not None:
        metrics["specificity"] = float(specificity)
        metrics["ppv"] = float(ppv)
        metrics["npv"] = float(npv)

    # AUC metrics (require probability scores)
    if probs is not None:
        probs = np.array(probs)
        try:
            if num_classes == 2:
                # Use probability of positive class
                pos_probs = probs[:, 1] if probs.ndim == 2 else probs
                roc_auc = roc_auc_score(labels, pos_probs)
                avg_precision = average_precision_score(labels, pos_probs)
                fpr, tpr, thresholds = roc_curve(labels, pos_probs)
                metrics["roc_auc"] = float(roc_auc)
                metrics["avg_precision"] = float(avg_precision)
                metrics["roc_curve"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thresholds.tolist(),
                }
            else:
                # One-vs-Rest AUC for multi-class
                roc_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
                metrics["roc_auc"] = float(roc_auc)
        except ValueError as e:
            print(f"Warning: Could not compute AUC: {e}")

    # Per-class F1 scores
    if class_names:
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        for i, name in enumerate(class_names):
            if i < len(per_class_f1):
                metrics[f"f1_{name}"] = float(per_class_f1[i])

    return metrics


def format_metrics(metrics: Dict, prefix: str = "") -> str:
    """
    Format metrics dict as a human-readable string for logging.

    Args:
        metrics: Dict from compute_metrics().
        prefix:  Optional prefix (e.g., 'Val' or 'Test').

    Returns:
        Formatted string.
    """
    skip_keys = {"confusion_matrix", "roc_curve"}
    parts = []
    for k, v in metrics.items():
        if k in skip_keys:
            continue
        if isinstance(v, float):
            parts.append(f"{k}: {v:.4f}")

    header = f"[{prefix}] " if prefix else ""
    return header + " | ".join(parts)


def compute_optimal_threshold(
    labels: List[int],
    probs: np.ndarray,
    metric: str = "f1",
) -> float:
    """
    Find the optimal classification threshold by maximising a metric.
    Useful for medical AI where you want to tune sensitivity vs specificity.

    Args:
        labels: Ground truth binary labels.
        probs:  Predicted probabilities for the positive class.
        metric: 'f1' | 'sensitivity' | 'youden' (Youden's J = sensitivity + specificity - 1).

    Returns:
        Optimal threshold value.
    """
    labels = np.array(labels)
    probs = np.array(probs)
    fpr, tpr, thresholds = roc_curve(labels, probs)

    if metric == "youden":
        # Maximise Youden's J statistic
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
    elif metric == "sensitivity":
        # Find threshold where sensitivity >= 0.95
        target_sensitivity = 0.95
        valid = tpr >= target_sensitivity
        if valid.any():
            best_idx = np.where(valid)[0][0]
        else:
            best_idx = np.argmax(tpr)
    else:  # f1
        # Maximise F1 score across thresholds
        f1_scores = []
        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)
            f1_scores.append(f1_score(labels, preds, zero_division=0))
        best_idx = np.argmax(f1_scores)

    optimal_threshold = float(thresholds[best_idx])
    print(f"Optimal threshold ({metric}): {optimal_threshold:.4f}")
    return optimal_threshold
