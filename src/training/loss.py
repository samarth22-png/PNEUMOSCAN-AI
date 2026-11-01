"""
loss.py
Loss functions for medical image classification.
Includes: CrossEntropyLoss, Weighted CrossEntropy, Focal Loss, Label Smoothing CE.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for class-imbalanced datasets (Lin et al., 2017).
    Down-weights easy examples so the model focuses on hard misclassified ones.

    FL(p_t) = -α(1 - p_t)^γ · log(p_t)

    Where:
        p_t = model probability for the true class
        γ   = focusing parameter (γ=2 is standard)
        α   = class balance weight

    Args:
        alpha:      Weighting factor for the rare class (0.25 default).
        gamma:      Focusing parameter (2.0 default).
        reduction:  'mean' | 'sum' | 'none'.
        num_classes: Number of classes (for multi-class support).
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        num_classes: int = 2,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  Raw logits, shape (B, num_classes).
            targets: Ground truth labels, shape (B,).

        Returns:
            Scalar focal loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # Probability of the true class
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting: α for positive class, (1-α) for negative
        if self.num_classes == 2:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        else:
            alpha_t = torch.full_like(ce_loss, self.alpha)

        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.
    Prevents overconfident predictions by softening the target distribution.

    Args:
        smoothing: Label smoothing factor ε (0.1 is standard).
        reduction: 'mean' | 'sum'.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        # Smooth targets: (1 - ε) for true class, ε/(C-1) for others
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(config: dict, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Factory function to build the loss function from config.

    Args:
        config:        Full config dict.
        class_weights: Optional tensor of class weights for imbalanced data.

    Returns:
        Loss function module.
    """
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "cross_entropy")
    num_classes = config["data"]["num_classes"]

    if loss_type == "focal":
        criterion = FocalLoss(
            alpha=loss_cfg.get("focal_alpha", 0.25),
            gamma=loss_cfg.get("focal_gamma", 2.0),
            num_classes=num_classes,
        )
        print(f"Using Focal Loss (γ={loss_cfg.get('focal_gamma', 2.0)}, α={loss_cfg.get('focal_alpha', 0.25)})")

    elif loss_type == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using Weighted CrossEntropyLoss")

    elif loss_type == "label_smoothing":
        smoothing = config["training"].get("label_smoothing", 0.1)
        criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        print(f"Using Label Smoothing CrossEntropy (ε={smoothing})")

    else:  # cross_entropy (default)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config["training"].get("label_smoothing", 0.0),
        )
        print("Using CrossEntropyLoss")

    return criterion
