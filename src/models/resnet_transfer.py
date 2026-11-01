"""
resnet_transfer.py
ResNet-50 fine-tuned for medical image classification.
Uses differential learning rates and two-stage training:
  Stage 1: Freeze backbone, train head only (feature extraction)
  Stage 2: Unfreeze layer4, fine-tune with small LR
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetTransfer(nn.Module):
    """
    ResNet-50 with custom classification head for medical imaging.

    Architecture:
        ResNet-50 backbone (pretrained on ImageNet)
        → Replace fc: Linear(2048, 512) → ReLU → Dropout → Linear(512, num_classes)

    Args:
        num_classes:      Number of output classes.
        freeze_backbone:  If True, freeze all backbone layers (Stage 1).
        dropout:          Dropout rate in the classifier head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()

        # Load pretrained ResNet-50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Freeze backbone for Stage 1 (feature extraction)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Replace the final fully-connected layer
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def unfreeze_layer4(self) -> None:
        """
        Stage 2: Unfreeze layer4 for fine-tuning.
        Use a smaller LR (1e-4) for these layers.
        """
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True
        print("Unfroze ResNet layer4 for fine-tuning.")

    def unfreeze_all(self) -> None:
        """Unfreeze entire backbone for full fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("Unfroze entire ResNet backbone.")

    def get_param_groups(
        self,
        head_lr: float = 1e-3,
        layer4_lr: float = 1e-4,
        backbone_lr: float = 1e-5,
    ) -> list:
        """
        Return parameter groups with differential learning rates.
        Smaller LR for earlier layers (they already have good features).

        Args:
            head_lr:     LR for the classification head.
            layer4_lr:   LR for layer4 (if unfrozen).
            backbone_lr: LR for layers 1–3 (if unfrozen).

        Returns:
            List of dicts for torch.optim.
        """
        head_params = list(self.backbone.fc.parameters())
        layer4_params = list(self.backbone.layer4.parameters())
        early_params = (
            list(self.backbone.layer1.parameters())
            + list(self.backbone.layer2.parameters())
            + list(self.backbone.layer3.parameters())
            + list(self.backbone.conv1.parameters())
            + list(self.backbone.bn1.parameters())
        )

        return [
            {"params": [p for p in head_params if p.requires_grad], "lr": head_lr},
            {"params": [p for p in layer4_params if p.requires_grad], "lr": layer4_lr},
            {"params": [p for p in early_params if p.requires_grad], "lr": backbone_lr},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def count_parameters(self) -> dict:
        """Return trainable and total parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def build_resnet(
    num_classes: int = 2,
    freeze_backbone: bool = True,
    dropout: float = 0.4,
) -> ResNetTransfer:
    """Factory function to build ResNet-50 transfer learning model."""
    model = ResNetTransfer(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )
    counts = model.count_parameters()
    print(
        f"ResNet-50 Transfer — "
        f"Total: {counts['total']:,} | Trainable: {counts['trainable']:,}"
    )
    return model
