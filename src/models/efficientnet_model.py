"""
efficientnet_model.py
EfficientNet-B4 fine-tuned for medical image classification.
EfficientNet uses compound scaling (depth × width × resolution) for
best accuracy-per-parameter ratio. B4 is the sweet spot for medical imaging.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetTransfer(nn.Module):
    """
    EfficientNet-B4 with custom classification head.

    Why B4?
      - Input: 380×380 (we resize to 224 for speed, still works well)
      - 19M parameters vs ResNet-50's 25M
      - Compound scaling: depth=1.4, width=1.8, resolution=1.3
      - Consistently outperforms ResNet-50 on medical imaging benchmarks

    Args:
        num_classes:     Number of output classes.
        freeze_backbone: If True, freeze feature extractor (Stage 1).
        dropout:         Dropout rate in the classifier head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()

        # Load pretrained EfficientNet-B4
        self.backbone = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )

        # Freeze feature extractor for Stage 1
        if freeze_backbone:
            for p in self.backbone.features.parameters():
                p.requires_grad = False

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features  # 1792
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def unfreeze_last_blocks(self, num_blocks: int = 3) -> None:
        """
        Progressively unfreeze the last N MBConv blocks.
        EfficientNet-B4 has 9 feature blocks (indices 0–8).

        Args:
            num_blocks: Number of blocks to unfreeze from the end.
        """
        feature_blocks = list(self.backbone.features.children())
        for block in feature_blocks[-num_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        print(f"Unfroze last {num_blocks} EfficientNet feature blocks.")

    def unfreeze_all(self) -> None:
        """Unfreeze entire backbone for full fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("Unfroze entire EfficientNet backbone.")

    def get_param_groups(
        self,
        head_lr: float = 1e-3,
        last_blocks_lr: float = 1e-4,
        backbone_lr: float = 1e-5,
    ) -> list:
        """
        Differential learning rates for EfficientNet.

        Returns:
            List of param group dicts for torch.optim.
        """
        feature_blocks = list(self.backbone.features.children())
        head_params = list(self.backbone.classifier.parameters())
        last_block_params = []
        for block in feature_blocks[-3:]:
            last_block_params += list(block.parameters())
        early_params = []
        for block in feature_blocks[:-3]:
            early_params += list(block.parameters())

        return [
            {"params": [p for p in head_params if p.requires_grad], "lr": head_lr},
            {"params": [p for p in last_block_params if p.requires_grad], "lr": last_blocks_lr},
            {"params": [p for p in early_params if p.requires_grad], "lr": backbone_lr},
        ]

    def get_target_layer(self):
        """Return the target layer for Grad-CAM (last conv block)."""
        return self.backbone.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def build_efficientnet(
    num_classes: int = 2,
    freeze_backbone: bool = True,
    dropout: float = 0.4,
) -> EfficientNetTransfer:
    """Factory function to build EfficientNet-B4 transfer learning model."""
    model = EfficientNetTransfer(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )
    counts = model.count_parameters()
    print(
        f"EfficientNet-B4 Transfer — "
        f"Total: {counts['total']:,} | Trainable: {counts['trainable']:,}"
    )
    return model
