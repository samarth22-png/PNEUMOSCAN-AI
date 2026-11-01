"""
cnn_model.py
Custom CNN architecture built from scratch.
Establishes the baseline before transfer learning.
Architecture: 4× ConvBlock → GlobalAvgPool → Classifier head
Channel progression: 3 → 32 → 64 → 128 → 256
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv2d → BatchNorm2d → ReLU → MaxPool2d.
    Halves spatial dimensions via MaxPool(2).

    Args:
        in_ch:  Number of input channels.
        out_ch: Number of output channels.
        dropout_p: Dropout probability (0 = disabled).
    """

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),  # Double 3×3
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.block(x))


class MedCNN(nn.Module):
    """
    Medical CNN baseline model.

    Input:  (B, 3, 224, 224)
    Output: (B, num_classes) — raw logits

    Spatial progression:
        224 → 112 → 56 → 28 → 14 → 1 (after GlobalAvgPool)

    Args:
        num_classes: Number of output classes (2 for binary).
        dropout:     Dropout rate in the classifier head.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),           # 224 → 112
            ConvBlock(32, 64),          # 112 → 56
            ConvBlock(64, 128),         # 56  → 28
            ConvBlock(128, 256),        # 28  → 14
        )

        self.pool = nn.AdaptiveAvgPool2d(1)  # 14 → 1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialisation for Conv layers, Xavier for Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def get_feature_maps(self, x: torch.Tensor) -> list:
        """Return intermediate feature maps for visualisation."""
        maps = []
        for block in self.features:
            x = block(x)
            maps.append(x)
        return maps

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_cnn(num_classes: int = 2, dropout: float = 0.4) -> MedCNN:
    """Factory function to build the baseline CNN."""
    model = MedCNN(num_classes=num_classes, dropout=dropout)
    print(f"MedCNN — Parameters: {model.count_parameters():,}")
    return model
