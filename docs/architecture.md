# Architecture Documentation

## Model Progression

### Stage 1 — Custom CNN Baseline (`src/models/cnn_model.py`)

```
Input (B, 3, 224, 224)
    │
    ▼
ConvBlock 1: Conv(3→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool(2)
    │  Output: (B, 32, 112, 112)
    ▼
ConvBlock 2: Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool(2)
    │  Output: (B, 64, 56, 56)
    ▼
ConvBlock 3: Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool(2)
    │  Output: (B, 128, 28, 28)
    ▼
ConvBlock 4: Conv(128→256) → BN → ReLU → Conv(256→256) → BN → ReLU → MaxPool(2)
    │  Output: (B, 256, 14, 14)
    ▼
AdaptiveAvgPool2d(1)
    │  Output: (B, 256, 1, 1)
    ▼
Flatten → Linear(256→512) → ReLU → Dropout(0.4) → Linear(512→num_classes)
    │
    ▼
Output: (B, num_classes) — raw logits
```

**Parameters:** ~2.5M  
**Receptive field:** 224×224 (full image)  
**Key design choices:**
- Double 3×3 convolutions instead of single 7×7 (same receptive field, more non-linearity, fewer params)
- BatchNorm after every Conv (stabilises training, allows higher LR)
- AdaptiveAvgPool instead of fixed Flatten (handles variable input sizes)
- Kaiming initialisation for Conv, Xavier for Linear

---

### Stage 2 — ResNet-50 Transfer Learning (`src/models/resnet_transfer.py`)

```
ResNet-50 Backbone (pretrained on ImageNet)
    ├── conv1 + bn1 + relu + maxpool
    ├── layer1 (3 × Bottleneck blocks)   ← Frozen in Stage 1
    ├── layer2 (4 × Bottleneck blocks)   ← Frozen in Stage 1
    ├── layer3 (6 × Bottleneck blocks)   ← Frozen in Stage 1
    └── layer4 (3 × Bottleneck blocks)   ← Unfrozen in Stage 2
         │
         ▼
    AdaptiveAvgPool2d(1)
         │
         ▼
    Custom Head:
    Linear(2048→512) → ReLU → Dropout(0.4) → Linear(512→num_classes)
```

**Two-stage training:**
1. Stage 1 (epochs 1–10): Freeze backbone, train head only. LR = 1e-3
2. Stage 2 (epochs 11–30): Unfreeze layer4, fine-tune. LR = 1e-4 for layer4, 1e-3 for head

**Differential learning rates:**
- Head: 1e-3
- layer4: 1e-4
- layers 1–3: 1e-5 (if unfrozen)

---

### Stage 3 — EfficientNet-B4 (`src/models/efficientnet_model.py`)

```
EfficientNet-B4 Backbone (pretrained on ImageNet)
    ├── features[0]: Stem Conv
    ├── features[1–8]: MBConv blocks (compound scaled)
    │     depth=1.4, width=1.8, resolution=1.3
    └── features[-1]: Last Conv + BN + SiLU
         │
         ▼
    AdaptiveAvgPool2d(1)
         │
         ▼
    Custom Head:
    Dropout(0.4) → Linear(1792→512) → ReLU → Dropout(0.2) → Linear(512→num_classes)
```

**Why EfficientNet-B4 over ResNet-50?**
- 19M params vs 25M (24% fewer)
- Compound scaling: simultaneously scales depth, width, resolution
- Better accuracy-per-parameter ratio on ImageNet
- Achieves 93.4% vs ResNet-50's 91.2% in our experiments

---

## Training Pipeline

```
config.yaml
    │
    ▼
DataLoader (WeightedRandomSampler for class imbalance)
    │
    ▼
Model (CNN / ResNet-50 / EfficientNet-B4)
    │
    ▼
Forward Pass → Focal Loss (γ=2, α=0.25)
    │
    ▼
Backward Pass (AMP with GradScaler)
    │
    ▼
Gradient Clipping (max_norm=1.0)
    │
    ▼
AdamW Optimizer (lr=1e-3, weight_decay=1e-4)
    │
    ▼
CosineAnnealingLR Scheduler
    │
    ▼
Early Stopping (patience=7)
    │
    ▼
Checkpoint Save (best_model.pth)
```

---

## Explainability: Grad-CAM

```
Input X-ray
    │
    ▼
Forward Pass → Class Score y^c
    │
    ▼
Backward Pass → ∂y^c/∂A^k (gradients w.r.t. last conv layer)
    │
    ▼
α^c_k = GAP(∂y^c/∂A^k)  (importance weight per channel)
    │
    ▼
L^c = ReLU(Σ_k α^c_k · A^k)  (class activation map)
    │
    ▼
Resize to input resolution → Overlay on X-ray
```

**Target layers:**
- MedCNN: `model.features[-1].block[-3]` (last Conv2d)
- ResNet-50: `model.backbone.layer4[-1]`
- EfficientNet-B4: `model.backbone.features[-1]`
