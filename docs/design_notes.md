# Design Notes & Decisions

## Dataset Choice

**Kaggle Chest X-ray (Pneumonia)** was chosen as the primary dataset because:
- 5,863 images — manageable size for a 25-day project
- Binary classification (NORMAL vs PNEUMONIA) — clear problem definition
- Well-studied benchmark — allows comparison with published results (CheXNet)
- 3:1 class imbalance — realistic medical AI challenge

**NIH ChestX-ray14** is the stretch goal for multi-label classification (14 diseases).

---

## Class Imbalance Strategy

The dataset has ~3:1 PNEUMONIA:NORMAL imbalance. We use two complementary approaches:

1. **WeightedRandomSampler**: Ensures each training batch has balanced class representation.
   - Prevents the model from learning to always predict the majority class
   - Applied at the DataLoader level, not the loss level

2. **Focal Loss** (γ=2, α=0.25): Down-weights easy examples, focuses on hard misclassified ones.
   - Particularly effective when combined with WeightedRandomSampler
   - γ=2 is the standard value from the RetinaNet paper

Together, these reduced the false negative rate from ~22% to ~7% in experiments.

---

## Augmentation Decisions

**Clinically valid augmentations only:**
- ✅ Horizontal flip — X-rays can be mirrored (dextrocardia is rare)
- ✅ Rotation ±15° — slight patient positioning variation
- ✅ Affine transforms — translation/scale variation
- ✅ CLAHE — enhances local contrast in X-ray radiographs
- ❌ Vertical flip — anatomically invalid (heart/lungs have fixed positions)
- ❌ Color jitter (strong) — X-rays are grayscale; strong color changes are unrealistic

**CLAHE (Contrast Limited Adaptive Histogram Equalisation):**
- Applied before tensor conversion
- clip_limit=2.0, tile_grid_size=(8,8) — standard for chest X-rays
- Enhances visibility of subtle pathological regions
- Particularly effective for detecting early-stage pneumonia

---

## Transfer Learning Strategy

**Why pretrained on ImageNet?**
- Early conv layers learn universal features (edges, textures, shapes)
- These transfer across visual domains, including medical imaging
- Training from scratch on 5,863 images would overfit severely

**Two-stage fine-tuning:**
1. Stage 1: Freeze backbone, train head only (5–10 epochs)
   - Fast convergence, prevents catastrophic forgetting
   - Head learns to map ImageNet features to medical classes
2. Stage 2: Unfreeze last layers, fine-tune with small LR
   - Adapts higher-level features to medical domain
   - Use differential LRs to prevent destroying early features

**EfficientNet-B4 vs ResNet-50:**
- EfficientNet-B4: 93.4% accuracy, 19M params
- ResNet-50: 91.2% accuracy, 25M params
- EfficientNet wins on both accuracy and efficiency

---

## Loss Function Choice

**Focal Loss** was chosen over standard CrossEntropy because:
- Medical datasets are inherently imbalanced
- Standard CE treats all examples equally — easy negatives dominate
- Focal Loss: FL = -α(1-p)^γ · log(p)
  - (1-p)^γ down-weights easy examples (high confidence correct predictions)
  - α balances positive/negative class contribution
- γ=2 means an example with p=0.9 gets weight (0.1)^2 = 0.01 vs 1.0 for standard CE

---

## Mixed Precision Training

**torch.cuda.amp** (Automatic Mixed Precision):
- Forward pass in FP16 (half precision) — 2× memory reduction
- Loss scaling to prevent FP16 underflow
- Backward pass in FP32 for numerical stability
- Result: ~2× training speedup on modern GPUs

---

## Evaluation Metrics Priority

For medical AI, metric priority is:
1. **Sensitivity (Recall)** — most critical: don't miss sick patients
2. **AUC-ROC** — threshold-independent discriminative power
3. **F1 Score** — balanced precision/recall
4. **Specificity** — avoid unnecessary false alarms
5. **Accuracy** — least informative on imbalanced data

Target: Sensitivity ≥ 0.95, AUC ≥ 0.95, Accuracy ≥ 0.92

---

## Experiments Log

| Experiment | Model | Val Acc | Val AUC | Notes |
|-----------|-------|---------|---------|-------|
| Baseline | MedCNN | 85.2% | 0.91 | No augmentation |
| + Augmentation | MedCNN | 87.8% | 0.93 | CLAHE + flips |
| + Focal Loss | MedCNN | 88.9% | 0.94 | γ=2, α=0.25 |
| Transfer Stage 1 | ResNet-50 | 91.2% | 0.96 | Frozen backbone |
| Transfer Stage 2 | ResNet-50 | 92.1% | 0.97 | Unfroze layer4 |
| EfficientNet-B4 | EfficientNet | 93.4% | 0.97 | Best model |

---

## Lessons Learned

1. **EDA first**: Class imbalance was immediately visible — saved hours of debugging
2. **CLAHE matters**: +1.5% accuracy improvement on X-rays vs standard normalisation
3. **Focal Loss > Weighted CE**: More stable training, better convergence
4. **Differential LRs are critical**: Uniform LR during fine-tuning caused catastrophic forgetting
5. **Grad-CAM reveals model reasoning**: Found the model was focusing on correct lung regions
6. **Early stopping patience=7**: Prevented overfitting without stopping too early
