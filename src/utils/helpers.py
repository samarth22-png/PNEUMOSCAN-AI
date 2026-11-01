"""
helpers.py
Utility functions used across the project.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml


def seed_everything(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seeded everything with seed={seed}")


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.

    Args:
        prefer_cuda: If True, use CUDA if available.

    Returns:
        torch.device.
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    elif prefer_cuda and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU available)")
    return device


def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Config dict.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from: {config_path}")
    return config


def compute_class_weights(
    labels: list,
    num_classes: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced datasets.
    Used with nn.CrossEntropyLoss(weight=...).

    Args:
        labels:      List of integer class labels.
        num_classes: Total number of classes.
        device:      Target device for the tensor.

    Returns:
        Tensor of class weights, shape (num_classes,).
    """
    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=num_classes).astype(float)
    total = class_counts.sum()

    # Inverse frequency: rare classes get higher weight
    weights = total / (num_classes * class_counts + 1e-8)
    weights = weights / weights.sum() * num_classes  # Normalise

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if device:
        weight_tensor = weight_tensor.to(device)

    print(f"Class weights: {weights.tolist()}")
    return weight_tensor


def save_json(data: dict, path: str) -> None:
    """Save dict to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON → {path}")


def load_json(path: str) -> dict:
    """Load JSON file to dict."""
    with open(path, "r") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters.

    Returns:
        Tuple of (total_params, trainable_params).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters — Total: {total:,} | Trainable: {trainable:,} | Frozen: {total - trainable:,}")
    return total, trainable


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Return model size in megabytes."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1e6
    print(f"Model size: {size_mb:.2f} MB")
    return size_mb


def denormalise_image(
    tensor: torch.Tensor,
    mean: list = None,
    std: list = None,
) -> np.ndarray:
    """
    Reverse ImageNet normalisation for visualisation.

    Args:
        tensor: Normalised image tensor (C, H, W).
        mean:   Normalisation mean.
        std:    Normalisation std.

    Returns:
        Denormalised numpy array (H, W, C), values in [0, 1].
    """
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]

    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)
    return img


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Build optimizer from config.

    Args:
        model:  PyTorch model.
        config: Full config dict.

    Returns:
        Configured optimizer.
    """
    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    print(f"Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    num_epochs: Optional[int] = None,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Build LR scheduler from config.

    Args:
        optimizer:  Optimizer instance.
        config:     Full config dict.
        num_epochs: Total training epochs (for cosine scheduler).

    Returns:
        LR scheduler or None.
    """
    train_cfg = config.get("training", {})
    scheduler_type = train_cfg.get("scheduler", "cosine")
    epochs = num_epochs or train_cfg.get("epochs", 30)

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        print(f"Scheduler: CosineAnnealingLR (T_max={epochs})")
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
        print("Scheduler: StepLR (step=10, gamma=0.1)")
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
        print("Scheduler: ReduceLROnPlateau")
    elif scheduler_type == "warmup_cosine":
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - 5, eta_min=1e-6
        )
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
        print("Scheduler: Warmup + CosineAnnealingLR")
    else:
        scheduler = None
        print("No LR scheduler")

    return scheduler
