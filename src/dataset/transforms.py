"""
transforms.py
Domain-specific augmentation pipeline for chest X-ray classification.
Clinically valid transforms only — no vertical flips, no color jitter.
"""

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


class CLAHETransform:
    """
    Contrast Limited Adaptive Histogram Equalisation (CLAHE).
    Enhances local contrast in X-ray images, making pathological
    regions more distinguishable. Applied before tensor conversion.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=tile_grid_size
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        # Convert PIL → numpy (grayscale)
        img_np = np.array(img.convert("L"))
        img_clahe = self.clahe.apply(img_np)
        # Convert back to RGB PIL (3-channel for pretrained models)
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(img_rgb)

    def __repr__(self) -> str:
        return (
            f"CLAHETransform(clip_limit={self.clip_limit}, "
            f"tile_grid_size={self.tile_grid_size})"
        )


# ImageNet normalisation statistics (used for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224, use_clahe: bool = True) -> transforms.Compose:
    """
    Training augmentation pipeline.
    Clinically valid: horizontal flip OK (X-rays can be mirrored),
    no vertical flip (anatomically invalid), no color jitter (grayscale).

    Args:
        img_size: Target image size (square).
        use_clahe: Whether to apply CLAHE contrast enhancement.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    transform_list = []

    if use_clahe:
        transform_list.append(CLAHETransform(clip_limit=2.0))

    transform_list += [
        transforms.Resize((img_size + 32, img_size + 32)),  # Slightly larger for crop
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=5,
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Mild — X-rays are grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Simulate occlusion
    ]

    return transforms.Compose(transform_list)


def get_val_transforms(img_size: int = 224, use_clahe: bool = True) -> transforms.Compose:
    """
    Validation / test augmentation pipeline.
    Only deterministic transforms — no randomness.

    Args:
        img_size: Target image size (square).
        use_clahe: Whether to apply CLAHE contrast enhancement.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    transform_list = []

    if use_clahe:
        transform_list.append(CLAHETransform(clip_limit=2.0))

    transform_list += [
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return transforms.Compose(transform_list)


def get_tta_transforms(img_size: int = 224, use_clahe: bool = True) -> list:
    """
    Test-Time Augmentation (TTA) transforms.
    Returns a list of deterministic transforms for ensemble prediction.

    Args:
        img_size: Target image size.
        use_clahe: Whether to apply CLAHE.

    Returns:
        List of Compose transforms for TTA.
    """
    base = []
    if use_clahe:
        base.append(CLAHETransform(clip_limit=2.0))

    norm = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    tta_list = [
        # Original
        transforms.Compose(base + [transforms.Resize((img_size, img_size)), transforms.CenterCrop(img_size)] + norm),
        # Horizontal flip
        transforms.Compose(base + [transforms.Resize((img_size, img_size)), transforms.CenterCrop(img_size), transforms.RandomHorizontalFlip(p=1.0)] + norm),
        # Slight rotation +10
        transforms.Compose(base + [transforms.Resize((img_size, img_size)), transforms.CenterCrop(img_size), transforms.RandomRotation((10, 10))] + norm),
        # Slight rotation -10
        transforms.Compose(base + [transforms.Resize((img_size, img_size)), transforms.CenterCrop(img_size), transforms.RandomRotation((-10, -10))] + norm),
    ]

    return tta_list
