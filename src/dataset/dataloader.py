"""
dataloader.py
Custom Dataset class and DataLoader factory for chest X-ray classification.
Supports folder-based and CSV-based label loading.
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for chest X-ray images.

    Supports two modes:
      1. Folder-based: data/processed/train/NORMAL/, data/processed/train/PNEUMONIA/
      2. CSV-based:    CSV with columns ['filepath', 'label']

    Args:
        root_dir:   Root directory of the dataset.
        split:      One of 'train', 'val', 'test'.
        transform:  torchvision transform pipeline.
        csv_file:   Optional path to CSV file with filepath/label columns.
        class_names: List of class names in label order.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        csv_file: Optional[str] = None,
        class_names: Optional[list] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.class_names = class_names or ["NORMAL", "PNEUMONIA"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

        if csv_file is not None:
            self._load_from_csv(csv_file)
        else:
            self._load_from_folder()

    def _load_from_folder(self) -> None:
        """Load image paths and labels from folder structure."""
        split_dir = self.root_dir / self.split
        self.samples = []

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                f"Expected structure: {self.root_dir}/{{train,val,test}}/{{CLASS_NAME}}/*.jpg"
            )

        for class_name in self.class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            label = self.class_to_idx[class_name]
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                for img_path in class_dir.glob(ext):
                    self.samples.append((str(img_path), label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {split_dir}. "
                "Check folder structure and image extensions."
            )

    def _load_from_csv(self, csv_file: str) -> None:
        """Load image paths and labels from a CSV file."""
        df = pd.read_csv(csv_file)
        required_cols = {"filepath", "label"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        df = df[df["split"] == self.split] if "split" in df.columns else df
        self.samples = list(zip(df["filepath"].tolist(), df["label"].tolist()))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]

        # Load image and convert to RGB (handles grayscale DICOM-derived PNGs)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> np.ndarray:
        """Compute inverse-frequency class weights for WeightedRandomSampler."""
        labels = [s[1] for s in self.samples]
        class_counts = np.bincount(labels, minlength=len(self.class_names))
        class_weights = 1.0 / (class_counts + 1e-6)
        return class_weights

    def get_sample_weights(self) -> np.ndarray:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        labels = [s[1] for s in self.samples]
        return class_weights[labels]


def create_stratified_splits(
    raw_dir: str,
    output_dir: str,
    class_names: list,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Create stratified train/val/test splits from a flat raw directory.
    Saves split CSVs to output_dir.

    Expected raw_dir structure:
        raw_dir/NORMAL/*.jpg
        raw_dir/PNEUMONIA/*.jpg

    Args:
        raw_dir:     Directory containing class subdirectories.
        output_dir:  Where to save split CSV files.
        class_names: List of class folder names.
        train_ratio: Fraction for training.
        val_ratio:   Fraction for validation.
        seed:        Random seed for reproducibility.
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_paths, all_labels = [], []
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    for class_name in class_names:
        class_dir = raw_path / class_name
        if not class_dir.exists():
            print(f"Warning: class directory not found: {class_dir}")
            continue
        label = class_to_idx[class_name]
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in class_dir.glob(ext):
                all_paths.append(str(p))
                all_labels.append(label)

    if len(all_paths) == 0:
        raise RuntimeError(f"No images found in {raw_dir}")

    # Stratified split: train vs (val + test)
    test_ratio = 1.0 - train_ratio - val_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_paths, all_labels,
        test_size=(val_ratio + test_ratio),
        stratify=all_labels,
        random_state=seed,
    )

    # Split temp into val and test
    val_frac = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1.0 - val_frac),
        stratify=y_temp,
        random_state=seed,
    )

    for split_name, paths, labels in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        df = pd.DataFrame({"filepath": paths, "label": labels, "split": split_name})
        csv_path = out_path / f"{split_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {split_name} split: {len(df)} samples → {csv_path}")


def get_dataloaders(
    config: dict,
    train_transform=None,
    val_transform=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function to create train, val, and test DataLoaders.

    Args:
        config:          Config dict (from config.yaml).
        train_transform: Transform for training set.
        val_transform:   Transform for val/test sets.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_cfg = config["data"]
    root = data_cfg["root"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    class_names = data_cfg["class_names"]

    # Check if CSV splits exist, otherwise use folder structure
    train_csv = os.path.join(root, "train.csv")
    val_csv = os.path.join(root, "val.csv")
    test_csv = os.path.join(root, "test.csv")

    use_csv = os.path.exists(train_csv)

    train_dataset = ChestXrayDataset(
        root_dir=root,
        split="train",
        transform=train_transform,
        csv_file=train_csv if use_csv else None,
        class_names=class_names,
    )

    val_dataset = ChestXrayDataset(
        root_dir=root,
        split="val",
        transform=val_transform,
        csv_file=val_csv if use_csv else None,
        class_names=class_names,
    )

    test_dataset = ChestXrayDataset(
        root_dir=root,
        split="test",
        transform=val_transform,
        csv_file=test_csv if use_csv else None,
        class_names=class_names,
    )

    # WeightedRandomSampler for class imbalance
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    print(f"Dataset sizes — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Class weights: {train_dataset.get_class_weights()}")

    return train_loader, val_loader, test_loader
