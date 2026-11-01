"""
train.py
Main training script for the CNN Medical Image Classifier.

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --model cnn
    python train.py --config configs/config.yaml --model resnet50 --resume outputs/models/latest_checkpoint.pth
"""

import argparse
import sys
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from src.dataset.dataloader import get_dataloaders
from src.dataset.transforms import get_train_transforms, get_val_transforms
from src.models.cnn_model import build_cnn
from src.models.efficientnet_model import build_efficientnet
from src.models.resnet_transfer import build_resnet
from src.training.loss import build_loss
from src.training.trainer import Trainer
from src.utils.helpers import (
    build_optimizer,
    build_scheduler,
    compute_class_weights,
    get_device,
    load_config,
    seed_everything,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file="outputs/logs/train.log")


def build_model(config: dict, device):
    """Build model based on config."""
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "efficientnet_b4")
    num_classes = config["data"]["num_classes"]
    dropout = model_cfg.get("dropout", 0.4)
    freeze = model_cfg.get("freeze_backbone", True)

    if model_name == "cnn":
        model = build_cnn(num_classes=num_classes, dropout=dropout)
    elif model_name == "resnet50":
        model = build_resnet(num_classes=num_classes, freeze_backbone=freeze, dropout=dropout)
    elif model_name == "efficientnet_b4":
        model = build_efficientnet(num_classes=num_classes, freeze_backbone=freeze, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: cnn, resnet50, efficientnet_b4")

    return model.to(device)


def main(args):
    # Load config
    config = load_config(args.config)

    # Override model from CLI if provided
    if args.model:
        config["model"]["name"] = args.model

    # Reproducibility
    seed_everything(config.get("seed", 42))

    # Device
    device = get_device()

    # Transforms
    img_size = config["data"]["img_size"]
    train_transform = get_train_transforms(img_size=img_size)
    val_transform = get_val_transforms(img_size=img_size)

    # DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        config, train_transform=train_transform, val_transform=val_transform
    )

    # Model
    model = build_model(config, device)

    # Class weights for imbalanced data
    all_labels = [s[1] for s in train_loader.dataset.samples]
    class_weights = compute_class_weights(
        all_labels, config["data"]["num_classes"], device=device
    )

    # Loss
    criterion = build_loss(config, class_weights=class_weights)

    # Optimizer
    optimizer = build_optimizer(model, config)

    # Scheduler
    scheduler = build_scheduler(optimizer, config)

    # TensorBoard
    model_name = config["model"]["name"]
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=f"runs/{model_name}")
        logger.info(f"TensorBoard: tensorboard --logdir runs/{model_name}")
    else:
        writer = None
        logger.info("TensorBoard not available — skipping logging")

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        scheduler=scheduler,
        writer=writer,
    )

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from=args.resume,
    )

    logger.info("Training complete.")
    logger.info(f"Best val accuracy: {max(history['val_acc']):.4f}")
    if history.get("val_auc"):
        logger.info(f"Best val AUC: {max(history['val_auc']):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN Medical Image Classifier")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["cnn", "resnet50", "efficientnet_b4"],
        help="Override model from config"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )
    args = parser.parse_args()
    main(args)
