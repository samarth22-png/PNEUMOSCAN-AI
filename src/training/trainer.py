"""
trainer.py
Core training engine for the CNN Medical Image Classifier.
Features:
  - Mixed precision training (torch.cuda.amp)
  - Gradient accumulation
  - Early stopping
  - Checkpoint save/load
  - TensorBoard logging
  - Cosine annealing LR scheduler
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from tqdm import tqdm

from src.training.metrics import compute_metrics, format_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.

    Args:
        patience:  Number of epochs to wait before stopping.
        min_delta: Minimum improvement to count as progress.
        mode:      'min' for loss, 'max' for accuracy/AUC.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement.")

        return self.should_stop


class Trainer:
    """
    Full training lifecycle manager.

    Args:
        model:          PyTorch model.
        optimizer:      Optimizer instance.
        criterion:      Loss function.
        device:         torch.device.
        config:         Full config dict.
        scheduler:      Optional LR scheduler.
        writer:         Optional TensorBoard SummaryWriter.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: dict,
        scheduler: Optional[_LRScheduler] = None,
        writer: Optional[SummaryWriter] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.writer = writer

        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 30)
        self.amp = train_cfg.get("amp", True) and device.type == "cuda"
        self.grad_accum_steps = train_cfg.get("grad_accumulation_steps", 1)
        self.grad_clip = train_cfg.get("gradient_clip", 1.0)
        self.patience = train_cfg.get("early_stopping_patience", 7)

        self.scaler = torch.amp.GradScaler("cuda") if self.amp else None
        self.early_stopping = EarlyStopping(patience=self.patience, mode="min")

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_f1": [], "val_f1": [],
            "val_auc": [], "lr": [],
        }

        output_cfg = config.get("outputs", {})
        self.model_dir = Path(output_cfg.get("model_dir", "outputs/models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
        self.optimizer.zero_grad()

        for step, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward pass with optional mixed precision
            if self.amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels) / self.grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels) / self.grad_accum_steps
                loss.backward()

            # Gradient accumulation
            if (step + 1) % self.grad_accum_steps == 0:
                if self.amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps

            # Collect predictions
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item() * self.grad_accum_steps:.4f}"})

        avg_loss = total_loss / len(loader)
        import numpy as np
        metrics = compute_metrics(all_preds, all_labels, np.array(all_probs),
                                  num_classes=self.config["data"]["num_classes"])
        metrics["loss"] = avg_loss
        return metrics

    @torch.no_grad()
    def validate_epoch(self, loader: DataLoader, epoch: int) -> Dict:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]  ", leave=False)

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(loader)
        import numpy as np
        metrics = compute_metrics(all_preds, all_labels, np.array(all_probs),
                                  num_classes=self.config["data"]["num_classes"])
        metrics["loss"] = avg_loss
        return metrics

    def save_checkpoint(self, epoch: int, val_metrics: Dict, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_metrics.get("loss", 0),
            "val_acc": val_metrics.get("accuracy", 0),
            "val_auc": val_metrics.get("roc_auc", 0),
            "config": self.config,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Always save latest
        latest_path = self.model_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)

        # Save best model separately
        if is_best:
            best_path = self.model_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model → {best_path} (val_acc={val_metrics.get('accuracy', 0):.4f})")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and return the epoch to resume from."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        # Only restore optimizer if parameter count matches (skip on backbone unfreeze)
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (ValueError, RuntimeError):
            print("Optimizer state mismatch (backbone unfrozen) — starting optimizer fresh.")
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception:
                pass
        epoch = checkpoint.get("epoch", 0)
        logger.info(f"Resumed from checkpoint: {checkpoint_path} (epoch {epoch})")
        return epoch

    def _log_to_tensorboard(self, train_m: Dict, val_m: Dict, epoch: int) -> None:
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return
        for key in ["loss", "accuracy", "f1", "roc_auc"]:
            if key in train_m:
                self.writer.add_scalar(f"Train/{key}", train_m[key], epoch)
            if key in val_m:
                self.writer.add_scalar(f"Val/{key}", val_m[key], epoch)
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("LR", current_lr, epoch)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: Optional[str] = None,
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader:   Validation DataLoader.
            resume_from:  Optional checkpoint path to resume from.

        Returns:
            Training history dict.
        """
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1

        logger.info(f"Starting training: {self.epochs} epochs, device={self.device}, AMP={self.amp}")

        for epoch in range(start_epoch, self.epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch + 1)

            # Validate
            val_metrics = self.validate_epoch(val_loader, epoch + 1)

            # LR scheduler step
            if self.scheduler:
                self.scheduler.step()

            # Log
            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - epoch_start

            logger.info(
                f"Epoch [{epoch+1:03d}/{self.epochs}] "
                f"({elapsed:.1f}s) | "
                f"Train — {format_metrics(train_metrics, 'Train')} | "
                f"Val — {format_metrics(val_metrics, 'Val')} | "
                f"LR: {current_lr:.2e}"
            )

            # TensorBoard
            self._log_to_tensorboard(train_metrics, val_metrics, epoch + 1)

            # Update history
            self.history["train_loss"].append(train_metrics.get("loss", 0))
            self.history["val_loss"].append(val_metrics.get("loss", 0))
            self.history["train_acc"].append(train_metrics.get("accuracy", 0))
            self.history["val_acc"].append(val_metrics.get("accuracy", 0))
            self.history["train_f1"].append(train_metrics.get("f1", 0))
            self.history["val_f1"].append(val_metrics.get("f1", 0))
            self.history["val_auc"].append(val_metrics.get("roc_auc", 0))
            self.history["lr"].append(current_lr)

            # Save checkpoint
            is_best = val_metrics.get("accuracy", 0) > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics.get("accuracy", 0)
                self.best_val_loss = val_metrics.get("loss", float("inf"))
            self.save_checkpoint(epoch + 1, val_metrics, is_best=is_best)

            # Early stopping
            if self.early_stopping(val_metrics.get("loss", float("inf"))):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Save training history
        history_path = self.model_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training complete. Best val_acc: {self.best_val_acc:.4f}")

        if self.writer:
            self.writer.close()

        return self.history
