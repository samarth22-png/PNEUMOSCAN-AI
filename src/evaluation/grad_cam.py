"""
grad_cam.py
Gradient-weighted Class Activation Mapping (Grad-CAM).
Selvaraju et al., 2017 — https://arxiv.org/abs/1610.02391

Generates spatial heatmaps showing which regions of an X-ray
drove the model's prediction. Critical for clinical trust and
explainability in medical AI.

Formula:
    α^c_k = GAP(∂y^c / ∂A^k)   (importance weight for channel k)
    L^c   = ReLU(Σ_k α^c_k · A^k)  (class activation map)
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM implementation using forward and backward hooks.

    Args:
        model:        PyTorch model (eval mode recommended).
        target_layer: The convolutional layer to hook (last conv layer).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output) -> None:
        """Forward hook: save feature map activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        """Backward hook: save gradients of the target layer."""
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            input_tensor: Preprocessed image tensor, shape (1, C, H, W).
            class_idx:    Target class index. If None, uses predicted class.

        Returns:
            Normalised heatmap as numpy array, shape (H, W), values in [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        # Compute importance weights: global average pooling of gradients
        # Shape: (1, C, H, W) → (1, C, 1, 1)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted sum of activation maps
        # Shape: (1, C, H, W) → (H, W)
        cam = (weights * self.activations).sum(dim=1).squeeze()

        # ReLU: only keep positive contributions
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()

    def overlay_heatmap(
        self,
        original_image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original image.

        Args:
            original_image: Original image as numpy array (H, W, 3), uint8.
            cam:            Grad-CAM heatmap, shape (H, W), values in [0, 1].
            alpha:          Heatmap transparency (0 = invisible, 1 = opaque).
            colormap:       OpenCV colormap for heatmap.

        Returns:
            Blended image as numpy array (H, W, 3), uint8.
        """
        # Resize CAM to match image dimensions
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # Apply colormap
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), colormap
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend with original image
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)

        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        return overlay

    def visualise(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        class_names: list,
        class_idx: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generate and display Grad-CAM visualisation.

        Args:
            input_tensor:   Preprocessed tensor (1, C, H, W).
            original_image: Original image array (H, W, 3).
            class_names:    List of class names.
            class_idx:      Target class (None = predicted class).
            save_path:      Optional path to save the figure.

        Returns:
            matplotlib Figure.
        """
        cam = self.generate_cam(input_tensor, class_idx)
        overlay = self.overlay_heatmap(original_image, cam)

        # Get prediction info
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_class = output.argmax(dim=1).item()
            pred_prob = probs[pred_class].item()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor("#070b14")

        for ax in axes:
            ax.set_facecolor("#0d1424")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e2d4a")

        axes[0].imshow(original_image, cmap="gray" if original_image.ndim == 2 else None)
        axes[0].set_title("Original X-ray", color="white", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", color="white", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(
            f"Overlay — Pred: {class_names[pred_class]} ({pred_prob:.2%})",
            color="#00ffa3", fontsize=12,
        )
        axes[2].axis("off")

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"Saved Grad-CAM → {save_path}")

        return fig

    def batch_visualise(
        self,
        images: list,
        tensors: list,
        class_names: list,
        save_dir: str,
        max_images: int = 20,
    ) -> None:
        """
        Generate Grad-CAM gallery for multiple images.

        Args:
            images:      List of original numpy images.
            tensors:     List of preprocessed tensors.
            class_names: List of class names.
            save_dir:    Directory to save individual heatmaps.
            max_images:  Maximum number of images to process.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for i, (img, tensor) in enumerate(zip(images[:max_images], tensors[:max_images])):
            fig = self.visualise(
                tensor, img, class_names,
                save_path=str(save_path / f"gradcam_{i:03d}.png"),
            )
            plt.close(fig)

        print(f"Saved {min(len(images), max_images)} Grad-CAM visualisations → {save_dir}")

    def remove_hooks(self) -> None:
        """Remove registered hooks to free memory."""
        self._forward_hook.remove()
        self._backward_hook.remove()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ (Chattopadhay et al., 2018).
    Improved version with better localisation for multiple object instances.
    Uses second-order gradients for more accurate importance weights.
    """

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap."""
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        # Grad-CAM++ weights use second-order gradient approximation
        grads = self.gradients  # (1, C, H, W)
        acts = self.activations  # (1, C, H, W)

        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3

        # Denominator for alpha computation
        sum_acts = acts.sum(dim=[2, 3], keepdim=True)
        eps = 1e-8
        alpha_denom = 2 * grads_power_2 + sum_acts * grads_power_3 + eps
        alpha = grads_power_2 / alpha_denom

        # Weights: sum of alpha * ReLU(gradients)
        weights = (alpha * F.relu(grads)).sum(dim=[2, 3], keepdim=True)

        cam = (weights * acts).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() + eps)

        return cam.cpu().numpy()
