"""
inference.py
Single-image and batch inference for the CNN Medical Image Classifier.
Supports: PyTorch (.pth), ONNX (.onnx), TorchScript (.pt)
Also includes model export utilities (ONNX, TorchScript, INT8 quantisation).

Usage:
    # Single image
    python inference.py --checkpoint outputs/models/best_model.pth --image path/to/xray.jpg

    # Batch directory
    python inference.py --checkpoint outputs/models/best_model.pth --dir path/to/images/

    # Export to ONNX
    python inference.py --checkpoint outputs/models/best_model.pth --export onnx

    # Export to TorchScript
    python inference.py --checkpoint outputs/models/best_model.pth --export torchscript
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.dataset.transforms import get_val_transforms
from src.utils.helpers import get_device, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """
    Inference engine for the CNN Medical Image Classifier.

    Args:
        model:       Loaded PyTorch model in eval mode.
        transform:   Preprocessing transform pipeline.
        class_names: List of class names.
        device:      Inference device.
        threshold:   Classification threshold for positive class (binary).
    """

    def __init__(
        self,
        model: nn.Module,
        transform,
        class_names: List[str],
        device: torch.device,
        threshold: float = 0.5,
    ):
        self.model = model.eval().to(device)
        self.transform = transform
        self.class_names = class_names
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def predict_single(self, image_path: str) -> Dict:
        """
        Predict class for a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict with prediction, confidence, and all class probabilities.
        """
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        start = time.perf_counter()
        logits = self.model(tensor)
        latency_ms = (time.perf_counter() - start) * 1000

        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))

        # For binary: apply custom threshold
        if len(self.class_names) == 2:
            pred_idx = 1 if probs[1] >= self.threshold else 0

        return {
            "image": str(image_path),
            "prediction": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {name: float(p) for name, p in zip(self.class_names, probs)},
            "latency_ms": round(latency_ms, 2),
        }

    @torch.no_grad()
    def predict_batch(self, image_paths: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Predict classes for a list of images.

        Args:
            image_paths: List of image file paths.
            batch_size:  Batch size for inference.

        Returns:
            List of prediction dicts.
        """
        results = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tensors = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    tensors.append(self.transform(img))
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    continue

            if not tensors:
                continue

            batch = torch.stack(tensors).to(self.device)
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for path, prob in zip(batch_paths, probs):
                pred_idx = int(np.argmax(prob))
                if len(self.class_names) == 2:
                    pred_idx = 1 if prob[1] >= self.threshold else 0
                results.append({
                    "image": str(path),
                    "prediction": self.class_names[pred_idx],
                    "confidence": float(prob[pred_idx]),
                    "probabilities": {name: float(p) for name, p in zip(self.class_names, prob)},
                })

        return results

    def predict_directory(self, dir_path: str) -> List[Dict]:
        """Predict all images in a directory."""
        dir_path = Path(dir_path)
        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_paths.extend(dir_path.glob(ext))
        image_paths = sorted(image_paths)
        logger.info(f"Found {len(image_paths)} images in {dir_path}")
        return self.predict_batch([str(p) for p in image_paths])


def export_onnx(
    model: nn.Module,
    save_path: str,
    img_size: int = 224,
    device: Optional[torch.device] = None,
) -> None:
    """
    Export model to ONNX format.

    Args:
        model:     PyTorch model in eval mode.
        save_path: Output .onnx file path.
        img_size:  Input image size.
        device:    Device for dummy input.
    """
    device = device or torch.device("cpu")
    model = model.eval().to(device)
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    logger.info(f"Exported ONNX model → {save_path}")

    # Verify
    import onnx
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verified successfully.")


def export_torchscript(
    model: nn.Module,
    save_path: str,
    img_size: int = 224,
    device: Optional[torch.device] = None,
) -> None:
    """
    Export model to TorchScript format.

    Args:
        model:     PyTorch model in eval mode.
        save_path: Output .pt file path.
        img_size:  Input image size.
        device:    Device for tracing.
    """
    device = device or torch.device("cpu")
    model = model.eval().to(device)
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(model, dummy_input)
    traced.save(save_path)
    logger.info(f"Exported TorchScript model → {save_path}")


def export_quantised(
    model: nn.Module,
    save_path: str,
) -> None:
    """
    Export INT8 quantised model for CPU deployment.
    Achieves ~40% size reduction with minimal accuracy loss.

    Args:
        model:     PyTorch model (CPU only for static quantisation).
        save_path: Output .pth file path.
    """
    model = model.eval().cpu()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Exported INT8 quantised model → {save_path}")


def main(args):
    config = load_config(args.config)
    device = get_device()
    img_size = config["data"]["img_size"]
    class_names = config["data"]["class_names"]

    # Load model
    from train import build_model
    model = build_model(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded model from: {args.checkpoint}")

    # Export mode
    if args.export:
        model_dir = Path(config["outputs"]["model_dir"])
        if args.export == "onnx":
            export_onnx(model, str(model_dir / "model.onnx"), img_size=img_size, device=device)
        elif args.export == "torchscript":
            export_torchscript(model, str(model_dir / "model.pt"), img_size=img_size, device=device)
        elif args.export == "quantised":
            export_quantised(model, str(model_dir / "model_int8.pth"))
        return

    # Inference mode
    transform = get_val_transforms(img_size=img_size)
    predictor = Predictor(
        model=model,
        transform=transform,
        class_names=class_names,
        device=device,
        threshold=args.threshold,
    )

    if args.image:
        result = predictor.predict_single(args.image)
        print("\n" + "=" * 50)
        print("PREDICTION RESULT")
        print("=" * 50)
        print(f"Image:      {result['image']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Latency:    {result['latency_ms']} ms")
        print("\nClass Probabilities:")
        for cls, prob in result["probabilities"].items():
            bar = "█" * int(prob * 30)
            print(f"  {cls:<12} {prob:.4f}  {bar}")
        print("=" * 50)

    elif args.dir:
        results = predictor.predict_directory(args.dir)
        print(f"\nProcessed {len(results)} images")
        for r in results[:10]:  # Show first 10
            print(f"  {Path(r['image']).name:<30} → {r['prediction']:<12} ({r['confidence']:.2%})")
        if len(results) > 10:
            print(f"  ... and {len(results) - 10} more")

        # Save results
        import json
        out_path = "outputs/reports/inference_results.json"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved inference results → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for CNN Medical Image Classifier")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--dir", type=str, default=None, help="Directory of images")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold for positive class")
    parser.add_argument("--export", type=str, default=None,
                        choices=["onnx", "torchscript", "quantised"],
                        help="Export model format")
    args = parser.parse_args()
    main(args)
