"""Inference and Anomaly Localization Pipeline Module.

This module provides high-performance single-image and batch inference for DeepInspect AI:
- Safe model weight deserialization with automatic fallback to CPU (`map_location=torch.device('cpu')`).
- Pixel residual map calculation (reconstruction discrepancy between original and autoencoder output).
- Spatial Gaussian smoothing and Jet colormap heatmap generation.
- Visual defect overlay blending (original inspection image + colored heatmap).
- Binary defect classification against calibrated thresholds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import yaml

from src.models.autoencoder import ConvAutoencoder, build_autoencoder, load_compatible_state_dict


def load_model(
    checkpoint_path: Union[str, Path] = "./models/best_model.pth",
    config_or_path: Optional[Union[str, Path, Dict[str, Any]]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[ConvAutoencoder, Dict[str, Any], torch.device]:
    """Safely load trained ConvAutoencoder weights with automatic CPU fallback.

    Args:
        checkpoint_path: Path to .pth model weights file.
        config_or_path: Optional configuration or path to config.yaml.
        device: Target device string or object.

    Returns:
        Tuple of (model, config, resolved_device).
    """
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Resolve target compute device with safe CPU fallback
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # Load checkpoint safely across hardware boundaries
    try:
        checkpoint = torch.load(ckpt_path, map_location=target_device)
    except Exception:
        target_device = torch.device("cpu")
        checkpoint = torch.load(ckpt_path, map_location=target_device)

    # Extract configuration
    if config_or_path is not None:
        if isinstance(config_or_path, (str, Path)):
            with open(Path(config_or_path).expanduser(), "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        else:
            config = config_or_path
    elif isinstance(checkpoint, dict) and "config" in checkpoint:
        config = checkpoint["config"]
    else:
        # Fallback to default project configuration
        default_cfg_path = Path("configs/config.yaml")
        if default_cfg_path.exists():
            with open(default_cfg_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        else:
            config = {}

    model = build_autoencoder(config)

    # Load state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        load_compatible_state_dict(model, checkpoint["model_state_dict"])
    else:
        load_compatible_state_dict(model, checkpoint)

    model = model.to(target_device).eval()
    return model, config, target_device


class AnomalyPredictor:
    """Inference engine for reconstruction anomaly detection and heatmap localization.

    Args:
        checkpoint_path: Path to trained PyTorch model weights (.pth).
        config_or_path: Optional configuration dictionary or path to config.yaml.
        threshold: Decision boundary threshold. If None, loaded from evaluation_results.json.
        device: Optional target compute device.
        gaussian_blur_ksize: Kernel size for heatmap smoothing. Default: 7.
        gaussian_blur_sigma: Gaussian filter sigma standard deviation. Default: 4.0.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "./models/best_model.pth",
        config_or_path: Optional[Union[str, Path, Dict[str, Any]]] = None,
        threshold: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
        gaussian_blur_ksize: int = 7,
        gaussian_blur_sigma: float = 4.0,
    ) -> None:
        self.model, self.config, self.device = load_model(
            checkpoint_path=checkpoint_path,
            config_or_path=config_or_path,
            device=device,
        )

        ds_cfg = self.config.get("dataset", {})
        self.image_size: Tuple[int, int] = tuple(ds_cfg.get("image_size", [256, 256]))
        self.gaussian_blur_ksize = (
            gaussian_blur_ksize if gaussian_blur_ksize % 2 == 1 else gaussian_blur_ksize + 1
        )
        self.gaussian_blur_sigma = float(gaussian_blur_sigma)

        # Resolve decision threshold
        if threshold is not None:
            self.threshold = float(threshold)
        else:
            self.threshold = self._load_calibrated_threshold()

        self.transform = T.Compose(
            [
                T.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
                T.ToTensor(),
            ]
        )

    def _load_calibrated_threshold(self) -> float:
        """Attempt to read calibrated optimal threshold from evaluation_results.json."""
        eval_path = Path("outputs/evaluation_results.json")
        if eval_path.exists():
            try:
                import json

                with open(eval_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return float(data.get("optimal_threshold", 0.015))
            except Exception:
                pass
        return 0.015  # Fallback default threshold for normalized MSE

    def preprocess_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Convert input image into a normalized PyTorch tensor and uint8 RGB numpy array.

        Args:
            image_input: File path, PIL Image, or numpy array.

        Returns:
            Tuple of (tensor_image of shape (1, 3, H, W), numpy_image of shape (H, W, 3) in [0, 255]).
        """
        if isinstance(image_input, (str, Path)):
            with Image.open(Path(image_input)) as img:
                pil_img = img.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Resize original image to target resolution
        pil_resized = pil_img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        np_img = np.array(pil_resized, dtype=np.uint8)

        tensor_img = self.transform(pil_resized).unsqueeze(0).to(self.device)
        return tensor_img, np_img

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        alpha: float = 0.5,
    ) -> Dict[str, Any]:
        """Run inspection inference on a single image.

        Args:
            image_input: Input image file path, PIL Image, or numpy array.
            alpha: Heatmap blend weight in overlay (0.0 = original image, 1.0 = heatmap).

        Returns:
            Inspection result dictionary containing:
                - is_anomaly (bool): True if anomaly_score >= threshold.
                - anomaly_score (float): Maximum pixel anomaly intensity.
                - threshold (float): Applied decision threshold.
                - original_image (np.ndarray): Input RGB image (H, W, 3).
                - reconstructed_image (np.ndarray): Model reconstruction RGB image (H, W, 3).
                - raw_residual (np.ndarray): Unprocessed 2D MSE residual map (H, W).
                - anomaly_heatmap (np.ndarray): Smoothed and normalized colormap heatmap (H, W, 3) RGB.
                - overlay (np.ndarray): Alpha-blended inspection overlay (H, W, 3) RGB.
        """
        tensor_img, orig_np = self.preprocess_image(image_input)

        with torch.no_grad():
            reconstruction_tensor = self.model(tensor_img)

            # Compute pixel-wise squared reconstruction difference
            diff_squared = (tensor_img - reconstruction_tensor) ** 2
            # Mean error across the 3 RGB color channels
            residual = torch.mean(diff_squared, dim=1, keepdim=True).squeeze(0).squeeze(0)
            residual_np = residual.cpu().numpy()

            # Reconstructed image array
            recon_np = (
                reconstruction_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
            )
            recon_uint8 = (recon_np * 255).astype(np.uint8)

        # 1. Apply Gaussian spatial smoothing to eliminate single-pixel sensor noise
        smoothed_residual = cv2.GaussianBlur(
            residual_np,
            (self.gaussian_blur_ksize, self.gaussian_blur_ksize),
            sigmaX=self.gaussian_blur_sigma,
            sigmaY=self.gaussian_blur_sigma,
        )

        # 2. Compute scalar anomaly score (maximum intensity in smoothed residual map)
        anomaly_score = float(np.max(smoothed_residual))
        is_anomaly = anomaly_score >= self.threshold

        # 3. Normalize smoothed residual map to [0, 255] for visual rendering
        res_min = smoothed_residual.min()
        res_max = smoothed_residual.max()
        if res_max > res_min:
            norm_map = ((smoothed_residual - res_min) / (res_max - res_min) * 255.0).astype(np.uint8)
        else:
            norm_map = np.zeros_like(smoothed_residual, dtype=np.uint8)

        # 4. Generate colored Jet heatmap
        heatmap_bgr = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # 5. Create alpha-blended overlay with original image
        overlay = cv2.addWeighted(orig_np, 1.0 - alpha, heatmap_rgb, alpha, 0)

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 6),
            "threshold": round(self.threshold, 6),
            "original_image": orig_np,
            "reconstructed_image": recon_uint8,
            "raw_residual": residual_np,
            "anomaly_heatmap": heatmap_rgb,
            "overlay": overlay,
        }


def predict_image(
    image_path: Union[str, Path],
    checkpoint_path: Union[str, Path] = "./models/best_model.pth",
    threshold: Optional[float] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Functional convenience helper to predict and optionally save overlay."""
    predictor = AnomalyPredictor(checkpoint_path=checkpoint_path, threshold=threshold)
    result = predictor.predict(image_path)

    if output_path is not None:
        out_file = Path(output_path).expanduser()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # Save overlay image
        Image.fromarray(result["overlay"]).save(out_file)

    return result


def main() -> None:
    """CLI Entrypoint for running single-image inspection."""
    parser = argparse.ArgumentParser(description="DeepInspect AI - Image Anomaly Inspection")
    parser.add_argument("--image", type=str, required=True, help="Path to inspection image")
    parser.add_argument("--checkpoint", type=str, default="./models/best_model.pth", help="Model weights path")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold")
    parser.add_argument("--output", type=str, default="./outputs/prediction_overlay.png", help="Overlay output file")

    args = parser.parse_args()

    result = predict_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        output_path=args.output,
    )

    verdict = "ANOMALY (DEFECT DETECTED)" if result["is_anomaly"] else "NORMAL (PASS)"
    print("\n" + "=" * 55)
    print(" Inspection Results")
    print("=" * 55)
    print(f" Verdict       : {verdict}")
    print(f" Anomaly Score : {result['anomaly_score']:.6f}")
    print(f" Threshold     : {result['threshold']:.6f}")
    print(f" Overlay Saved : {args.output}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
