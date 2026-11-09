"""Evaluation and Anomaly Scoring Metrics Module.

This module provides an OOP-based evaluation engine for DeepInspect AI, computing:
- Image-level AUROC (Area Under the Receiver Operating Characteristic).
- Pixel-level AUROC for spatial defect segmentation.
- Optimal threshold calibration (F1-score maximization or normal validation percentile).
- Precision, Recall, F1-Score, and Confusion Matrix.
- ROC curve generation and metrics serialization to JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.data.dataset import get_dataloaders
from src.models.autoencoder import ConvAutoencoder, build_autoencoder, load_compatible_state_dict


class AnomalyEvaluator:
    """Evaluates reconstruction-based autoencoders for image- and pixel-level anomaly detection.

    Args:
        model: Trained ConvAutoencoder instance.
        test_loader: DataLoader containing test images (normal + defects).
        val_loader: Optional DataLoader with normal validation images for threshold calibration.
        device: Target compute device ('cuda' or 'cpu').
        gaussian_blur_ksize: Kernel size for smoothing pixel residual anomaly maps.
        gaussian_blur_sigma: Gaussian filter sigma standard deviation.
    """

    def __init__(
        self,
        model: ConvAutoencoder,
        test_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[Union[str, torch.device]] = None,
        gaussian_blur_ksize: int = 7,
        gaussian_blur_sigma: float = 4.0,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device).eval()
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.gaussian_blur_ksize = (
            gaussian_blur_ksize if gaussian_blur_ksize % 2 == 1 else gaussian_blur_ksize + 1
        )
        self.gaussian_blur_sigma = float(gaussian_blur_sigma)

    def _smooth_anomaly_map(self, residual_map: np.ndarray) -> np.ndarray:
        """Apply Gaussian spatial blur to smooth reconstruction noise.

        Args:
            residual_map: 2D numpy array (H, W).

        Returns:
            Smoothed 2D numpy array (H, W).
        """
        smoothed = cv2.GaussianBlur(
            residual_map,
            (self.gaussian_blur_ksize, self.gaussian_blur_ksize),
            sigmaX=self.gaussian_blur_sigma,
            sigmaY=self.gaussian_blur_sigma,
        )
        return smoothed

    def compute_predictions(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute pixel residual maps, anomaly scores, and collect ground truth masks.

        Returns:
            Tuple of:
                - true_image_labels: 1D array of shape (N,) {0: normal, 1: anomaly}.
                - pred_image_scores: 1D array of shape (N,) representing max anomaly residual.
                - true_pixel_masks: 1D flattened array of all ground truth pixels.
                - pred_pixel_maps: 1D flattened array of all predicted anomaly map pixels.
        """
        true_labels: List[int] = []
        pred_scores: List[float] = []

        flat_true_masks: List[np.ndarray] = []
        flat_pred_maps: List[np.ndarray] = []

        pbar = tqdm(self.test_loader, desc="Evaluating Test Set", leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for images, labels, masks, _ in pbar:
                images = images.to(self.device)
                reconstructed = self.model(images)

                # Channel-wise MSE residual: (B, 1, H, W)
                residuals = torch.mean((images - reconstructed) ** 2, dim=1, keepdim=True)
                residuals_np = residuals.cpu().numpy()
                masks_np = masks.cpu().numpy()
                labels_np = labels.cpu().numpy()

                batch_size = images.size(0)
                for b in range(batch_size):
                    res_map = residuals_np[b, 0]  # (H, W)
                    smoothed_map = self._smooth_anomaly_map(res_map)
                    gt_mask = masks_np[b, 0]  # (H, W)

                    # Image-level score: maximum intensity in smoothed residual map
                    image_score = float(np.max(smoothed_map))

                    true_labels.append(int(labels_np[b]))
                    pred_scores.append(image_score)

                    flat_true_masks.append(gt_mask.flatten())
                    flat_pred_maps.append(smoothed_map.flatten())

        y_true_img = np.array(true_labels, dtype=np.int32)
        y_score_img = np.array(pred_scores, dtype=np.float32)

        y_true_pix = (
            np.concatenate(flat_true_masks).astype(np.int32)
            if flat_true_masks
            else np.array([], dtype=np.int32)
        )
        y_score_pix = (
            np.concatenate(flat_pred_maps).astype(np.float32)
            if flat_pred_maps
            else np.array([], dtype=np.float32)
        )

        return y_true_img, y_score_img, y_true_pix, y_score_pix

    def calibrate_threshold(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        method: str = "f1_optimal",
        percentile: float = 98.0,
    ) -> float:
        """Calibrate decision boundary threshold between normal and anomalous images.

        Args:
            y_true: Binary ground truth labels (0 = normal, 1 = defect).
            y_score: Continuous anomaly prediction scores.
            method: 'f1_optimal' to maximize F1-score, or 'percentile' based on validation/normal data.
            percentile: Percentile value if method is 'percentile'.

        Returns:
            Calculated optimal scalar threshold.
        """
        if method == "percentile" and self.val_loader is not None and len(self.val_loader.dataset) > 0:
            val_scores: List[float] = []
            with torch.no_grad():
                for images, _, _, _ in self.val_loader:
                    images = images.to(self.device)
                    reconstruction = self.model(images)
                    residuals = torch.mean((images - reconstruction) ** 2, dim=1, keepdim=True).cpu().numpy()
                    for b in range(images.size(0)):
                        smoothed = self._smooth_anomaly_map(residuals[b, 0])
                        val_scores.append(float(np.max(smoothed)))
            threshold = float(np.percentile(val_scores, percentile))
            return threshold

        # Default: F1-score optimization over precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
        # Avoid division by zero
        f1_scores = np.divide(
            2 * (precisions * recalls),
            (precisions + recalls),
            out=np.zeros_like(precisions),
            where=(precisions + recalls) > 0,
        )
        best_idx = int(np.argmax(f1_scores))
        if best_idx < len(thresholds):
            optimal_thresh = float(thresholds[best_idx])
        else:
            optimal_thresh = float(np.median(y_score))

        return optimal_thresh

    def evaluate(self) -> Dict[str, Any]:
        """Compute full benchmark evaluation suite."""
        y_true_img, y_score_img, y_true_pix, y_score_pix = self.compute_predictions()

        # 1. Image-Level AUROC
        if len(np.unique(y_true_img)) > 1:
            image_auroc = float(roc_auc_score(y_true_img, y_score_img))
        else:
            image_auroc = 1.0

        # 2. Pixel-Level AUROC
        if len(y_true_pix) > 0 and len(np.unique(y_true_pix)) > 1:
            pixel_auroc = float(roc_auc_score(y_true_pix, y_score_pix))
        else:
            pixel_auroc = 1.0

        # 3. Threshold Calibration & Classification Metrics
        threshold = self.calibrate_threshold(y_true_img, y_score_img, method="f1_optimal")
        y_pred_img = (y_score_img >= threshold).astype(np.int32)

        f1 = float(f1_score(y_true_img, y_pred_img, zero_division=0))
        precision = float(precision_score(y_true_img, y_pred_img, zero_division=0))
        recall = float(recall_score(y_true_img, y_pred_img, zero_division=0))
        cm = confusion_matrix(y_true_img, y_pred_img).tolist()

        results: Dict[str, Any] = {
            "image_auroc": round(image_auroc, 4),
            "pixel_auroc": round(pixel_auroc, 4),
            "optimal_threshold": round(threshold, 6),
            "f1_score": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "confusion_matrix": cm,
            "total_test_samples": int(len(y_true_img)),
            "anomaly_samples": int(np.sum(y_true_img)),
            "normal_samples": int(len(y_true_img) - np.sum(y_true_img)),
            "raw_scores": {
                "y_true_img": y_true_img.tolist(),
                "y_score_img": y_score_img.tolist(),
            },
        }

        return results

    def plot_roc_curves(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path] = "./outputs/roc_curves.png",
    ) -> Path:
        """Generate and save publication-quality ROC and Precision-Recall curves.

        Args:
            results: Dictionary produced by evaluate().
            output_path: Target image file path.

        Returns:
            Saved Path object.
        """
        output_file = Path(output_path).expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        y_true = np.array(results["raw_scores"]["y_true_img"])
        y_score = np.array(results["raw_scores"]["y_score_img"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 1. ROC Curve
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc_val = auc(fpr, tpr)
            ax1.plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"ROC (AUC = {roc_auc_val:.3f})")
            ax1.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--")
        ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("False Positive Rate", fontsize=11)
        ax1.set_ylabel("True Positive Rate", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="lower right", fontsize=10)

        # 2. Precision-Recall Curve
        if len(np.unique(y_true)) > 1:
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            pr_auc_val = auc(rec, prec)
            ax2.plot(rec, prec, color="#27ae60", lw=2.5, label=f"PR (AUC = {pr_auc_val:.3f})")
        ax2.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Recall", fontsize=11)
        ax2.set_ylabel("Precision", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="lower left", fontsize=10)

        plt.suptitle("DeepInspect AI - Inspection Performance Curves", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        return output_file


def evaluate(
    config_path: Union[str, Path] = "configs/config.yaml",
    checkpoint_path: Union[str, Path] = "./models/best_model.pth",
    data_dir: Optional[str] = None,
    output_dir: Union[str, Path] = "./outputs",
) -> Dict[str, Any]:
    """Load checkpoint and run complete benchmark evaluation.

    Args:
        config_path: Path to YAML config.
        checkpoint_path: Path to trained model weights.
        data_dir: Optional dataset path override.
        output_dir: Path to save evaluation metrics JSON and plots.

    Returns:
        Evaluation metrics dictionary.
    """
    config_file = Path(config_path).expanduser()
    ckpt_file = Path(checkpoint_path).expanduser()
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if data_dir is not None:
        config["dataset"]["root_dir"] = data_dir

    # Build model and dataloaders
    _, val_loader, test_loader = get_dataloaders(config)
    model = build_autoencoder(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = AnomalyEvaluator(model=model, test_loader=test_loader, val_loader=val_loader, device=device)

    # Robust model loading with CPU fallback
    try:
        checkpoint = torch.load(ckpt_file, map_location=device)
    except Exception:
        checkpoint = torch.load(ckpt_file, map_location=torch.device("cpu"))

    if "model_state_dict" in checkpoint:
        load_compatible_state_dict(model, checkpoint["model_state_dict"])
    else:
        load_compatible_state_dict(model, checkpoint)
    results = evaluator.evaluate()

    # Save results JSON
    json_path = out_dir / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # Plot and save curves
    plot_path = evaluator.plot_roc_curves(results, output_path=out_dir / "roc_curves.png")

    print("\n" + "=" * 65)
    print(" DeepInspect AI - Evaluation Benchmark Results")
    print("=" * 65)
    print(f" Image-Level AUROC     : {results['image_auroc']:.4f}")
    print(f" Pixel-Level AUROC     : {results['pixel_auroc']:.4f}")
    print(f" Optimal F1 Threshold  : {results['optimal_threshold']:.6f}")
    print(f" Precision @ Threshold : {results['precision']:.4f}")
    print(f" Recall @ Threshold    : {results['recall']:.4f}")
    print(f" F1-Score @ Threshold  : {results['f1_score']:.4f}")
    print(f" Confusion Matrix      : {results['confusion_matrix']}")
    print(f" Report JSON Saved     : {json_path}")
    print(f" ROC Plot Saved        : {plot_path}")
    print("=" * 65 + "\n")

    return results


def main() -> None:
    """CLI entrypoint for evaluation script."""
    parser = argparse.ArgumentParser(description="DeepInspect AI - Model Evaluation & Metrics")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default="./models/best_model.pth", help="Path to checkpoint")
    parser.add_argument("--data_dir", type=str, default=None, help="Root path of MVTec AD category")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")

    args = parser.parse_args()

    evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
