"""Multi-Type Bottle Anomaly Detector — PatchCore-lite Implementation.

بدلاً من تدريب نموذج لكل نوع زجاجة، يستخدم هذا النهج:
1. شبكة EfficientNet-B0 مُدرَّبة مسبقاً على ImageNet لاستخراج الخصائص.
2. مكتبة مرجعية (Memory Bank) لصور الزجاجات السليمة من كل نوع.
3. مقارنة k-NN: إذا كانت خصائص الصورة الجديدة بعيدة عن المرجعية → معيبة.

لا يتطلب إعادة تدريب عند إضافة نوع زجاجة جديد!
تكفي 5-15 صورة سليمة لكل نوع زجاجة.

AUROC مرجعي لـ PatchCore على MVTec AD (Bottle): ~98.6%
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image


# ==============================================================================
# 1. Feature Extractor using pretrained EfficientNet-B0
# ==============================================================================
class BottleFeatureExtractor(nn.Module):
    """Extracts multi-scale patch features from EfficientNet-B0.

    Uses features from two intermediate layers (layer 3 & layer 4) with
    spatial average pooling to capture neighborhood context (PatchCore architecture).
    Deterministic: no random layers or projection matrices.

    Args:
        device: Computation device ('cpu' or 'cuda').
    """

    LAYER_INDICES = [3, 4]   # layer 3 (40 ch, 28x28), layer 4 (80 ch, 14x14)

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)

        # Load pretrained EfficientNet-B0 (ImageNet weights)
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.features = self.features.to(self.device).eval()

        # Freeze ALL pretrained weights
        for param in self.features.parameters():
            param.requires_grad = False

        # Neighborhood pooling for patch context (standard PatchCore practice)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale patch features with context pooling.

        Args:
            x: Input batch tensor (B, 3, H, W) normalized.

        Returns:
            L2-normalized feature map tensor (B, 120, 28, 28).
        """
        x = x.to(self.device)
        maps: List[torch.Tensor] = []
        feat = x
        for i, layer in enumerate(self.features):
            feat = layer(feat)
            if i in self.LAYER_INDICES:
                maps.append(self.pool(feat))

        # Target spatial size is layer 3's resolution (28x28 for 224x224 input)
        target_h, target_w = maps[0].shape[2], maps[0].shape[3]
        aligned = [
            maps[0],
            F.interpolate(maps[1], size=(target_h, target_w), mode="bilinear", align_corners=False),
        ]
        concat = torch.cat(aligned, dim=1)   # (B, 40 + 80 = 120, 28, 28)

        # L2-normalize along channel dim so cosine distance == L2 distance
        out = F.normalize(concat, p=2, dim=1)
        return out


# ==============================================================================
# 2. PatchCore-lite Memory Bank per Bottle Type
# ==============================================================================
class PatchCoreDetector:
    """Training-free anomaly detector based on patch feature k-NN retrieval.

    Usage:
        detector = PatchCoreDetector()

        # Enroll normal samples for a bottle type
        detector.enroll("glass_bottle", list_of_pil_images)
        detector.enroll("plastic_bottle", other_pil_images)

        # Detect anomaly in new image
        score, heatmap = detector.predict(pil_image, bottle_type="glass_bottle")

    Attributes:
        extractor: Pretrained feature extractor.
        memory_banks: Dict mapping bottle type name → stored feature vectors.
        input_size: Image resolution passed to the network.
    """

    # ImageNet normalization (EfficientNet was pretrained with these stats)
    NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __init__(
        self,
        device: str = "cpu",
        input_size: Tuple[int, int] = (224, 224),
        k_neighbors: int = 5,
        subsample_ratio: float = 0.25,
    ) -> None:
        self.device = device
        self.input_size = input_size
        self.k = k_neighbors
        self.subsample_ratio = subsample_ratio
        self.extractor = BottleFeatureExtractor(device=device)
        self.memory_banks: Dict[str, torch.Tensor] = {}   # type_name → (N_patches, 256)
        self.thresholds: Dict[str, float] = {}

        self.transform = T.Compose([
            T.Resize(input_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(input_size),
            T.ToTensor(),
            self.NORMALIZE,
        ])

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------
    def enroll(
        self,
        bottle_type: str,
        images: List[Image.Image],
        verbose: bool = True,
    ) -> None:
        """Build memory bank from normal (defect-free) reference images.

        Args:
            bottle_type: Unique name for this bottle type (e.g. 'beer_glass').
            images: List of PIL images of normal bottles.
            verbose: Print progress.
        """
        if verbose:
            print(f"[PatchCore] Enrolling '{bottle_type}' with {len(images)} images...")

        all_patches: List[torch.Tensor] = []

        for img in images:
            x = self.transform(img.convert("RGB")).unsqueeze(0)
            feat_map = self.extractor.extract(x)   # (1, 256, h, w)
            # Reshape to (h*w, 256) patch vectors
            B, C, H, W = feat_map.shape
            patches = feat_map.permute(0, 2, 3, 1).reshape(-1, C)   # (H*W, 256)
            all_patches.append(patches.cpu())

        all_patches_tensor = torch.cat(all_patches, dim=0)   # (N_total, 256)

        # Coreset subsampling — keep only subsample_ratio fraction of patches
        # This reduces memory and speeds up k-NN without sacrificing accuracy much
        n_keep = max(1, int(len(all_patches_tensor) * self.subsample_ratio))
        indices = torch.randperm(len(all_patches_tensor))[:n_keep]
        memory = all_patches_tensor[indices]

        self.memory_banks[bottle_type] = memory   # (n_keep, 256)

        if verbose:
            print(f"[PatchCore] Memory bank '{bottle_type}': {memory.shape[0]} patch vectors stored.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        bottle_type: str,
    ) -> Tuple[float, np.ndarray]:
        """Compute anomaly score and pixel-level heatmap for a test image.

        Args:
            image: PIL image to inspect.
            bottle_type: Which memory bank to compare against.

        Returns:
            Tuple of (scalar_anomaly_score, heatmap_uint8_array HxWx3).

        Raises:
            KeyError: If bottle_type not enrolled.
        """
        if bottle_type not in self.memory_banks:
            raise KeyError(
                f"Bottle type '{bottle_type}' not enrolled. "
                f"Available: {list(self.memory_banks.keys())}"
            )

        bank = self.memory_banks[bottle_type].to(self.device)   # (N, 256)

        x = self.transform(image.convert("RGB")).unsqueeze(0)
        feat_map = self.extractor.extract(x)   # (1, 256, h, w)
        _, C, H, W = feat_map.shape
        patches = feat_map.permute(0, 2, 3, 1).reshape(-1, C)   # (H*W, 256)

        # k-NN distance for each test patch against memory bank
        # Compute pairwise L2 distances in batches for memory efficiency
        BATCH = 512
        min_dists = []
        for i in range(0, len(patches), BATCH):
            q = patches[i:i + BATCH]   # (b, 256)
            # ||q - bank||^2 = ||q||^2 + ||bank||^2 - 2 * q @ bank.T
            dists = torch.cdist(q.float(), bank.float(), p=2)   # (b, N)
            knn_dists, _ = dists.topk(self.k, dim=1, largest=False)
            min_dists.append(knn_dists.mean(dim=1))   # mean of k nearest

        dist_map = torch.cat(min_dists).reshape(H, W).cpu().numpy()   # (H, W)

        # Anomaly score = 95th percentile of patch distances (robust to noise)
        anomaly_score = float(np.percentile(dist_map, 95))

        # Normalize distance map to [0, 255] heatmap
        import cv2
        dist_up = cv2.resize(
            dist_map,
            (self.input_size[1], self.input_size[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        d_min, d_max = dist_up.min(), dist_up.max()
        if d_max > d_min:
            norm = ((dist_up - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(dist_up, dtype=np.uint8)

        heatmap_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        return anomaly_score, heatmap_rgb

    # ------------------------------------------------------------------
    # Threshold calibration
    # ------------------------------------------------------------------
    def calibrate_threshold(
        self,
        bottle_type: str,
        normal_images: List[Image.Image],
        percentile: float = 97.5,
    ) -> float:
        """Auto-calibrate decision threshold from normal image scores.

        Uses max normal score + safety margin to prevent false rejects on normal samples,
        while maintaining high sensitivity to anomalies.

        Args:
            bottle_type: Which bottle type to calibrate for.
            normal_images: Images KNOWN to be normal (from user or MVTec).
            percentile: Percentile fallback if many images are provided.

        Returns:
            Calibrated threshold value.
        """
        scores = [self.predict(img, bottle_type)[0] for img in normal_images]
        if len(scores) < 10:
            # Small sample regime: max normal score + 30% margin (min 0.10 margin)
            max_s = float(np.max(scores))
            threshold = float(max(max_s * 1.30, max_s + 0.10))
        else:
            # Large sample regime: percentile
            threshold = float(np.percentile(scores, percentile))
        self.thresholds[bottle_type] = threshold
        print(f"[PatchCore] Calibrated threshold for '{bottle_type}': {threshold:.4f} (max normal={np.max(scores):.4f})")
        return threshold

    def get_threshold(self, bottle_type: str, default: float = 1.5) -> float:
        """Return calibrated threshold or default."""
        return self.thresholds.get(bottle_type, default)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save all memory banks and thresholds to disk."""
        data = {
            "memory_banks": {k: v.numpy() for k, v in self.memory_banks.items()},
            "thresholds": self.thresholds,
            "k": self.k,
            "subsample_ratio": self.subsample_ratio,
            "input_size": self.input_size,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[PatchCore] Saved to {path}")

    def load(self, path: str) -> None:
        """Load memory banks and thresholds from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.memory_banks = {k: torch.from_numpy(v) for k, v in data["memory_banks"].items()}
        self.thresholds = data.get("thresholds", {})
        self.k = data.get("k", self.k)
        self.input_size = data.get("input_size", self.input_size)
        print(f"[PatchCore] Loaded. Types: {list(self.memory_banks.keys())}")

    @property
    def enrolled_types(self) -> List[str]:
        """List of enrolled bottle type names."""
        return list(self.memory_banks.keys())
