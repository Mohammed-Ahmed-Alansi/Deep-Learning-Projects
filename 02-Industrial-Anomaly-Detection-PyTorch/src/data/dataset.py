"""MVTec AD Dataset and DataLoader Module.

This module provides an OOP-based PyTorch Dataset implementation for the MVTec Anomaly
Detection (AD) dataset. It enforces strict data integrity rules, guarantees zero
leakage between normal training data and defect evaluation samples, and provides
cross-platform compatibility across Windows and Linux (Google Colab).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import yaml


class MVTecDataset(Dataset):
    """PyTorch Dataset for MVTec Anomaly Detection (AD).

    Args:
        root_dir: Root directory of category data (e.g. './data/mvtec_ad/bottle').
        split: One of 'train', 'val', or 'test'.
        image_size: Target tuple (height, width) for image resizing. Default is (256, 256).
        val_ratio: Fraction of normal training images reserved for validation split.
        seed: Random seed for deterministic train/val splitting.
        transform: Optional custom transform for input images.
        target_transform: Optional custom transform for ground truth masks.
    """

    VALID_SPLITS = ("train", "val", "test")

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        val_ratio: float = 0.2,
        seed: int = 42,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Expected one of {self.VALID_SPLITS}.")

        self.root_dir = Path(root_dir).expanduser()
        self.split = split
        self.image_size = tuple(image_size)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)

        # Configure default transforms if not explicitly provided
        self.transform = transform or self._default_image_transform()
        self.target_transform = target_transform or self._default_mask_transform()

        # Samples list storing metadata dictionaries
        self.samples: List[Dict[str, Any]] = []
        self._load_samples()

    def _default_image_transform(self) -> T.Compose:
        """Construct standard image transformation: Resize and normalize to [0, 1]."""
        return T.Compose(
            [
                T.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
                T.ToTensor(),
            ]
        )

    def _default_mask_transform(self) -> T.Compose:
        """Construct standard binary mask transformation."""
        return T.Compose(
            [
                T.Resize(self.image_size, interpolation=InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )

    def _load_samples(self) -> None:
        """Discover and index files based on the specified split.

        Data Integrity Rules:
        - 'train': Only normal ('good') images from the 'train' folder.
        - 'val': Held-out normal ('good') images from the 'train' folder.
        - 'test': All test images ('good' + defect categories) with corresponding GT masks.
        """
        train_dir = self.root_dir / "train" / "good"
        test_dir = self.root_dir / "test"
        gt_dir = self.root_dir / "ground_truth"

        if self.split in ("train", "val"):
            if not train_dir.exists():
                return  # Directory not yet created or empty

            # Sort files to guarantee deterministic indexing across platforms
            train_images = sorted(
                [
                    p
                    for p in train_dir.iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                ]
            )

            if len(train_images) == 0:
                return

            # Deterministic train/validation split
            rng = np.random.RandomState(self.seed)
            indices = np.arange(len(train_images))
            rng.shuffle(indices)

            val_count = int(np.floor(len(train_images) * self.val_ratio))
            if self.split == "val":
                selected_indices = indices[:val_count]
            else:
                selected_indices = indices[val_count:]

            for idx in selected_indices:
                img_path = train_images[idx]
                self.samples.append(
                    {
                        "image_path": img_path,
                        "label": 0,  # 0 indicates normal / healthy
                        "defect_type": "good",
                        "mask_path": None,
                    }
                )

        elif self.split == "test":
            if not test_dir.exists():
                return

            # Discover all subdirectories under test/
            defect_types = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

            for defect in defect_types:
                defect_path = test_dir / defect
                is_anomaly = defect != "good"
                label = 1 if is_anomaly else 0

                images = sorted(
                    [
                        p
                        for p in defect_path.iterdir()
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                    ]
                )

                for img_path in images:
                    mask_path: Optional[Path] = None
                    if is_anomaly:
                        # Standard MVTec AD mask naming: <defect_name>/<stem>_mask.png
                        expected_mask = gt_dir / defect / f"{img_path.stem}_mask{img_path.suffix}"
                        if expected_mask.exists():
                            mask_path = expected_mask
                        else:
                            # Fallback if filename matches image filename exactly
                            alt_mask = gt_dir / defect / img_path.name
                            if alt_mask.exists():
                                mask_path = alt_mask

                    self.samples.append(
                        {
                            "image_path": img_path,
                            "label": label,
                            "defect_type": defect,
                            "mask_path": mask_path,
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor, str]:
        """Fetch item by index.

        Returns:
            Tuple of:
                - image (torch.Tensor): Shape (C, H, W) normalized to [0, 1].
                - label (int): 0 for normal ('good'), 1 for anomaly.
                - mask (torch.Tensor): Shape (1, H, W) binary mask {0.0, 1.0}.
                - image_path (str): Absolute file path to the source image.
        """
        sample_meta = self.samples[index]
        img_path = sample_meta["image_path"]
        label = sample_meta["label"]
        mask_path = sample_meta["mask_path"]

        # Load RGB Image
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        image_tensor = self.transform(image)

        # Load or generate binary mask
        h, w = self.image_size
        if mask_path is not None and mask_path.exists():
            with Image.open(mask_path) as mask_img:
                mask_tensor = self.target_transform(mask_img.convert("L"))
                # Binarize mask to guarantee strict {0.0, 1.0} values
                mask_tensor = (mask_tensor > 0.5).float()
        else:
            mask_tensor = torch.zeros((1, h, w), dtype=torch.float32)

        return image_tensor, label, mask_tensor, str(img_path)


def get_dataloaders(
    config_or_path: Union[str, Path, Dict[str, Any]],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Factory function to build train, val, and test DataLoaders from configuration.

    Supports both file path to YAML and parsed config dictionaries. Works seamlessly
    on local machines and cloud environments like Google Colab.

    Args:
        config_or_path: Path to YAML config file, or dictionary with configurations.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    if isinstance(config_or_path, (str, Path)):
        config_path = Path(config_or_path).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = config_or_path

    ds_cfg = config.get("dataset", {})
    root_dir = ds_cfg.get("root_dir", "./data/mvtec_ad/bottle")
    image_size = tuple(ds_cfg.get("image_size", [256, 256]))
    batch_size = int(ds_cfg.get("batch_size", 16))
    num_workers = int(ds_cfg.get("num_workers", 2))
    val_ratio = float(ds_cfg.get("val_split_ratio", 0.2))
    pin_memory = bool(ds_cfg.get("pin_memory", True) and torch.cuda.is_available())
    seed = int(config.get("project", {}).get("seed", 42))

    # Construct datasets
    train_dataset = MVTecDataset(
        root_dir=root_dir,
        split="train",
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
    )
    val_dataset = MVTecDataset(
        root_dir=root_dir,
        split="val",
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
    )
    test_dataset = MVTecDataset(
        root_dir=root_dir,
        split="test",
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
    )

    # Build DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_dataset) > batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader

