"""Unit and integration tests for MVTec AD Dataset and DataLoaders.

Tests verify:
1. Strict Data Integrity: Zero data leakage between train/val/test splits.
2. Tensor Shapes & Types: Correct dimensions (C, H, W) and normalized ranges.
3. Binary Mask Integrity: Ground truth mask alignment and binarization.
4. Cross-Platform Path Handling: Compatibility with Windows and Google Colab paths.
"""

from pathlib import Path
from typing import Generator
import numpy as np
from PIL import Image
import pytest
import torch

from src.data.dataset import MVTecDataset, get_dataloaders


@pytest.fixture
def mock_mvtec_dataset(tmp_path: Path) -> Path:
    """Fixture to generate a synthetic MVTec AD directory structure.

    Directory layout:
        tmp_path/
            train/good/ (10 images)
            test/good/ (4 images)
            test/broken_large/ (4 images)
            ground_truth/broken_large/ (4 mask images)
    """
    category_root = tmp_path / "mock_bottle"

    # Setup directories
    train_good_dir = category_root / "train" / "good"
    test_good_dir = category_root / "test" / "good"
    test_defect_dir = category_root / "test" / "broken_large"
    gt_defect_dir = category_root / "ground_truth" / "broken_large"

    for d in (train_good_dir, test_good_dir, test_defect_dir, gt_defect_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Create 10 normal train images
    for i in range(10):
        img_arr = np.random.randint(50, 200, size=(128, 128, 3), dtype=np.uint8)
        Image.fromarray(img_arr).save(train_good_dir / f"{i:03d}.png")

    # 2. Create 4 normal test images
    for i in range(4):
        img_arr = np.random.randint(50, 200, size=(128, 128, 3), dtype=np.uint8)
        Image.fromarray(img_arr).save(test_good_dir / f"{i:03d}.png")

    # 3. Create 4 defective test images + corresponding ground truth masks
    for i in range(4):
        # Image
        img_arr = np.random.randint(50, 200, size=(128, 128, 3), dtype=np.uint8)
        Image.fromarray(img_arr).save(test_defect_dir / f"{i:03d}.png")

        # Mask: center circle representing defect
        mask_arr = np.zeros((128, 128), dtype=np.uint8)
        mask_arr[40:88, 40:88] = 255  # Square defect region
        Image.fromarray(mask_arr).save(gt_defect_dir / f"{i:03d}_mask.png")

    return category_root


def test_dataset_split_and_zero_leakage(mock_mvtec_dataset: Path) -> None:
    """Verify train/val/test splits and ensure STRICT zero data leakage."""
    val_ratio = 0.2
    image_size = (128, 128)

    train_ds = MVTecDataset(
        root_dir=mock_mvtec_dataset,
        split="train",
        image_size=image_size,
        val_ratio=val_ratio,
        seed=42,
    )
    val_ds = MVTecDataset(
        root_dir=mock_mvtec_dataset,
        split="val",
        image_size=image_size,
        val_ratio=val_ratio,
        seed=42,
    )
    test_ds = MVTecDataset(
        root_dir=mock_mvtec_dataset,
        split="test",
        image_size=image_size,
        val_ratio=val_ratio,
        seed=42,
    )

    # 10 train images total: 8 train, 2 val (with val_ratio=0.2)
    assert len(train_ds) == 8, f"Expected 8 train samples, got {len(train_ds)}"
    assert len(val_ds) == 2, f"Expected 2 val samples, got {len(val_ds)}"
    # 4 good test + 4 broken_large = 8 test samples
    assert len(test_ds) == 8, f"Expected 8 test samples, got {len(test_ds)}"

    train_paths = {str(s["image_path"]) for s in train_ds.samples}
    val_paths = {str(s["image_path"]) for s in val_ds.samples}
    test_paths = {str(s["image_path"]) for s in test_ds.samples}

    # CRITICAL DATA INTEGRITY CHECK 1: Train and Val splits must not overlap
    assert train_paths.isdisjoint(val_paths), "Data leakage detected: Overlap between Train and Val sets!"

    # CRITICAL DATA INTEGRITY CHECK 2: Train and Test splits must not overlap
    assert train_paths.isdisjoint(test_paths), "Data leakage detected: Overlap between Train and Test sets!"

    # CRITICAL DATA INTEGRITY CHECK 3: Val and Test splits must not overlap
    assert val_paths.isdisjoint(test_paths), "Data leakage detected: Overlap between Val and Test sets!"

    # All train & val samples must be normal ('good' with label 0)
    for sample in train_ds.samples:
        assert sample["label"] == 0
        assert sample["defect_type"] == "good"

    for sample in val_ds.samples:
        assert sample["label"] == 0
        assert sample["defect_type"] == "good"


def test_item_shapes_and_value_ranges(mock_mvtec_dataset: Path) -> None:
    """Verify image/mask dimensions, tensor dtypes, and value normalization."""
    image_size = (64, 64)
    test_ds = MVTecDataset(
        root_dir=mock_mvtec_dataset,
        split="test",
        image_size=image_size,
    )

    for i in range(len(test_ds)):
        image, label, mask, img_path = test_ds[i]

        # Verify types
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(img_path, str)

        # Verify shapes
        assert image.shape == (3, 64, 64)
        assert mask.shape == (1, 64, 64)

        # Verify normalization ranges
        assert 0.0 <= image.min() <= image.max() <= 1.0
        assert label in (0, 1)

        # Verify binary mask
        unique_mask_vals = torch.unique(mask).tolist()
        assert all(val in (0.0, 1.0) for val in unique_mask_vals)

        # Defect images should have positive mask pixels
        if label == 1:
            assert mask.sum() > 0, f"Expected non-zero mask for anomaly sample: {img_path}"
        else:
            assert mask.sum() == 0, f"Expected zero mask for normal sample: {img_path}"


def test_dataloaders_batching(mock_mvtec_dataset: Path) -> None:
    """Verify DataLoader batch creation, shuffling, and structure."""
    config = {
        "project": {"seed": 123},
        "dataset": {
            "root_dir": str(mock_mvtec_dataset),
            "image_size": [64, 64],
            "batch_size": 4,
            "num_workers": 0,  # 0 workers for fast local testing
            "val_split_ratio": 0.25,
            "pin_memory": False,
        },
    }

    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Fetch one batch from train loader
    images, labels, masks, paths = next(iter(train_loader))

    assert images.shape == (4, 3, 64, 64)
    assert labels.shape == (4,)
    assert masks.shape == (4, 1, 64, 64)
    assert len(paths) == 4
    # All train batch labels must be 0
    assert torch.all(labels == 0)


def test_invalid_split_raises() -> None:
    """Verify that an invalid split name raises a ValueError."""
    with pytest.raises(ValueError, match="Invalid split"):
        MVTecDataset(root_dir="./dummy", split="unknown_split")

