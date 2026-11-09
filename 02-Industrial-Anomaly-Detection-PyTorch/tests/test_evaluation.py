"""Unit tests for the evaluation metrics module."""

from pathlib import Path
import pytest
import torch

from src.data.dataset import get_dataloaders
from src.evaluation.metrics import AnomalyEvaluator
from src.models.autoencoder import ConvAutoencoder
from tests.test_data import mock_mvtec_dataset


def test_anomaly_evaluator_metrics(mock_mvtec_dataset: Path, tmp_path: Path) -> None:
    """Verify evaluation metrics computation including AUROC, F1, and threshold calibration."""
    config = {
        "project": {"name": "TestInspect", "seed": 42},
        "dataset": {
            "root_dir": str(mock_mvtec_dataset),
            "image_size": [64, 64],
            "batch_size": 4,
            "num_workers": 0,
            "val_split_ratio": 0.25,
            "pin_memory": False,
        },
        "training": {
            "device": "cpu",
            "checkpoint_dir": str(tmp_path / "models"),
            "output_dir": str(tmp_path / "outputs"),
        },
    }

    _, val_loader, test_loader = get_dataloaders(config)

    model = ConvAutoencoder(
        in_channels=3,
        input_size=(64, 64),
        channels=[16, 32],
        latent_dim=32,
    )

    evaluator = AnomalyEvaluator(
        model=model,
        test_loader=test_loader,
        val_loader=val_loader,
        device="cpu",
        gaussian_blur_ksize=5,
        gaussian_blur_sigma=2.0,
    )

    results = evaluator.evaluate()

    # 1. Assert keys exist
    for key in (
        "image_auroc",
        "pixel_auroc",
        "optimal_threshold",
        "f1_score",
        "precision",
        "recall",
        "confusion_matrix",
    ):
        assert key in results, f"Missing metric key: {key}"

    # 2. Check metric bounds
    assert 0.0 <= results["image_auroc"] <= 1.0
    assert 0.0 <= results["pixel_auroc"] <= 1.0
    assert 0.0 <= results["f1_score"] <= 1.0
    assert results["optimal_threshold"] > 0.0
    assert len(results["confusion_matrix"]) == 2

    # 3. Test ROC Plotting
    plot_file = tmp_path / "test_roc.png"
    saved_plot = evaluator.plot_roc_curves(results, output_path=plot_file)
    assert saved_plot.exists()
    assert saved_plot.stat().st_size > 0

