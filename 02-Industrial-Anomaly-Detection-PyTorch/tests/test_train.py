"""Unit and integration tests for the training and validation pipeline."""

from pathlib import Path
import json
import pytest
import torch

from src.data.dataset import get_dataloaders
from src.models.autoencoder import ConvAutoencoder
from src.training.train import EarlyStopping, Trainer
from tests.test_data import mock_mvtec_dataset


def test_early_stopping_logic() -> None:
    """Verify EarlyStopping correctly tracks patience and triggers stopping."""
    patience = 3
    es = EarlyStopping(patience=patience, min_delta=1e-3)

    # Initial loss: best loss initialized
    assert not es(1.0)
    assert es.counter == 0

    # Improved loss
    assert not es(0.8)
    assert es.counter == 0

    # No improvement (increment counter)
    assert not es(0.81)
    assert es.counter == 1

    assert not es(0.805)
    assert es.counter == 2

    # Patience exhausted
    assert es(0.82)
    assert es.counter == 3
    assert es.early_stop is True


def test_trainer_fit_and_checkpointing(mock_mvtec_dataset: Path, tmp_path: Path) -> None:
    """Verify end-to-end training loop execution, checkpoint saving, and metrics logging."""
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "outputs"

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
            "epochs": 2,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "device": "cpu",
            "checkpoint_dir": str(checkpoint_dir),
            "output_dir": str(output_dir),
            "scheduler": {"factor": 0.5, "patience": 1},
            "early_stopping": {"enabled": True, "patience": 2, "min_delta": 1e-4},
        },
    }

    train_loader, val_loader, _ = get_dataloaders(config)
    model = ConvAutoencoder(
        in_channels=3,
        input_size=(64, 64),
        channels=[16, 32],
        latent_dim=32,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device="cpu",
    )

    history = trainer.fit()

    # 1. Verify history records
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert len(history["lr"]) == 2

    # 2. Verify checkpoint files exist
    best_ckpt = checkpoint_dir / "best_model.pth"
    last_ckpt = checkpoint_dir / "last_model.pth"
    assert best_ckpt.exists()
    assert last_ckpt.exists()

    # 3. Verify checkpoint contents
    loaded = torch.load(best_ckpt, map_location="cpu")
    assert "model_state_dict" in loaded
    assert "optimizer_state_dict" in loaded
    assert "epoch" in loaded
    assert "val_loss" in loaded

    # 4. Verify outputs exist
    assert (output_dir / "training_history.json").exists()
    assert (output_dir / "loss_curve.png").exists()

    # 5. Verify checkpoint weights load into a fresh model
    new_model = ConvAutoencoder(
        in_channels=3,
        input_size=(64, 64),
        channels=[16, 32],
        latent_dim=32,
    )
    new_model.load_state_dict(loaded["model_state_dict"])
    new_model.eval()
    dummy_input = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        out = new_model(dummy_input)
    assert out.shape == dummy_input.shape

