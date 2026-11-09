"""Unit tests for the inference and heatmap localization module."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from src.inference.predict import AnomalyPredictor, load_model, predict_image
from src.models.autoencoder import ConvAutoencoder


@pytest.fixture
def mock_checkpoint(tmp_path: Path) -> Path:
    """Fixture to produce a valid model checkpoint on disk."""
    model = ConvAutoencoder(
        in_channels=3,
        input_size=(64, 64),
        channels=[16, 32],
        latent_dim=32,
    )
    ckpt_path = tmp_path / "test_model.pth"
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "config": {
                "dataset": {"channels": 3, "image_size": [64, 64]},
                "model": {"channel_multipliers": [16, 32], "latent_dim": 32},
            },
        },
        ckpt_path,
    )
    return ckpt_path


def test_load_model_cpu_fallback(mock_checkpoint: Path) -> None:
    """Verify that model loading functions properly with CPU fallback."""
    model, config, device = load_model(mock_checkpoint, device="cpu")
    assert isinstance(model, ConvAutoencoder)
    assert not model.training
    assert device.type == "cpu"


def test_anomaly_predictor_pipeline(mock_checkpoint: Path, tmp_path: Path) -> None:
    """Verify full inference prediction pipeline including heatmap and overlay generation."""
    predictor = AnomalyPredictor(
        checkpoint_path=mock_checkpoint,
        threshold=0.02,
        device="cpu",
    )

    # Create dummy synthetic image
    img_arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img_arr)

    result = predictor.predict(pil_img, alpha=0.5)

    # 1. Verify dictionary keys
    expected_keys = (
        "is_anomaly",
        "anomaly_score",
        "threshold",
        "original_image",
        "reconstructed_image",
        "raw_residual",
        "anomaly_heatmap",
        "overlay",
    )
    for k in expected_keys:
        assert k in result, f"Key {k} missing from inference output."

    # 2. Verify types & shapes
    assert isinstance(result["is_anomaly"], bool)
    assert isinstance(result["anomaly_score"], float)
    assert result["original_image"].shape == (64, 64, 3)
    assert result["reconstructed_image"].shape == (64, 64, 3)
    assert result["raw_residual"].shape == (64, 64)
    assert result["anomaly_heatmap"].shape == (64, 64, 3)
    assert result["overlay"].shape == (64, 64, 3)

    # 3. Test saving overlay through functional interface
    img_path = tmp_path / "test_sample.png"
    pil_img.save(img_path)
    out_overlay_path = tmp_path / "overlay_out.png"

    pred_res = predict_image(
        image_path=img_path,
        checkpoint_path=mock_checkpoint,
        threshold=0.02,
        output_path=out_overlay_path,
    )
    assert out_overlay_path.exists()
    assert out_overlay_path.stat().st_size > 0
    assert pred_res["anomaly_score"] >= 0.0

