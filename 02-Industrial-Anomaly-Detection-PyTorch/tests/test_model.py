"""Unit tests for the Convolutional Autoencoder architecture."""

import torch
import torch.nn as nn
from src.models.autoencoder import (
    ConvAutoencoder,
    Decoder,
    Encoder,
    build_autoencoder,
    load_compatible_state_dict,
)


def test_encoder_output_shape() -> None:
    """Verify that Encoder compresses image to expected latent dimension."""
    batch_size = 4
    in_channels = 3
    input_size = (256, 256)
    latent_dim = 128

    encoder = Encoder(
        in_channels=in_channels,
        channels=[32, 64, 128, 256],
        latent_dim=latent_dim,
        input_size=input_size,
    )

    x = torch.randn(batch_size, in_channels, *input_size)
    z = encoder(x)

    assert z.shape == (batch_size, latent_dim)
    assert not torch.isnan(z).any()


def test_decoder_output_shape_and_bounds() -> None:
    """Verify that Decoder reconstructs exact original image dimensions with Sigmoid bounds."""
    batch_size = 4
    out_channels = 3
    latent_dim = 128
    bottleneck_shape = (256, 16, 16)

    decoder = Decoder(
        latent_dim=latent_dim,
        out_channels=out_channels,
        channels=[256, 128, 64, 32],
        bottleneck_shape=bottleneck_shape,
    )

    z = torch.randn(batch_size, latent_dim)
    reconstructed = decoder(z)

    assert reconstructed.shape == (batch_size, out_channels, 256, 256)
    # Output must be strictly bounded in [0.0, 1.0] by Sigmoid
    assert 0.0 <= reconstructed.min().item()
    assert reconstructed.max().item() <= 1.0


def test_autoencoder_end_to_end_forward() -> None:
    """Verify full end-to-end forward pass preserves spatial resolution."""
    batch_size = 2
    model = ConvAutoencoder(
        in_channels=3,
        input_size=(256, 256),
        channels=[32, 64, 128, 256],
        latent_dim=128,
    )

    x = torch.rand(batch_size, 3, 256, 256)
    out = model(x)

    assert out.shape == x.shape
    assert 0.0 <= out.min().item() <= out.max().item() <= 1.0


def test_anomaly_map_generation() -> None:
    """Verify anomaly map calculation produces (B, 1, H, W) residual map."""
    model = ConvAutoencoder(
        in_channels=3,
        input_size=(128, 128),
        channels=[32, 64],
        latent_dim=64,
    )

    x = torch.rand(2, 3, 128, 128)
    anomaly_map = model.compute_anomaly_map(x)

    assert anomaly_map.shape == (2, 1, 128, 128)
    assert anomaly_map.min().item() >= 0.0


def test_gradient_flow_backpropagation() -> None:
    """Verify that gradients properly flow through all encoder and decoder parameters."""
    model = ConvAutoencoder(
        in_channels=3,
        input_size=(64, 64),
        channels=[16, 32],
        latent_dim=32,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    x = torch.rand(2, 3, 64, 64)
    optimizer.zero_grad()
    reconstruction = model(x)
    loss = criterion(reconstruction, x)
    loss.backward()
    optimizer.step()

    # Check that every learnable parameter received a valid gradient
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_build_autoencoder_from_config() -> None:
    """Verify factory instantiation using project configs/config.yaml."""
    model = build_autoencoder("configs/config.yaml")
    assert isinstance(model, ConvAutoencoder)
    assert model.latent_dim == 128
    assert model.input_size == (256, 256)


def test_load_legacy_checkpoint_state_dict() -> None:
    """Verify checkpoints with legacy layer names load into the current model."""
    model = ConvAutoencoder(
        in_channels=3,
        input_size=(32, 32),
        channels=[4, 8],
        latent_dim=6,
    )
    expected_state_dict = model.state_dict()
    legacy_state_dict = {}
    for key, value in expected_state_dict.items():
        legacy_key = key
        for current_prefix, legacy_prefix in (
            ("encoder.fc_latent.", "encoder.fc."),
            ("decoder.fc_unflatten.", "decoder.fc."),
            ("decoder.deconv_net.", "decoder.deconv."),
        ):
            if key.startswith(current_prefix):
                legacy_key = legacy_prefix + key[len(current_prefix):]
                break
        legacy_state_dict[legacy_key] = value.clone()

    loaded_model = ConvAutoencoder(
        in_channels=3,
        input_size=(32, 32),
        channels=[4, 8],
        latent_dim=6,
    )
    load_compatible_state_dict(loaded_model, legacy_state_dict)

    for key, value in expected_state_dict.items():
        assert torch.equal(loaded_model.state_dict()[key], value)
