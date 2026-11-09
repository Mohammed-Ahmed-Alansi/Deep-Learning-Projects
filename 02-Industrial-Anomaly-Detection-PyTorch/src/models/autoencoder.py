"""Convolutional Autoencoder Architecture Module.

This module implements a modular PyTorch Convolutional Autoencoder (CAE) tailored
for unsupervised industrial visual inspection (MVTec AD). The architecture consists of:
1. Encoder: Conv2D layers with Batch Normalization, ReLU activation, and MaxPooling2D.
2. Bottleneck: Dense latent projection preserving global geometry and texture distributions.
3. Decoder: ConvTranspose2D layers with Batch Normalization, ReLU, and a Sigmoid output
   ensuring the reconstructed images match the exact spatial dimensions (e.g., 256x256)
   and [0, 1] range of the input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import yaml


class Encoder(nn.Module):
    """Convolutional Encoder network.

    Downsamples the input image through successive Conv2D -> ReLU -> MaxPooling2D blocks,
    culminating in a compact latent vector representation.

    Args:
        in_channels: Number of image channels (3 for RGB). Default: 3.
        channels: Channel multiplier progression. Default: [32, 64, 128, 256].
        latent_dim: Bottleneck latent dimension. Default: 128.
        input_size: Spatial dimensions (H, W). Default: (256, 256).
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Optional[List[int]] = None,
        latent_dim: int = 128,
        input_size: Tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels or [32, 64, 128, 256]
        self.latent_dim = latent_dim
        self.input_size = tuple(input_size)

        conv_blocks: List[nn.Module] = []
        prev_ch = self.in_channels

        for i, ch in enumerate(self.channels):
            conv_blocks.extend(
                [
                    nn.Conv2d(prev_ch, ch, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            prev_ch = ch

        self.features = nn.Sequential(*conv_blocks)

        # Dynamically deduce bottleneck feature map dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, *self.input_size)
            dummy_out = self.features(dummy_input)
            self.bottleneck_shape: Tuple[int, int, int] = (
                dummy_out.shape[1],
                dummy_out.shape[2],
                dummy_out.shape[3],
            )
            self.flattened_dim: int = dummy_out.numel() // dummy_out.shape[0]

        self.fc_latent = nn.Linear(self.flattened_dim, self.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to extract latent vector.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Latent tensor of shape (B, latent_dim).
        """
        feats = self.features(x)
        flat = feats.flatten(start_dim=1)
        latent = self.fc_latent(flat)
        return latent


class Decoder(nn.Module):
    """Transposed Convolutional Decoder network.

    Reconstructs the original spatial resolution from the latent bottleneck
    using Linear projection and ConvTranspose2D layers with Sigmoid activation.

    Args:
        latent_dim: Bottleneck latent dimension. Default: 128.
        out_channels: Reconstructed image channels (3 for RGB). Default: 3.
        channels: Reverse channel progression matching encoder. Default: [256, 128, 64, 32].
        bottleneck_shape: Feature map shape (C, H, W) before flattening in Encoder.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 3,
        channels: Optional[List[int]] = None,
        bottleneck_shape: Tuple[int, int, int] = (256, 16, 16),
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.channels = channels or [256, 128, 64, 32]
        self.bottleneck_shape = bottleneck_shape
        self.flattened_dim = int(torch.tensor(bottleneck_shape).prod().item())

        self.fc_unflatten = nn.Linear(self.latent_dim, self.flattened_dim)

        deconv_blocks: List[nn.Module] = []
        for i in range(len(self.channels) - 1):
            deconv_blocks.extend(
                [
                    nn.ConvTranspose2d(
                        self.channels[i],
                        self.channels[i + 1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.channels[i + 1]),
                    nn.ReLU(inplace=True),
                ]
            )

        # Final Transposed Convolution projecting to original image channels
        deconv_blocks.extend(
            [
                nn.ConvTranspose2d(
                    self.channels[-1],
                    self.out_channels,
                    kernel_size=2,
                    stride=2,
                    bias=True,
                ),
                nn.Sigmoid(),  # Bounds outputs strictly to [0.0, 1.0]
            ]
        )

        self.deconv_net = nn.Sequential(*deconv_blocks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass to reconstruct image from latent code.

        Args:
            z: Latent tensor of shape (B, latent_dim).

        Returns:
            Reconstructed image tensor of shape (B, C, H, W).
        """
        flat = self.fc_unflatten(z)
        unflattened = flat.view(-1, *self.bottleneck_shape)
        reconstruction = self.deconv_net(unflattened)
        return reconstruction


class ConvAutoencoder(nn.Module):
    """End-to-End Convolutional Autoencoder for Visual Anomaly Detection.

    Args:
        in_channels: Number of image channels (e.g., 3 for RGB). Default: 3.
        input_size: Spatial dimensions (H, W). Default: (256, 256).
        channels: Channel multiplier sequence for encoder blocks. Default: [32, 64, 128, 256].
        latent_dim: Latent representation dimensionality. Default: 128.
    """

    def __init__(
        self,
        in_channels: int = 3,
        input_size: Tuple[int, int] = (256, 256),
        channels: Optional[List[int]] = None,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.input_size = tuple(input_size)
        self.channels = channels or [32, 64, 128, 256]
        self.latent_dim = latent_dim

        # 1. Modular Encoder
        self.encoder = Encoder(
            in_channels=self.in_channels,
            channels=self.channels,
            latent_dim=self.latent_dim,
            input_size=self.input_size,
        )

        # 2. Modular Decoder matching Encoder topology
        rev_channels = list(reversed(self.channels))
        self.decoder = Decoder(
            latent_dim=self.latent_dim,
            out_channels=self.in_channels,
            channels=rev_channels,
            bottleneck_shape=self.encoder.bottleneck_shape,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input image to latent code."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to reconstructed image."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full reconstruction forward pass.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Reconstructed image tensor of shape (B, C, H, W).
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction

    def compute_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pixel-wise reconstruction residual anomaly heatmap.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Single-channel anomaly residual map of shape (B, 1, H, W).
        """
        self.eval()
        with torch.no_grad():
            reconstruction = self.forward(x)
            residual = torch.mean((x - reconstruction) ** 2, dim=1, keepdim=True)
        return residual


def load_compatible_state_dict(
    model: ConvAutoencoder,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Load model weights, translating parameter names used by older checkpoints."""
    legacy_prefixes = (
        ("encoder.fc.", "encoder.fc_latent."),
        ("decoder.fc.", "decoder.fc_unflatten."),
        ("decoder.deconv.", "decoder.deconv_net."),
    )
    compatible_state_dict: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        compatible_key = key
        for legacy_prefix, current_prefix in legacy_prefixes:
            if key.startswith(legacy_prefix):
                compatible_key = current_prefix + key[len(legacy_prefix):]
                break

        if compatible_key in compatible_state_dict:
            raise ValueError(f"Checkpoint contains multiple entries for {compatible_key!r}")
        compatible_state_dict[compatible_key] = value

    model.load_state_dict(compatible_state_dict)


def build_autoencoder(
    config_or_path: Union[str, Path, Dict[str, Any]],
) -> ConvAutoencoder:
    """Convenience factory function to build ConvAutoencoder from config file or dictionary.

    Args:
        config_or_path: Path to YAML file or configuration dictionary.

    Returns:
        Instantiated ConvAutoencoder instance.
    """
    if isinstance(config_or_path, (str, Path)):
        cfg_path = Path(config_or_path).expanduser()
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = config_or_path

    model_cfg = config.get("model", {})
    ds_cfg = config.get("dataset", {})

    in_channels = int(model_cfg.get("in_channels", ds_cfg.get("channels", 3)))
    input_size = tuple(ds_cfg.get("image_size", [256, 256]))
    channels = list(model_cfg.get("channel_multipliers", [32, 64, 128, 256]))
    latent_dim = int(model_cfg.get("latent_dim", 128))

    model = ConvAutoencoder(
        in_channels=in_channels,
        input_size=input_size,
        channels=channels,
        latent_dim=latent_dim,
    )
    return model
