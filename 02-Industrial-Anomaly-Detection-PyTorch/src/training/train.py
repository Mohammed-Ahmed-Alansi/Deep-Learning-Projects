"""Training and Validation Pipeline Module for DeepInspect AI.

This module provides a production-grade training orchestrator tailored for both
local workstations and Google Colab Linux GPU environments. It features:
- Automatic hardware device detection (CUDA GPU fallback to CPU).
- Unsupervised reconstruction optimization using MSE Loss and Adam optimizer.
- Validation monitoring on normal calibration samples.
- Model checkpointing saving the best weights to './models/best_model.pth'.
- Early stopping to prevent overfitting.
- Clean, informative Colab-friendly console outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.data.dataset import get_dataloaders
from src.models.autoencoder import ConvAutoencoder, build_autoencoder


class EarlyStopping:
    """Monitors validation loss and signals when training should terminate.

    Args:
        patience: Number of epochs to wait without improvement before stopping.
        min_delta: Minimum change in monitored loss to qualify as improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class Trainer:
    """Orchestrates model training, validation, checkpointing, and metric logging.

    Args:
        model: Instantiated ConvAutoencoder model.
        train_loader: DataLoader containing normal training images.
        val_loader: DataLoader containing normal validation images.
        config: Full configuration dictionary.
        checkpoint_dir: Directory where checkpoints are saved. Default: './models'.
        device: PyTorch device. If None, auto-detects CUDA else CPU.
    """

    def __init__(
        self,
        model: ConvAutoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        checkpoint_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.config = config
        train_cfg = config.get("training", {})

        # 1. Device Detection (Crucial for Colab GPU / local execution)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 2. Hyperparameters
        self.epochs = int(train_cfg.get("epochs", 50))
        self.lr = float(train_cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-5))

        # 3. Loss & Optimizer (Strictly MSE Loss & Adam optimizer)
        self.criterion = nn.MSELoss()
        self.optimizer: Optimizer = Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # 4. Learning Rate Scheduler
        sched_cfg = train_cfg.get("scheduler", {})
        self.scheduler: _LRScheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 5)),
        )

        # 5. Early Stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping_enabled = bool(es_cfg.get("enabled", True))
        self.early_stopping = EarlyStopping(
            patience=int(es_cfg.get("patience", 10)),
            min_delta=float(es_cfg.get("min_delta", 1e-4)),
        )

        # 6. Checkpoint & Output Directories
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        else:
            self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "./models")).expanduser()
        self.output_dir = Path(train_cfg.get("output_dir", "./outputs")).expanduser()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 7. Metrics & State History
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": []}
        self.best_val_loss = float("inf")

    def train_epoch(self, epoch: int) -> float:
        """Run a single training epoch."""
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:02d}/{self.epochs:02d} [Train]",
            leave=False,
            dynamic_ncols=True,
        )
        for images, _, _, _ in pbar:
            # Transfer images to target compute device
            images = images.to(self.device)

            self.optimizer.zero_grad()
            reconstructed = self.model(images)
            loss = self.criterion(reconstructed, images)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        epoch_loss = running_loss / max(len(self.train_loader.dataset), 1)
        return epoch_loss

    def validate(self, epoch: int) -> float:
        """Evaluate reconstruction error on the held-out normal validation set."""
        self.model.eval()
        running_loss = 0.0

        if len(self.val_loader.dataset) == 0:
            return 0.0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch:02d}/{self.epochs:02d} [Val]",
            leave=False,
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for images, _, _, _ in pbar:
                images = images.to(self.device)
                reconstructed = self.model(images)
                loss = self.criterion(reconstructed, images)

                running_loss += loss.item() * images.size(0)
                pbar.set_postfix({"val_loss": f"{loss.item():.5f}"})

        val_loss = running_loss / max(len(self.val_loader.dataset), 1)
        return val_loss

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        """Save model checkpoint to disk, targeting './models/best_model.pth'."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else None,
            "val_loss": val_loss,
            "config": self.config,
        }

        # Save last model checkpoint
        last_path = self.checkpoint_dir / "last_model.pth"
        torch.save(checkpoint, last_path)

        # Save best model to ./models/best_model.pth
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)

    def plot_loss_curve(self) -> Path:
        """Generate and save the training and validation reconstruction loss curve."""
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs_range = range(1, len(self.history["train_loss"]) + 1)

        ax.plot(epochs_range, self.history["train_loss"], label="Train Loss (MSE)", color="#007acc", lw=2)
        if any(v > 0 for v in self.history["val_loss"]):
            ax.plot(epochs_range, self.history["val_loss"], label="Val Loss (MSE)", color="#e67e22", lw=2)

        ax.set_title("DeepInspect AI - Reconstruction Loss Curve", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=10)
        plt.tight_layout()

        plot_path = self.output_dir / "loss_curve.png"
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        return plot_path

    def fit(self) -> Dict[str, List[float]]:
        """Run the complete training pipeline across all epochs with clean Colab logging."""
        device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        print("=" * 78)
        print(f" DeepInspect AI - Training Pipeline")
        print(f" Target Compute Device : {self.device} ({device_name})")
        print(f" Total Planned Epochs  : {self.epochs}")
        print(f" Learning Rate & Decay : {self.lr} | {self.weight_decay}")
        print(f" Best Checkpoint Target: {self.checkpoint_dir / 'best_model.pth'}")
        print("=" * 78)

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            # Step scheduler based on monitored loss
            monitored_loss = val_loss if val_loss > 0 else train_loss
            self.scheduler.step(monitored_loss)

            # Determine whether this epoch attained the best loss
            is_best = monitored_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = monitored_loss

            self.save_checkpoint(epoch, monitored_loss, is_best=is_best)

            # Clean and informative single-line output per epoch for Colab / terminal
            best_tag = " -> [SAVED BEST]" if is_best else ""
            print(
                f"[Epoch {epoch:03d}/{self.epochs:03d}]  "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {current_lr:.6f}{best_tag}"
            )

            # Early Stopping Check
            if self.early_stopping_enabled and self.early_stopping(monitored_loss):
                print(f"\n[Early Stopping] No improvement for {self.early_stopping.patience} epochs. Stopping.")
                break

        # Save training summary
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)

        plot_path = self.plot_loss_curve()
        print("=" * 78)
        print(" Training Session Finished Successfully!")
        print(f"   Best Validation Loss : {self.best_val_loss:.6f}")
        print(f"   Best Model Weights   : {self.checkpoint_dir / 'best_model.pth'}")
        print(f"   Loss Curve Plot      : {plot_path}")
        print("=" * 78 + "\n")

        return self.history


def train(
    config_path: Union[str, Path] = "configs/config.yaml",
    data_dir: Optional[str] = None,
    checkpoint_dir: Union[str, Path] = "./models",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    device: Optional[str] = None,
) -> Trainer:
    """Convenience function to load config and run model training.

    Args:
        config_path: Path to YAML config file.
        data_dir: Optional dataset path override (e.g. for Google Colab).
        checkpoint_dir: Path to directory where checkpoints are saved.
        epochs: Optional epoch count override.
        batch_size: Optional batch size override.
        lr: Optional learning rate override.
        device: Optional device string ('cuda' or 'cpu').

    Returns:
        Trained Trainer instance.
    """
    config_file = Path(config_path).expanduser()
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Overrides
    if data_dir is not None:
        config["dataset"]["root_dir"] = data_dir
    if epochs is not None:
        config["training"]["epochs"] = int(epochs)
    if batch_size is not None:
        config["dataset"]["batch_size"] = int(batch_size)
    if lr is not None:
        config["training"]["learning_rate"] = float(lr)

    train_loader, val_loader, _ = get_dataloaders(config)
    model = build_autoencoder(config)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )
    trainer.fit()
    return trainer


def main() -> None:
    """CLI Entrypoint for running training via command line."""
    parser = argparse.ArgumentParser(description="DeepInspect AI - Autoencoder Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Root path of MVTec AD category")
    parser.add_argument("--checkpoint_dir", type=str, default="./models", help="Directory to save best_model.pth")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")

    args = parser.parse_args()

    train(
        config_path=args.config,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
