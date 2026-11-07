"""Display helpers for the Pneumonia PyTorch classroom notebook.

The notebook contains the training and prediction logic. This module contains
only plotting and the optional Grad-CAM demonstration.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from IPython.display import display


# Show sample images and class labels before training.
# Why: catch image preparation or label issues early.
# Use after creating the DataLoader; this previews samples, not the full dataset.
def show_batch(images, labels, class_names, mean, std, output="assets/batch_examples.png"):
    """Display six normalized tensors after reversing normalization."""
    # Shape normalization values for the three RGB channels.
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    # Preview up to six images from the batch.
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, image, label in zip(axes.flat, images[:6], labels[:6]):
        # Undo normalization and change CHW to HWC for display.
        rgb = (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0)
        ax.imshow(rgb)
        ax.set_title(class_names[int(label)], fontsize=13)
        ax.axis("off")
    fig.tight_layout()
    # Save and display the figure.
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    display(fig)
    plt.close(fig)


# Plot training and validation loss and accuracy across epochs.
# Why: check progress and spot a growing train-validation gap.
# Use after training or loading saved history to look for signs of overfitting.
def plot_history(history, best_epoch, output="assets/learning_curves.png"):
    """Plot measured training and validation metrics."""
    # Columns: train loss, validation loss, train accuracy, validation accuracy.
    values = np.asarray(history, dtype=float)
    # Check the history shape and metric values before plotting.
    if values.ndim != 2 or values.shape[1] != 4 or not len(values):
        raise ValueError("history must contain Train loss, Val loss, Train accuracy, Val accuracy.")
    if not np.isfinite(values).all():
        raise ValueError("history contains a non-finite metric.")
    epochs = np.arange(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    # Compare training and validation in two plots.
    for ax, pair, title in zip(axes, [(0, 1), (2, 3)], ["Loss", "Accuracy"]):
        ax.plot(epochs, values[:, pair[0]], "o-", label="Train", color="#174E66")
        ax.plot(epochs, values[:, pair[1]], "o-", label="Validation", color="#D98043")
        # Mark the selected epoch with a dashed line.
        ax.axvline(best_epoch, color="#778899", linestyle="--", linewidth=1)
        ax.set(xlabel="Epoch", ylabel=title, xticks=epochs)
        ax.grid(alpha=.15)
        ax.legend()
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 1.02)
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    fig.suptitle(f"Measured history | Selected epoch: {best_epoch}", fontsize=12)
    fig.tight_layout()
    # Save and display the figure.
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    display(fig)
    plt.close(fig)


# Show correct predictions and mistakes in a confusion matrix.
# Why: reveal class-specific errors that overall accuracy can hide.
# Use after computing test results; rows are true labels, columns are predictions.
def plot_confusion(cm, class_names, output="assets/confusion_matrix.png"):
    """Rows are ground truth; columns are model predictions."""
    # Rows: true labels. Columns: predicted labels.
    count = len(class_names)
    if np.shape(cm) != (count, count):
        raise ValueError("Confusion matrix shape must match class_names.")
    fig, ax = plt.subplots(figsize=(max(5.6, count * 2), 4.8))
    ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set(xticks=range(count), yticks=range(count), xticklabels=class_names,
           yticklabels=class_names, xlabel="Predicted label", ylabel="True label")
    # Write the image count inside each cell.
    for (row, col), value in np.ndenumerate(cm):
        color = "white" if value > np.max(cm) / 2 else "#17364A"
        ax.text(col, row, str(value), ha="center", va="center", color=color,
                fontsize=22, fontweight="bold")
    fig.tight_layout()
    # Save and display the figure.
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    display(fig)
    plt.close(fig)


# Show one image alongside its predicted class and scores.
# Why: connect the model output to the actual image.
# Use after prediction; this helper displays an existing result.
def show_prediction(image_path, result, output="assets/prediction.png", true_label=None):
    """Display the image and its uncalibrated Softmax scores."""
    image = Image.open(image_path).convert("RGB")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title(f"Dataset label: {true_label}" if true_label else "Input X-ray")
    # Extract class names and scores from the result dictionary.
    names, scores = list(result["scores"]), list(result["scores"].values())
    palette = ["#174E66", "#D98043", "#4C8C75"]
    colors = [palette[i % len(palette)] for i in range(len(names))]
    # Display model scores, not calibrated medical confidence.
    axes[1].barh(names, scores, color=colors)
    axes[1].set(xlim=(0, 1), xlabel="Model score (not calibrated)",
                title=f'Prediction: {result["label"]}')
    for index, score in enumerate(scores):
        axes[1].text(min(score + .02, .77), index, f"{score:.1%}", va="center")
    fig.tight_layout()
    # Save and display the figure.
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    display(fig)
    plt.close(fig)


# Compute Grad-CAM: a coarse map of positive influence on a class score.
# Why: inspect whether highlighted regions lie in the lungs or elsewhere.
# Example: a correct prediction with a bright border region deserves checking.
# Use after prediction; compare both correct and incorrect examples.
# Red means higher activation, blue lower, within this image only.
# The map suggests an explanation; it proves neither causality nor disease location.
def show_gradcam(model, image_path, transform, class_names, device,
                 target_class=None, output="assets/gradcam.png"):
    """Compute actual Grad-CAM from the last MobileNetV2 feature block.

    This coarse class-dependent map is not a lesion mask or a medical diagnosis.
    target_class=None explains the predicted class. A supplied index explains
    that class instead. The function also works when all backbone weights are
    frozen: the input explicitly requires gradients. No model weights change.
    """
    image = Image.open(image_path).convert("RGB")
    # Prepare the image, add a batch dimension, and move it to the device.
    x = transform(image).unsqueeze(0).to(device)
    # Enable input gradients even when the backbone is frozen.
    x.requires_grad_(True)
    # Switch to evaluation mode.
    model.eval()
    # Store captured features in a dictionary.
    captured = {}

    # Keep the last block's features for the Grad-CAM calculation.
    # The hook calls this automatically during the forward pass.
    def capture_features(_module, _inputs, output_tensor):
        captured["features"] = output_tensor

    # Attach a hook to MobileNetV2's last feature block.
    handle = model.features[-1].register_forward_hook(capture_features)
    try:
        with torch.enable_grad():
            logits = model(x)
            # Choose the class whose score we want to explain.
            selected = int(logits.argmax(dim=1).item()) if target_class is None else int(target_class)
            if selected not in range(len(class_names)):
                raise ValueError("target_class must be a valid class index.")
            features = captured["features"]
            # Compute how the selected class score changes with the features.
            gradients = torch.autograd.grad(logits[0, selected], features)[0]
            # Average gradients over space to weight each feature channel.
            channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
            # Combine weighted channels and keep positive contributions.
            cam = (channel_weights * features).sum(dim=1, keepdim=True).relu()
            # Resize the map to match the input image.
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()
            peak = float(cam.max())
            # Scale the map to 0-1; keep an all-zero map if no positive signal exists.
            cam = cam / peak if peak > 0 else np.zeros_like(cam)
    finally:
        # Remove the hook after use.
        handle.remove()
    resized = image.resize((x.shape[-1], x.shape[-2]), Image.Resampling.BILINEAR)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    axes[0].imshow(resized)
    axes[0].set_title("Input X-ray")
    # Blue: lower activation. Red: higher activation within this image.
    heatmap = axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"Grad-CAM: {class_names[selected]}")
    axes[2].imshow(resized)
    # Overlay the heatmap with transparency.
    axes[2].imshow(cam, cmap="jet", vmin=0, vmax=1, alpha=.42)
    axes[2].set_title("Coarse explanatory overlay")
    for ax in axes:
        ax.axis("off")
    colorbar = fig.colorbar(heatmap, ax=list(axes[1:]), shrink=.78, pad=.025)
    colorbar.set_ticks([0, .5, 1])
    colorbar.set_ticklabels(["0  Low", "0.5", "1  High"])
    colorbar.set_label("Relative positive Grad-CAM activation")
    note = ("All-zero map: no positive activation for this image and class"
            if peak == 0 else "Blue = lower, red = higher; normalized within this image")
    fig.suptitle(f"{note}\nA coarse model explanation, not a lesion boundary", fontsize=11)
    # Save and display the figure.
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    display(fig)
    plt.close(fig)
    return {"target_class": class_names[selected], "cam_min": float(cam.min()),
            "cam_max": float(cam.max()), "all_zero": peak == 0}
