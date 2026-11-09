"""DeepInspect AI - Streamlit Industrial Anomaly Detection Dashboard.

This application provides a web-based graphical user interface for visual defect
inspection of MVTec AD samples using a custom PyTorch Convolutional Autoencoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torchvision.transforms as T

from src.models.autoencoder import ConvAutoencoder, load_compatible_state_dict


# ==============================================================================
# 1. Page Configuration & Title
# ==============================================================================
st.set_page_config(
    page_title="DeepInspect AI: Industrial Anomaly Detection",
    page_icon="🔍",
    layout="wide",
)

st.title("DeepInspect AI: Industrial Anomaly Detection")
st.markdown(
    "Automated surface defect inspection and localization powered by a custom "
    "**PyTorch Convolutional Autoencoder** on the **MVTec AD** dataset."
)
st.markdown("---")


# ==============================================================================
# 2. Model Loading with CPU Fallback (@st.cache_resource)
# ==============================================================================
@st.cache_resource(show_spinner="Loading best model checkpoint from disk...")
def load_cached_model(checkpoint_path: str = "models/best_model.pth") -> Optional[ConvAutoencoder]:
    """Load model checkpoint with explicit CPU mapping for local execution."""
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        return None

    # Load default architecture configuration or instantiate model
    model = ConvAutoencoder(
        in_channels=3,
        input_size=(256, 256),
        channels=[32, 64, 128, 256],
        latent_dim=128,
    )

    # Explicitly map checkpoint to CPU
    checkpoint = torch.load(ckpt_file, map_location=torch.device("cpu"))

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        load_compatible_state_dict(model, checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        load_compatible_state_dict(model, checkpoint)

    model.eval()
    return model


# ==============================================================================
# 3. Sidebar Configuration & Thresholding
# ==============================================================================
st.sidebar.header("⚙️ Inspection Settings")

checkpoint_path = st.sidebar.text_input("Model Path", value="models/best_model.pth")
model = load_cached_model(checkpoint_path)

if model is not None:
    st.sidebar.success(f"✅ Model loaded from `{checkpoint_path}`")
else:
    st.sidebar.warning(f"⚠️ `{checkpoint_path}` not found. (Using baseline autoencoder for demonstration)")
    # Fallback to fresh autoencoder if checkpoint not yet generated
    model = ConvAutoencoder(
        in_channels=3,
        input_size=(256, 256),
        channels=[32, 64, 128, 256],
        latent_dim=128,
    )
    model.eval()

# Optimal anomaly threshold computed from evaluation (F1-score optimization)
predefined_threshold = 0.0452
threshold = st.sidebar.slider(
    "Anomaly Score Threshold",
    min_value=0.0010,
    max_value=0.0800,
    value=predefined_threshold,
    step=0.0010,
    format="%.4f",
    help="Images with reconstruction error >= threshold are classified as defective.",
)

heatmap_alpha = st.sidebar.slider(
    "Heatmap Overlay Blend (Alpha)",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Train the model on Google Colab or locally using `python src/training/train.py`.")


# ==============================================================================
# 4. Image Preprocessing & Inference Functions
# ==============================================================================
def preprocess_image(pil_img: Image.Image, target_size: Tuple[int, int] = (256, 256)) -> Tuple[torch.Tensor, np.ndarray]:
    """Preprocess uploaded image: resize to 256x256, convert to RGB array and normalized tensor."""
    resized_pil = pil_img.convert("RGB").resize(target_size, Image.BILINEAR)
    orig_np = np.array(resized_pil, dtype=np.uint8)

    transform = T.Compose([
        T.ToTensor(),  # Scales pixel values to [0.0, 1.0]
    ])
    tensor_img = transform(resized_pil).unsqueeze(0)  # Shape: (1, 3, 256, 256)
    return tensor_img, orig_np


def generate_anomaly_heatmap(
    orig_np: np.ndarray,
    residual_error: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute smoothed anomaly heatmap and alpha-blended overlay using OpenCV.

    Args:
        orig_np: Original image array (H, W, 3) in [0, 255].
        residual_error: 2D Mean Squared Error array (H, W).
        alpha: Blending weight for heatmap overlay.

    Returns:
        Tuple of (heatmap_rgb, overlay_rgb, scalar_anomaly_score).
    """
    # 1. Apply Gaussian Blur to smooth out sensor noise
    smoothed_error = cv2.GaussianBlur(residual_error, (7, 7), sigmaX=4.0, sigmaY=4.0)

    # Anomaly score is the maximum residual error peak
    anomaly_score = float(np.max(smoothed_error))

    # 2. Normalize smoothed error to [0, 255] for colormap visualization
    res_min, res_max = smoothed_error.min(), smoothed_error.max()
    if res_max > res_min:
        norm_map = ((smoothed_error - res_min) / (res_max - res_min) * 255.0).astype(np.uint8)
    else:
        norm_map = np.zeros_like(smoothed_error, dtype=np.uint8)

    # 3. Apply Jet colormap
    heatmap_bgr = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 4. Overlay heatmap onto the original image
    overlay = cv2.addWeighted(orig_np, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return heatmap_rgb, overlay, anomaly_score


# ==============================================================================
# 5. User Upload & Main Execution
# ==============================================================================
sample_options = {
    "🍾 زجاجة كوكاكولا زجاجية كاملة (تصوير حقيقي من الجانب)": "data/demo_samples/full_coca_cola_bottle.png",
    "🍾 زجاجة خضراء كاملة (تصوير حقيقي من الجانب)": "data/demo_samples/full_green_bottle.jpg",
    "🔴 فوهة زجاجة مكسورة (كاميرا صناعية حقيقية)": "data/demo_samples/real_broken_bottle.png",
    "🔴 فوهة زجاجة ملوثة بشوائب (كاميرا صناعية حقيقية)": "data/demo_samples/real_contaminated_bottle.png",
    "🟢 فوهة زجاجة سليمة (كاميرا صناعية حقيقية)": "data/demo_samples/real_good_bottle.png",
    "📁 رفع صورة من جهازي (Upload Custom Image)": None,
}

selected_label = st.selectbox(
    "🧪 اختر عينة زجاجة جاهزة للفحص الفوري أو اختر رفع صورة من جهازك:",
    options=list(sample_options.keys()),
    index=0,
)

sample_path = sample_options[selected_label]
uploaded_file = None

if sample_path is None:
    uploaded_file = st.file_uploader(
        "Upload a bottle inspection image to test for defects (PNG / JPG):",
        type=["png", "jpg", "jpeg"],
    )

if uploaded_file is not None or (sample_path and Path(sample_path).exists()):
    # Load and preprocess image
    if uploaded_file is not None:
        input_pil = Image.open(uploaded_file)
    else:
        input_pil = Image.open(sample_path)
    input_tensor, orig_np = preprocess_image(input_pil, target_size=(256, 256))

    # Pass image through autoencoder model
    with torch.no_grad():
        reconstructed_tensor = model(input_tensor)

        # Pixel-level Mean Squared Error across color channels: shape (256, 256)
        diff_sq = (input_tensor - reconstructed_tensor) ** 2
        residual_error = torch.mean(diff_sq, dim=1).squeeze(0).cpu().numpy()

        # Reconstructed image array for display
        recon_np = (
            reconstructed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0) * 255.0
        ).astype(np.uint8)

    # Generate anomaly heatmap and overlay
    heatmap_rgb, overlay_rgb, anomaly_score = generate_anomaly_heatmap(
        orig_np=orig_np,
        residual_error=residual_error,
        alpha=heatmap_alpha,
    )

    is_anomaly = anomaly_score >= threshold

    # --------------------------------------------------------------------------
    # 6. Visual Status Alert (Success / Error)
    # --------------------------------------------------------------------------
    st.markdown("### Inspection Status")
    if is_anomaly:
        st.error(
            f"🚨 **DEFECT DETECTED (REJECT)** — Anomaly Score: **`{anomaly_score:.5f}`** "
            f"exceeds threshold (`{threshold:.5f}`)."
        )
    else:
        st.success(
            f"✅ **NORMAL SAMPLE (PASS)** — Anomaly Score: **`{anomaly_score:.5f}`** "
            f"is within acceptable threshold (`{threshold:.5f}`)."
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 7. Side-by-Side 3-Column Display
    # --------------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Original Image")
        st.image(orig_np, caption="Input Surface Image (256x256)", use_container_width=True)

    with col2:
        st.subheader("2. Reconstructed Image")
        st.image(recon_np, caption="Autoencoder Output", use_container_width=True)

    with col3:
        st.subheader("3. Anomaly Heatmap Overlay")
        st.image(overlay_rgb, caption="Defect Localization Heatmap", use_container_width=True)

    # Telemetry metrics row
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Calculated Anomaly Score", f"{anomaly_score:.5f}")
    m2.metric("Decision Threshold", f"{threshold:.5f}")
    m3.metric("Final Verdict", "DEFECTIVE (REJECT)" if is_anomaly else "NORMAL (PASS)")

else:
    st.info("👆 Please upload an image file (PNG/JPG) using the uploader above to begin inspection.")
