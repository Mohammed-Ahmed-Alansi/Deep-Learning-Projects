# DeepInspect AI: Industrial Anomaly Detection System

DeepInspect AI is an end-to-end industrial visual anomaly detection platform powered by PyTorch. Built using a custom Convolutional Autoencoder (CAE) architecture, it targets high-precision surface defect inspection on the **MVTec AD (Bottle)** dataset.

---

## 📌 Architecture Overview

DeepInspect AI operates on an **unsupervised reconstruction paradigm**:
1. **Training (Normal only)**: The model is trained exclusively on defect-free ("good") samples to learn the nominal distribution of structural textures and geometries.
2. **Inference & Defect Localization**: When an anomalous sample with defects (e.g., broken large/small, contamination) passes through the autoencoder, the model fails to accurately reconstruct the anomalous regions.
3. **Anomaly Map & Scoring**: Pixel-wise reconstruction residual maps (MSE/L1) are smoothed via Gaussian filtering to produce anomaly heatmaps and decision boundaries (AUROC, F1-optimal threshold).

```
   Normal Image ──► [Encoder (Conv2D)] ──► Latent Space ──► [Decoder (ConvTranspose2D)] ──► Reconstruction
                                                                                                  │
                                                                         Input Image ─────────────┴─► [Residual Map] ──► [Anomaly Heatmap]
```

---

## 📁 Project Structure

```text
DeepInspect_AI/
├── configs/
│   └── config.yaml             # Hyperparameters, data paths, model configuration
├── models/                     # Saved model checkpoints and weights
├── notebooks/
│   └── DeepInspect_AI_Colab.ipynb # Interactive Google Colab GPU training notebook
├── outputs/                    # Visualizations, anomaly heatmaps, and metrics
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py          # PyTorch MVTec AD Dataset and DataLoaders
│   ├── models/
│   │   ├── __init__.py
│   │   └── autoencoder.py      # Convolutional Autoencoder Architecture
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py            # Training loop with validation and early stopping
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          # Image-level and Pixel-level AUROC & F1 evaluation
│   └── inference/
│       ├── __init__.py
│       └── predict.py          # Single image reconstruction and heatmap overlay
├── app/
│   └── app.py                  # Interactive Streamlit inspection dashboard
├── tests/
│   ├── __init__.py
│   └── test_data.py            # Data pipeline unit and integration tests
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation and usage guide
```

---

## 🚀 Quickstart Guide

### 1. Environment & Setup

The project is pre-configured to use Python 3.11 with PyTorch (CUDA acceleration enabled). Dependencies are tracked in `requirements.txt`:

```bash
# If running locally
pip install -r requirements.txt
```

### 2. Google Colab Training Workflow

To run training smoothly on Google Colab with GPU acceleration:

```python
# 1. Mount Google Drive or upload project folder
# 2. Install requirements
!pip install -r requirements.txt

# 3. Train on MVTec AD Bottle dataset
!python src/training/train.py --config configs/config.yaml --data_dir /content/data/mvtec_ad/bottle --epochs 50

# 4. Run Benchmark Evaluation
!python src/evaluation/metrics.py --config configs/config.yaml --checkpoint ./models/best_model.pth
```

### 3. Local Streamlit Dashboard

Launch the interactive industrial inspection dashboard:

```bash
streamlit run app/app.py
```

---

## ⚙️ Engineering Principles

- **Zero Data Leakage**: The training split strictly ingests non-defective samples (`train/good/`).
- **Modularity & OOP**: Components are decoupled into clear interfaces with full type hinting and docstrings.
- **Factory Floor Simulation & Video**: High-precision 60 FPS conveyor belt simulation with real-time AI optical scanning and pneumatic ejector arm for defective glass separation (`factory_inspection_simulation.mp4`).
- **Interactive Technical Keynote**: 17-slide standalone defense presentation deck (`DeepInspect_AI_Presentation.html`) with embedded live canvas digital twin, timer, and slide index.

- **Reproducibility**: Global seeding is enforced across PyTorch, NumPy, and Python standard libraries.

