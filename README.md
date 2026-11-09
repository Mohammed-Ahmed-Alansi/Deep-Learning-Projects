# Deep Learning & Computer Vision Projects

A curated repository of deep learning, computer vision, and neural network projects developed with **PyTorch**. This repository highlights practical end-to-end implementations featuring **Transfer Learning**, **Model Interpretability (Explainable AI / Grad-CAM)**, **Unsupervised Anomaly Detection**, and **Industrial Visual Inspection**.

**Author:** Mohammed Ahmed Al-ansi  
**GitHub:** [@Mohammed-Ahmed-Alansi](https://github.com/Mohammed-Ahmed-Alansi)

---

## Projects Directory

| # | Project Name | Task / Domain | Architecture | Key Technologies |
|---|---|---|---|---|
| 01 | [Pneumonia Detection from Chest X-Rays](./01-Pneumonia-Classification-PyTorch) | Medical Image Classification (2-Class & 3-Class) | MobileNetV2 (Pretrained ImageNet) | PyTorch, Transfer Learning, Grad-CAM, Scikit-Learn |
| 02 | [Industrial Anomaly Detection & Defect Localization](./02-Industrial-Anomaly-Detection-PyTorch) | Industrial Visual Quality Inspection (MVTec AD) | Custom Convolutional Autoencoder (CAE) | PyTorch, Unsupervised Reconstruction, AUROC, Heatmaps, Streamlit |

---

## Tech Stack
* **Framework:** PyTorch, Torchvision
* **Model Architectures:** MobileNetV2, Convolutional Autoencoders (CAE), CNNs
* **Evaluation & Metrics:** Image & Pixel AUROC, F1-Score, Confusion Matrix, Precision, Recall/Sensitivity, Specificity
* **Explainable AI & Inspection:** Grad-CAM, Pixel Residual Heatmap Overlay (OpenCV Jet Colormap)
* **Interactive Dashboards:** Streamlit, Gradio
* **Visualization:** Matplotlib, Pillow, Seaborn
