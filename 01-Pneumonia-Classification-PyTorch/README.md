# Pneumonia Detection & Classification from Chest X-Rays using PyTorch

**Author:** Mohammed Ahmed Al-ansi  

## Overview
This project presents an end-to-end deep learning pipeline for detecting pneumonia from chest X-ray radiographs using **Transfer Learning** with **MobileNetV2**. The project addresses both binary classification (Normal vs. Pneumonia) and 3-class subtype classification (Normal vs. Bacterial Pneumonia vs. Viral Pneumonia), accompanied by model interpretability using **Grad-CAM**.

---

## Key Achievements & Results

### 1. Binary Classification (Normal vs. Pneumonia)
* **Test Accuracy:** 82.4%
* **Pneumonia Sensitivity / Recall:** **99.7%** (Crucial in clinical workflows to minimize false negatives).
* **Fine-Tuning:** Unfreezing the final feature block (`features[-1]`) with a reduced learning rate (`0.0001`) improved validation accuracy from **94.05%** to **95.14%** and decreased validation loss from **0.1436** to **0.1309**.

### 2. Multi-Class Subtype Classification (Normal / Bacterial / Viral)
* **Test Accuracy:** **78.5%**
* **Macro Recall:** **77.8%**
* **Class-Specific Recalls:**
  * **Bacterial Pneumonia:** 94.6% (229 / 242)
  * **Viral Pneumonia:** 74.3% (110 / 148)
  * **Normal:** 64.5% (151 / 234)

---

## Model Interpretability: Grad-CAM
Using **Gradient-weighted Class Activation Mapping (Grad-CAM)**, the model visualizes the spatial regions responsible for its predictions without modifying the network weights. Heatmaps confirm the model attends to opacity and infiltrates in lung fields rather than radiographic artifacts.

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/Mohammed-Ahmed-Alansi/Deep-Learning-Projects.git
cd Deep-Learning-Projects/01-Pneumonia-Classification-PyTorch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter:
```bash
jupyter lab
```
