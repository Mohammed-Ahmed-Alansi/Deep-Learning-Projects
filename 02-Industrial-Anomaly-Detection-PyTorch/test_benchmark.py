"""DeepInspect AI — Model Evaluation & Benchmark Comparison.

Compares ConvAutoencoder vs PatchCore on real industrial bottle test samples:
- Inference Latency (ms)
- Anomaly Scores
- Decision Thresholds
- Final Verdict (PASS / REJECT)
- Classification Accuracy
"""
from __future__ import annotations

import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from src.models.autoencoder import ConvAutoencoder, load_compatible_state_dict
from src.models.patchcore import PatchCoreDetector

CAE_PATH = "models/best_model.pth"
PATCHCORE_BANK = "models/patchcore_bank.pkl"

def main():
    print("=" * 75)
    print("🔍 DEEPINSPECT AI — BENCHMARK & COMPARATIVE EVALUATION")
    print("=" * 75)

    # 1. Load Models
    print("\n[1/3] Loading models...")
    # ConvAutoencoder
    cae = ConvAutoencoder(in_channels=3, input_size=(256, 256),
                          channels=[32, 64, 128, 256], latent_dim=128)
    ckpt = torch.load(CAE_PATH, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        load_compatible_state_dict(cae, ckpt["model_state_dict"])
    elif isinstance(ckpt, dict):
        load_compatible_state_dict(cae, ckpt)
    cae.eval()
    print("  ✓ ConvAutoencoder loaded from", CAE_PATH)

    # PatchCore
    patchcore = PatchCoreDetector(device="cpu", subsample_ratio=0.5)
    patchcore.load(PATCHCORE_BANK)
    btype = patchcore.enrolled_types[0]
    thresh_pc = patchcore.get_threshold(btype)
    print(f"  ✓ PatchCore loaded. Enrolled types: {patchcore.enrolled_types}")
    print(f"    Selected type: '{btype}' (Threshold: {thresh_pc:.4f})")

    # 2. Test Samples
    test_suite = [
        {"name": "MVTec Normal Bottle 1", "path": "data/demo_samples/real_good_bottle.png", "label": "NORMAL"},
        {"name": "MVTec Normal Bottle 2", "path": "data/demo_samples/train_good_000.png",   "label": "NORMAL"},
        {"name": "MVTec Broken Glass",     "path": "data/demo_samples/real_broken_bottle.png", "label": "DEFECT"},
        {"name": "MVTec Contaminated",    "path": "data/demo_samples/real_contaminated_bottle.png", "label": "DEFECT"},
    ]

    thresh_cae = 0.0040

    # 3. Benchmark Execution
    print("\n[2/3] Running inference benchmark...")
    print("-" * 75)
    header = f"{'Sample Name':<24} | {'Actual':<7} | {'CAE Score':<10} | {'CAE Ver':<7} | {'PC Score':<10} | {'PC Ver':<7}"
    print(header)
    print("-" * 75)

    cae_latencies = []
    pc_latencies = []
    cae_correct = 0
    pc_correct = 0

    for item in test_suite:
        p = Path(item["path"])
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        actual = item["label"]

        # CAE Inference
        t0 = time.perf_counter()
        resized_cae = img.resize((256, 256), Image.BILINEAR)
        t_tensor = T.ToTensor()(resized_cae).unsqueeze(0)
        with torch.no_grad():
            recon = cae(t_tensor)
            diff = (t_tensor - recon) ** 2
            residual = diff.mean(dim=1).squeeze().numpy()
        smoothed = cv2.GaussianBlur(residual, (7, 7), 4.0, 4.0)
        cae_score = float(np.mean(smoothed))
        cae_lat = (time.perf_counter() - t0) * 1000
        cae_latencies.append(cae_lat)
        cae_verdict = "REJECT" if cae_score >= thresh_cae else "PASS"

        # PatchCore Inference
        t0 = time.perf_counter()
        pc_score, _ = patchcore.predict(img, btype)
        pc_lat = (time.perf_counter() - t0) * 1000
        pc_latencies.append(pc_lat)
        pc_verdict = "REJECT" if pc_score >= thresh_pc else "PASS"

        # Check accuracy
        cae_match = (cae_verdict == "REJECT" and actual == "DEFECT") or (cae_verdict == "PASS" and actual == "NORMAL")
        pc_match = (pc_verdict == "REJECT" and actual == "DEFECT") or (pc_verdict == "PASS" and actual == "NORMAL")
        if cae_match: cae_correct += 1
        if pc_match: pc_correct += 1

        print(f"{item['name']:<24} | {actual:<7} | {cae_score:<10.5f} | {cae_verdict:<7} | {pc_score:<10.4f} | {pc_verdict:<7}")

    print("-" * 75)

    # 4. Summary Metrics
    print("\n[3/3] Performance & Accuracy Summary:")
    print(f"  • ConvAutoencoder:")
    print(f"    - Decision Threshold: {thresh_cae:.5f}")
    print(f"    - Mean Latency:       {np.mean(cae_latencies):.1f} ms / image")
    print(f"    - Accuracy:           {cae_correct}/{len(test_suite)} ({cae_correct/len(test_suite)*100:.1f}%)")

    print(f"  • PatchCore (Multi-Type):")
    print(f"    - Decision Threshold: {thresh_pc:.4f}")
    print(f"    - Mean Latency:       {np.mean(pc_latencies):.1f} ms / image")
    print(f"    - Accuracy:           {pc_correct}/{len(test_suite)} ({pc_correct/len(test_suite)*100:.1f}%)")

    print("\n" + "=" * 75)
    print("✅ Benchmark completed successfully!")
    print("=" * 75)

if __name__ == "__main__":
    main()
