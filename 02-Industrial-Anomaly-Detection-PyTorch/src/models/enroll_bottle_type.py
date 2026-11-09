"""Script to enroll bottle types into PatchCore memory bank.

الاستخدام:
    # تسجيل نوع زجاجة جديد باستخدام صور من مجلد
    python src/models/enroll_bottle_type.py --type "زجاجة_بيرة" --images_dir path/to/normal/images

    # الاستخدام التفاعلي عبر التطبيق (انظر تبويب "إضافة نوع زجاجة" في Streamlit)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from src.models.patchcore import PatchCoreDetector

BANK_PATH = "models/patchcore_bank.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll bottle type into PatchCore memory bank.")
    parser.add_argument("--type", required=True, help="Name/label for this bottle type.")
    parser.add_argument("--images_dir", required=True, help="Folder of NORMAL (defect-free) images.")
    parser.add_argument("--bank", default=BANK_PATH, help="Path to save/load memory bank.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Auto-calibrate threshold from provided images.")
    args = parser.parse_args()

    detector = PatchCoreDetector()

    # Load existing bank if it exists
    if Path(args.bank).exists():
        detector.load(args.bank)

    # Load images
    img_dir = Path(args.images_dir)
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in supported]
    if not image_paths:
        print(f"ERROR: No images found in {img_dir}")
        sys.exit(1)

    images = [Image.open(p).convert("RGB") for p in image_paths]
    print(f"Loaded {len(images)} images from {img_dir}")

    # Enroll
    detector.enroll(args.type, images, verbose=True)

    if args.calibrate:
        t = detector.calibrate_threshold(args.type, images)
        print(f"Threshold for '{args.type}': {t:.4f}")

    # Save updated bank
    Path(args.bank).parent.mkdir(parents=True, exist_ok=True)
    detector.save(args.bank)
    print(f"Done! Enrolled types: {detector.enrolled_types}")


if __name__ == "__main__":
    main()

