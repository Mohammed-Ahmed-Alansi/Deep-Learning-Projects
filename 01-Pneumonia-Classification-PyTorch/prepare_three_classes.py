"""Prepare NORMAL / BACTERIAL / VIRAL folders from the supplied image metadata.

The original train/val/test membership and image bytes stay unchanged.
Use prepare_dataset() once at the start of the homework notebook.
"""
from pathlib import Path
import csv
import hashlib
import shutil
import tempfile

CLASSES = ["BACTERIAL", "NORMAL", "VIRAL"]
EXPECTED = {
    "train": {"BACTERIAL": 1771, "NORMAL": 1071, "VIRAL": 992},
    "val": {"BACTERIAL": 404, "NORMAL": 270, "VIRAL": 251},
    "test": {"BACTERIAL": 242, "NORMAL": 234, "VIRAL": 148},
}


# Read the subtype already encoded in the source filename, never from pixels.
def class_from_source(path):
    if path.parent.name == "NORMAL":
        return "NORMAL"
    name = path.name.lower()
    if path.parent.name == "PNEUMONIA" and "_bacteria_" in name and "_virus_" not in name:
        return "BACTERIAL"
    if path.parent.name == "PNEUMONIA" and "_virus_" in name and "_bacteria_" not in name:
        return "VIRAL"
    raise ValueError(f"Unknown source label: {path}. No label was guessed.")


# Copy images into three class folders; safe to rerun without overwriting changes.
def prepare_dataset(source="data/chest_xray", destination="data/chest_xray_3class"):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise FileNotFoundError("Open Jupyter from the supplied project folder.")
    if source == destination or source in destination.parents:
        raise ValueError("Use a separate destination beside the original dataset.")
    records = []
    counts = {split: dict.fromkeys(CLASSES, 0) for split in EXPECTED}
    for split in EXPECTED:
        for old_class in ["NORMAL", "PNEUMONIA"]:
            folder = source / split / old_class
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
                    continue
                new_class = class_from_source(path)
                counts[split][new_class] += 1
                relative = Path(split) / new_class / path.name
                records.append((path, relative, hashlib.sha256(path.read_bytes()).hexdigest()))
    if counts != EXPECTED:
        raise ValueError(f"Counts differ from this classroom dataset: {counts}")
    if len({str(r[1]) for r in records}) != len(records):
        raise ValueError("Duplicate destination names found.")

    # An existing folder must match exactly; never replace a student's edits.
    if destination.exists():
        actual = {p.relative_to(destination) for p in destination.rglob("*") if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}
        if actual != {r[1] for r in records}:
            raise ValueError("Existing three-class folder differs. Choose a new destination.")
        for _, relative, digest in records:
            if hashlib.sha256((destination / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Existing image changed: {relative}. Choose a new destination.")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="three_class_", dir=destination.parent))
        try:
            for original, relative, _ in records:
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
            with (staging / "label_mapping.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["source", "destination", "sha256"])
                writer.writerows((str(a.relative_to(source)), str(b), c) for a, b, c in records)
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    print("Three-class dataset ready. Original images and splits are unchanged.")
    print("Split     BACTERIAL  NORMAL  VIRAL  Total")
    for split, row in counts.items():
        print(f"{split:8}  {row['BACTERIAL']:9}  {row['NORMAL']:6}  {row['VIRAL']:5}  {sum(row.values()):5}")
    return destination


if __name__ == "__main__":
    prepare_dataset()
