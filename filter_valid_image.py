import os
import shutil
from pathlib import Path

# ===== CONFIG =====
IMAGE_DIR = Path("inference/images_resized")
LABEL_DIR = Path("inference/labels_yolo")

OUTPUT_DIR = Path("inference/filtered")
OUT_IMG_DIR = OUTPUT_DIR / "images"
OUT_LBL_DIR = OUTPUT_DIR / "labels"

# ==================

def has_objects(label_path):
    """Return True if label file contains at least one object"""
    if not label_path.exists():
        return False

    with open(label_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    # remove empty lines
    lines = [l for l in lines if l]

    return len(lines) > 0


def main():
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    kept = 0
    total = len(image_files)

    for i, img_name in enumerate(image_files, start=1):
        img_path = IMAGE_DIR / img_name
        label_path = LABEL_DIR / (Path(img_name).stem + ".txt")

        if has_objects(label_path):
            # copy image
            shutil.copy(img_path, OUT_IMG_DIR / img_name)

            # copy label
            shutil.copy(label_path, OUT_LBL_DIR / label_path.name)

            kept += 1

        print(f"[{i}/{total}] {img_name}")

    print("\nDone.")
    print(f"Kept {kept}/{total} images with detections.")


if __name__ == "__main__":
    main()