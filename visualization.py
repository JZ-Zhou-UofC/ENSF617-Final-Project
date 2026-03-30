import os
import cv2
from pathlib import Path

# ===== CONFIG =====
IMAGE_DIR = Path("inference/filtered/images")
LABEL_DIR = Path("inference/filtered/labels")
OUTPUT_DIR = Path("inference/visualized")

CLASS_NAMES = {
    0: "speedlimit",
    1: "crosswalk",
    2: "trafficlight",
    3: "stop"
}
# ==================

def yolo_to_xyxy(x_c, y_c, w, h, img_w, img_h):
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    return x1, y1, x2, y2


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    total = len(image_files)
    print(f"Total images: {total}")

    for i, img_name in enumerate(image_files, start=1):
        img_path = IMAGE_DIR / img_name
        label_path = LABEL_DIR / (Path(img_name).stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipping unreadable: {img_name}")
            continue

        img_h, img_w = img.shape[:2]

        # ===== Draw boxes =====
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue

                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])

                    x1, y1, x2, y2 = yolo_to_xyxy(x_c, y_c, w, h, img_w, img_h)

                    # draw rectangle
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # draw label
                    label = CLASS_NAMES.get(cls_id, str(cls_id))
                    cv2.putText(
                        img,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

        # ===== Save image =====
        out_path = OUTPUT_DIR / img_name
        cv2.imwrite(str(out_path), img)

        print(f"[{i}/{total}] {img_name}")

    print(f"\nDone. Visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()