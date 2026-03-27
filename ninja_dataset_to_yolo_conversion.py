import json
from pathlib import Path
import shutil

# ===== CONFIG =====
INPUT_DIR = Path("./raw_data/road-sign-detection-DatasetNinja/ds")
OUTPUT_DIR = Path("./processed_data")

CLASS_MAP = {
    "speedlimit": 0,
    "crosswalk": 1,
    "trafficlight": 2,
    "stop": 3
}
# ==================


def convert_bbox(img_w, img_h, bbox):
    """Convert bbox from absolute (x1,y1,x2,y2) to YOLO format"""
    x1, y1, x2, y2 = bbox

    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h

    return x_center, y_center, width, height


def process():
    # create output folders
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    ann_dir = INPUT_DIR / "ann"
    img_dir = INPUT_DIR / "img"
    
    for ann_path in sorted(ann_dir.glob("*.json")):
        with open(ann_path) as f:
            data = json.load(f)

        # ===== FIX: get image name from JSON safely =====
        img_name = ann_path.name.replace(".json", "")
        img_path = img_dir / img_name

        if not img_name:
            print(f"❌ Missing description in {ann_path}")
            continue

        img_path = img_dir / img_name

        if not img_path.exists():
            print(f"❌ Image not found: {img_path}")
            continue

        if not img_path.is_file():
            print(f"⚠️ Not a file: {img_path}")
            continue

        # ===== image size =====
        img_w = data["size"]["width"]
        img_h = data["size"]["height"]

        label_lines = []

        # ===== process objects =====
        for obj in data.get("objects", []):
            cls_name = obj.get("classTitle", "").lower()

            if cls_name not in CLASS_MAP:
                continue

            class_id = CLASS_MAP[cls_name]

            # bounding box points
            try:
                (x1, y1), (x2, y2) = obj["points"]["exterior"]
            except:
                print(f"⚠️ Bad bbox format in {ann_path}")
                continue

            bbox = convert_bbox(img_w, img_h, (x1, y1, x2, y2))

            label_lines.append(
                f"{class_id} {' '.join(map(str, bbox))}"
            )

        # ===== copy image =====
        dst_img = OUTPUT_DIR / "images" / img_name
        shutil.copy(img_path, dst_img)

        # ===== write label file =====
        label_path = OUTPUT_DIR / "labels" / (Path(img_name).stem + ".txt")

        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

    print("✅ Conversion complete!")


if __name__ == "__main__":
    process()