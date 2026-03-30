from ultralytics import YOLO
import os

def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height


def main():
    # ==== Paths ====
    model_path = "inference/best.pt"
    input_dir = "inference/images_resized" 
    output_dir = "inference/labels_yolo"

    os.makedirs(output_dir, exist_ok=True)

    # ==== Load model ====
    model = YOLO(model_path)

    # ==== Get image list ====
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    total = len(image_files)
    print(f"Total images: {total}")

    # ==== Run inference (safe settings) ====
    results = model(
        input_dir,
        batch=1,
        stream=True,
        imgsz=640,
        verbose=False
    )

    # ==== Process results ====
    for i, r in enumerate(results, start=1):
        image_name = os.path.basename(r.path)
        label_name = os.path.splitext(image_name)[0] + ".txt"

        label_lines = []

        # ✅ use resized image size (correct)
        img_h, img_w = r.orig_img.shape[:2]

        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                x_c, y_c, w, h = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)

                label_lines.append(
                    f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                )

        # ==== Save label file (empty if no detections) ====
        output_path = os.path.join(output_dir, label_name)
        with open(output_path, "w") as f:
            f.write("\n".join(label_lines))

        # ==== Progress ====
        print(f"[{i}/{total}] {image_name}")

    print(f"\nDone. Labels saved to: {output_dir}")


if __name__ == "__main__":
    main()