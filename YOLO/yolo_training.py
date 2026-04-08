

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import wandb
import torch


def main():
    # ==== 1. Check GPU ====
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())

    # ==== 2. Initialize W&B ====
    wandb.init(
        project="yolo-training",
        name="yolov8s-run1",
        config={
            "epochs": 50,
            "imgsz": 640,
            "batch": 32,
            "model": "yolov8s.pt"
        }
    )

    # ==== 3. Load Model ====
    model = YOLO("yolov8s.pt")

    # ==== 4. Train ====
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=32,
        device=0,
        project="runs",
        name="detect_exp",
        exist_ok=True,

        # performance tweaks
        cache=True,
        workers=4,
        verbose=True
    )

    # ==== 5. Validate ====
    metrics = model.val(data="data.yaml", split="test")

    # ==== 6. Print clean results ====
    print("\n=== Validation Results ===")
    print(f"mAP50       : {metrics.box.map50:.4f}")
    print(f"mAP50-95    : {metrics.box.map:.4f}")
    print(f"Precision   : {metrics.box.mp:.4f}")
    print(f"Recall      : {metrics.box.mr:.4f}")

    # ==== 7. Log metrics to W&B ====
    wandb.log({
        "test/mAP50": metrics.box.map50,
        "test/mAP50-95": metrics.box.map,
        "test/precision": metrics.box.mp,
        "test/recall": metrics.box.mr
    })

    # ==== 8. Log per-class AP ====
    print("\n=== Per-Class AP50 ===")
    for i, name in model.names.items():
        ap = metrics.box.ap50[i]
        print(f"{name}: {ap:.4f}")
        wandb.log({f"AP50/{name}": ap})

    # ==== 9. Optional: log sample prediction ====
    sample_image = "test.jpg"  # change to your test image
    if os.path.exists(sample_image):
        results = model(sample_image)
        for r in results:
            wandb.log({
                "prediction": wandb.Image(r.plot())
            })

    # ==== 10. Finish W&B ====
    wandb.finish()


if __name__ == "__main__":
    main()
