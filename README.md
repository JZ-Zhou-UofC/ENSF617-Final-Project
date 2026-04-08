# Cross-Dataset Road Sign Detection (YOLO → Faster R-CNN Pipeline)

## 📌 Overview
This project implements a **cross-dataset object detection pipeline** for road sign detection in ADAS systems.

The goal is to improve **out-of-distribution (OOD) generalization** by combining:
- YOLOv8 (high recall)
- Faster R-CNN (high precision)

The system uses:
- DatasetNinja → training dataset
- Mapillary → OOD evaluation dataset

---

## 🚀 Pipeline Summary

1. Train YOLOv8 on DatasetNinja  
2. Use YOLO to run inference on Mapillary images  
3. Filter valid detections (pseudo-labels)  
4. Manually select high-quality samples → OOD test set  
5. Train Faster R-CNN on DatasetNinja  
6. Evaluate Faster R-CNN on OOD dataset  

---

## 📂 Project Structure
├── data.yaml # YOLO dataset config
├── yolo_training.py # Train YOLOv8
├── ninja_dataset_to_yolo_conversion.py
├── yolo_train_val_test_split.py

├── infer.py # YOLO inference on Mapillary
├── filter_valid_image.py # Keep images with detections
├── file_name_change.py # Rename images
├── visualization.py # Visualize bounding boxes

├── hand_picked_test_dataset.py # Extract selected OOD samples

├── frcnn_train_on_ninja_dataset.py # Train Faster R-CNN
├── frcnn_test_on_ninja_dataset.ipynb
├── frcnn_test_on_hand_picked_dataset.ipynb


---

## 🧠 Methodology

## Model Training

### YOLO Training
- Model: YOLOv8s
- Dataset: DatasetNinja
- Epochs: 50
- Output: trained YOLO model
- File_name: YOLO/yolo_training.py

```python
model.train(data="data.yaml", epochs=50, imgsz=640, batch=32)
```

### Faster R-CNN Training

- Model: Backbone: ResNet50-FPN (pretrained) + Custom ROI head (2 FC layers + dropout)
- Dataset: DatasetNinja
- Epochs: 15
- Output: trained frcnn model
- File_name: FRCNN/frcnn_train_on_ninja_dataset.py

Two-phase training:
- Phase 1
Freeze backbone
Train ROI head

- Phase 2
Partially unfreeze backbone
Fine-tune


## Model Testing


