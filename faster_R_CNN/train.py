import os
import json
import torch
import cv2
import torchvision
import wandb

from torch.utils.data import Dataset, DataLoader, random_split

# =========================
# CONFIG
# =========================
IMG_DIR = r"D:/617 project/raw_data/road-sign-detection-DatasetNinja/ds/img"
ANN_DIR = r"D:/617 project/raw_data/road-sign-detection-DatasetNinja/ds/ann"

CLASS_MAP = {
    "speedlimit": 1,
    "crosswalk": 2,
    "trafficlight": 3,
    "stop": 4
}

NUM_CLASSES = 5  # 4 classes + background
EPOCHS = 20


# =========================
# DATASET
# =========================
class NinjaDataset(Dataset):
    def __init__(self, img_dir, ann_dir):
        self.img_dir = img_dir
        self.ann_dir = ann_dir

        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith(".png")
        ])

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, img_name + ".json")

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        boxes = []
        labels = []

        # Load annotation
        if os.path.exists(ann_path):
            with open(ann_path) as f:
                data = json.load(f)

            for obj in data.get("objects", []):
                x1, y1 = obj["points"]["exterior"][0]
                x2, y2 = obj["points"]["exterior"][1]

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(CLASS_MAP[obj["classTitle"]])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        img = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)

        return img, target

    def __len__(self):
        return len(self.images)


# =========================
# DATALOADER
# =========================
def collate_fn(batch):
    return tuple(zip(*batch))


# =========================
# MODEL
# =========================
def get_model(num_classes=5, freeze_backbone=False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        num_classes
    )

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model


# =========================
# TRAINING
# =========================
def train_model(train_loader, val_loader, epochs=20):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(NUM_CLASSES)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    wandb.init(
        project="faster-rcnn-traffic",
        name="run2",
        mode="offline"
    )
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
                model.train()  # 🔴 temporarily enable loss computation
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                model.eval()   # 🔴 switch back

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        if val_loss < best_val_loss:
         best_val_loss = val_loss
         torch.save(model.state_dict(), "best_model.pth")
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss
        })

    wandb.finish()

    return model


# =========================
# MAIN
# =========================
def main():
    dataset = NinjaDataset(IMG_DIR, ANN_DIR)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    train_model(train_loader, val_loader, EPOCHS)


if __name__ == "__main__":
    main()