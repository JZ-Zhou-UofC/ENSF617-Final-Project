import os
import random
import shutil

# ==== CONFIG ====
images_dir = "processed_data/images"
labels_dir = "processed_data/labels"

output_dir = "yolo_data_set"

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

random.seed(42)  # reproducible split

# ==== GET FILES ====
images = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
images.sort()  # optional before shuffle
random.shuffle(images)

# ==== SPLIT ====
n = len(images)
train_end = int(n * train_ratio)
val_end = int(n * (train_ratio + val_ratio))

train_files = images[:train_end]
val_files = images[train_end:val_end]
test_files = images[val_end:]

# ==== COPY FUNCTION ====
def copy_files(file_list, split):
    os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    for img in file_list:
        label = os.path.splitext(img)[0] + ".txt"

        shutil.copy(os.path.join(images_dir, img),
                    f"{output_dir}/images/{split}/{img}")

        label_path = os.path.join(labels_dir, label)
        if os.path.exists(label_path):
            shutil.copy(label_path,
                        f"{output_dir}/labels/{split}/{label}")
        else:
            print(f"Warning: missing label for {img}")

# ==== EXECUTE ====
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("Done splitting dataset.")