import os
import cv2

input_dir =  r"D:/617 project/cleaned_up/dataset/hand_picked/images"
output_dir = r"D:/617 project/cleaned_up/dataset/hand_picked/image_png"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg")):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed: {filename}")
            continue

        new_name = os.path.splitext(filename)[0] + ".png"
        save_path = os.path.join(output_dir, new_name)

        cv2.imwrite(save_path, img)
        print(f"Converted: {filename} → {new_name}")

print("Done.")