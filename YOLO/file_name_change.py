import os

def rename_images(folder_path, prefix="Mapillary", start_index=1):
    # Get all files in folder
    files = os.listdir(folder_path)

    # Keep only image files (adjust extensions if needed)
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in files if f.lower().endswith(image_exts)]

    # Sort to ensure consistent ordering
    images.sort()

    # First pass: rename to temporary names (avoid overwrite issues)
    temp_names = []
    for i, filename in enumerate(images):
        old_path = os.path.join(folder_path, filename)
        temp_name = f"temp_{i}{os.path.splitext(filename)[1]}"
        temp_path = os.path.join(folder_path, temp_name)

        os.rename(old_path, temp_path)
        temp_names.append(temp_name)

    # Second pass: rename to final names
    for i, temp_name in enumerate(temp_names, start=start_index):
        ext = os.path.splitext(temp_name)[1]
        new_name = f"{prefix}{i}{ext}"

        old_path = os.path.join(folder_path, temp_name)
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)

    print(f"Renamed {len(images)} images.")


if __name__ == "__main__":
    folder = r".\raw_data\test\images"
    rename_images(folder)