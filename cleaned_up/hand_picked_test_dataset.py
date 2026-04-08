import os
import shutil

# --- your list ---
ids = [
    1722,1736,1742,1755,1780,1791,1793,1800,1821,1832,1836,
    1860,1871,1876,1891,1904,1944,2096,2116,2156,2165,2207,
    2212,2221,2230,2264,2266,2947,2967,3033,3038,3065,3140,
    3219,3277,3322,3333,3381,3479,3498,3604,4572,4645,4689,
    4702,4715,4740,4765,4790,4798
]

# convert to set for faster lookup
id_set = set(ids)

# --- paths ---
source_dir = r"D:\617 project\inference\filtered_new\labels"
target_dir = r"D:\617 project\cleaned_up\dataset\hand_picked\labels"

os.makedirs(target_dir, exist_ok=True)

# --- copy matching files ---
for filename in os.listdir(source_dir):

    # example: Mapillary4798.jpg
    if filename.startswith("Mapillary"):
        try:
            number = int(filename.replace("Mapillary", "").split(".")[0])

            if number in id_set:
                src_path = os.path.join(source_dir, filename)
                dst_path = os.path.join(target_dir, filename)

                shutil.copy2(src_path, dst_path)
                print(f"Copied: {filename}")

        except ValueError:
            continue

print("Done.")