import os
import cv2
import random
import shutil
from tqdm import tqdm
from collections import defaultdict

# CONFIG
dataset_dirs = [
    'D:\RIN\PRJCT\ISL_Dataset_D',
    'D:\RIN\PRJCT\ISL_Dataset_B',
    'D:\RIN\PRJCT\ISL_Dataset_W',
]

output_dir = 'D:\RIN\PRJCT\ISL_DATASET_C'
max_per_class = 600
img_size = (128, 128)
target_labels = [str(i) for i in range(10)] + [chr(c) for c in range(ord('A'), ord('Z') + 1)]
target_labels_set = set(target_labels)

# CLEAN OUTPUT FOLDER
if os.path.exists(output_dir):
    print(f"⚠️ Clearing existing data in '{output_dir}'...")
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# NORMALIZE LABEL
def normalize_label(label):
    label = label.upper().strip()
    return label if label in target_labels_set else None

# FIND IMAGES GROUPED BY LABEL FROM ALL DATASETS
label_image_map = defaultdict(list)
used_image_paths = set()

print("🔍 Scanning all datasets...")

for dataset in dataset_dirs:
    for root, dirs, files in os.walk(dataset):
        if not files:
            continue

        label = os.path.basename(root)
        norm_label = normalize_label(label)
        if not norm_label:
            continue

        # Collect valid image paths
        for file in files:
            full_path = os.path.join(root, file)
            label_image_map[norm_label].append(full_path)

# SHUFFLE EACH LABEL LIST
for label in label_image_map:
    random.shuffle(label_image_map[label])

# BUILD FINAL DATASET
print("\n🔧 Building combined dataset...")

final_class_counts = defaultdict(int)

for label in tqdm(target_labels):
    saved = 0
    class_dir = os.path.join(output_dir, label)
    os.makedirs(class_dir, exist_ok=True)

    # Try pulling up to max_per_class unique images
    for img_path in label_image_map.get(label, []):
        if img_path in used_image_paths:
            continue  # Already used elsewhere

        img = cv2.imread(img_path)
        if img is None:
            continue

        try:
            img = cv2.resize(img, img_size)
            save_path = os.path.join(class_dir, f"{label}_{saved}.jpg")
            cv2.imwrite(save_path, img)

            used_image_paths.add(img_path)
            saved += 1

            if saved >= max_per_class:
                break
        except Exception as e:
            print(f"Error with {img_path}: {e}")

    if saved < max_per_class:
        print(f"⚠️ WARNING: Class '{label}' only has {saved}/{max_per_class} images.")

    final_class_counts[label] = saved

# SUMMARY
print("\n📊 Final count per class:")
for label in sorted(final_class_counts):
    print(f"{label}: {final_class_counts[label]}")

print("\n✅ Final dataset saved in:", output_dir)



