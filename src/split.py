from pathlib import Path
import random
import shutil

# ==============================
# KONFIGURASI
# ==============================

SOURCE_DIR = Path("dataset_new")
OUTPUT_DIR = Path("dataset_split_new")

CLASSES = ["MG90S", "PCA9685"]

TRAIN_RATIO = 0.8
SEED = 42

# ==============================
# RANDOM SEED
# ==============================

random.seed(SEED)

# ==============================
# PROSES SPLIT
# ==============================

print("=" * 60)
print("DATASET SPLIT - P2 TRANSFER LEARNING")
print("=" * 60)

for class_name in CLASSES:

    source_class = SOURCE_DIR / class_name

    train_class = OUTPUT_DIR / "train" / class_name
    val_class = OUTPUT_DIR / "val" / class_name

    train_class.mkdir(parents=True, exist_ok=True)
    val_class.mkdir(parents=True, exist_ok=True)

    images = sorted(source_class.glob("*.jpg"))

    if len(images) == 0:
        print(f"[ERROR] Tidak ada gambar di {source_class}")
        continue

    random.shuffle(images)

    train_count = int(len(images) * TRAIN_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:]

    print()
    print(f"Kelas       : {class_name}")
    print(f"Total       : {len(images)}")
    print(f"Train       : {len(train_images)}")
    print(f"Validation  : {len(val_images)}")

    # Copy train
    for image in train_images:
        shutil.copy2(image, train_class / image.name)

    # Copy validation
    for image in val_images:
        shutil.copy2(image, val_class / image.name)

print()
print("=" * 60)
print("SPLIT DATASET SELESAI")
print("=" * 60)
print(f"Output: {OUTPUT_DIR.resolve()}")