import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import shutil
import random
import matplotlib.pyplot as plt

# ============================================================
# P2 TRANSFER LEARNING V2
# MG90S vs PCA9685
#
# DATASET LAMA : 60 + 60
# DATASET BARU : 75 + 75
#
# 20 gambar dataset_new per kelas disimpan sebagai
# EXTERNAL TEST dan TIDAK ikut training.
#
# Sisa:
# 60 lama + 55 baru = 115 per kelas
#
# Kemudian split:
# 80% TRAIN
# 20% VALIDATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OLD_DATASET = BASE_DIR / "dataset"
NEW_DATASET = BASE_DIR / "dataset_new"

SPLIT_DIR = BASE_DIR / "dataset_split_v2"
EXTERNAL_DIR = BASE_DIR / "external_test"

MODEL_PATH = BASE_DIR / "mg90s_pca9685_model_v2.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 15
SEED = 42

CLASSES = ["MG90S", "PCA9685"]

random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print("TRANSFER LEARNING V2 - MG90S vs PCA9685")
print("=" * 70)

# ============================================================
# HAPUS SPLIT LAMA
# ============================================================

if SPLIT_DIR.exists():
    shutil.rmtree(SPLIT_DIR)

if EXTERNAL_DIR.exists():
    shutil.rmtree(EXTERNAL_DIR)

# ============================================================
# BUAT FOLDER
# ============================================================

for split in ["train", "val"]:
    for cls in CLASSES:
        (SPLIT_DIR / split / cls).mkdir(
            parents=True,
            exist_ok=True
        )

for cls in CLASSES:
    (EXTERNAL_DIR / cls).mkdir(
        parents=True,
        exist_ok=True
    )

# ============================================================
# SIAPKAN DATASET
# ============================================================

print("\nMENYIAPKAN DATASET...")

for cls in CLASSES:

    old_files = (
        list((OLD_DATASET / cls).glob("*.jpg")) +
        list((OLD_DATASET / cls).glob("*.jpeg")) +
        list((OLD_DATASET / cls).glob("*.png"))
    )

    new_files = (
        list((NEW_DATASET / cls).glob("*.jpg")) +
        list((NEW_DATASET / cls).glob("*.jpeg")) +
        list((NEW_DATASET / cls).glob("*.png"))
    )

    random.shuffle(new_files)

    # --------------------------------------------------------
    # 20 DATASET BARU -> EXTERNAL TEST
    # --------------------------------------------------------

    external_files = new_files[:20]
    remaining_new = new_files[20:]

    for img in external_files:
        shutil.copy2(
            img,
            EXTERNAL_DIR / cls / img.name
        )

    # --------------------------------------------------------
    # DATA TRAINING POOL
    # 60 lama + 55 baru = 115
    # --------------------------------------------------------

    combined = old_files + remaining_new

    random.shuffle(combined)

    split_index = int(len(combined) * 0.8)

    train_files = combined[:split_index]
    val_files = combined[split_index:]

    print(f"\n{cls}")
    print(f"  Dataset lama       : {len(old_files)}")
    print(f"  Dataset baru       : {len(new_files)}")
    print(f"  External test      : {len(external_files)}")
    print(f"  Training pool      : {len(combined)}")
    print(f"  Train              : {len(train_files)}")
    print(f"  Validation         : {len(val_files)}")

    for img in train_files:
        shutil.copy2(
            img,
            SPLIT_DIR / "train" / cls / img.name
        )

    for img in val_files:
        shutil.copy2(
            img,
            SPLIT_DIR / "val" / cls / img.name
        )

# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOAD DATASET")
print("=" * 70)

train_ds = tf.keras.utils.image_dataset_from_directory(
    SPLIT_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    SPLIT_DIR / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names

print("\nCLASS:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
])

# ============================================================
# MOBILENETV2
# ============================================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# ============================================================
# BUILD MODEL
# ============================================================

inputs = keras.Input(
    shape=IMG_SIZE + (3,)
)

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(
    inputs,
    outputs
)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nMODEL:")
model.summary()

# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("START TRAINING V2")
print("=" * 70)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============================================================
# SAVE MODEL
# ============================================================

model.save(MODEL_PATH)

print("\nModel berhasil disimpan:")
print(MODEL_PATH)

# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_file = BASE_DIR / "class_names_v2.txt"

with open(
    class_file,
    "w",
    encoding="utf-8"
) as f:
    for name in class_names:
        f.write(name + "\n")

print("Class names:")
print(class_file)

# ============================================================
# GRAPH ACCURACY
# ============================================================

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy - V2")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

accuracy_path = BASE_DIR / "accuracy_v2.png"

plt.savefig(
    accuracy_path,
    dpi=150
)

plt.close()

print("Grafik accuracy:")
print(accuracy_path)

# ============================================================
# GRAPH LOSS
# ============================================================

plt.figure()

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss - V2")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

loss_path = BASE_DIR / "loss_v2.png"

plt.savefig(
    loss_path,
    dpi=150
)

plt.close()

print("Grafik loss:")
print(loss_path)

# ============================================================
# SELESAI
# ============================================================

print("\n" + "=" * 70)
print("TRAINING V2 SELESAI")
print("=" * 70)