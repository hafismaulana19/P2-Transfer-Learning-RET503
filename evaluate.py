import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# P2 TRANSFER LEARNING - MODEL EVALUATION
# ============================================================

MODEL_PATH = "mg90s_pca9685_model.keras"
VAL_DIR = Path("dataset_split_new/val")
IMG_SIZE = (224, 224)

CLASS_NAMES = ["MG90S", "PCA9685"]

print("=" * 60)
print("P2 TRANSFER LEARNING - MODEL EVALUATION")
print("=" * 60)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Model berhasil dimuat.")


# ------------------------------------------------------------
# Evaluasi setiap kelas
# ------------------------------------------------------------

confusion_matrix = np.zeros((2, 2), dtype=int)

total = 0
correct = 0

for true_id, class_name in enumerate(CLASS_NAMES):

    class_dir = VAL_DIR / class_name

    if not class_dir.exists():
        print(f"\nERROR: Folder tidak ditemukan: {class_dir}")
        continue

    image_files = []

    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(class_dir.glob(ext))

    print(f"\nKelas: {class_name}")
    print(f"Jumlah gambar: {len(image_files)}")

    class_correct = 0

    for image_path in image_files:

        img = tf.keras.utils.load_img(
            image_path,
            target_size=IMG_SIZE
        )

        img = tf.keras.utils.img_to_array(img)

        # Tambahkan batch dimension
        img = np.expand_dims(img, axis=0)

        # Prediction
        prediction = model.predict(img, verbose=0)

        predicted_id = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_id])

        # Simpan confusion matrix
        confusion_matrix[true_id, predicted_id] += 1

        total += 1

        if predicted_id == true_id:
            correct += 1
            class_correct += 1

    # Accuracy kelas
    if len(image_files) > 0:
        accuracy = class_correct / len(image_files) * 100
    else:
        accuracy = 0

    print(
        f"Accuracy {class_name}: "
        f"{class_correct}/{len(image_files)} "
        f"({accuracy:.2f}%)"
    )

# ============================================================
# HASIL AKHIR
# ============================================================

print("\n" + "=" * 60)
print("HASIL EVALUASI")

overall_accuracy = correct / total * 100

print(f"\nTotal gambar      : {total}")
print(f"Prediksi benar    : {correct}")
print(f"Prediksi salah    : {total - correct}")
print(f"Accuracy          : {overall_accuracy:.2f}%")

print("\nConfusion Matrix:")
print("                 Predicted")
print("               MG90S  PCA9685")
print(
    f"Actual MG90S    {confusion_matrix[0,0]:5d}"
    f"  {confusion_matrix[0,1]:7d}"
)
print(
    f"Actual PCA9685  {confusion_matrix[1,0]:5d}"
    f"  {confusion_matrix[1,1]:7d}"
)


# ============================================================
# CONFUSION MATRIX GRAPH
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

ax.imshow(confusion_matrix)

ax.set_title(
    f"Confusion Matrix - P2 Transfer Learning\n"
    f"Accuracy: {overall_accuracy:.2f}%"
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(CLASS_NAMES)
ax.set_yticklabels(CLASS_NAMES)

# Tampilkan angka pada matrix
for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(confusion_matrix[i, j]),
            ha="center",
            va="center",
            fontsize=18
        )

plt.tight_layout()

output_path = "confusion_matrix.png"
plt.savefig(output_path, dpi=150)

print(f"\nConfusion matrix disimpan:")
print(f"{Path(output_path).resolve()}")

print("\nEvaluasi selesai.")