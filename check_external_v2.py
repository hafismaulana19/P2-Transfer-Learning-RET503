import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras.utils import load_img, img_to_array

MODEL_PATH = "mg90s_pca9685_model_v2.keras"
TEST_DIR = Path("external_test")

IMG_SIZE = (224, 224)

CLASSES = ["MG90S", "PCA9685"]

print("=" * 70)
print("EXTERNAL TEST - MODEL V2")
print("=" * 70)

model = tf.keras.models.load_model(MODEL_PATH)

total = 0
correct = 0

class_correct = {
    "MG90S": 0,
    "PCA9685": 0
}

class_total = {
    "MG90S": 0,
    "PCA9685": 0
}

for true_class in CLASSES:

    files = sorted(
        list((TEST_DIR / true_class).glob("*.jpg")) +
        list((TEST_DIR / true_class).glob("*.jpeg")) +
        list((TEST_DIR / true_class).glob("*.png"))
    )

    print("\n" + "-" * 70)
    print(f"TRUE CLASS: {true_class}")
    print("-" * 70)

    for path in files:

        img = load_img(path, target_size=IMG_SIZE)
        img = img_to_array(img)

        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)[0]

        predicted_index = np.argmax(prediction)

        predicted_class = CLASSES[predicted_index]

        mg90s = prediction[0] * 100
        pca9685 = prediction[1] * 100

        is_correct = predicted_class == true_class

        if is_correct:
            correct += 1
            class_correct[true_class] += 1

        total += 1
        class_total[true_class] += 1

        status = "BENAR" if is_correct else "SALAH"

        print(
            f"{path.name}"
            f" -> MG90S: {mg90s:6.2f}%"
            f" | PCA9685: {pca9685:6.2f}%"
            f" | Prediksi: {predicted_class}"
            f" | {status}"
        )

# ============================================================
# HASIL
# ============================================================

print("\n" + "=" * 70)
print("HASIL EXTERNAL TEST")
print("=" * 70)

for cls in CLASSES:

    acc = (
        class_correct[cls] /
        class_total[cls] *
        100
    )

    print(
        f"{cls:8s}: "
        f"{class_correct[cls]}/{class_total[cls]} "
        f"= {acc:.2f}%"
    )

overall = correct / total * 100

print("-" * 70)
print(
    f"TOTAL    : "
    f"{correct}/{total} "
    f"= {overall:.2f}%"
)

print("=" * 70)