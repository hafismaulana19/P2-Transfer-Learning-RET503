import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "mg90s_pca9685_model.keras"
CLASS_FILE = "class_names.txt"

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60


# =========================
# LOAD MODEL
# =========================
print("=" * 60)
print("P2 TRANSFER LEARNING - CAMERA TEST")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_FILE, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

print("Model berhasil dimuat.")
print("Classes:", class_names)


# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Kamera tidak dapat dibuka.")
    exit()

print("\nKamera aktif.")
print("Letakkan objek MG90S atau PCA9685 di depan kamera.")
print("Tekan Q untuk keluar.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Tidak dapat membaca frame.")
        break

    # Resize untuk MobileNetV2
    image = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Ubah ke float tanpa normalisasi tambahan
    image = image.astype(np.float32)

    # Tambahkan batch dimension
    image = np.expand_dims(image, axis=0)

    # =========================
    # PREDICTION
    # =========================
    prediction = model.predict(image, verbose=0)[0]

    class_id = int(np.argmax(prediction))
    confidence = float(prediction[class_id])

    if confidence >= CONFIDENCE_THRESHOLD:
        label = class_names[class_id]
    else:
        label = "Unknown"

    # =========================
    # DISPLAY
    # =========================
    text = f"{label} : {confidence * 100:.2f}%"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("P2 Transfer Learning - Object Detection", frame)

    # Q = keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("\nProgram selesai.")