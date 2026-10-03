# P2 Transfer Learning RET503

## Image Classification MG90S vs PCA9685 using MobileNetV2

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-green)
![Project](https://img.shields.io/badge/Project-Transfer%20Learning-success)

---

## 📌 Deskripsi Project

Project ini merupakan pengerjaan Praktikum 2 (P2) mata kuliah RET503 dengan topik Transfer Learning untuk Image Classification.

Sistem dikembangkan untuk melakukan klasifikasi citra dua kelas objek, yaitu:

- MG90S
- PCA9685

Model klasifikasi menggunakan arsitektur MobileNetV2 dengan pendekatan Transfer Learning.

Project mencakup proses pengumpulan dataset, preprocessing, pembagian dataset, training model, evaluasi model, pengujian menggunakan data eksternal, serta implementasi klasifikasi citra.

---

## 🎯 Tujuan

Tujuan dari project ini adalah:

1. Memahami konsep Transfer Learning pada Computer Vision.
2. Menggunakan MobileNetV2 untuk melakukan image classification.
3. Membuat dan menyiapkan dataset untuk objek MG90S dan PCA9685.
4. Melakukan preprocessing dan pembagian dataset.
5. Melakukan training model menggunakan dataset yang telah disiapkan.
6. Mengevaluasi performa model.
7. Membandingkan hasil eksperimen model.
8. Melakukan pengujian menggunakan external dataset.
9. Mengimplementasikan model untuk melakukan klasifikasi citra.

---

## 🧠 Metode

### Transfer Learning

Model yang digunakan dalam project ini adalah MobileNetV2.

MobileNetV2 digunakan sebagai model dasar untuk mengekstraksi fitur dari citra. Dengan pendekatan Transfer Learning, fitur yang telah dipelajari oleh model dapat digunakan dan disesuaikan untuk tugas klasifikasi objek MG90S dan PCA9685.

Pipeline sistem:

```text
Input Image
     │
     ▼
Image Preprocessing
     │
     ▼
MobileNetV2
     │
     ▼
Feature Extraction
     │
     ▼
Classification Layer
     │
     ▼
MG90S / PCA9685