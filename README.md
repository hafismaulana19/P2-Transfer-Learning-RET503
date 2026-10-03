\# P2 Transfer Learning RET503



\## Image Classification: MG90S vs PCA9685



Project ini merupakan implementasi Transfer Learning untuk melakukan klasifikasi gambar komponen elektronik menggunakan TensorFlow dan MobileNetV2.



Model digunakan untuk membedakan dua kelas:



\- MG90S

\- PCA9685



\---



\## 1. Tujuan



Membangun model klasifikasi citra menggunakan metode Transfer Learning dengan arsitektur MobileNetV2 untuk mengenali komponen MG90S dan PCA9685.



\---



\## 2. Dataset



Dataset terdiri dari dua kelas:



| Class | Dataset Awal | Dataset Baru |

|---|---:|---:|

| MG90S | 60 | 75 |

| PCA9685 | 60 | 75 |

| \*\*Total\*\* | \*\*120\*\* | \*\*150\*\* |



Dataset baru digunakan untuk proses training model V2.



\---



\## 3. Metode



Model menggunakan:



\- TensorFlow / Keras

\- MobileNetV2

\- ImageNet pretrained weights

\- Image size: 224 × 224 pixels

\- Batch size: 8

\- Epochs: 15

\- Data augmentation

\- Adam optimizer

\- Sparse Categorical Crossentropy



Data augmentation yang digunakan:



\- Random Flip

\- Random Rotation

\- Random Zoom

\- Random Contrast



Base model MobileNetV2 menggunakan pretrained ImageNet weights dan pada tahap training awal dibuat non-trainable.



\---



\## 4. Pembagian Dataset



Dataset dibagi menjadi:



\- 80% training

\- 20% validation



Dataset validation digunakan untuk mengevaluasi performa model selama proses training.



\---



\## 5. Hasil Training



Pada epoch terakhir:



\- Training Accuracy: 98.37%

\- Validation Accuracy: 100%

\- Validation Loss: 0.0285



\### Training Accuracy



!\[Training Accuracy](accuracy\_v2.png)



\### Training Loss



!\[Training Loss](loss\_v2.png)



\---



\## 6. Validation Test



Hasil pengujian validation:



| Class | Correct | Total | Accuracy |

|---|---:|---:|---:|

| MG90S | 11 | 12 | 91.67% |

| PCA9685 | 12 | 12 | 100% |

| \*\*Total\*\* | \*\*23\*\* | \*\*24\*\* | \*\*95.83%\*\* |



\---



\## 7. External Test



Pengujian dilakukan menggunakan 40 gambar eksternal yang tidak digunakan dalam proses training.



| Class | Correct | Total | Accuracy |

|---|---:|---:|---:|

| MG90S | 20 | 20 | 100% |

| PCA9685 | 20 | 20 | 100% |

| \*\*Total\*\* | \*\*40\*\* | \*\*40\*\* | \*\*100%\*\* |



\---



\## 8. Model



Model hasil training disimpan dalam format:



```text

mg90s\_pca9685\_model\_v2.keras

