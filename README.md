# 🤖 P2 Transfer Learning — RET503

### Computer Vision and Deep Learning
**Mata Kuliah:** RET503 — Computer Vision and Deep Learning  
**Program Studi:** Teknologi Rekayasa Robotika  
**Institusi:** Politeknik Negeri Batam  

> Implementasi transfer learning dan fine-tuning untuk klasifikasi komponen robot menggunakan dataset **MG90S** dan **PCA9685**.

---

## 📌 Deskripsi

Project ini merupakan implementasi **Transfer Learning dan Fine-Tuning Model Visi** untuk tugas P2 mata kuliah **RET503 Computer Vision and Deep Learning**.

Model digunakan untuk melakukan klasifikasi citra dua jenis komponen robot:

- 🔧 **MG90S Servo**
- 🔌 **PCA9685 Servo Driver**

Eksperimen utama menggunakan **ResNet18 pretrained ImageNet** dengan tiga strategi pembelajaran:

1. **Feature Extraction**
2. **Partial Fine-Tuning**
3. **Training from Scratch**

Selain eksperimen utama tersebut, project juga dilengkapi dengan implementasi **MobileNetV2** dan pengujian kamera secara real-time.

Tujuan akhirnya adalah mendapatkan model yang memiliki kombinasi **akurasi yang baik dan latency yang sesuai untuk aplikasi robotika**.

---

## 🎯 Tujuan

Project ini bertujuan untuk:

- Memahami konsep transfer learning pada computer vision.
- Menggunakan model CNN pretrained untuk klasifikasi objek.
- Membandingkan Feature Extraction, Partial Fine-Tuning, dan Training from Scratch.
- Menganalisis pengaruh pretrained weights terhadap performa model.
- Membandingkan performa ResNet18 dan MobileNetV2.
- Mengukur performa model berdasarkan accuracy dan inference latency.
- Menguji model menggunakan data external yang tidak digunakan selama training.
- Mengimplementasikan model untuk real-time camera classification.

---

## 🧠 Konsep Transfer Learning

Transfer learning memanfaatkan pengetahuan yang telah diperoleh model dari
dataset sumber untuk menyelesaikan tugas baru pada dataset target.

Pada project ini, model **ResNet18 pretrained ImageNet** digunakan sebagai
backbone untuk klasifikasi dua jenis komponen robot, yaitu MG90S dan PCA9685.

```text
                 PRETRAINED MODEL
                    ImageNet
                       │
                       ▼
                ┌─────────────┐
                │  ResNet18   │
                └─────────────┘
                       │
                       ▼
             Dataset Komponen Robot
                       │
              ┌────────┴────────┐
              ▼                 ▼
           MG90S             PCA9685


## 📊 Analisis Hasil ResNet18

Berdasarkan hasil eksperimen tiga strategi pembelajaran pada ResNet18,
diperoleh hasil sebagai berikut:

| Metode | Accuracy |
|---|---:|
| Feature Extraction | 97.50% |
| Partial Fine-Tuning | 97.50% |
| Training from Scratch | 75.00% |

Feature Extraction dan Partial Fine-Tuning menghasilkan accuracy yang sama,
yaitu **97.50%**, sedangkan Training from Scratch menghasilkan accuracy
sebesar **75.00%**.

Hasil tersebut menunjukkan bahwa penggunaan pretrained weights dari ImageNet
memberikan keuntungan pada dataset yang digunakan. Dengan memanfaatkan fitur
yang telah dipelajari sebelumnya, model dapat memperoleh performa yang lebih
baik dibandingkan model yang dilatih dari awal.

Pada eksperimen ini, Partial Fine-Tuning belum memberikan peningkatan accuracy
dibandingkan Feature Extraction. Hal tersebut menunjukkan bahwa membuka
sebagian layer backbone belum memberikan keuntungan tambahan pada dataset dan
konfigurasi training yang digunakan.

Sebaliknya, Training from Scratch menghasilkan accuracy yang lebih rendah.
Hal ini menunjukkan bahwa pada dataset dengan jumlah data yang relatif
terbatas, penggunaan pretrained model dapat membantu proses pembelajaran
fitur visual secara lebih efektif.

Dengan demikian, hasil eksperimen ResNet18 menunjukkan bahwa pendekatan
berbasis transfer learning lebih efektif dibandingkan Training from Scratch
untuk dataset yang digunakan pada project ini.


## 🧪 Analisis External Testing MobileNetV2

Setelah proses training, model MobileNetV2 V2 diuji menggunakan dataset
external yang tidak digunakan selama proses training.

Dataset external terdiri dari:

| Kelas | Jumlah Data |
|---|---:|
| MG90S | 20 |
| PCA9685 | 20 |
| **Total** | **40** |

Hasil external testing:

| Parameter | Hasil |
|---|---:|
| Total images | 40 |
| Correct prediction | 40 |
| Wrong prediction | 0 |
| External Accuracy | **100.00%** |
| MG90S | **100.00% (20/20)** |
| PCA9685 | **100.00% (20/20)** |
| Average Latency | **84.136 ms/image** |

Model berhasil mengklasifikasikan seluruh 40 gambar external test dengan benar.
Tidak terdapat kesalahan klasifikasi pada kedua kelas.

Hasil tersebut menunjukkan bahwa model MobileNetV2 V2 mampu melakukan
generalisasi dengan baik terhadap data yang tidak digunakan selama training.

Namun, nilai accuracy yang tinggi pada external test tetap perlu
diinterpretasikan berdasarkan ukuran dataset. External test yang digunakan
berjumlah 40 gambar, sehingga hasil 100% menunjukkan bahwa seluruh sampel
pada dataset external berhasil diklasifikasikan dengan benar, tetapi belum
dapat secara mutlak merepresentasikan performa model pada seluruh kondisi
lingkungan nyata.

Pengujian ini kemudian dilanjutkan dengan camera test untuk melihat kemampuan
model dalam melakukan klasifikasi secara real-time.


## 📷 Analisis Real-Time Camera

Model MobileNetV2 V2 juga diuji menggunakan kamera secara real-time untuk
melihat kemampuan model dalam melakukan klasifikasi langsung terhadap objek.

Beberapa hasil pengujian kamera menunjukkan:

| Objek | Confidence |
|---|---:|
| PCA9685 | 99.77% |
| MG90S | 97.37% |
| MG90S | 99.38% |
| PCA9685 | 96.91% |

Model dapat mengenali kedua objek dengan confidence yang tinggi pada kondisi
pengujian kamera.

Hasil tersebut menunjukkan bahwa model tidak hanya dapat bekerja pada dataset
external, tetapi juga dapat digunakan untuk melakukan klasifikasi objek secara
langsung melalui kamera.

Meskipun demikian, performa real-time dapat dipengaruhi oleh beberapa faktor,
seperti pencahayaan, posisi objek, jarak kamera, sudut pengambilan gambar,
background, dan kondisi objek.


## ⚖️ Perbandingan Model

Hasil akhir perbandingan model menunjukkan:

| Model | Accuracy | Latency |
|---|---:|---:|
| MobileNetV2 | **100.00%** | 85.225 ms |
| ResNet18 Feature Extraction | 97.50% | 5.658 ms |
| ResNet18 Partial Fine-Tuning | 97.50% | 0.673 ms |
| ResNet18 Scratch | 75.00% | 0.638 ms |

MobileNetV2 menghasilkan accuracy tertinggi, yaitu **100.00%** pada external
test yang digunakan. Sementara itu, ResNet18 Feature Extraction dan Partial
Fine-Tuning memperoleh accuracy **97.50%**.

Training from Scratch memiliki latency paling rendah pada pengujian tersebut,
tetapi accuracy-nya hanya **75.00%**. Oleh karena itu, latency yang rendah
tidak selalu berarti model tersebut merupakan pilihan terbaik apabila
accuracy menjadi prioritas utama.

Untuk aplikasi robotika, pemilihan model perlu mempertimbangkan trade-off
antara accuracy dan inference latency. Model dengan accuracy tinggi lebih
cocok apabila kesalahan klasifikasi harus diminimalkan, sedangkan model
dengan latency rendah lebih sesuai apabila respons real-time menjadi
prioritas utama.

Berdasarkan hasil eksperimen project ini, **MobileNetV2 memberikan accuracy
tertinggi**, sedangkan **ResNet18 Partial Fine-Tuning memberikan latency
yang lebih rendah pada pengujian yang dilakukan**.