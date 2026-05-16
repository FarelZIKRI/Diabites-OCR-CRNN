# CRNN OCR: Indonesian Nutrition Label

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0+-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

Proyek ini menghadirkan solusi **End-to-End OCR** menggunakan arsitektur **CRNN (Convolutional Recurrent Neural Network)** untuk membaca teks pada label informasi nilai gizi (nutrition labels) produk makanan di Indonesia. Fokus utama adalah menangani variasi kualitas gambar dan kompleksitas teks pada kemasan retail.

---

## Project Highlights

- **Custom Architecture**: Integrasi MobileNetV2 dengan Bidirectional LSTM dan algoritma CTC.
- **Real-world Robustness**: Dilatih untuk menangani gangguan visual seperti _blur_, _glare_ (pantulan cahaya), dan _overlapping text_.
- **Production Ready**: Dilengkapi dengan script inference mandiri dan export model dalam format `.keras` dan `SavedModel`.
- **Balanced Data**: Dataset yang dikurasi secara manual dengan distribusi kelas yang sangat seimbang.

---

## Technical Architecture

Model ini mengadopsi struktur hybrid yang efisien untuk pengenalan urutan karakter:

1.  **Backbone (CNN)**:
    - Menggunakan **MobileNetV2** (Pre-trained pada ImageNet).
    - Ekstraksi fitur dihentikan pada **`block_3_expand_relu`**. Pemilihan layer ini krusial untuk mempertahankan resolusi fitur horizontal (480px input -> 120 timesteps), memungkinkan model "melihat" karakter yang sangat kecil sekalipun.
2.  **Dimension Bridge**:
    - Penambahan layer **Conv2D** kustom untuk mereduksi tinggi gambar menjadi 1 pixel tanpa mengurangi resolusi lebar. Hal ini mengubah data spasial menjadi representasi urutan (_sequence_).
3.  **Sequence Modeling (RNN)**:
    - **2x Bidirectional LSTM** (128 units). Lapisan ini menangkap dependensi karakter secara dua arah (maju dan mundur), sangat efektif untuk konteks kata-kata nutrisi seperti "Karbohidrat" atau "Kalori".
4.  **Prediction (CTC)**:
    - Menggunakan **CTC Loss** yang memungkinkan pelatihan tanpa anotasi posisi tiap karakter. Hasil akhir didekode menggunakan _Greedy Search_ untuk konversi probabilitas menjadi teks string.

---

## Exploratory Data Analysis (EDA)

### Dataset Composition

Dataset terdiri dari **420 citra label** yang dikategorikan ke dalam 5 grup nutrisi utama.

| Kategori         | Jumlah | Deskripsi                    |
| :--------------- | :----- | :--------------------------- |
| **Calories**     | 84     | Label energi total (Kkal/Kj) |
| **Carbohydrate** | 84     | Informasi serat dan gula     |
| **Fat**          | 84     | Lemak total dan lemak jenuh  |
| **Sodium**       | 84     | Kandungan garam (mg)         |
| **Sugar**        | 84     | Kandungan gula (g)           |

![Distribusi Data](foto/distribusi&kualitas-data.png)

### Data Augmentation Strategy

Kami menggunakan library **Albumentations** untuk mensimulasikan tantangan kamera smartphone:

- **Gaussian Noise & Blur**: Meniru sensor kamera murah atau getaran tangan.
- **Brightness/Contrast**: Menangani kondisi pencahayaan minim atau berlebih (_glare_).

![Augmentasi](foto/augmentasi.png)

---

## Training Performance

- **Learning Rate**: 0.0002 (Adam Optimizer).
- **Epochs**: 200 (dengan Early Stopping).
- **Input Resolution**: 64 x 480 (RGB).

Pelatihan menunjukkan konvergensi yang stabil dengan penurunan CTC Loss yang signifikan pada data validasi, menandakan model tidak mengalami _overfitting_ meski menggunakan arsitektur yang cukup kompleks.

![Kurva Pelatihan](foto/hasil-training,%20val,%20LR.png)

---

## Inference Results

Model diuji pada data yang belum pernah dilihat sebelumnya (_unseen data_) untuk memvalidasi generalisasi.

### Original Test Set

Visualisasi hasil prediksi model pada dataset testing yang menunjukkan akurasi tinggi pada teks rapat.
![Hasil Model](foto/hasil-model.png)

### Test Data

Pengujian model menggunakan dataset pada folder `test`:
![Test Inference](foto/test-inference-model.png)

### Pengujian pada Data Baru

Pengujian data baru menggunakan script `inference.py` pada folder `test-data-baru`. Hasil prediksi visual disimpan secara otomatis di folder `hasil_data_baru`.

![Hasil Data Baru](hasil_data_baru/hasil-data-baru.png)
![Hasil Data Baru](hasil_data_baru/hasil-data-baru2.png)

---

## Project Structure & Setup

```text
.
├── foto/                   # Visualisasi EDA dan training
├── saved_model/            # Model weights & vocab mapping (JSON)
├── test/                   # Dataset uji original
├── test-data-baru/         # Folder untuk testing gambar baru Anda
├── hasil_data_baru/        # Output visual hasil prediksi
├── valid/                  # Dataset validasi
├── CRNN_OCR_Final.ipynb    # Pipeline pelatihan lengkap
├── inference.py            # Script prediksi (Run this!)
└── readme.md               # Dokumentasi utama
```

### Quick Setup

1.  Pastikan Python 3.9+ terinstal.
2.  Install library: `pip install tensorflow opencv-python pandas numpy albumentations`
3.  Jalankan prediksi: `python inference.py`

---

## Author

**Muhammad Farel Zikri** - _AI Engineer Enthusiast_
