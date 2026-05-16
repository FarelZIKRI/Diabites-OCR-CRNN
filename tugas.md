Kriteria:

Main Quest:

1. Membangun model Deep Learning menggunakan TensorFlow Functional API atau Model Subclassing, yang disesuaikan dengan dataset dan permasalahan bisnis yang telah ditentukan oleh tim Data Science (jika ada).
2. Mengimplementasikan setidaknya satu komponen kustom lanjutan dalam proses pengembangan model, seperti:

- Custom Layer
- Custom Loss Function
- Custom Callback

3. Menyimpan dan mengekspor model yang telah dilatih secara penuh dalam format TensorFlow siap produksi (.keras atau SavedModel).
4. Membuat kode sederhana untuk proses inference model.

Side Quest:

1. Mengimplementasikan training dan evaluation loop kustom secara penuh dari awal menggunakan tf.GradientTape.
2. Mengintegrasikan TensorBoard untuk memantau dan memvisualisasikan metrik pelatihan secara menyeluruh, serta menyertakan log yang dihasilkan dalam repository akhir.
3. Memastikan model memiliki performa yang baik, dengan ketentuan minimum:

- Akurasi minimal: 85%
- MAE maksimal: 0,02

Dilarang:

1. Menggunakan model yang sudah tersedia dari TensorFlow Hub atau sumber serupa (tetapi diperbolehkan jika di fine tuning dan pre trained model sertakan link sumber repository).
2. Menggunakan model langsung dari layanan API seperti ChatGPT API, Gemini API, dan sejenisnya.
3. Menggunakan AutoML untuk membuat model AI diskriminatif (Vertex AI hanya diperbolehkan untuk use case Generative AI).
