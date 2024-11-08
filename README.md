# Ch. 2: Machine Learning and Deep Learning

README ini menjelaskan deliverables dari tugas Bab 2 tentang Machine Learning yang mencakup dua kasus, yaitu klasifikasi dan klasterisasi. Tugas ini dikerjakan dalam dua file terpisah, masing-masing berfokus pada jenis masalah berbeda (klasifikasi dan klasterisasi).

## Isi

1. [Assignment Chapter 2 - Case 1: Supervised Learning Classification (02_Kelompok_B_1.ipynb)](#1)
2. [Assignment Chapter 2 - Case 2: Unsupervised Learning Clustering (02_Kelompok_B_2.ipynb)](#2)
3. [Assignment Chapter 2 - Case 3: Regression (02_Kelompok_B_3.ipynb)](#3)
4. [Assignment Chapter 2 - Case 4: Classification (02_Kelompok_B_4.ipynb)](#4)

---

## 1

### Ringkasan Tugas
Kasus pertama dalam tugas ini adalah membuat model klasifikasi menggunakan algoritma Machine Learning untuk memprediksi `Exited` dari nasabah dalam dataset *bank data*. 

### Dataset
Dataset yang digunakan adalah `SC_HW1_bank_data.csv`.

### Library yang Digunakan
- **Pandas**: Untuk manipulasi data.
- **Numpy**: Untuk operasi numerik.
- **Scikit-learn**: Untuk membangun model machine learning (RandomForestClassifier, SVC, GradientBoostingClassifier).

### Persyaratan
- **Modul dan Versi**: Pastikan semua modul yang diperlukan sudah diinstal.
- **Kolom dan Data yang Digunakan**: Hilangkan kolom yang tidak relevan dan lakukan preprocessing untuk menyiapkan data.

### Langkah Pengerjaan
1. **Data Preprocessing**: Menghilangkan kolom yang tidak relevan, melakukan *One-Hot Encoding*, dan normalisasi menggunakan `MinMaxScaler`.
2. **Pemodelan dan Evaluasi**:
   - **Model #1: Random Forest**
   - **Model #2: Support Vector Classifier (SVC)**
   - **Model #3: Gradient Boosting Classifier**
3. **Hyperparameter Tuning**: Melakukan Grid Search untuk mencari parameter terbaik.
4. **Evaluasi**:
   - **Metrics**: Menggunakan `accuracy_score`, `classification_report`, dan `confusion_matrix`.
   - Membandingkan hasil akurasi dari ketiga model dan menarik kesimpulan.

### Hasil dan Kesimpulan
Setiap model dijelaskan secara singkat dalam *notebook*. Berdasarkan hasil, **Gradient Boosting Classifier** menunjukkan performa terbaik dengan akurasi yang lebih tinggi serta waktu pemrosesan lebih cepat dibanding model lainnya.

---

## 2

### Ringkasan Tugas
Kasus kedua dalam tugas ini adalah melakukan *clustering* atau segmentasi data menggunakan *unsupervised learning*, khususnya menggunakan algoritma *KMeans Clustering*.

### Dataset
Dataset yang digunakan adalah `cluster_s1.csv`.

### Library yang Digunakan
- **Pandas**: Untuk manipulasi data.
- **Numpy**: Untuk operasi numerik.
- **Matplotlib & Seaborn**: Untuk visualisasi data.
- **Scikit-learn**: Untuk algoritma clustering KMeans.

### Persyaratan
- **Modul dan Versi**: Pastikan semua modul yang diperlukan sudah diinstal.

### Langkah Pengerjaan
1. **Data Preparation**: Menghapus kolom yang tidak relevan.
2. **Penentuan Jumlah Cluster Terbaik**:
   - Menggunakan *Silhouette Score* untuk menentukan nilai *k* terbaik di antara rentang nilai tertentu.
3. **Pemodelan dengan KMeans**:
   - Melatih model KMeans dengan jumlah cluster terbaik yang didapatkan.
4. **Evaluasi dan Visualisasi**:
   - Memvisualisasikan hasil *clustering* menggunakan Seaborn, menampilkan *scatter plot* dengan warna berbeda untuk setiap cluster.

### Hasil dan Kesimpulan
Nilai *k* terbaik diperoleh berdasarkan Silhouette Score tertinggi. Hasil clustering divisualisasikan dalam bentuk scatter plot yang menunjukkan distribusi dan segmentasi data yang dihasilkan.

---

## 3

### Ringkasan Tugas
Kasus ketiga dalam tugas ini adalah membuat model regresi menggunakan TensorFlow-Keras untuk memprediksi harga rumah di California berdasarkan dataset *California House Price*. Model ini dibangun menggunakan arsitektur *Multilayer Perceptron* dengan input ganda.

### Dataset
Dataset yang digunakan adalah California House Price dari *Scikit-Learn*, dengan variabel target berupa harga rumah.

### Library yang Digunakan
- **Pandas**: Digunakan untuk manipulasi data dalam bentuk DataFrame.
- **Numpy**: Digunakan untuk operasi array dan matriks.
- **TensorFlow**: Framework utama untuk membangun dan melatih model deep learning.
- **Keras**: API untuk membangun model deep learning dalam TensorFlow.
- **Scikit-learn**: Digunakan untuk pre-processing data, seperti standardisasi dan pembagian dataset.
- **Matplotlib**: Digunakan untuk visualisasi grafik dari hasil pelatihan model.

### Langkah Penyelesaian
1. **Persiapan dan Pemisahan Data**: 
   - Konversi data ke bentuk DataFrame.
   - Pisahkan data menjadi *train*, *validation*, dan *test set*.
   - Lakukan standarisasi dan normalisasi data.

2. **Membangun Model Neural Network**:
   - Buat dua hidden layer dengan 30 neuron menggunakan fungsi aktivasi ReLU.
   - Gabungkan input ganda (input A dan B) sebelum memasukkan ke output layer.

3. **Training dan Evaluasi Model**:
   - Tentukan parameter *learning rate*, *epochs*, dan *batch size*.
   - Lakukan *training* pada model dan tampilkan hasil *loss* untuk memastikan model tidak *overfitting*.

4. **Menyimpan Model**:
   - Simpan model yang telah dilatih dan prediksi beberapa sampel baru.

---

## 4

### Ringkasan Tugas
Kasus keempat dalam tugas ini adalah membangun model klasifikasi menggunakan PyTorch untuk mendeteksi transaksi fraud pada dataset *Credit Card Fraud 2023*. Model ini menggunakan *Multilayer Perceptron* untuk mengklasifikasikan apakah suatu transaksi termasuk fraud atau tidak.

### Dataset
Dataset yang digunakan adalah *Credit Card Fraud 2023*, dengan variabel target berupa kolom *Class* (fraud/non-fraud).

### Library yang Digunakan
- **Pandas**: Digunakan untuk manipulasi data dalam bentuk DataFrame.
- **cuDF**: Versi GPU-accelerated dari Pandas untuk memproses data di GPU.
- **cuML**: Digunakan untuk pemrosesan data secara parallel di GPU, khususnya untuk scaling.
- **Numpy (cuPy)**: Digunakan sebagai library untuk operasi array di GPU.
- **Scikit-learn**: Digunakan untuk standardisasi dan pembagian dataset.
- **PyTorch**: Framework untuk membangun model deep learning dan melakukan training serta evaluasi.

### Langkah Penyelesaian
1. **Impor Dataset dengan GPU**:
   - Unduh dan *unzip* dataset, lalu baca menggunakan cuDF (Pandas versi GPU).
   - Hilangkan kolom ID dan lakukan standarisasi menggunakan GPU.

2. **Pemisahan Data dan Konversi ke Tensor**:
   - Tentukan fitur X dan target Y.
   - Lakukan pemisahan data *train* dan *test* menggunakan GPU.
   - Konversi data ke Tensor untuk digunakan pada *DataLoader* PyTorch.

3. **Membangun Model Neural Network**:
   - Bangun arsitektur *Multilayer Perceptron* dengan 4 hidden layers menggunakan PyTorch.
   - Tentukan parameter model seperti *epochs*, *num_layers*, dan *learning rate*.

4. **Training dan Evaluasi Model**:
   - Lakukan *training* model dan evaluasi akurasi model.
   - Pastikan akurasi model mencapai setidaknya 95%, atau lakukan fine-tuning jika dibutuhkan.

---

## Cara Menjalankan Kode

1. Buka Google Colab.
2. Unggah file `02_Kelompok_B_1.ipynb` dan `02_Kelompok_B_2.ipynb`.
3. Untuk masing-masing file, jalankan sel secara berurutan dan ikuti instruksi yang terdapat dalam *notebook*.
