# Batik Classification Dashboard

## Demo

🌐 **Live Demo**: [Batik Classification Dashboard](https://uap-machine-learninggit-wqfmutcqdxysvqqvtxxxdd.streamlit.app/)

Akses dashboard secara langsung tanpa instalasi lokal melalui link di atas.

## Deskripsi Proyek

Proyek ini merupakan sistem klasifikasi citra batik menggunakan pendekatan _deep learning_ dengan tiga arsitektur model: CNN kustom, MobileNetV2, dan ResNet50. Sistem dibagi menjadi dua bagian utama:  
1. Skrip pelatihan model (training) yang dijalankan di Google Colab.  
2. Dashboard web interaktif berbasis Streamlit untuk visualisasi hasil evaluasi dan uji coba prediksi gambar batik secara langsung.

Tujuan utama proyek ini adalah:
- Mengklasifikasikan citra batik ke dalam 5 kelas: **Batik Poleng, Batik Kawung, Batik Parang, Batik Megamendung, dan Batik Dayak**.  
- Membandingkan performa tiga arsitektur model yang berbeda.  
- Menyediakan antarmuka visual yang memudahkan analisis performa dan demonstrasi model kepada pengguna.

---

## Dataset dan Preprocessing

### Struktur Dataset

Dataset disusun dalam struktur direktori sebagai berikut:

```
DATASET/
├── TRAIN/
│   ├── Batik Poleng/
│   ├── Batik Kawung/
│   ├── Batik Parang/
│   ├── Batik Megamendung/
│   └── Batik Dayak/
└── TEST/
    ├── Batik Poleng/
    ├── Batik Kawung/
    ├── Batik Parang/
    ├── Batik Megamendung/
    └── Batik Dayak/
```

- Folder `TRAIN` berisi data latih, sedangkan `TEST` berisi data uji.  
- Pada subset evaluasi yang digunakan dalam laporan, terdapat **100 citra uji** (20 citra per kelas).

### Preprocessing Umum

Seluruh model menggunakan `ImageDataGenerator` untuk melakukan:
- **Rescaling** nilai piksel dari [0, 255] ke [0, 1] dengan `rescale=1./255`.  
- **Resize** citra ke ukuran:
  - CNN Custom: **128 × 128 piksel**.  
  - MobileNetV2 & ResNet50: **224 × 224 piksel** (ukuran standar untuk banyak arsitektur pretrained).  

Generator data juga diatur dengan:
- `class_mode='categorical'` untuk klasifikasi multi-kelas.  
- `shuffle=True` pada data latih dan `shuffle=False` pada data uji untuk menjaga urutan label saat evaluasi.

### Augmentasi Data

#### 1. CNN Custom

Menggunakan augmentasi **moderate** untuk menambah variasi tanpa mengubah karakteristik citra secara berlebihan:

```
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

- `IMG_SIZE = (128, 128)`  
- `BATCH_SIZE = 16`  
- `steps_per_epoch = 200` untuk memaksa augmentasi berjalan agresif tiap epoch.

#### 2. MobileNetV2

Menggunakan augmentasi **standard–aggressive** yang cocok untuk transfer learning:

```
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

- `IMG_SIZE = (224, 224)`  
- `BATCH_SIZE = 32`  
- `steps_per_epoch = len(train_generator)` sehingga satu epoch mencakup seluruh batch train.

#### 3. ResNet50

Menggunakan augmentasi **aggressive** karena model lebih dalam dan berisiko tinggi overfitting:

```
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
```

- `IMG_SIZE = (224, 224)`  
- `BATCH_SIZE = 32`  
- Digunakan konsisten pada tiga fase training (head-only, fine-tuning 30 layer, deep fine-tuning 100 layer + class weights).

---

## Arsitektur Model

### CNN Custom ("Compact Leaky-GAP")

CNN ini dibangun dengan arsitektur `Sequential` yang relatif ringkas namun cukup dalam untuk belajar pola tekstur batik:

- 4 blok konvolusi utama:
  - Blok 1: `Conv2D(32) → BatchNormalization → LeakyReLU → MaxPooling2D`  
  - Blok 2: `Conv2D(64) → BatchNormalization → LeakyReLU → MaxPooling2D → Dropout(0.2)`  
  - Blok 3: `Conv2D(128) → BatchNormalization → LeakyReLU → MaxPooling2D → Dropout(0.3)`  
  - Blok 4: `Conv2D(256) → BatchNormalization → LeakyReLU → MaxPooling2D → Dropout(0.3)`  

- Bagian akhir (head):
  - `GlobalAveragePooling2D` untuk mengurangi jumlah parameter dan overfitting.  
  - `Dense(128) → BatchNormalization → LeakyReLU → Dropout(0.5)`  
  - Output: `Dense(5, activation='softmax')`  

- Optimizer: **SGD** dengan `learning_rate=0.01`, `momentum=0.9`, `nesterov=True`.  
- Loss: `categorical_crossentropy`, input shape: **128×128×3**.  

Model ini dilatih dari nol (_from scratch_) tanpa pretrained weights.

### MobileNetV2 (Transfer Learning)

Contoh konfigurasi MobileNetV2 yang digunakan:

```
inputs = Input(shape=(224, 224, 3))
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_tensor=inputs
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(5, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
```

- Base model menggunakan bobot pretrained ImageNet dan dibekukan pada tahap awal.  
- Head kustom menyesuaikan ke 5 kelas batik.  
- Optimizer: **Adam** (misalnya `learning_rate=0.0001`), loss: `categorical_crossentropy`.

### ResNet50 (Transfer Learning, 3 Phase Fine-Tuning)

Contoh konfigurasi dasar ResNet50 yang digunakan:

```
inputs = Input(shape=(224, 224, 3), name='input_layer')
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_tensor=inputs
)

x = base_model.output
x = GlobalAveragePooling2D(name='global_avg_pool')(x)
x = Dense(256, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(0.01),
          name='dense_256')(x)
x = BatchNormalization(name='bn_256')(x)
x = Dropout(0.5, name='dropout_256')(x)
x = Dense(128, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(0.01),
          name='dense_128')(x)
x = BatchNormalization(name='bn_128')(x)
x = Dropout(0.4, name='dropout_128')(x)
outputs = Dense(5, activation='softmax', name='output_layer')(x)
model = Model(inputs=inputs, outputs=outputs, name='ResNet50_Batik')
```

- Menggunakan pretrained weights ImageNet dengan head kustom dan regularisasi L2 + dropout.  
- Dilatih dengan pendekatan bertahap (freeze–unfreeze layer dan penyesuaian learning rate).

---

## Hasil Evaluasi dan Analisis Perbandingan

Berdasarkan classification report yang diberikan, ringkasan performa ketiga model adalah sebagai berikut.

### Ringkasan Kuantitatif Global

| Aspek                    | CNN Custom | MobileNetV2 | ResNet50 |
|--------------------------|-----------:|------------:|---------:|
| Akurasi                  | 0.5900     | 0.8700      | 0.6100   |
| Precision (macro avg)    | 0.6309     | 0.8800      | 0.6468   |
| Recall (macro avg)       | 0.5900     | 0.8700      | 0.6100   |
| F1-score (macro avg)     | 0.5856     | 0.8685      | 0.6056   |
| Precision (weighted avg) | 0.6309     | 0.8800      | 0.6468   |
| Recall (weighted avg)    | 0.5900     | 0.8700      | 0.6100   |
| F1-score (weighted avg)  | 0.5856     | 0.8685      | 0.6056   |

### Performa per Kelas (F1-score)

| Kelas             | CNN Custom | MobileNetV2 | ResNet50 |
|-------------------|----------:|-----------:|---------:|
| Batik Poleng      | 0.5306    | 0.9524     | 0.8000   |
| Batik Kawung      | 0.5000    | 0.8095     | 0.5000   |
| Batik Parang      | 0.6667    | 0.8235     | 0.5000   |
| Batik Megamendung | 0.5806    | 0.9000     | 0.5854   |
| Batik Dayak       | 0.6500    | 0.8571     | 0.6429   |

### Analisis per Model

| Model | Akurasi | Macro F1 | Kekuatan | Kelemahan |
|-------|---------|----------|----------|-----------|
| **CNN Custom** | 59% | 0.5856 | - Kelas Parang dan Dayak cukup baik (F1 ≈ 0.66-0.65)<br>- Baseline sederhana untuk perbandingan<br>- Dapat dilatih dari nol tanpa pretrained weights | - Performa terendah di antara ketiga model<br>- Kawung lemah (F1 = 0.50)<br>- Sangat bergantung pada jumlah data<br>- Kurang stabil dibanding transfer learning |
| **MobileNetV2** | 87% | 0.8685 | - **Akurasi tertinggi** di antara semua model<br>- Semua kelas F1 > 0.80<br>- Poleng sangat baik (F1 = 0.95, recall 1.0)<br>- Performa paling konsisten dan stabil<br>- **Cocok untuk deployment** | - Beberapa sampel Parang masih salah prediksi (recall 0.70) |
| **ResNet50** | 61% | 0.6056 | - Poleng cukup baik (F1 = 0.80)<br>- Potensi besar untuk peningkatan<br>- Arsitektur paling dalam | - Performa jauh di bawah MobileNetV2<br>- Kawung dan Parang lemah (F1 = 0.50)<br>- Konfigurasi fine-tuning belum optimal<br>- Model besar tidak menjamin hasil terbaik |

### Kesimpulan Perbandingan

| Kategori | Model | Alasan |
|----------|-------|--------|
| **Terbaik untuk Deployment** | MobileNetV2 | Akurasi 87%, macro F1 = 0.8685, performa konsisten di semua kelas |
| **Baseline Sederhana** | CNN Custom | Model dari nol dengan hasil moderat (59%), cocok sebagai pembanding |
| **Potensial untuk Eksplorasi** | ResNet50 | Perlu tuning lanjut (layer unfreezing, learning rate, epoch), belum optimal saat ini |

### Rekomendasi

- **Gunakan MobileNetV2** untuk aplikasi produksi karena balance terbaik antara akurasi dan efisiensi
- **CNN Custom** tetap berguna untuk memahami pembelajaran dari nol dan sebagai baseline
- **ResNet50** dapat ditingkatkan dengan:
  - Penyesuaian jumlah layer yang di-unfreeze per fase
  - Tuning learning rate yang lebih halus
  - Penambahan data training
  - Eksperimen dengan class weights yang berbeda
---

## Struktur Proyek

Struktur direktori minimal untuk menjalankan dashboard:

```
.
├── app.py
├── models/
│   ├── model_batik_cnn.h5
│   ├── cnn_batik.pkl
│   ├── model_batik_mobilenetv2_fixed.h5
│   ├── model_batik_mobilenetv2_fixed.pkl
│   ├── model_batik_resnet50_final.h5
│   └── batik_model_dashboard_data.pkl
├── dataset/
└── README.md
```

---

## Panduan Menjalankan Sistem Secara Lokal

### 1. Membuat Virtual Environment

```
python -m venv venv
```

Aktivasi environment:

**Windows:**
```
venv\Scripts\activate
```

**Linux / macOS:**
```
source venv/bin/activate
```

### 2. Instalasi Dependensi

Jika tersedia `requirements.txt`:

```
pip install -r requirements.txt
```

Jika belum ada, instal manual:

```
pip install streamlit tensorflow numpy pandas plotly pillow scikit-learn seaborn
```

### 3. Menjalankan Dashboard

```
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

### 4. Fitur Dashboard

**Tab Model Evaluation:**
- Grafik training vs validation accuracy dan loss
- Confusion matrix
- Per-class accuracy
- Classification report lengkap

**Tab Image Prediction:**
- Upload citra batik (JPG/PNG)
- Prediksi kelas dengan confidence score
- Distribusi probabilitas semua kelas

---

## Catatan Penting

Jika muncul pesan error tentang confusion matrix, pastikan file `.pkl` berisi:
- `confusion_matrix`
- `precision`, `recall`, `f1_score`
- `per_class_accuracy`
- `macro_avg` dan `weighted_avg`

Untuk retraining, simpan model `.h5` dan metadata `.pkl` yang lengkap, lalu salin ke folder `models/`.

---

## Lisensi

Proyek ini dibuat untuk keperluan akademis dan pembelajaran.
```

