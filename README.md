# 🩺 Diabetes Prediction App

Aplikasi web prediksi risiko diabetes berbasis **Machine Learning** menggunakan **XGBoost** dan metode **undersampling (Tomek Links)** untuk menangani ketidakseimbangan kelas pada data. Dibangun menggunakan **Streamlit**.

## 📌 Deskripsi Project

Project ini merupakan implementasi model klasifikasi biner untuk memprediksi apakah seorang pasien berisiko diabetes atau tidak berdasarkan data klinis. Pembuatan model menggunakan berbagai teknik resampling (oversampling & undersampling) dan dibandingkan performanya, dengan model terbaik disimpan dan di-deploy melalui Streamlit.

**Fitur Input:**

- Usia, BMI, HbA1c Level, Blood Glucose Level
- Jenis Kelamin, Ras, Riwayat Merokok
- Hipertensi, Penyakit Jantung

**Output:** Prediksi positif/negatif diabetes beserta nilai probabilitasnya.

## 🛠️ Tech Stack

| Komponen        | Library                                          |
| --------------- | ------------------------------------------------ |
| Model           | XGBoost                                          |
| Resampling      | imbalanced-learn (Tomek Links, ENN, SMOTE, dll.) |
| Web App         | Streamlit                                        |
| Data Processing | Pandas, NumPy                                    |

## 📁 Struktur Project

```
undersampling-tomlek/
├── app.py               # Aplikasi Streamlit
├── model_xgb.pkl        # Model XGBoost terlatih
├── requirements.txt     # Daftar dependensi
├── .gitignore           # File ignore
└── venv/                # Virtual environment
```

## 🚀 Cara Menjalankan

### 1. Clone / Download Project

```bash
git clone <url-repo>
cd undersampling-tomlek
```

### 2. Buat Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# atau
venv\Scripts\activate           # Windows
```

### 3. Install Dependensi

```bash
pip install streamlit pandas xgboost scikit-learn imbalanced-learn
# atau jika menggunakan uv:
uv pip install streamlit pandas xgboost scikit-learn imbalanced-learn
```

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

## 📊 Dataset

Dataset menggunakan **Diabetes Dataset** dengan fitur-fitur berikut:

| Kolom                 | Keterangan                       |
| --------------------- | -------------------------------- |
| `age`                 | Usia pasien                      |
| `gender`              | Jenis kelamin (Female / Male)    |
| `race:*`              | Ras pasien (5 kategori)          |
| `bmi`                 | Body Mass Index                  |
| `hbA1c_level`         | Kadar HbA1c dalam darah          |
| `blood_glucose_level` | Kadar glukosa darah              |
| `hypertension`        | Riwayat hipertensi (0/1)         |
| `heart_disease`       | Riwayat penyakit jantung (0/1)   |
| `smoking_history`     | Riwayat merokok                  |
| `diabetes`            | Label target (0 = Tidak, 1 = Ya) |
