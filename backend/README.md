# FastAPI Multi-Label NLP Engine

Proyek ini adalah layanan backend berkinerja tinggi untuk **Klasifikasi Teks Multi-Label** menggunakan **FastAPI** dan **LightGBM (via Classifier Chains)**.

## 🏗️ Arsitektur: Modular Monolith

Proyek ini dibangun dengan struktur yang memisahkan konfigurasi inti (`core`) dari logika bisnis (`modules`).

```plaintext
nlp-multilabel-backend/
├── app/
│   ├── core/                   # Konfigurasi Global (Settings, Logging, Exceptions)
│   ├── modules/                # Domain Logic
│   │   └── inference/          # Module ML (Service, Router, Schemas)
│   └── main.py                 # Entry point aplikasi
├── models/                     # Tempat menyimpan artifact model (.pkl)
├── tests/                      # Unit tests
├── Dockerfile                  # Konfigurasi Deployment Container
├── Pipfile                     # Manajemen Dependensi (Python 3.12)
└── .env                        # Environment Variables
```

## 🚀 Cara Menjalankan (Local Development)

### 1. Prasyarat
- Python 3.12
- Pipenv (`pip install pipenv`)

### 2. Instalasi Dependensi
Masuk ke folder proyek dan install paket yang dibutuhkan:

```bash
cd nlp-multilabel-backend
pipenv install --dev
```

### 3. Setup Model
Pastikan file model yang sudah dilatih (`github_classifier_cc_lgbm_stratified.pkl`) berada di dalam folder `models/`.

> **Catatan Teknis (Memory):**
> Model ini dilatih menggunakan versi `scikit-learn` yang lebih lama. Aplikasi ini memiliki **patch otomatis** di `app/modules/inference/service.py` untuk menangani perubahan nama parameter `base_estimator` menjadi `estimator` pada `ClassifierChain` di scikit-learn versi terbaru.

### 4. Jalankan Server
Jalankan server menggunakan Uvicorn dengan fitur hot-reload:

```bash
pipenv run uvicorn app.main:app --reload
```

Akses dokumentasi API di:
- **Swagger UI:** [http://localhost:8000/](http://localhost:8000/) (Redirect ke /api/v1/docs)
- **ReDoc:** [http://localhost:8000/api/v1/redoc](http://localhost:8000/api/v1/redoc)

---

## 🐳 Cara Menjalankan (Docker)

Untuk lingkungan produksi yang konsisten, gunakan Docker.

1. **Build Image:**
   ```bash
   docker build -t nlp-backend .
   ```

2. **Run Container:**
   ```bash
   docker run -p 8000:8000 nlp-backend
   ```

---

## 📡 Penggunaan API

### Endpoint: `POST /api/v1/predict`

Menerima judul dan isi isu GitHub, lalu mengembalikan label prediksi beserta skor keyakinannya.

#### Contoh Request (cURL)

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "title": "Error in LightGBM installation",
  "body": "I cannot install lightgbm on my mac m1 machine. It gives an error about libomp."
}'
```

#### Contoh Response

```json
{
  "labels": [
    "enhancement"
  ],
  "confidence_scores": {
    "bug": 0.02,
    "enhancement": 0.85,
    "question": 0.12,
    "documentation": 0.01
  }
}
```

## 🛠️ Detail Implementasi Penting

### 1. Model Loading & Patching
File: `app/modules/inference/service.py`

Saat memuat model `.pkl`, sistem melakukan pengecekan kompatibilitas versi `scikit-learn`. Jika terdeteksi model lama yang menggunakan `base_estimator`, sistem akan secara dinamis memetakan atribut tersebut ke `estimator` dan menandai `base_estimator` sebagai "deprecated" agar validasi internal scikit-learn berhasil.

### 2. Preprocessing
Input `title` dan `body` diproses (lowercase, strip) secara terpisah, kemudian di-vektorisasi menggunakan `TfidfVectorizer` masing-masing, dan digabungkan (`hstack`) sebelum masuk ke model prediksi.

### 3. Confidence Scores
Skor keyakinan dihitung menggunakan method `predict_proba` dari `ClassifierChain`. Karena ini adalah masalah multi-label, skor ini merepresentasikan probabilitas marjinal untuk setiap label.
