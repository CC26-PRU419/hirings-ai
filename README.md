# Hirings AI Service
**Know Your Gap. Own Your Future.**

Hirings adalah aplikasi capstone DBS Foundation Coding Camp 2026 Tim CC26-PRU419 yang membantu fresh graduate memahami skill yang sudah dimiliki, membandingkannya dengan kebutuhan karier, dan mendapatkan rekomendasi jalur karier yang lebih terarah. Repository ini berisi seluruh pipeline AI Engineer mulai dari proses build model, training, hingga deployment sebagai REST API yang digunakan oleh backend Hirings.

## Project Status

Fitur AI yang sudah tersedia:

- Job Recommendation berbasis TF-IDF + Sentence Embedding (hybrid similarity)
- Skill Trend Forecasting menggunakan model LSTM Deep Learning
- CV Parsing otomatis dari file PDF
- Career Suggestion berbasis skill user
- AI Career Chat menggunakan Gemini Generative AI
- REST API lengkap dengan FastAPI

---

## Repository Structure

```
.
├── Capstone_Project_AI_Engineer.ipynb              
├── app.py                         
├── skill_forecast_model.keras    
├── tfidf_vectorizer.pkl           
├── forecast_scaler.pkl            
├── job_salary_mean.csv
├── logs.zip        
├── requirements.txt               
├── runtime.txt                    
├── Dockerfile                     
└── README.md
```

> **Catatan:** File besar seperti `final_jobs_data.csv`, `job_embeddings.npy`, dan `tfidf_matrix.npz` tidak disimpan di repo ini. File-file tersebut di-download otomatis dari Google Drive saat `app.py` dijalankan pertama kali via `gdown`.

---

## Model Overview

### 1. Skill Trend Forecasting (Deep Learning — Main Model)

Model LSTM dibangun menggunakan **TensorFlow Functional API** untuk memprediksi tren permintaan skill berdasarkan data historis.

**Arsitektur:**
```
Input (shape: 3, 1)
    └── LSTM (64 units)
        └── Dense (32, relu)
            └── Dense (1, linear)
```

**Komponen kustom yang diimplementasikan:**
- `CustomCallback` — meng-extend `tf.keras.callbacks.Callback` untuk logging loss per epoch
- `tf.GradientTape` — training loop kustom dari awal (side quest)
- `TensorBoard` — monitoring metrik pelatihan secara real-time

**Format model:** `.keras`

**Target performa:** MAE ≤ 0.02

---

### 2. Job Recommendation (Hybrid Similarity)

Sistem rekomendasi pekerjaan menggunakan dua pendekatan yang digabungkan:

| Komponen | Bobot | Keterangan |
|---|---|---|
| Sentence Embedding (`all-MiniLM-L6-v2`) | 45% | Semantic similarity antar skill dan job description |
| TF-IDF Cosine Similarity | 55% | Keyword matching berbasis frekuensi kata |

Hasil akhir dikombinasikan dengan **title boost** untuk meningkatkan relevansi berdasarkan kecocokan judul pekerjaan.

---

### 3. AI Career Advisor (Generative AI)

Menggunakan **Gemini 2.5 Flash Lite** untuk menghasilkan career advice yang dipersonalisasi berdasarkan skill user, skill gap, dan rekomendasi karir yang tersedia.

---

## Tech Stack

| Komponen | Library / Framework |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Deep Learning | TensorFlow 2.16 (CPU) |
| NLP / Embedding | Sentence Transformers, Scikit-learn TF-IDF |
| Generative AI | Google Generative AI (Gemini) |
| Data Processing | Pandas, NumPy, SciPy |
| CV Parsing | pdfplumber |
| Model Serialization | Joblib |
| Deployment | HuggingFace Spaces (Docker) |

---

## API Endpoints

Base URL (HuggingFace): `https://ainurroshida-hirings-ai.hf.space`

### Health Check
```
GET /
```
Response:
```json
{ "message": "Hirings AI API is running!" }
```

### Job Recommendation
```
POST /recommend
Content-Type: application/json
```
Request:
```json
{ "skills": "SQL React Node.js Machine Learning" }
```
Response:
```json
[
  {
    "job_id": 3902823365,
    "company_id": 359901,
    "title": "GCP Platform Engineer",
    "location": "Johns Creek, GA",
    "formatted_experience_level": 0,
    "min_salary": 60000,
    "max_salary": 80000,
    "avg_salary": 70000,
    "salary_range": "60000 - 80000",
    "remote_allowed": 0,
    "similarity_score": 0.11679133772850037,
    "roadmap": [
      "Learn Communication Skills",
      "Build Portfolio Projects",
      "Learn Problem Solving",
      "Practice Interview Preparation"
    ],
    "industry": "Technology"
  }
]
```

### Forecast
```
POST /forecast
Content-Type: application/json
```
Request:
```json
{
    "trend": [70, 80, 90]
}
```
Response:
```json
{
    "prediction": 80.54
}
```

### Parse CV
```
POST /parse-cv
Content-Type: multipart/form-data
```
Form field: `file` (PDF)

Response:
```json
{
  "success": true,
  "data": {
    "skills": ["SQL"],
    "raw_text": "..."
  }
}
```

### Career Suggestion
```
POST /career-suggestion
Content-Type: application/json
```
Request:
```json
{ "skills": "Python SQL Machine Learning" }
```

### AI Career Chat
```
POST /ai-career-chat
Content-Type: application/json
```
Request:
```json
{
  "user_message": "Saya cocok kerja apa?",
  "context": {
    "skills": ["React", "Node.js", "SQL"],
    "cv_raw_text": "...",
    "skill_gap": {
      "missing_skills": ["Docker", "CI/CD"] },
    "career_recommendations": ["Frontend Developer", "Fullstack Developer"]
  }
}
```
Response:
```json
{
  "success": true,
  "reply": "...",
  "recommendations": ["Frontend Developer", "Fullstack Developer"],
  "next_steps": ["Docker", "CI/CD"]
}
```

### All Jobs
```
GET /all-job
```

---

## Local Setup

### Prerequisites
- Python 3.10+
- pip

### Instalasi

```bash
# Clone repo
git clone https://github.com/ainurroshida/hirings-ai.git
cd hirings-ai

# Install dependencies
pip install -r requirements.txt

# Set environment variable untuk Gemini API Key
export GEMINI_API_KEY=your_gemini_api_key_here

# Jalankan API
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

API berjalan di: `http://localhost:7860`

### Environment Variables

| Variable | Keterangan |
|---|---|
| `GEMINI_API_KEY` | API key dari Google AI Studio |

> **Penting:** Jangan pernah hardcode API key di dalam kode. Gunakan environment variable atau file `.env` (tambahkan `.env` ke `.gitignore`).

---

## Notebook

File `Capstone_Project_AI_Engineer.ipynb` berisi dokumentasi lengkap seluruh proses pengembangan model:

1. **Data Loading & Understanding** — eksplorasi dataset LinkedIn Jobs dan salary data
2. **Data Cleaning & Preprocessing** — text cleaning, deduplication, feature engineering
3. **TF-IDF Training** — vectorizer untuk job recommendation
4. **Deep Learning Model** — build, training, dan evaluasi model LSTM dengan Functional API
5. **Custom Components** — implementasi CustomCallback dan GradientTape custom training loop
6. **TensorBoard Monitoring** — visualisasi metrik pelatihan
7. **Model Export** — simpan model ke format `.keras`
8. **FastAPI Integration** — kode inference dan endpoint API

---

## Deployment

Service ini di-deploy di **HuggingFace Spaces** menggunakan Docker.

File dataset besar (`final_jobs_data.csv`, `job_embeddings.npy`, `tfidf_matrix.npz`) tidak di-commit ke repo. File tersebut di-download otomatis oleh `app.py` dari Google Drive saat container pertama kali dijalankan.

---

## Checklist AI Engineer

### Main Quest ✅
- [x] Model Deep Learning dengan TensorFlow Functional API
- [x] Custom Callback (`CustomCallback` extend `tf.keras.callbacks.Callback`)
- [x] Model diekspor ke format `.keras` (produksi)
- [x] Kode inference model tersedia di `app.py`

### Side Quest ✅
- [x] REST API mandiri dengan FastAPI
- [x] `tf.GradientTape` untuk custom training loop
- [x] Generative AI (Gemini) untuk AI Career Chat
- [x] TensorBoard untuk monitoring metrik pelatihan
- [x] Target performa MAE ≤ 0.02

---

## Integration Notes

- Backend Hirings memanggil AI Service via HTTP ke base URL yang dikonfigurasi di `AI_SERVICE_BASE_URL`.
- Endpoint `/recommend` mengembalikan salary dalam USD — konversi ke IDR dilakukan oleh backend.
- Endpoint `/skill-gap` benchmark dinamis belum tersedia; saat ini backend menggunakan benchmark internal sambil menunggu endpoint dari AI/DS.
- Jika ngrok digunakan untuk development, URL berubah setiap restart — update `AI_SERVICE_BASE_URL` di backend setiap kali restart.

---

## License

Proyek ini dibuat untuk Capstone Project DBS Foundation Coding Camp 2026 — Tim CC26-PRU419.
