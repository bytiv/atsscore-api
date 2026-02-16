# ATS Score API

A Flask-based REST API that analyzes resumes against job descriptions using TF-IDF vectorization and a pre-trained classifier. Returns an **ATS compatibility score**, a **predicted job category**, and a **confidence score**.

---

## Features

- **PDF Resume Parsing**: Upload a PDF and extract text automatically
- **ATS Score Calculation**: Keyword-overlap scoring between job description and resume
- **Category Prediction**: ML classifier predicts the resume's job category
- **Confidence Score**: How confident the model is in its prediction
- **Text-only endpoint**: Quick ATS scoring without file upload

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API info and available endpoints |
| `GET` | `/health` | Health check (model loaded?) |
| `POST` | `/analyze-resume` | Full analysis: PDF upload → ATS score + category |
| `POST` | `/calculate-ats-only` | ATS score from plain text (no file) |

### POST `/analyze-resume`

**Content-Type**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `job_description` | text | Yes | The job description to score against |
| `resume_file` | file | Yes | PDF resume file |

**Response**:
```json
{
  "success": true,
  "ats_score": 72.5,
  "predicted_category": "Data Science",
  "confidence": 89.3,
  "resume_text_length": 4521,
  "cleaned_text_length": 3102
}
```

### POST `/calculate-ats-only`

**Content-Type**: `application/json`

```json
{
  "job_description": "Looking for a Python developer with...",
  "resume_text": "Experienced software engineer with..."
}
```

**Response**:
```json
{
  "success": true,
  "ats_score": 65.2
}
```

---

## Setup

### 1. Clone & install

```bash
git clone <your-repo-url>
cd atsscore
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
PORT=5000
FLASK_DEBUG=true
ALLOWED_ORIGINS=http://localhost:8080
```

### 3. Ensure model files exist

The repo includes two pickle files:
- `resume_classifier.pkl` — trained classification model
- `tfidf_vectorizer.pkl` — TF-IDF vectorizer

These must be in the project root (or set `MODEL_PATH` / `VECTORIZER_PATH` env vars).

### 4. Run locally

```bash
python app.py
```

API will be available at `http://localhost:5000`

---

## Deployment (Railway)

1. Push the repo to GitHub
2. Create a new Railway project → Deploy from GitHub
3. Set environment variables in Railway dashboard:

| Variable | Value |
|----------|-------|
| `PORT` | `5000` (Railway auto-sets this) |
| `ALLOWED_ORIGINS` | `https://your-frontend-domain.com` |

4. Railway auto-detects the Python project and deploys

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Server port |
| `FLASK_DEBUG` | `false` | Enable debug mode |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `MODEL_PATH` | `resume_classifier.pkl` | Path to the classifier model |
| `VECTORIZER_PATH` | `tfidf_vectorizer.pkl` | Path to the TF-IDF vectorizer |

---

## How the Scoring Works

### ATS Score (keyword matching)
1. Tokenize the job description and resume into words
2. Remove common stop words (the, and, is, etc.)
3. Calculate: `matched_keywords / total_job_keywords × 100`
4. Cap at 100%

### Category Prediction (ML)
1. Clean the resume text (remove short words, non-alpha chars, lowercase)
2. Vectorize using the pre-trained TF-IDF vectorizer
3. Predict category using the pre-trained classifier
4. Return the highest-probability class and its confidence

---

## Integration with Clever Hire Connect

The frontend calls this API in two places:

1. **On application submit** (`useApplications.tsx`): Downloads the user's resume from Supabase Storage, sends it here with the job description, saves the results back to the `applications` table.

2. **HR manual trigger** (`useHRApplications.ts`): HR can click "Calculate ATS Score" on any application to (re)calculate scores.

Set the frontend's `VITE_ATS_API_URL` to your deployed Railway URL.
# atsscore-api
