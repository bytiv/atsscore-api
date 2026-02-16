import os
import io
import re
import pickle

from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2

app = Flask(__name__)

# ---------------------------------------------------------------------------
# CORS — allow the frontend origin (set via env var, defaults to allow all)
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
CORS(app, origins=ALLOWED_ORIGINS.split(","))

# ---------------------------------------------------------------------------
# Load ML model & vectorizer
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "resume_classifier.pkl")
VECTORIZER_PATH = os.environ.get("VECTORIZER_PATH", "tfidf_vectorizer.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        tfidf = pickle.load(f)
    print("Model and vectorizer loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    model = None
    tfidf = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
STOP_WORDS = frozenset({
    "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "must", "can", "a", "an", "this", "that", "it", "its", "we",
    "our", "you", "your", "they", "their", "from", "as", "not", "all",
    "also", "more", "about", "up", "so", "no", "if", "out", "who", "which",
    "when", "what", "there", "each", "how", "then", "them", "than", "into",
    "over", "such", "any", "very", "own", "just",
})


def clean_resume_for_ml(text: str) -> str:
    """Clean resume text for the ML classifier (keeps existing model compatibility)."""
    text = re.sub(r"\b\w{1,2}\b", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text.lower()


def normalize_text_for_ats(text: str) -> set:
    """
    Normalize text for ATS keyword matching.
    - Lowercase everything
    - Keep alphanumeric chars and common separators (+, #, .)
    - Extract meaningful tokens (keeps things like 'c++', 'c#', 'node.js', '5+')
    - Remove stop words
    """
    text = text.lower()
    # Replace common separators with spaces, but keep +, #, . within words
    text = re.sub(r"[/\\|,;:()[\]{}\"\']", " ", text)
    # Tokenize: allow word chars plus +, #, .
    tokens = re.findall(r"[\w+#.]+", text)
    # Also extract compound terms (e.g., "ci/cd" -> "ci/cd" and "cicd")
    compounds = re.findall(r"[\w+#]+[/\-][\w+#]+", text.lower())
    all_tokens = set(tokens) | set(compounds)
    # Remove stop words and very short meaningless tokens
    return {t for t in all_tokens if t not in STOP_WORDS and len(t) > 0}


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from raw PDF bytes."""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        return " ".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def calculate_ats_score(job_description: str, resume_text: str) -> dict:
    """
    Calculate ATS score based on keyword matching.
    Both texts are normalized the same way for fair comparison.
    Returns score + matched/missing keywords for transparency.
    """
    job_keywords = normalize_text_for_ats(job_description)
    resume_keywords = normalize_text_for_ats(resume_text)

    if not job_keywords:
        return {"score": 0.0, "matched": [], "missing": [], "total_job_keywords": 0}

    matched = job_keywords & resume_keywords
    missing = job_keywords - resume_keywords

    score = len(matched) / len(job_keywords) * 100
    score = min(score, 100.0)

    return {
        "score": round(score, 2),
        "matched": sorted(matched)[:20],      # top 20 for response size
        "missing": sorted(missing)[:20],
        "total_job_keywords": len(job_keywords),
        "matched_count": len(matched),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "ATS Scoring API is running!",
        "endpoints": {
            "health": "/health",
            "analyze_resume": "/analyze-resume (POST)",
            "calculate_ats_only": "/calculate-ats-only (POST)",
        },
    })


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": tfidf is not None,
    })


@app.route("/analyze-resume", methods=["POST"])
def analyze_resume():
    """Full analysis: PDF upload -> ATS score + category prediction."""
    try:
        if model is None or tfidf is None:
            return jsonify({"error": "Model not loaded"}), 500

        job_description = request.form.get("job_description", "")
        if not job_description:
            return jsonify({"error": "job_description is required"}), 400

        if "resume_file" not in request.files:
            return jsonify({"error": "resume_file is required"}), 400

        resume_file = request.files["resume_file"]
        if resume_file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        pdf_bytes = resume_file.read()
        resume_text = extract_text_from_pdf(pdf_bytes)
        if not resume_text.strip():
            return jsonify({"error": "Could not extract text from PDF"}), 400

        # ML prediction uses the old cleaning (model was trained on it)
        cleaned_for_ml = clean_resume_for_ml(resume_text)
        vectorized = tfidf.transform([cleaned_for_ml])
        prediction = model.predict(vectorized)
        proba = model.predict_proba(vectorized)
        confidence = float(max(proba[0]) * 100)

        # ATS score uses the NEW normalization (applied equally to both texts)
        ats_result = calculate_ats_score(job_description, resume_text)

        return jsonify({
            "success": True,
            "predicted_category": prediction[0],
            "confidence": round(confidence, 2),
            "ats_score": ats_result["score"],
            "matched_keywords": ats_result["matched"],
            "missing_keywords": ats_result["missing"],
            "total_job_keywords": ats_result["total_job_keywords"],
            "matched_count": ats_result["matched_count"],
            "resume_text_length": len(resume_text),
        })

    except Exception as e:
        print(f"Error in analyze_resume: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/calculate-ats-only", methods=["POST"])
def calculate_ats_only():
    """Quick ATS score from plain text (no file upload)."""
    try:
        data = request.get_json()
        job_description = data.get("job_description", "")
        resume_text = data.get("resume_text", "")

        if not job_description or not resume_text:
            return jsonify({"error": "Both job_description and resume_text are required"}), 400

        ats_result = calculate_ats_score(job_description, resume_text)

        return jsonify({
            "success": True,
            "ats_score": ats_result["score"],
            "matched_keywords": ats_result["matched"],
            "missing_keywords": ats_result["missing"],
        })

    except Exception as e:
        print(f"Error in calculate_ats_only: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)