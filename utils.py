import pickle
import re
import string
from pathlib import Path
from urllib.parse import urlparse

import nltk
from nltk.corpus import stopwords


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model.pkl"
REAL_SIGNAL_TERMS = {
    "official",
    "officials",
    "government",
    "ministry",
    "department",
    "reported",
    "statement",
    "confirmed",
    "announced",
    "according",
    "agency",
    "minister",
    "police",
    "court",
    "hospital",
    "researchers",
    "published",
    "election",
    "parliament",
}
FAKE_SIGNAL_TERMS = {
    "secret",
    "shocking",
    "miracle",
    "instantly",
    "conspiracy",
    "mind",
    "control",
    "hidden",
    "anonymous",
    "hoax",
    "fabricated",
    "unbelievable",
    "clickbait",
    "rumor",
    "rumour",
    "viral",
}
TRUSTED_NEWS_DOMAINS = {
    "ndtv.com",
    "bbc.com",
    "reuters.com",
    "apnews.com",
    "thehindu.com",
    "indianexpress.com",
    "cnn.com",
    "nytimes.com",
    "aljazeera.com",
}


def ensure_nltk_resources() -> None:
    """Download required NLTK data if it is missing."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def preprocess_text(text: str) -> str:
    """Lowercase text, remove punctuation, stopwords, and extra spaces."""
    ensure_nltk_resources()

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = re.findall(r"\b[a-zA-Z]+\b", text)

    stop_words = set(stopwords.words("english"))
    cleaned_tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

    return " ".join(cleaned_tokens)


def load_model_bundle() -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model file not found. Please run model_train.py first.")

    with open(MODEL_FILE, "rb") as model_file:
        return pickle.load(model_file)


def predict_news(text: str, source_url: str = "") -> dict:
    """Run preprocessing, vectorization, and prediction for a single article."""
    cleaned_text = preprocess_text(text)

    if not cleaned_text.strip():
        raise ValueError("The scraped article did not contain enough clean text for prediction.")

    model_bundle = load_model_bundle()
    vectorizer = model_bundle["vectorizer"]
    model = model_bundle["model"]

    features = vectorizer.transform([cleaned_text])
    class_probabilities = dict(zip(model.classes_, model.predict_proba(features)[0]))
    real_score = float(class_probabilities.get("REAL", 0.5))
    fake_score = float(class_probabilities.get("FAKE", 0.5))

    heuristic_adjustment, reason = compute_heuristic_adjustment(text, source_url)
    real_score = min(max(real_score + heuristic_adjustment, 0.01), 0.99)
    fake_score = 1 - real_score

    prediction = "REAL" if real_score >= fake_score else "FAKE"
    confidence = real_score if prediction == "REAL" else fake_score

    return {
        "label": prediction,
        "confidence": confidence,
        "model_accuracy": float(model_bundle.get("accuracy", 0.0)),
        "reason": reason,
    }


def get_model_summary() -> dict:
    """Expose saved model metadata for the frontend."""
    model_bundle = load_model_bundle()
    return {
        "accuracy": round(float(model_bundle.get("accuracy", 0.0)) * 100, 2),
        "dataset_size": int(model_bundle.get("dataset_size", 0)),
        "train_size": int(model_bundle.get("train_size", 0)),
        "test_size": int(model_bundle.get("test_size", 0)),
        "classes": model_bundle.get("classes", []),
    }


def compute_heuristic_adjustment(text: str, source_url: str) -> tuple[float, str]:
    """Lightweight rule layer to stabilize predictions on tiny demo datasets."""
    lowered_text = text.lower()
    words = set(re.findall(r"\b[a-zA-Z]+\b", lowered_text))

    real_hits = len(words & REAL_SIGNAL_TERMS)
    fake_hits = len(words & FAKE_SIGNAL_TERMS)
    domain = extract_domain(source_url)

    adjustment = 0.0
    reasons = []

    if domain in TRUSTED_NEWS_DOMAINS:
        adjustment += 0.12
        reasons.append(f"trusted source domain: {domain}")

    if real_hits:
        adjustment += min(real_hits * 0.02, 0.12)
        reasons.append(f"real-news signals: {real_hits}")

    if fake_hits:
        adjustment -= min(fake_hits * 0.03, 0.18)
        reasons.append(f"fake-news signals: {fake_hits}")

    reason = ", ".join(reasons) if reasons else "model prediction only"
    return adjustment, reason


def extract_domain(source_url: str) -> str:
    if not source_url:
        return ""

    netloc = urlparse(source_url).netloc.lower()
    return netloc.replace("www.", "")
