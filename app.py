from pathlib import Path

from flask import Flask, jsonify, render_template, request

from model_train import MODEL_FILE, train_and_save_model
from scraper import scrape_article_text
from utils import get_model_summary, predict_news


app = Flask(__name__)


def ensure_model_exists() -> None:
    """Train the model once if the pickle file is missing."""
    if not Path(MODEL_FILE).exists():
        train_and_save_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model-info", methods=["GET"])
def model_info():
    ensure_model_exists()
    return jsonify(get_model_summary())


@app.route("/predict", methods=["POST"])
def predict():
    ensure_model_exists()

    payload = request.get_json(silent=True) or {}
    url = (payload.get("url") or "").strip()
    text = (payload.get("text") or "").strip()

    if not url and not text:
        return jsonify({"error": "Please provide either a news URL or article text."}), 400

    try:
        if url:
            article = scrape_article_text(url)
            article_text = article["text"]
            source = "url"
        else:
            article = {"title": "Manual Text Input", "text": text}
            article_text = text
            source = "text"

        result = predict_news(article_text, source_url=url)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify(
            {
                "error": "Something went wrong while checking the article. Please try again."
            }
        ), 500

    return jsonify(
        {
            "label": result["label"],
            "confidence": round(result["confidence"] * 100, 2),
            "preview": article["text"][:600] + ("..." if len(article["text"]) > 600 else ""),
            "title": article["title"],
            "word_count": len(article["text"].split()),
            "model_accuracy": round(result["model_accuracy"] * 100, 2),
            "source": source,
            "reason": result["reason"],
        }
    )


if __name__ == "__main__":
    ensure_model_exists()
    app.run(debug=True)
