import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils import preprocess_text


BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "dataset" / "fake_real_news.csv"
MODEL_FILE = BASE_DIR / "model.pkl"


def load_dataset() -> pd.DataFrame:
    """Load the dataset and keep only rows needed by the classifier."""
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

    data = pd.read_csv(DATASET_FILE)
    required_columns = {"text", "label"}

    if not required_columns.issubset(data.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    data = data.dropna(subset=["text", "label"]).copy()
    data["label"] = data["label"].astype(str).str.upper().str.strip()
    data = data[data["label"].isin(["REAL", "FAKE"])]

    if data.empty:
        raise ValueError("Dataset is empty after cleaning.")

    return data


def train_and_save_model() -> dict:
    """Train the TF-IDF + Logistic Regression pipeline and save it to disk."""
    data = load_dataset()
    data["clean_text"] = data["text"].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        data["clean_text"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)

    model_bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "accuracy": accuracy,
        "classes": list(model.classes_),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "dataset_size": int(len(data)),
    }

    with open(MODEL_FILE, "wb") as model_file:
        pickle.dump(model_bundle, model_file)

    return model_bundle


if __name__ == "__main__":
    bundle = train_and_save_model()
    print(f"Model trained successfully and saved to {MODEL_FILE}")
    print(f"Validation accuracy: {bundle['accuracy']:.2%}")
