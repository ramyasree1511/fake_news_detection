# Fake News Detection Web Application

This project is a beginner-friendly Flask web app that:

- scrapes article text from a news URL using `requests` and `BeautifulSoup`
- preprocesses the text using `nltk`
- converts text into TF-IDF features
- trains a `LogisticRegression` classifier with `scikit-learn`
- predicts whether the article is `REAL` or `FAKE`

It also supports manual text input in case a page blocks scraping.

## Project Structure

```text
project/
|-- app.py
|-- model_train.py
|-- scraper.py
|-- utils.py
|-- model.pkl
|-- requirements.txt
|-- README.md
|-- templates/
|   |-- index.html
|-- static/
|   |-- style.css
|   `-- script.js
`-- dataset/
    `-- fake_real_news.csv
```

## Technologies Used

- Python
- Flask
- pandas
- scikit-learn
- BeautifulSoup
- requests
- nltk
- HTML
- CSS
- JavaScript

## Features

- Analyze a news article directly from URL
- Paste article text manually for classification
- Web scraping with paragraph extraction
- Text cleaning and stopword removal
- TF-IDF vectorization
- Logistic Regression model
- Confidence score display
- Extracted article preview
- Model accuracy and dataset summary
- Responsive modern UI with loading animation

## How To Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python model_train.py
```

This creates `model.pkl`.

### 3. Start the Flask app

```bash
python app.py
```

### 4. Open in browser

Visit:

```text
http://127.0.0.1:5000
```

## How It Works

### URL mode

1. User enters a news article URL
2. The backend scrapes paragraph text from the page
3. The text is cleaned and vectorized
4. The ML model predicts `REAL` or `FAKE`
5. The UI shows the result, confidence, preview, and metadata

### Text mode

1. User pastes article content
2. The backend skips scraping and directly preprocesses the text
3. The model returns a prediction

## API Endpoints

### `GET /`

Loads the main web page.

### `GET /api/model-info`

Returns model metadata:

```json
{
  "accuracy": 60.0,
  "dataset_size": 24,
  "train_size": 19,
  "test_size": 5,
  "classes": ["FAKE", "REAL"]
}
```

### `POST /predict`

Request body for URL mode:

```json
{
  "url": "https://example.com/news-article"
}
```

Request body for text mode:

```json
{
  "text": "Paste the article content here."
}
```

Example response:

```json
{
  "label": "REAL",
  "confidence": 58.97,
  "preview": "Officials announced a new public transport plan after a council vote...",
  "title": "Manual Text Input",
  "word_count": 15,
  "model_accuracy": 60.0,
  "source": "text"
}
```

## Sample Input

### URL input

```text
https://www.example.com/sample-news-article
```

### Manual text input

```text
Officials announced a new public transport plan after a council vote and published the policy details on the city website.
```

## Sample Output

```text
Prediction: REAL
Confidence: 58.97%
Source: Analyzed from pasted text
Word Count: 15
Model Accuracy: 60.00%
```

## Important Note

The included CSV is a small starter dataset for demonstration and learning. For better prediction quality, replace `dataset/fake_real_news.csv` with a larger real-world fake/real news dataset that still uses:

- `text`
- `label`

The `label` values should be `REAL` or `FAKE`.
