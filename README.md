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


