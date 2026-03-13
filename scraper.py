import requests
from bs4 import BeautifulSoup


REQUEST_HEADERS = [
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": "https://www.google.com/",
    },
    {
        "User-Agent": (
            "Mozilla/5.0 (Linux; Android 13; Pixel 7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Mobile Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
        "Referer": "https://news.google.com/",
    },
]


def scrape_article_text(url: str) -> dict:
    """Fetch the article page and collect readable paragraph content."""
    try:
        response = fetch_url_with_retries(normalize_url(url))
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code in {401, 403}:
            raise ValueError(
                "This website is blocking automated scraping requests. Try another news URL or use 'Paste Article Text'."
            ) from exc
        raise ValueError("Unable to fetch the article from the provided URL.") from exc
    except requests.RequestException as exc:
        raise ValueError(
            "Unable to fetch the article from the provided URL. Please check the link or use 'Paste Article Text'."
        ) from exc

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = extract_title(soup)

    article_tag = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=lambda value: value and "article" in value.lower())
        or soup.find("div", class_=lambda value: value and "content" in value.lower())
    )
    paragraph_tags = article_tag.find_all("p") if article_tag else soup.find_all("p")
    paragraphs = [p.get_text(" ", strip=True) for p in paragraph_tags if p.get_text(strip=True)]

    article_text = " ".join(paragraphs)
    article_text = " ".join(article_text.split())

    if len(article_text.split()) < 60:
        raise ValueError(
            "The page did not contain enough article text. Please try a different news URL."
        )

    return {"title": title, "text": article_text}


def normalize_url(url: str) -> str:
    """Add a scheme when a user pastes a bare domain."""
    cleaned_url = url.strip()
    if not cleaned_url.startswith(("http://", "https://")):
        cleaned_url = f"https://{cleaned_url}"
    return cleaned_url


def fetch_url_with_retries(url: str) -> requests.Response:
    """Retry the request with multiple browser-like headers."""
    last_error = None
    session = requests.Session()

    for headers in REQUEST_HEADERS:
        try:
            response = session.get(url, headers=headers, timeout=20, allow_redirects=True)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    raise requests.RequestException("Unable to fetch URL.")


def extract_title(soup: BeautifulSoup) -> str:
    """Try common title sources before falling back to the page title."""
    title_sources = [
        soup.find("meta", property="og:title"),
        soup.find("meta", attrs={"name": "twitter:title"}),
    ]

    for source in title_sources:
        if source and source.get("content"):
            return source["content"].strip()

    heading = soup.find("h1")
    if heading and heading.get_text(strip=True):
        return heading.get_text(strip=True)

    return soup.title.get_text(strip=True) if soup.title else "Untitled Article"
