import re
import requests
from bs4 import BeautifulSoup


def slugify_term(term):
    term = term.lower().strip()
    term = re.sub(r"[^a-z0-9\s-]", "", term)
    term = re.sub(r"\s+", "-", term)
    return term


def clean_text(text):
    text = re.sub(r"\s+", " ", text).strip()

    bad_values = {
        "APA Dictionary of Psychology",
        "Dictionary of Psychology",
    }

    if text in bad_values:
        return None

    if len(text.split()) < 5:
        return None

    return text


def get_apa_dictionary_definition(term):
    slug = slugify_term(term)
    url = f"https://dictionary.apa.org/{slug}"

    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Try meta description first
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        definition = clean_text(meta["content"])

        if definition and term.lower() in definition.lower():
            return {
                "term": term,
                "definition": definition,
                "source": "APA Dictionary of Psychology",
                "url": url,
            }

    # Try visible paragraph/list text
    candidates = []

    for tag in soup.find_all(["p", "div", "span", "li"]):
        text = clean_text(tag.get_text(" ", strip=True))

        if not text:
            continue

        lower = text.lower()

        if term.lower() in lower and "apa dictionary" not in lower:
            candidates.append(text)

    if candidates:
        definition = candidates[0]

        return {
            "term": term,
            "definition": definition,
            "source": "APA Dictionary of Psychology",
            "url": url,
        }

    return {
    "term": term,
    "definition": "APA Dictionary page found, but definition could not be extracted automatically.",
    "source": "APA Dictionary of Psychology",
    "url": url,
    }