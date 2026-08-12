import html
import re

import requests
from bs4 import BeautifulSoup


APA_NOT_FOUND_PATTERNS = [
    r"not\s+in\s+the\s+dictionary\s+of\s+psychology",
    r"not\s+in\s+the\s+apa\s+dictionary",
    r"please\s+report\s+to\s+apa",
    r"sorry.{0,200}not\s+in\s+the\s+dictionary",
]


def slugify_term(term):
    """
    Convert a term into an APA Dictionary URL slug.
    """

    term = term.lower().strip()
    term = re.sub(r"[^a-z0-9\s-]", "", term)
    term = re.sub(r"\s+", "-", term)

    return term


def normalize_page_text(text):
    """
    Normalize HTML entities, whitespace, and case
    for reliable page-message detection.
    """

    if not text:
        return ""

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\\[nrt]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def clean_text(text):
    """
    Clean candidate definition text and reject
    generic APA page labels.
    """

    if not text:
        return None

    text = re.sub(r"\s+", " ", text).strip()

    bad_values = {
        "apa dictionary of psychology",
        "dictionary of psychology",
    }

    if text.lower() in bad_values:
        return None

    if len(text.split()) < 5:
        return None

    return text


def is_missing_apa_entry(response_text, soup):
    """
    Detect an APA page indicating that the requested
    term does not exist.

    APA may return HTTP 200 for missing entries, so
    the response status alone is insufficient.
    """

    visible_text = soup.get_text(
        " ",
        strip=True,
    )

    title_text = ""

    if soup.title:
        title_text = soup.title.get_text(
            " ",
            strip=True,
        )

    meta_texts = []

    for meta in soup.find_all("meta"):
        content = meta.get("content")

        if content:
            meta_texts.append(content)

    searchable_text = " ".join(
        [
            response_text,
            visible_text,
            title_text,
            *meta_texts,
        ]
    )

    searchable_text = normalize_page_text(
        searchable_text
    )

    return any(
        re.search(
            pattern,
            searchable_text,
            flags=re.IGNORECASE,
        )
        for pattern in APA_NOT_FOUND_PATTERNS
    )


def get_apa_dictionary_definition(term):
    """
    Retrieve a definition from the APA Dictionary
    of Psychology.

    Returns a dictionary containing one of these
    statuses:

    - found
    - not_found
    - request_error
    - parse_error
    """

    slug = slugify_term(term)
    url = f"https://dictionary.apa.org/{slug}"

    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

        response.raise_for_status()


# ---------- DEBUG ----------
        with open(
            f"apa_debug_{slug}.html",
            "w",
            encoding="utf-8",
        ) as file:
            file.write(response.text)

        print(
            f"APA response: status={response.status_code}, "
            f"url={response.url}, "
            f"length={len(response.text)}"
        )
# ---------------------------
    except requests.RequestException as error:
        return {
            "term": term,
            "definition": (
                "The APA Dictionary of Psychology "
                "could not be accessed."
            ),
            "source": "APA Dictionary of Psychology",
            "url": url,
            "status": "request_error",
            "error": str(error),
        }

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    # Detect APA's missing-entry page.
    if is_missing_apa_entry(
        response.text,
        soup,
    ):
        return {
            "term": term,
            "definition": (
                f'The term "{term}" does not have an entry '
                "in the APA Dictionary of Psychology."
            ),
            "source": "APA Dictionary of Psychology",
            "url": url,
            "status": "not_found",
        }

    # Try the meta description.
    meta = soup.find(
        "meta",
        attrs={"name": "description"},
    )

    if meta and meta.get("content"):
        definition = clean_text(
            meta["content"]
        )

        if (
            definition
        ):
            return {
                "term": term,
                "definition": definition,
                "source": (
                    "APA Dictionary of Psychology"
                ),
                "url": url,
                "status": "found",
            }


    # Try structured definition content first.
    candidates = []

    selectors = [
        "[class*='definition']",
        "[id*='definition']",
        "article p",
        "main p",
    ]

    for selector in selectors:
        for tag in soup.select(selector):
            text = clean_text(
                tag.get_text(" ", strip=True)
            )

            if not text:
                continue

            lower_text = text.lower()

            if any(
                phrase in lower_text
                for phrase in [
                    "apa dictionary of psychology",
                    "not in the dictionary",
                    "please report to apa",
                    "browse dictionary",
                    "sign in",
                    "log in",
                ]
            ):
                continue

            candidates.append(text)


    candidates = [
        candidate
        for candidate in candidates
        if 5 <= len(candidate.split()) <= 250
    ]


    if candidates:
        candidates.sort(
            key=lambda candidate: len(candidate.split())
        )

        return {
            "term": term,
            "definition": candidates[0],
            "source": "APA Dictionary of Psychology",
            "url": url,
            "status": "found",
        }

    return {
        "term": term,
        "definition": (
            f'No APA Dictionary definition was found for "{term}".'
        ),
        "source": "APA Dictionary of Psychology",
        "url": url,
        "status": "not_found",
    }