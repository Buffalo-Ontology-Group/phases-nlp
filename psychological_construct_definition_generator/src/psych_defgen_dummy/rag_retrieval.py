from sentence_transformers import (
    SentenceTransformer,
)
from sklearn.metrics.pairwise import (
    cosine_similarity,
)


DEFINITION_PATTERNS = [
    "occurs when",
    "defined as",
    "is defined as",
    "refers to",
    "stems from",
    "is characterized by",
    "is conceptualized as",
    "is understood as",
    "determines",
]


CONCEPT_HINTS = [
    "risk",
    "disadvantage",
    "conditions",
    "factors",
    "capacity",
    "vulnerability",
    "susceptibility",
    "exposure",
    "outcomes",
    "social",
]


def is_useful_text(text):
    """
    Remove article titles, very short snippets,
    and non-informative evidence.
    """

    if not text:
        return False

    text = text.strip()
    words = text.split()
    text_lower = text.lower()

    # Remove very short text.
    if len(words) < 12:
        return False

    # Remove likely title-only text unless it
    # contains definitional language.
    if (
        len(words) < 30
        and not any(
            pattern in text_lower
            for pattern in DEFINITION_PATTERNS
        )
    ):
        return False

    # Always keep definition-like passages.
    if any(
        pattern in text_lower
        for pattern in DEFINITION_PATTERNS
    ):
        return True

    # Keep longer conceptual passages.
    if (
        len(words) >= 30
        and any(
            hint in text_lower
            for hint in CONCEPT_HINTS
        )
    ):
        return True

    return False


def normalize_evidence_item(item):
    """
    Convert an evidence item into a consistent
    dictionary format.

    Dictionary records preserve article metadata.
    Plain strings remain supported for backward
    compatibility.
    """

    if isinstance(item, str):
        return {
            "text": item,
            "title": None,
            "authors": [],
            "journal": None,
            "year": None,
            "pmid": None,
            "pmcid": None,
            "source": None,
            "pubmed_url": None,
            "url": None,
        }
        

    if not isinstance(item, dict):
        return None

    text = item.get("text")

    if not text:
        return None

    return {
        "text": text,
        "title": item.get("title"),
        "authors": item.get("authors", []),
        "journal": item.get("journal"),
        "year": item.get("year"),
        "pmid": item.get("pmid"),
        "pmcid": item.get("pmcid"),
        "source": item.get("source"),
        "pubmed_url": item.get("pubmed_url"),
        "url": item.get("url"),
    }


def retrieve_relevant_texts(
    term,
    items,
    top_k=5,
):
    """
    Retrieve top-k relevant evidence passages
    while preserving source-article metadata.
    """

    normalized_items = []

    for item in items:
        normalized_item = (
            normalize_evidence_item(item)
        )

        if normalized_item is None:
            continue

        if not is_useful_text(
            normalized_item["text"]
        ):
            continue

        normalized_items.append(
            normalized_item
        )

    if not normalized_items:
        return []

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    query = (
        f"Meaning, definition, characteristics, "
        f"conceptual explanation, causes, and "
        f"consequences of {term}"
    )

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
    )

    evidence_texts = [
        item["text"]
        for item in normalized_items
    ]

    text_embeddings = model.encode(
        evidence_texts,
        convert_to_numpy=True,
    )

    similarities = cosine_similarity(
        query_embedding,
        text_embeddings,
    )[0]

    scored_items = []

    for index, item in enumerate(
        normalized_items
    ):
        text = item["text"]
        score = float(
            similarities[index]
        )
        text_lower = text.lower()

        # Strong boost for definitional text.
        if any(
            pattern in text_lower
            for pattern in DEFINITION_PATTERNS
        ):
            score += 0.20

        # Small boost for conceptual content.
        if any(
            hint in text_lower
            for hint in CONCEPT_HINTS
        ):
            score += 0.05

        scored_item = item.copy()
        scored_item["score"] = score

        scored_items.append(
            scored_item
        )

    scored_items.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_items[:top_k]