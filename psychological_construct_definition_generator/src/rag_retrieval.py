from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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

    # Remove very short text
    if len(words) < 12:
        return False

    # Remove likely title-only texts unless they contain definitional language
    if len(words) < 30 and not any(
        pattern in text_lower
        for pattern in DEFINITION_PATTERNS
    ):
        return False

    # Always keep definition-like passages
    if any(
        pattern in text_lower
        for pattern in DEFINITION_PATTERNS
    ):
        return True

    # Keep longer conceptual passages
    if len(words) >= 30 and any(
        hint in text_lower
        for hint in CONCEPT_HINTS
    ):
        return True

    return False


def retrieve_relevant_texts(term, texts, top_k=5):
    """
    Retrieve top-k relevant evidence passages.
    """

    texts = [
        text
        for text in texts
        if is_useful_text(text)
    ]

    if not texts:
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
        convert_to_numpy=True
    )

    text_embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        query_embedding,
        text_embeddings
    )[0]

    scored_items = []

    for index, text in enumerate(texts):
        score = float(similarities[index])
        text_lower = text.lower()

        # Strong boost for definitional sentences
        if any(
            pattern in text_lower
            for pattern in DEFINITION_PATTERNS
        ):
            score += 0.20

        # Small boost for conceptual content
        if any(
            hint in text_lower
            for hint in CONCEPT_HINTS
        ):
            score += 0.05

        scored_items.append(
            {
                "text": text,
                "score": score
            }
        )

    scored_items.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored_items[:top_k]
