import re


DEFINITION_START_PATTERNS = [
    "is defined as",
    "has been defined as",
    "can be defined as",
    "refers to",
    "is a",
    "is an",
    "is the",
    "involves",
    "is characterized by",
    "is conceptualized as",
    "is understood as",
    "occurs when",
    "stems from",
    "typically defined by",
    "defined by",
]


BAD_STUDY_PATTERNS = [
    "systematic review",
    "meta-analysis",
    "association between",
    "associated with",
    "found that",
    "study investigating",
    "study examined",
    "study investigates",
    "intervention",
    "cognitive function",
    "risk of developing",
    "predictor of",
    "tested a",
    "tested as",
]


def split_sentences(text):
    return re.split(
        r"(?<=[.!?])\s+",
        text,
    )


def clean_sentence(sentence):
    sentence = re.sub(
        r"^\d+\s+",
        "",
        sentence,
    )

    sentence = re.sub(
        r"\s+",
        " ",
        sentence,
    )

    return sentence.strip()

def clean_sentence(sentence):
    sentence = re.sub(
        r"^\d+\s+",
        "",
        sentence,
    )

    sentence = re.sub(
        r"\s+",
        " ",
        sentence,
    )

    return sentence.strip()


def normalize_for_matching(
    sentence,
    term,
):
    """
    Normalize section-heading artifacts only
    for definition matching.

    The original sentence is not modified.
    """

    normalized = clean_sentence(
        sentence
    )

    term_lower = term.lower().strip()
    lower = normalized.lower()

    repeated_term = (
        f"{term_lower} {term_lower}"
    )

    if lower.startswith(
        repeated_term
    ):
        normalized = normalized[
            len(term):
        ].strip()

    return normalized


def is_definition_sentence(
    sentence,
    term,
):

    original_sentence = clean_sentence(
        sentence
    )

    matching_sentence = (
        normalize_for_matching(
            original_sentence,
            term,
        )
    )

    lower = matching_sentence.lower()
    term_lower = term.lower().strip()


    if term_lower not in lower:
        return False

    if any(
        pattern in lower
        for pattern in BAD_STUDY_PATTERNS
    ):
        return False

    word_count = len(
        sentence.split()
    )

    if (
        word_count < 6
        or word_count > 60
    ):
        return False

    if not lower.startswith(
        term_lower
    ):
        return False

    return any(
        pattern in lower
        for pattern in DEFINITION_START_PATTERNS
    )


def definition_score(
    sentence,
    term,
):

    original_sentence = clean_sentence(
        sentence
    )

    matching_sentence = (
        normalize_for_matching(
            original_sentence,
            term,
        )
    )

    lower = matching_sentence.lower()
    term_lower = term.lower().strip()


    if not is_definition_sentence(
        sentence,
        term,
    ):
        return -1

    score = 0

    strong_patterns = [
        f"{term_lower} is defined as",
        f"{term_lower} has been defined as",
        f"{term_lower} can be defined as",
        f"{term_lower} refers to",
    ]

    if any(
        lower.startswith(pattern)
        for pattern in strong_patterns
    ):
        score += 100

    conceptual_patterns = [
        f"{term_lower} is conceptualized as",
        f"{term_lower} is understood as",
        f"{term_lower} is characterized by",
    ]

    if any(
        lower.startswith(pattern)
        for pattern in conceptual_patterns
    ):
        score += 80

    noun_definition_patterns = [
        f"{term_lower} is a ",
        f"{term_lower} is an ",
        f"{term_lower} is the ",
    ]

    if any(
        lower.startswith(pattern)
        for pattern in noun_definition_patterns
    ):
        score += 70

    other_patterns = [
        f"{term_lower} involves",
        f"{term_lower} occurs when",
        f"{term_lower} stems from",
    ]

    if any(
        lower.startswith(pattern)
        for pattern in other_patterns
    ):
        score += 50

    return score

def select_best_definition_sentence(
    term,
    retrieved_items,
):
    """
    Select the strongest explicit definition.

    Definition-pattern score is the primary criterion.
    Retrieval similarity is used only as a tie-breaker.
    """

    candidates = []

    for item in retrieved_items:

        text = item.get(
            "text",
            "",
        )

        retrieval_score = item.get(
            "score",
            0.0,
        )

        for sentence in split_sentences(
            text
        ):
            original_sentence = clean_sentence(
                sentence
            )

            output_sentence = (
                normalize_for_matching(
                    original_sentence,
                    term,
                )
            )

            score = definition_score(
                output_sentence,
                term,
            )

            if score >= 0:
                candidates.append(
                    (
                        score,
                        retrieval_score,
                        output_sentence,
                    )
                )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    return candidates[0][2]