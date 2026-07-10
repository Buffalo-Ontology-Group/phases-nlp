import re


DEFINITION_START_PATTERNS = [
    "is defined as",
    "refers to",
    "is a",
    "is an",
    "is the",
    "involves",
    "is characterized by",
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
    "intervention",
    "cognitive function",
    "risk of developing",
]


def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)


def clean_sentence(sentence):
    sentence = re.sub(r"^\d+\s+", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip()


def is_definition_sentence(sentence, term):
    sentence = clean_sentence(sentence)
    lower = sentence.lower()
    term_lower = term.lower()

    if term_lower not in lower:
        return False

    if any(pattern in lower for pattern in BAD_STUDY_PATTERNS):
        return False

    word_count = len(sentence.split())

    if word_count < 6 or word_count > 60:
        return False

    # Case 1: sentence starts with the term
    if lower.startswith(term_lower):
        return any(pattern in lower for pattern in DEFINITION_START_PATTERNS)

    # Case 2: sentence defines/forms the term using "term as a/an/the"
    term_as_patterns = [
        f"{term_lower} as a",
        f"{term_lower} as an",
        f"{term_lower} as the",
    ]

    if any(pattern in lower for pattern in term_as_patterns):
        return True

    return False


def select_best_definition_sentence(term, retrieved_items):
    for item in retrieved_items:
        for sentence in split_sentences(item["text"]):
            sentence = clean_sentence(sentence)

            if is_definition_sentence(sentence, term):
                return sentence

    return None