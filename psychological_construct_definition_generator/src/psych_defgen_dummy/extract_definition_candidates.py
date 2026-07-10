import re


DEFINITION_KEYWORDS = [
    "defined as",
    "refers to",
    "is defined",
    "is a subjective",
    "is the subjective",
    "subjective and distressing experience",
    "resulting from",
    "characterized by",
    "conceptualized as",
    "understood as",
    "social factors",
    "social conditions",
    "social determinants",
    "risk",
    "disadvantage",
    "exposure",
    "capacity to cope",
    "ability to recover",
    "population characteristics",
]


def clean_sentence(sentence):
    sentence = re.sub(r'^\d+\s*,\s*\d+\s*', '', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence.strip()


def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)


def extract_definition_candidates(text, term):
    candidates = []
    term_lower = term.lower()

    for sentence in split_sentences(text):
        sentence_clean = clean_sentence(sentence)
        sentence_lower = sentence_clean.lower()

        if term_lower in sentence_lower:
            candidates.append(sentence_clean)

        elif any(keyword in sentence_lower for keyword in DEFINITION_KEYWORDS):
            if any(word in sentence_lower for word in term_lower.split()):
                candidates.append(sentence_clean)

    return candidates
