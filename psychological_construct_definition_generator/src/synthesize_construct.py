import re


DEFINITION_PATTERNS = [
    "occurs when",
    "is defined as",
    "defined as",
    "refers to",
    "stems from",
    "is characterized by",
    "is conceptualized as",
    "is understood as",
]


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\bBACKGROUND:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bMETHODS:\s*.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bRESULTS:\s*.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bCONCLUSIONS:\s*.*', '', text, flags=re.IGNORECASE)
    return text.strip()


def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)


def synthesize_construct(term, retrieved_items):
    if not retrieved_items:
        return f"No evidence was retrieved for {term}."

    term_lower = term.lower()

    for item in retrieved_items:
        text = clean_text(item["text"])
        sentences = split_sentences(text)

        for sentence in sentences:
            sentence = sentence.strip()
            sentence_lower = sentence.lower()

            if term_lower in sentence_lower:
                if any(pattern in sentence_lower for pattern in DEFINITION_PATTERNS):
                    return sentence

    best_text = clean_text(retrieved_items[0]["text"])
    sentences = split_sentences(best_text)

    for sentence in sentences:
        if term_lower in sentence.lower() and len(sentence.split()) > 8:
            return sentence.strip()

    return best_text
