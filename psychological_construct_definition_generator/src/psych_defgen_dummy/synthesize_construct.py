import re


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^\d+\s+", "", text)
    text = re.sub(r"\bBACKGROUND:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bMETHODS:\s*.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bRESULTS:\s*.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bCONCLUSIONS:\s*.*", "", text, flags=re.IGNORECASE)
    return text.strip()


def synthesize_construct(term, retrieved_items):
    if not retrieved_items:
        return f"No evidence was retrieved for {term}."

    top_text = clean_text(retrieved_items[0]["text"])

    return (
        f"{term.capitalize()} is a construct discussed in the retrieved literature "
        f"in relation to the following evidence: {top_text}"
    )