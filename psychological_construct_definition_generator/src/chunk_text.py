from src.models import TextChunk


def chunk_article(article, chunk_size=180, overlap=40):
    words = article.text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        if chunk_text.strip():
            chunks.append(
                TextChunk(
                    pmid=article.pmid,
                    title=article.title,
                    pmcid=article.pmcid,
                    source=article.source,
                    text=chunk_text
                )
            )

        start = end - overlap

    return chunks
