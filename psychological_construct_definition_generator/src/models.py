from dataclasses import dataclass


@dataclass
class Article:
    pmid: str
    title: str
    source: str
    text: str
    pmcid: str | None = None


@dataclass
class TextChunk:
    pmid: str
    title: str
    text: str
    source: str
    pmcid: str | None = None
    score: float | None = None
