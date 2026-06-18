# Psychological Construct Definition Generator

This project is a Python application that retrieves scientific literature related to psychological constructs and generates ontology-style definitions using a Retrieval-Augmented Generation (RAG) workflow.

## Overview

The Psychological Construct Definition Generator is a retrieval-augmented generation (RAG) pipeline that automatically generates ontology-style definitions for psychological constructs using evidence retrieved from PubMed and PubMed Central (PMC). The system searches the biomedical literature for a target construct, retrieves relevant articles and abstracts, extracts candidate definition statements, ranks evidence using semantic similarity, and generates a concise evidence-based definition suitable for ontology development and expert curation.

## Features


- Search PubMed for articles related to a psychological construct.
- Retrieve available full-text articles from PubMed Central (PMC).
- Retrieve PubMed abstracts as fallback evidence.
- Extract candidate definition statements from the literature.
- Chunk and index retrieved text passages.
- Perform semantic retrieval using sentence-transformer embeddings.
- Rank evidence passages based on relevance to the construct.
- Generate an ontology-style definition from the highest-ranked evidence.
- Export the definition and supporting evidence to a Markdown file.



## Usage

Generate a definition for a psychological construct:

```bash
python -m src.main loneliness
```

Multi-word constructs are supported:

```bash
python -m src.main social vulnerability
```

Optional parameters:

```bash
python -m src.main loneliness --max-results 20 --top-k 5
```

## Generated outputs are written to:

```text
data/outputs/
```



