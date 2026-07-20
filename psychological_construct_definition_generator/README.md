# Psychological Construct Definition Generator

This project is a Python application that retrieves scientific literature related to psychological constructs and generates evidence-based ontology-style definitions using a Retrieval-Augmented Generation (RAG) workflow over PubMed and PubMed Central literature.

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


---

# Installation

Install the package from TestPyPI:

```bash
pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple \
    psych-defgen-dummy
```


## Verify Installation

Verify that the package is installed correctly:

```bash
python -c "import psych_defgen_dummy; print('Package installed successfully')"
```

---

# Configure NCBI Email

This package uses the NCBI Entrez API to retrieve PubMed and PubMed Central articles. Before running the package, configure your email address as an environment variable.

### macOS / Linux

```bash
export NCBI_EMAIL="your_email@example.com"
```

### Windows PowerShell

```powershell
$env:NCBI_EMAIL="your_email@example.com"
```

Replace `your_email@example.com` with your own email address.

For higher request limits, you may optionally configure an NCBI API key:

```bash
export NCBI_API_KEY="your_api_key"
```

If no API key is provided, the package uses the standard NCBI request limits.

> **Note**
>
> The package does **not** store or transmit your email except when making requests to the official NCBI Entrez API. The environment variable is used only to identify your requests to NCBI, in accordance with their API guidelines.

---

# Usage

Generate a definition for a psychological construct:

```bash
python -m psych_defgen_dummy.main loneliness
```

Multi-word constructs are supported:

```bash
python -m psych_defgen_dummy.main "social vulnerability"
```

Specify the number of retrieved articles and evidence passages:

```bash
python -m psych_defgen_dummy.main loneliness --max-results 20 --top-k 5
```

---


# Output

Generated definitions are saved as Markdown files containing:

- the generated ontology-style definition
- supporting evidence passages
- article metadata (title, authors, journal, PMID/PMCID)
- retrieval scores

Files are written to:

```text
outputs/loneliness_definition.md
```
The output filename is automatically generated from the requested psychological construct.
---

# Requirements

- Python 3.11+
- Valid email address for NCBI Entrez access

---

# License

- This project is licensed under the MIT License. 

# Citation

Citation information will be updated once the accompanying manuscript is published.