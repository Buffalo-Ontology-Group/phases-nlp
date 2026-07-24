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

## Installation

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

## Development Installation

To install the project from GitHub:


```bash
git clone https://github.com/Buffalo-Ontology-Group/psychological_construct_definition_generator.git
cd psychological_construct_definition_generator

pip install -e .
```

Editable installation allows you to make changes to the source code and use them immediately without reinstalling the package.



---

# Configure NCBI Credentials

This package uses the NCBI Entrez API to retrieve PubMed and PubMed Central articles. An NCBI email address is required to access the NCBI Entrez API. You can provide your credentials either through environment variables or as command-line arguments.

## Option 1: Environment variables (recommended)

### macOS / Linux

```bash
export NCBI_EMAIL="YOUR_NCBI_EMAIL"
```

### Windows PowerShell

```powershell
$env:NCBI_EMAIL="YOUR_NCBI_EMAIL"
```

Replace `YOUR_NCBI_EMAIL` with your own email address.

For higher request limits, you may optionally configure an NCBI API key.

### macOS / Linux

```bash
export NCBI_API_KEY="YOUR_API_KEY"
```

### Windows PowerShell

```powershell
$env:NCBI_API_KEY="YOUR_API_KEY"
```

If no API key is provided, the package uses the standard NCBI request limits.

## Option 2: Command-line arguments

Instead of environment variables, you can provide your NCBI email directly when running the program:

```bash
python -m psych_defgen_dummy.main loneliness \
    --email YOUR_NCBI_EMAIL
```

To also use an NCBI API key:

```bash
python -m psych_defgen_dummy.main loneliness \
    --email YOUR_NCBI_EMAIL \
    --api-key YOUR_API_KEY
```

If both environment variables and command-line arguments are provided, the command-line arguments take precedence.

> **Note**
>
> The package does **not** store or transmit your email address or API key except when making requests to the official NCBI Entrez API. These credentials are used only to identify your requests in accordance with NCBI API guidelines.

---

# Usage

Generate a definition for a psychological construct:

```bash
python -m psych_defgen_dummy.main loneliness
```

Alternatively, specify your NCBI email directly:

```bash
python -m psych_defgen_dummy.main loneliness \
    --email YOUR_NCBI_EMAIL
```

Multi-word constructs are supported:

```bash
python -m psych_defgen_dummy.main "social vulnerability"
```

Specify the number of retrieved articles and evidence passages:

```bash
python -m psych_defgen_dummy.main loneliness \
    --max-results 20 \
    --top-k 5
```

Specify a custom output file:

```bash
python -m psych_defgen_dummy.main loneliness \
    --output results/loneliness_definition.md
```

The NCBI email can be combined with other options:

```bash
python -m psych_defgen_dummy.main loneliness \
    --email YOUR_NCBI_EMAIL \
    --max-results 20 \
    --top-k 5 \
    --output results/loneliness_definition.md
```

Display all available command-line options:

```bash
python -m psych_defgen_dummy.main --help
```

---

# Output


By default, generated definitions are saved as Markdown files in the `outputs` directory. A custom output file can be specified using the `--output` option.


```text
outputs/loneliness_definition.md
```

The output filename is automatically generated from the requested psychological construct.

---

# Requirements

- Python 3.11+
- Valid NCBI email address

An NCBI API key is optional but recommended for higher request rate limits.

---

# License

This project is licensed under the MIT License. 

# Citation

Citation information will be updated once the accompanying manuscript is published.