# PHASES Article Retrieval

This project is a Python application that retrieves articles related to specific topics, downloads their PDFs, and saves them to a local directory. The topics are "gerotranscendence" and "solitude".

## Features

- Retrieves articles from PubMed.
    - The number of articles retrieved for each topic can be customized by adjusting the 'max_results' parameter. It can be modified to retrieve a higher or lower number of articles based  on the need.
- Searches for available PDF links (via PubMed Central, Google, and Google Scholar).
- Downloads and saves PDFs to a user-defined directory.


## Requirements

Before running the application, ensure that the following dependencies are installed:

- Python 3.x
    - 'requests'
    - 'python-dotenv'
    - 'biopython'
    - 'googlesearch-python'
    - 'scholarly'

You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt

## Installation
1. Clone the repository:

    ```bash
    git clone https://github.com/YOUR_USERNAME/article_retrieval.git
    cd article_retrieval


 