# PHASES NLP

Natural language processing methods used for the PHASES project.

## Motivation

PHASES project aims at bridging the gap between Solitude and Gerotranscendence literatures to promote novel research into healthy aging. The aim is to develop Solitude Ontology and Gerotranscendence Ontology each extending from the upper-level Behavior Change Intervention Ontology. Ontologies provide explicit representation of any domain transforming information to knowledge and the Ontology development process requires a great deal of time and resources. Modeling Solitude and Gerotranscendence Ontologies require information to be acquired from available scientific resources and it necessitates domain experts to devote substantial time and effort to find the information. The aim of PHASES-NLP is to apply natural language analysis and machine learning techniques is to reduce the effort in the information acquisition process. Subsequently we aim at streamlining the process of knowledge acquisition.

## Projects

This repository is a Python application that supports PHASES project by retrieving and processing scientific articles.

### PHASES Articles Retrieval

Retrieves articles related to specific topics,"gerotranscendence" and "solitude" and downloads their PDFs.

### Topic Modeling on Abstracts Related to PHASES Articles 

Searches for scientific articles related to specific topics such as gerotranscendence, solitude and healthy aging, retrieve their abstracts and titles, and performs topic modeling using Latent Dirichlet Allocation (LDA).

### Installation

To get started with the **phases-nlp**, follow these steps:

1. Clone the repository:

    `git clone https://github.com/YOUR_USERNAME/phases-nlp.git ~/myfolder/phases-nlp`

2. Navigate into the repository folders:

    `cd ~/myfolder/phases-nlp`

3. Install the required dependencies:

    `pip install -r requirements.txt`

### Requirements

Before running the application, ensure that the following dependencies are installed:

- Python 3.x
  
    - `requests`
    - `python-dotenv`
    - `gensim`
    - `biopython`
    - `googlesearch-python`
    - `scholarly`
    - `click`
    - `nltk`
    - `pandas`
    - `openpyxl`
    - `matplotlib`
    - `seaborn`
    - `python-docx`
    - `pyLDAvis`
      

### Usage

Once the installation is complete, the project can be used by following the instructions. Below are the steps to run the application:

1. Set up your `.env` file:

    Create a `.env` file in the root of the project directory and add the environment variables for the directory path for saving PDFs, abstracts and for the API key for PubMed.

2. **Run the application**:

    After installing the dependencies, you can run the script by executing the following command:

    `python main.py`

### Contributing

To track changes made to this project, it is best maintained by following these steps:

1. Submit an issue detailing the problem.
2. Create a branch to address this issue that uses the same number as the issue tracker. For example, if the issue is `#50` in the issue tracker, name the branch `issue-50`. This allows other developers to easily know which branch needs to be checked out to contribute.
3. Create a pull request that fixes the issue. If possible, create a draft (or WIP) branch early in the process.
4. Merge pull request once all the necessary changes have been made. If needed, tag other developers to review the pull request.
5. Delete the issue branch (e.g., branch `issue-50`).
