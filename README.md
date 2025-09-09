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

### Term Frequency-Inverse Document Frequency (TF_IDF) computation on research abstracts of related to the PHASES project

Computes TF-IDF scores for a collection of research abstracts, related to specific topics such as gerotranscendence and solitude. 

### Guided Latent Dirichlet Allocation (LDA) on Abstracts Related to PHASES: Standard LDA, Subject Matter Guided (SME) LDA, and TF-IDF guided LDA

Performs guided topic modeling using LDA on research abstracts related to gerotranscendence and solitude. This project explores topic modeling across different phases of research using three distinct approaches to Latent Dirichlet Allocation (LDA). The first approach applies standard LDA without any seed words. The second approach uses SME-guided LDA, where domain-specific seed words are provided by SMEs to guide topic generation toward thematically relevant concepts. The third approach implements TF-IDF-guided LDA, using automatically extracted keywords based on TF-IDF scores from the corpus to serve as seed words. By applying and comparing these methods across research phases, the project aims to understand how guidance affects topic quality, coherence, and interpretability in a domain-specific context.

### BERTopic Modeling on Solitude and Gerotranscendence books

Identifies and analyzes latent themes related to solitude and gerotranscendence from two key books, The Handbook of Solitude: Psychological Perspectives on Social Isolation, Social Withdrawal, and Being Alone and Gerotranscendence: A Developmental Theory of Positive Aging. Topics are extracted using BERTopic (both default and custom preprocessing pipelines), and after validation by subject matter experts (SMEs), the refined topics are to be incorporated as concepts within the PHASES ontology. Also, the bertopic modelig output text files from solitude and gerotranscendence are processed to compute cross domain similairity.

### BERTopic Modeling on Plotkins Vaccines book

This project is a Python application that builds a BERTopic-based pipeline for analyzing chapters from Plotkin’s Vaccines. It extracts and explores latent themes related to vaccines using BERTopic with a custom preprocessing workflow. The pipeline generates hierarchical topic trees that organize concepts found in the text, and these trees are then merged into a unified directed ontology graph. In addition, the project provides visual outputs such as keyword bar charts, hierarchical clustering dendrograms, and BART-based summaries to support deeper interpretation of the topics.

### Installation

To get started with the **phases-nlp**, follow these steps:

1. Clone the repository:

    `git clone https://github.com/YOUR_USERNAME/phases-nlp.git ~/myfolder/phases-nlp`

2. Navigate into the repository folders:

    `cd ~/myfolder/phases-nlp`

3. Install the required dependencies:

    `pip install -r requirements.txt`
    `python -m spacy download en_core_web_sm`

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
    - `bertopic`
    - `sentence-transformers`
    - `spacy`
    - `scikit-learn`
    - `hdbscan`
    - `umap-learn`
    - `keybert`
    - `transformers`
    - `plotly`
    - `pyyaml`
    - `Local Stanford CoreNLP server running with the OpenIE annotator`. The server is accessible at `http://localhost:9000`.
 
### Stanford CoreNLP Setup

- Download Stanford CoreNLP from `http://stanfordnlp.github.io/CoreNLP/`
- Unzip and navigate to the CoreNLP directory.
- Start the server with OpenIE annotator enabled:
  ```bash
   java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
        -port 9000 -timeout 15000 -annotators tokenize,ssplit,pos,lemma,depparse,natlog,openie
      
### Usage

Once the installation is complete, the project can be used by following the instructions. Below are the steps to run the application:

1. Set up your `.env` file:

    Create a `.env` file in the root of the project directory and add the environment variables for the directory path for saving PDFs, abstracts and for the API key for PubMed.

2. **Run the application**:

   After installing the dependencies, run the script by executing the following command:

    `python main.py`
   
   **For BERTopic modeling:**
   
   `python bertopic_default_solitude.py'
   
   'python bertopic_custom_preprocessing_solitude.py`
   
   `python bertopic_default_gerotranscendence.py`
   
   `python bertopic_custom_preprocessing_gerotranscendence.py`

   `python overlap_topic_sol_gero.py`
   

### Contributing

To track changes made to this project, it is best maintained by following these steps:

1. Submit an issue detailing the problem.
2. Create a branch to address this issue that uses the same number as the issue tracker. For example, if the issue is `#50` in the issue tracker, name the branch `issue-50`. This allows other developers to easily know which branch needs to be checked out to contribute.
3. Create a pull request that fixes the issue. If possible, create a draft (or WIP) branch early in the process.
4. Merge pull request once all the necessary changes have been made. If needed, tag other developers to review the pull request.
5. Delete the issue branch (e.g., branch `issue-50`).
