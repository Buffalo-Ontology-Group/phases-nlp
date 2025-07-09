# PHASES Abstracts Guided LDA (Latent Dirichlet Allocation)

This project features three subprojects, Standard LDA, SME (Subject Matter Experts) guided LDA and TF-IDF (Term Frequency-Inverse Document Frequency) guided LDA.

Standard LDA: A basic Latent Dirichlet Allocation model for topic modeling.

SME guided LDA: A variant of LDA that incorporates subject matter expert guidance for improved performance.

TF-IDF guided LDA: Another variant of LDA, enhanced with TF-IDF weighting to improve topic quality.

Each of these subprojects applies LDA (Latent Dirichlet Allocation) for topic modeling on scientific abstracts related to topics such as gerotranscendence and solitude.

## Features

- Topic modeling using basic LDA and via guided LDA incorporating seed word priors and top words
- In case of SME guided LDA, common words extraction from highlighted word document files
- In case of TF-IDF guided LDA, top words generation is using TF-IDF computing. 
- Coherence, perplexity, and topic variance score calculation
- PyLDAvis interactive visualizations
- Heatmap generation for topic-word distribution

## Configuration

The number of topics and number of words for each topic to be generated are prompted at the command line. When running the script, the user will be asked to enter the desired number of topics and words, which can be customized interactively.

### Installation

To get started with the **phases_abstracts_guided_LDA**, follow these steps:

1. Clone the repository:

    `git clone https://github.com/YOUR_USERNAME/phases_abstracts_guided_LDA.git ~/myfolder/phases_abstracts_guided_LDA`

2. Navigate into the repository folders:

    `cd ~/myfolder/phases_abstracts_guided_LDA`

3. Install the required dependencies:

    `pip install -r requirements.txt`

### Requirements

Before running the application, ensure that the following dependencies are installed:

- Python 3.x
  
    - `click`
    - `python-dotenv`
    - `pandas`
    - `nltk`
    - `gensim`
    - `pyLDAvis`
    - `matplotlib`
    - `seaborn`
    - `python-docx`
    - `openpyxl`
   
### Usage

Once the installation is complete, the project can be used by following the instructions. Below are the steps to run the application:

1. Set up your `.env` file:

    Create a `.env` file in the root of the project directory and add the environment variables for the directory path for input and output files.

2. **Run the application**:

    After installing the dependencies, you can run the scripts by executing the following command for the corresponding projects:

    - For standard_LDA - `python main_standard_LDA.py`
    - For sme_guided_LDA - `python main_sme_guided_LDA.py`
    - For tf_idf_guided_LDA - `python main_tf_idf_guided_LDA.py`

### Contributing

To track changes made to this project, it is best maintained by following these steps:

1. Submit an issue detailing the problem.
2. Create a branch to address this issue that uses the same number as the issue tracker. For example, if the issue is `#50` in the issue tracker, name the branch `issue-50`. This allows other developers to easily know which branch needs to be checked out to contribute.
3. Create a pull request that fixes the issue. If possible, create a draft (or WIP) branch early in the process.
4. Merge pull request once all the necessary changes have been made. If needed, tag other developers to review the pull request.
5. Delete the issue branch (e.g., branch `issue-50`).
