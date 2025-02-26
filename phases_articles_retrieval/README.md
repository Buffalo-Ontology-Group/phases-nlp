# Phases Articles Retrieval

This project is a Python application that retrieves articles related to specific topics,"gerotranscendence" and "solitude", downloads their PDFs, and saves them to a local directory. 

## Features

- Retrieves articles from PubMed.
    - The number of articles retrieved for each topic can be customized by adjusting the 'max_results' parameter. It can be modified to retrieve a higher or lower number of articles.
- Searches for available PDF links (via PubMed Central, Google, and Google Scholar).
- Downloads and saves PDFs to a user-defined directory.

## Installation

To get started with the **Phases Articles Retrieval** project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/YOUR_USERNAME/phases_articles_retrieval.git
    ```

2. Navigate into the repository folder:

    ```bash
    cd phases_articles_retrieval
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Requirements

Before running the application, ensure that the following dependencies are installed:

- Python 3.x
    - `requests`
    - `python-dotenv`
    - `biopython`
    - `googlesearch-python`
    - `scholarly`

## Usage

Once the installation is complete, the project can be used by following the instructions. Below are the steps to run the application:

1. Set up your `.env` file:

    Create a `.env` file in the root of the project directory and add the environment variables for the directory path for saving abstracts.

2. **Run the application**:

    After installing the dependencies, you can run the script by executing the following command:

    ```bash
    python main.py
    ```

3. **Configuration**:

    Adjust the `max_results` parameter to customize the number rof articles retrieved.
   
## Contributing

To track changes made to this project, it is best maintained by following these steps:

1. Submit an issue detailing the problem.
2. Create a branch to address this issue that uses the same number as the issue tracker. For example, if the issue is `#50` in the issue tracker, name the branch `issue-50`. This allows other developers to easily know which branch needs to be checked out to contribute.
3. Create a pull request that fixes the issue. If possible, create a draft (or WIP) branch early in the process.
4. Merge pull request once all the necessary changes have been made. If needed, tag other developers to review the pull request.
5. Delete the issue branch (e.g., branch `issue-50`).







 
