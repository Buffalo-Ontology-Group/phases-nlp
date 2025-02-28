import os
import requests
import time
import random
import click
from dotenv import load_dotenv
from topic_modeling_abstracts import search_pubmed, retrieve_abstract_and_title, preprocess_text, perform_lda, display_topics_for_abstract

# Load environment variables from .env file
load_dotenv()

# Click command to fetch PubMed abstracts for given keywords
@click.command()
@click.option('--keywords', default='gerotranscendence,solitude,healthy aging', help='Comma-separated list of keywords for PubMed search')
@click.option('--folder', default=None, help='Directory to save abstracts (default is from .env or current directory)')
def fetch_pubmed_abstracts(keywords, folder):
    # If folder is not provided, use the folder from environment variables
    folder_path = folder if folder else os.getenv('PUBMED_DOWNLOAD_DIR')

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    # Split the keywords into a list and fetch abstracts
    keyword_list = keywords.split(',')
    topic_texts = []
    titles = []

    for keyword in keyword_list:
        print(f"Searching for articles related to: {keyword.strip()}")
        pmids = search_pubmed(keyword.strip())
        
        if pmids:
            print(f"Found {len(pmids)} articles for keyword '{keyword.strip()}'. Retrieving abstracts and titles...")
            for pubmed_id in pmids:
                title, abstract, authors, source = retrieve_abstract_and_title(pubmed_id, folder_path)
                if title and abstract:
                    topic_texts.append(abstract)  # Collect abstracts for topic modeling
                    titles.append(title)

    # Perform LDA topic modeling on all abstracts
    if topic_texts:
        print("Performing topic modeling on the abstracts...")
        lda = perform_lda(topic_texts)

        # Display topics for each abstract
        for i, abstract in enumerate(zip(topic_texts, titles)):
            display_topics_for_abstract(lda, abstract, titles[i])

# Run the Click command
if __name__ == '__main__':
    fetch_pubmed_abstracts()
