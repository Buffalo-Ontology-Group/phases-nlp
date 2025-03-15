import os
import requests
import time
import random
import click
from dotenv import load_dotenv 
from abstracts_retrieval import search_pubmed, retrieve_abstract_and_title
from topic_modeling import perform_topic_modeling_on_downloaded_texts  # Importing the topic modeling function

# Load environment variables from .env file
load_dotenv()

# Click command to get PubMed abstracts for given keywords
@click.command()
@click.option('--keywords', default='gerotranscendence,solitude,"gerotranscendence AND solitude"',
              help='Comma-separated list of keywords for PubMed search')
@click.option('--folder', default=None, help='Directory to save abstracts (default is from .env or current directory)')
@click.option('--num_abstracts', prompt='Number of abstracts to be downloaded', type=int,
              help='Number of abstracts to be downloaded from PubMed')
@click.option('--num_topics', prompt='Number of topics', type=int,
              help='Number of topics to be generated in topic modeling')

def get_pubmed_abstracts(keywords, folder, num_abstracts, num_topics):
    # If folder is not provided, use the folder from environment variables
    folder_path = folder if folder else os.getenv('PUBMED_DOWNLOAD_DIR')

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    # Create/overwrite the list.txt file to store titles and authors
    list_file_path = os.path.join(folder_path, 'list.txt')
    with open(list_file_path, 'w') as list_file:
        pass

    # Split the keywords into a list and fetch abstracts
    keyword_list = keywords.split(',')
    processed_pmids = set()

    total_articles = 0  # Count total articles downloaded

    for keyword in keyword_list:
        print(f"Searching for articles related to: {keyword.strip()}")
        pmids = search_pubmed(keyword.strip())
        
        if pmids:
            print(f"Found {len(pmids)} articles for keyword '{keyword.strip()}'. Retrieving abstracts and titles...")
            for pubmed_id in pmids[:num_abstracts]:  # Limit the number of abstracts to `num_abstracts`
                # Debugging: Show which PubMed ID is being processed
                print(f"Processing PubMed ID: {pubmed_id}")

                # Skip PubMed ID if already processed
                if pubmed_id in processed_pmids:
                    print(f"Skipping already processed PubMed ID: {pubmed_id}")
                    continue

                title, abstract, authors = retrieve_abstract_and_title(pubmed_id, folder_path)
                if title and abstract:

                    # Save only the abstract in a .txt file, excluding title and authors
                    file_name = f"{pubmed_id}_{title[:50]}.txt"  # Use the PubMed ID and truncated title for file name
                    file_path = os.path.join(folder_path, file_name)

                    with open(file_path, 'w') as file:
                        file.write(f"{abstract}\n")  # Only write the abstract to the file

                    # Save the title and authors to list.txt with space between them
                    with open(list_file_path, 'a') as list_file:
                        list_file.write(f"{title}\n\n{authors}\n\n........................\n\n")  # Add space between title and authors
                    
                    # Add the PubMed ID to the set to track processed articles
                    processed_pmids.add(pubmed_id)
                    total_articles += 1  # Increment total articles counter

                if total_articles >= num_abstracts:
                    break  # Stop once the desired number of abstracts are downloaded

    # Debugging: Print the total number of articles processed
    print(f"Total articles downloaded: {total_articles}")

    # After downloading abstracts, perform topic modeling and save results
    perform_topic_modeling_on_downloaded_texts(folder_path, num_topics) # Pass num_topics to topic modeling function

# Run the Click command
if __name__ == '__main__':
    get_pubmed_abstracts()  # Run the command to download and process the abstracts
