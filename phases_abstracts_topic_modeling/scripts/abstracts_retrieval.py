import os
import requests
import time
import random
from dotenv import load_dotenv

# Download NLTK resources (if required for other processes)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Define the base URL for PubMed search and fetch
search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Load PubMed API Key from environment
API_KEY = os.getenv("PUBMED_API_KEY")  # API key is to be stored in the .env file

# Function to search PubMed for articles using the keyword
def search_pubmed(keyword):
    params = {
        'db': 'pubmed',
        'term': keyword,  # The search term 
        'retmax': 5,  # number of results 
        'retmode': 'xml',
        'api_key': API_KEY  # Use the API key for requests
    }
    response = requests.get(search_url, params=params)
    
    if response.status_code == 200:
        try:
            from xml.etree import ElementTree as ET
            tree = ET.ElementTree(ET.fromstring(response.text))
            root = tree.getroot()
            
            # Extract PubMed IDs (PMIDs) from the search results
            pmids = [id.text for id in root.findall(".//Id")]
            return pmids
        except Exception as e:
            print(f"Error parsing search response for '{keyword}': {e}")
            return []
    else:
        print(f"Search request failed for '{keyword}'. Status code: {response.status_code}")
        return []

# Function to retrieve and save the abstract and title using PubMed ID
def retrieve_abstract_and_title(pubmed_id, folder_path, retries=5, backoff=60):
    for attempt in range(retries):
        params = {
            'db': 'pubmed',
            'id': pubmed_id,
            'retmode': 'xml',         # Response format (XML)
            'rettype': 'abstract',    # Only retrieve abstract and title
            'api_key': API_KEY        # Include API key in the request
        }
        
        response = requests.get(fetch_url, params=params)
        
        if response.status_code == 200:
            try:
                from xml.etree import ElementTree as ET
                tree = ET.ElementTree(ET.fromstring(response.text))
                root = tree.getroot()

                # Find the abstract and title in the XML response
                abstract = root.find(".//AbstractText")
                title = root.find(".//ArticleTitle")
                authors = root.findall(".//Author")
                
                # Safe check for None before using `strip()` on title and abstract
                if abstract is not None:
                    abstract_text = abstract.text.strip() if abstract.text else "No abstract available"
                else:
                    abstract_text = "No abstract available"

                if title is not None:
                    clean_title = title.text.strip() if title.text else "No title available"
                    clean_title = clean_title.replace('/', '_').replace('\\', '_').replace(':', '_')
                else:
                    clean_title = "No title available"
                
                # Extract authors
                author_list = []
                for author in authors:
                    last_name = author.find("LastName")
                    fore_name = author.find("ForeName")
                    if last_name is not None and fore_name is not None:
                        author_name = f"{fore_name.text} {last_name.text}"
                    elif last_name is not None:
                        author_name = last_name.text
                    elif fore_name is not None:
                        author_name = fore_name.text
                    else:
                        author_name = "Unknown Author"
                    author_list.append(author_name)

                # Join authors with commas
                authors_text = ', '.join(author_list) if author_list else "No authors available"

                # Combine PubMed ID and title to create the file name
                file_name = f"{pubmed_id}_{clean_title[:50]}.txt"  # Truncate title to avoid overly long names
                file_path = os.path.join(folder_path, file_name)
                
                # Save the title and abstract in the file
                with open(file_path, 'w') as file:
                    file.write(f"Title: {clean_title}\n\n")  # Add the title at the top
                    file.write(f"Authors: {authors_text}\n\n")  # Add authors below the title
                    file.write(f"Abstract: {abstract_text}\n")  # Save abstract text below the title
                
                print(f"Saved abstract, title, authors, and source for PMID {pubmed_id}")

                return clean_title, abstract_text, authors_text # Return title, abstract, authors, and source for further processing
            except Exception as e:
                print(f"Error parsing response for PMID {pubmed_id}: {e}")
                return None, None, None

        elif response.status_code == 429:
            # Exponential backoff strategy with a random delay added to avoid retry bursts
            print(f"Rate limit exceeded for PMID {pubmed_id}, retrying in {backoff} seconds...")
            time.sleep(backoff + random.uniform(0, 5))  # Adding randomness to the backoff time
            backoff *= 2  # Exponentially increase the wait time
        else:
            print(f"Failed to fetch data for PMID {pubmed_id}. Status code: {response.status_code}")
            break
    
    return None, None, None  # Return None if all retries fail
