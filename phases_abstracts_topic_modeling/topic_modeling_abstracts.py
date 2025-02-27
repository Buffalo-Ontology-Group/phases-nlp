import os
import requests
import time
import random
import gensim
import nltk
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict

# Download NLTK resources
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
        'retmax': 20,  # number of results 
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

                # Combine PubMed ID and title to create the file name
                file_name = f"{pubmed_id}_{clean_title[:50]}.txt"  # Truncate title to avoid overly long names
                file_path = os.path.join(folder_path, file_name)
                
                # Save the abstract in the file
                with open(file_path, 'w') as file:
                    file.write(abstract_text)  # Save abstract text in the file
                print(f"Saved abstract and title for PMID {pubmed_id}")

                return clean_title, abstract_text  # Return title and abstract for topic modeling
            except Exception as e:
                print(f"Error parsing response for PMID {pubmed_id}: {e}")
                return None, None
        elif response.status_code == 429:
            # Exponential backoff strategy with a random delay added to avoid retry bursts
            print(f"Rate limit exceeded for PMID {pubmed_id}, retrying in {backoff} seconds...")
            time.sleep(backoff + random.uniform(0, 5))  # Adding randomness to the backoff time
            backoff *= 2  # Exponentially increase the wait time
        else:
            print(f"Failed to fetch data for PMID {pubmed_id}. Status code: {response.status_code}")
            break
    
    return None, None  # Return None if all retries fail

# Preprocess the abstracts for topic modeling (tokenize, remove stopwords)
def preprocess_text(text):
    if isinstance(text, tuple):
        text = text[0]
        
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())  # Tokenize the text and convert to lowercase
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stopwords and non-alphabetic words
    return tokens

# Function for LDA Topic Modeling
def perform_lda(topic_texts):
    # Preprocess the abstracts (tokenization)
    processed_texts = [preprocess_text(text) for text in topic_texts]
    
    # Create a dictionary and corpus for topic modeling
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # Perform LDA
    lda = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

    return lda

# Function to display topics for each abstract
def display_topics_for_abstract(lda, abstract, title):
    print(f"Abstract Title: {title}")
    topics = lda[corpora.Dictionary([preprocess_text(abstract)]).doc2bow(preprocess_text(abstract))]

    # Sort topics by probability in descending order (most relevant to least)
    sorted_topics = sorted(topics, key=lambda x: x[0], reverse=True)

    print("Topics (sorted by relevance):")
    for topic_num, prob in sorted_topics:
        print(f"Topic {topic_num}: {lda.print_topic(topic_num, 5)} (Relevance: {prob:.4f})")  # Print top 5 words of each topic with relevance
    print("="*50)
