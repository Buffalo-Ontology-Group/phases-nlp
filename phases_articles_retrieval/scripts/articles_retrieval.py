import os
import requests
import re
from Bio import Entrez
from googlesearch import search
from scholarly import scholarly
from dotenv import load_dotenv
import ssl

# Disable SSL verification temporarily to handle certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables
load_dotenv()
Entrez.email = os.getenv('NCBI_EMAIL')

# Define the parent directory for downloading PDFs
download_directory = os.getenv("DOWNLOAD_DIRECTORY")
if not os.path.exists(download_directory):
    os.makedirs(download_directory)

# Create subdirectories for each topic
gerotranscendence_dir = os.path.join(download_directory, "gerotranscendence")
solitude_dir = os.path.join(download_directory, "solitude")

# Create directories if they do not exist
if not os.path.exists(gerotranscendence_dir):
    os.makedirs(gerotranscendence_dir)

if not os.path.exists(solitude_dir):
    os.makedirs(solitude_dir)

# Keywords for searching articles
keywords_gerotranscendence = "gerotranscendence"
keywords_solitude = "solitude"

# Function to sanitize file name (remove special characters)
def sanitize_filename(title):
    """Sanitize the title to be a valid file name."""
    title = re.sub(r'[\\/*?:"<>|]', "", title)  # Remove invalid characters from file name
    return title

# Function to download PDFs
def download_pdf(pdf_url, article_id, title, topic_dir, failed_downloads):
    """Download the PDF file for the article, if available."""
    try:
        if not pdf_url:
            print(f"No PDF URL found for article {article_id}")
            failed_downloads.append((article_id, title, "No PDF URL"))
            return None
        
        # Sanitize the title to be a valid filename
        sanitized_title = sanitize_filename(title)
        pdf_filename = f"{sanitized_title}.pdf"
        pdf_filepath = os.path.join(topic_dir, pdf_filename)
        
        # Check if the file already exists
        if os.path.exists(pdf_filepath):
            print(f"PDF already exists for article {article_id}: {pdf_filename}. Skipping download.")
            return pdf_filepath  # Skip download if file already exists
        
        print(f"Attempting to download PDF for article {article_id} from {pdf_url}")
        response = requests.get(pdf_url, stream=True)

        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' not in content_type.lower():
                print(f"Warning: The content at {pdf_url} is not a PDF. Skipping download.")
                failed_downloads.append((article_id, title, "Not a PDF"))
                return None
            
            # Download the PDF if the response is OK and content type is PDF
            with open(pdf_filepath, 'wb') as pdf_file:
                for chunk in response.iter_content(chunk_size=128):
                    pdf_file.write(chunk)

            print(f"Downloaded PDF: {pdf_filename}")
            return pdf_filepath
        else:
            print(f"Failed to download PDF for article {article_id}. Status code: {response.status_code}")
            failed_downloads.append((article_id, title, f"Failed with status code {response.status_code}"))
            return None
    except Exception as e:
        print(f"Error downloading PDF for article {article_id}: {e}")
        failed_downloads.append((article_id, title, f"Error: {e}"))
        return None

# Function to fetch article IDs from PubMed
def retrieve_and_download_articles(query, max_results=10):
    """Fetch articles from PubMed based on the query."""
    try:
        print(f"Retrieving articles for query: {query}...")
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, usehistory="y")
        record = Entrez.read(handle)
        handle.close()

        id_list = record.get("IdList", [])
        if not id_list:
            print(f"No articles found for the query '{query}'.")
            return []  # Return an empty list if no articles are found
        
        return id_list
    
    except Exception as e:
        print(f"Error during PubMed search: {e}")
        return []  # Return an empty list on error

# Function to get the PMC PDF URL from PubMed
def get_pmc_pdf_url(article_id):
    """Fetch the PMC link for a given PubMed article ID."""
    try:
        handle = Entrez.elink(dbfrom="pubmed", id=article_id, linkname="pubmed_pubmed_literature")
        links = Entrez.read(handle)
        handle.close()

        for link in links[0].get("LinkSetDb", []):
            for link_data in link.get("Link", []):
                if 'pmc' in link_data['Id']:  # Look for PMC articles
                    pmc_article_id = link_data["Id"]
                    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_article_id}/pdf/"
                    print(f"Found PMC PDF link: {pdf_url}")
                    return pdf_url
        print(f"No PMC PDF link found for article {article_id}")
        return None
    except Exception as e:
        print(f"Error getting PMC PDF URL for article {article_id}: {e}")
        return None

# Function to search Google for potential PDFs
def search_google_for_pdf(title):
    """Search Google for the article title to find a potential PDF link."""
    query = f"{title} filetype:pdf"
    search_results = search(query, num_results=5)
    for url in search_results:
        if 'pdf' in url.lower():  # Only consider URLs that contain 'pdf'
            print(f"Google Search found PDF link: {url}")
            return url
    return None

# Function to search Google Scholar for PDFs
def search_scholar_for_pdf(title):
    """Search Google Scholar for the article and return a link to the PDF."""
    search_query = scholarly.search_pubs(title)
    try:
        first_result = next(search_query)
        print(f"Google Scholar found article: {first_result['bib']['title']}")
        pdf_link = first_result.get('url_pdf')
        if pdf_link:
            print(f"Google Scholar found PDF link: {pdf_link}")
            return pdf_link
    except StopIteration:
        print("No results found in Google Scholar.")
    return None

# Function to process articles and attempt PDF downloads
def process_articles(id_list, topic_dir):
    """Process articles by fetching summaries and downloading PDFs."""
    articles = []
    failed_downloads = []

    if not id_list:
        return articles, failed_downloads
    
    for article_id in id_list:
        try:
            handle = Entrez.esummary(db="pubmed", id=article_id)
            summary = Entrez.read(handle)
            handle.close()

            article_title = summary[0]["Title"]
            article_url = f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/"

            # First, try to get PDF URL from PubMed Central (PMC)
            pdf_url = get_pmc_pdf_url(article_id)

            # If no PDF found in PubMed Central, try Google Search and Google Scholar
            if not pdf_url:
                pdf_url = search_google_for_pdf(article_title)
            
            if not pdf_url:
                pdf_url = search_scholar_for_pdf(article_title)

            # If no PDF URL found, back to the PubMed article URL
            if not pdf_url:
                pdf_url = article_url

            if pdf_url:
                pdf_filepath = download_pdf(pdf_url, article_id, article_title, topic_dir, failed_downloads)
                if pdf_filepath:
                    articles.append((article_id, article_title, article_url, pdf_filepath))
            else:
               failed_downloads.append((article_id, article_title, "No PDF URL"))
            
        except Exception as e:
            print(f"Error processing article {article_id}: {e}")
            failed_downloads.append((article_id, "", f"Error: {e}"))
    
    return articles, failed_downloads
