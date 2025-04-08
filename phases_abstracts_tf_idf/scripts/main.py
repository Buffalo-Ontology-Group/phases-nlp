import os
import click
from dotenv import load_dotenv
from tfidf import compute_tfidf_from_excel  # Import the modified TF-IDF function

# Load environment variables from .env file
load_dotenv()

@click.command()
@click.option('--folder', default=None, help='Directory to save results (default is from .env or current directory)')
@click.option('--num_terms', prompt='Number of top terms to be retrieved', type=int, help='Number of top terms to display in TF-IDF results')

def process_abstracts(folder, num_terms):
    # If folder is not provided, use the folder from environment variables
    folder_path = folder if folder else os.getenv('RESULTS_DIR')

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # After categorizing, compute TF-IDF and save results
    print(f"Processing abstracts from Excel and computing TF-IDF...")
    
    # Call the TF-IDF function with the specified number of terms
    compute_tfidf_from_excel(folder_path, num_terms)  # Now passing num_terms to the function

# Run the Click command
if __name__ == '__main__':
    process_abstracts()  # Run the command to process the abstracts and compute TF-IDF
