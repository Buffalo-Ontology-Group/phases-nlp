import os
import pandas as pd
from gensim import corpora
from gensim.models import TfidfModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess the text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words]

# Compute and save TF-IDF scores
def compute_tfidf_and_save(abstracts, folder_path=None, num_terms=30):

    folder_path = folder_path or os.getenv('TF_IDF_RESULTS')
    # Ensure the directory exists
    os.makedirs(folder_path, exist_ok=True)

    processed_texts = [preprocess_text(abstract) for abstract in abstracts]
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    tfidf = TfidfModel(corpus)
    tfidf_corpus = tfidf[corpus]

    term_tfidf_scores = {}
    for doc in tfidf_corpus:
        for term_id, weight in doc:
            word = dictionary[term_id]
            term_tfidf_scores.setdefault(word, []).append(weight)

    avg_tfidf_scores = {term: sum(weights) / len(weights) for term, weights in term_tfidf_scores.items()}
    sorted_terms = sorted(avg_tfidf_scores.items(), key=lambda x: x[1], reverse=True)

    term_file_path = os.path.join(folder_path, "tf_idf_results.txt")
    with open(term_file_path, 'w') as file:
            for term, _ in sorted_terms[:num_terms]:
                file.write(f"{term}\n")
    
    print(f"TF-IDF results saved to: {term_file_path}")

# Read Excel and compute TF-IDF
def compute_tfidf_from_excel(folder_path, num_terms):
    excel_file = os.getenv('EXCEL_FILE_PATH')
    df = pd.read_excel(excel_file)

    if 'Abstracts' not in df.columns:
        print("Error: The Excel file must contain an 'Abstracts' column.")
        return

    abstracts = df['Abstracts'].dropna().tolist()
    if abstracts:
        print("Computing TF-IDF scores...")
        compute_tfidf_and_save(abstracts, folder_path, num_terms)
    else:
        print("No abstracts found.")

# Main function
def main():
    folder_path = os.getenv('TF_IDF_RESULTS')
    os.makedirs(folder_path, exist_ok=True)

    num_terms = 30  # Pass the number of top terms here
    print("Starting processing...")
    compute_tfidf_from_excel(folder_path, num_terms)

if __name__ == '__main__':
    main()
