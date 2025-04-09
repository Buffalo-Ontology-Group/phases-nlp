import os
import pandas as pd
from gensim import corpora
from gensim.models import TfidfModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK resources if not already present
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess the text (tokenize, remove stopwords)
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())  # Tokenize the text and convert to lowercase
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stopwords and non-alphabetic words
    return tokens


# Function to compute and save TF-IDF scores for the entire collection of abstracts
def compute_tfidf_and_save(abstracts, folder_path, num_terms):
    # Preprocess the abstracts
    processed_texts = [preprocess_text(abstract) for abstract in abstracts]
    
    # Create a dictionary and corpus for TF-IDF modeling
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    # Create TF-IDF model based on the entire corpus
    tfidf = TfidfModel(corpus)
    
    # Apply TF-IDF transformation to the entire corpus
    tfidf_corpus = tfidf[corpus]

    # Create a dictionary to store TF-IDF scores for all terms
    term_tfidf_scores = {}

    # Iterate through the entire corpus and collect TF-IDF scores for all terms
    for doc in tfidf_corpus:
        for term_id, weight in doc:
            word = dictionary[term_id]
            if word not in term_tfidf_scores:
                term_tfidf_scores[word] = []
            term_tfidf_scores[word].append(weight)

    # Calculate the average TF-IDF score for each term across all documents
    avg_tfidf_scores = {term: sum(weights) / len(weights) for term, weights in term_tfidf_scores.items()}

    # Sort the terms based on their average TF-IDF scores
    sorted_terms = sorted(avg_tfidf_scores.items(), key=lambda x: x[1], reverse=True)

    # Save the top TF-IDF terms for the entire corpus
    term_file_path = os.path.join(folder_path, "tfidf_results.txt")
    
    with open(term_file_path, 'w') as file:
        # Write TF-IDF results for the entire corpus
        for term, avg_score in sorted_terms[:num_terms]:  # Top 10 terms
            file.write(f"{term}: {avg_score:.4f}\n")
    
    print(f"TF-IDF results saved to: {term_file_path}")


# Function to process the downloaded Excel file and compute TF-IDF for abstracts
def compute_tfidf_from_excel(folder_path, num_terms):
    # Load the Excel file
    excel_file = os.getenv('EXCEL_FILE_PATH')  # Fetch the value from the .env file
    df = pd.read_excel(excel_file)

    # Check if the required column exists
    if 'Abstracts' not in df.columns:
        print("Error: The Excel file must contain an 'Abstracts' column.")
        return

    # Extract the abstracts from the Excel file
    abstracts = df['Abstracts'].dropna().tolist()  # Extract abstracts and remove NaN values
    
    # Compute TF-IDF if abstracts are available
    if abstracts:
        print("Computing TF-IDF scores for the abstracts...")
        compute_tfidf_and_save(abstracts, folder_path, num_terms)
    else:
        print("No abstracts found to process for TF-IDF.")

# Function to generate and save a word cloud from the top TF-IDF terms
def generate_wordcloud(folder_path, num_terms):
    # Load the TF-IDF results
    term_file_path = os.path.join(folder_path, "tfidf_results.txt")
    
    with open(term_file_path, 'r') as file:
        lines = file.readlines()
    
    # Parse the terms and their scores
    terms = {}
    for line in lines:
        term, score = line.strip().split(": ")
        terms[term] = float(score)
    
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(terms)

    # Save the word cloud as an image
    wordcloud_image_path = os.path.join(folder_path, "wordcloud.png")
    wordcloud.to_file(wordcloud_image_path)

    # Display the word cloud (optional)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    print(f"Word cloud saved to: {wordcloud_image_path}")
