import os
import pandas as pd
import gensim
from gensim import corpora
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import CoherenceModel, LdaModel
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess the text (tokenize, remove stopwords)
def preprocess_text(text: str) -> list[str]:
    """
    Tokenizes and removes stopwords from a given text.

    Args:
        text (str): The input text to preprocess.

    Returns:
        List[str]: A list of cleaned, lowercased tokens with stopwords and non-alphabetic words removed. 
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

# Function to construct a custom eta matrix using seed words
def construct_eta(dictionary: corpora.Dictionary, seed_words_per_topic: List[List[str]], num_topics: int, eta_value: float = 0.9, default_eta: float = 0.01) -> np.ndarray:
    """
    Constructs a custom eta matrix for guided LDA using seed words.

    Args:
        dictionary (corpora.Dictionary): Gensim dictionary mapping of word IDs.
        seed_words_per_topic (List[List[str]]): List of seed words per topic.
        num_topics (int): Number of topics.
        eta_value (float): Value to assign for seed words.
        default_eta (float): Default value for other words.

    Returns:
        np.ndarray: A 2D array representing the eta matrix.
    """
    vocab_size = len(dictionary)
    eta = np.full((num_topics, vocab_size), default_eta)

    for topic_idx, seed_words in enumerate(seed_words_per_topic):
        for word in seed_words:
            if word in dictionary.token2id:
                word_id = dictionary.token2id[word]
                eta[topic_idx, word_id] = eta_value

    return eta

# Perform semi-guided LDA
def perform_guided_lda(topic_texts: List[str], num_topics: int, seed_words_per_topic: List[List[str]]) -> Tuple[LdaModel, corpora.Dictionary, List[List[Tuple[int, int]]], List[List[str]]]:
    """
    Performs semi-guided LDA using provided seed words.

    Args:
        topic_texts (List[str]): List of input documents.
        num_topics (int): Number of topics to extract.
        seed_words_per_topic (List[List[str]]): Seed words per topic for guidance.

    Returns:
        Tuple: (LdaModel, dictionary, corpus, processed_texts)
    """
    
    processed_texts = [preprocess_text(text) for text in topic_texts]
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    eta = construct_eta(dictionary, seed_words_per_topic, num_topics)

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=20,
        iterations=400,
        random_state=42,
        alpha='auto',
        eta=eta
    )

    return lda, dictionary, corpus, processed_texts

# Coherence
def compute_coherence_score(lda: LdaModel, corpus: List[List[Tuple[int, int]]], dictionary: corpora.Dictionary, processed_texts: List[List[str]]) -> float:
    """
    Computes the coherence score for the LDA model.

    Args:
        lda (LdaModel): Trained LDA model.
        corpus (List[List[Tuple[int, int]]]): Bag-of-words representation of documents.
        dictionary (corpora.Dictionary): Mapping of word IDs.
        processed_texts (List[List[str]]): Tokenized texts.

    Returns:
        float: Coherence score using the 'c_v' measure.
    """
    coherence_model = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, texts=processed_texts, coherence='c_v')
    return coherence_model.get_coherence()


def compute_perplexity(lda: LdaModel, corpus: List[List[Tuple[int, int]]]) -> float:
    """
    Computes the perplexity of the LDA model.

    Args:
        lda (LdaModel): Trained LDA model.
        corpus (List[List[Tuple[int, int]]]): Bag-of-words representation of documents.

    Returns:
        float: Exponentiated perplexity score.
    """
    log_perplexity = lda.log_perplexity(corpus)
    return np.exp(-log_perplexity)  # Exponentiate the negative log perplexity

# Topic variance
def compute_topic_variance(lda: LdaModel) -> float:
    """
    Computes the average variance across topic-word distributions.

    Args:
        lda (LdaModel): Trained LDA model.

    Returns:
        float: Mean variance across topics.
    """
    topic_word_distributions = lda.get_topics()
    variance = np.var(topic_word_distributions, axis=0).mean()
    return variance

# Save all metrics and topics
def save_topics_and_coherence_to_file(lda_model: LdaModel, coherence_score: float, perplexity_score: float, topic_variance_score: float, results_folder_path: str, num_words: int) -> None:
    """
    Saves the topics and evaluation metrics to a text file.

    Args:
        lda_model (LdaModel): Trained LDA model.
        coherence_score (float): Coherence score.
        perplexity_score (float): Perplexity score.
        topic_variance_score (float): Topic variance score.
        results_folder_path (str): Directory to save results.
        num_words (int): Number of words to display per topic.
    """

    os.makedirs(results_folder_path, exist_ok=True)
    topic_file_path = os.path.join(results_folder_path, "sme_guided_LDA_topic_modeling_results.txt")

    with open(topic_file_path, 'w') as file:
        file.write(f"SME Guided LDA Topic Modeling Results\n")
        file.write("=" * 50 + "\n")
        for i in range(lda_model.num_topics):
            file.write(f"Topic {i + 1}:\n")
            file.write(f"{lda_model.print_topic(i, num_words)}\n")
            file.write("=" * 50 + "\n")

        file.write(f"\nCoherence Score: {coherence_score}\n")
        file.write(f"Perplexity Score: {perplexity_score}\n")
        file.write(f"Topic Variance Score: {topic_variance_score}\n")
        file.write("=" * 50 + "\n")

    print(f"Topic modeling results and scores saved to: {topic_file_path}")

# Main function to call from CLI script
def perform_topic_modeling_on_excel_data_with_guided_lda(excel_file: str, num_topics: int, seed_words_per_topic: List[List[str]], num_words: int) -> Dict[str, Any]:
  
    """
    Loads text data from an Excel file and performs guided topic modeling.

    Args:
        excel_file (str): Path to Excel file containing abstracts.
        num_topics (int): Number of topics to extract.
        seed_words_per_topic (List[List[str]]): Seed words per topic.
        num_words (int): Number of words to display per topic in the results.

    Returns:
        Dict[str, Any]: Dictionary containing LDA model, corpus, dictionary, and evaluation scores.
    """
    try:
        df = pd.read_excel(excel_file)
        if 'Abstracts' not in df.columns:
            print("Error: The Excel file must contain an 'Abstracts' column.")
            return

        topic_texts = df['Abstracts'].dropna().tolist()
        if not topic_texts:
            print("Error: No text data found in the 'Abstracts' column of the Excel file.")
            return

        print(f"Performing topic modeling on {len(topic_texts)} abstracts...")

        lda, dictionary, corpus, processed_texts = perform_guided_lda(topic_texts, num_topics, seed_words_per_topic)
        coherence_score = compute_coherence_score(lda, corpus, dictionary, processed_texts)
        perplexity_score = compute_perplexity(lda, corpus)
        topic_variance_score = compute_topic_variance(lda)

        return {
            'lda': lda,
            'corpus': corpus,
            'dictionary': dictionary,
            'coherence_score': coherence_score,
            'perplexity_score': perplexity_score,
            'topic_variance_score': topic_variance_score
        }

    except Exception as e:
        print(f"Error while performing topic modeling: {e}")
        return None
