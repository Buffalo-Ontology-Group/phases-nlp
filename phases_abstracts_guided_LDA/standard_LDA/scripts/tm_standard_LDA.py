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

# Download NLTK resources if not already present
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text: str) -> List[str]:
    """
    Tokenize and preprocess the input text:
    
    - Convert to lowercase
    - Remove stopwords
    - Keep only alphabetic words.
    
    Args:
        text (str): Raw input text.
        
    Returns:
        List[str]: List of cleaned and tokenized words.        
    """

    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

def perform_lda(topic_texts: List[str], num_topics: int, num_words: int) -> Tuple[LdaModel, corpora.Dictionary, List[List[Tuple[int, int]]], List[List[str]], List[List[Tuple[str, float]]]]:
    """
    Perform Latent Dirichlet Allocation (LDA) topic modeling on a list of abstracts.
    
    Args:
        topic_texts (list of str): List of abstracts.
        num_topic (int): The number of topics to generate.
        num_words (int): The number of top words to display per topic.

    Returns:
        Tuple containing:
            - lda (LdaModel): Trained LDA model.
            - dictionary (corpora.Dictionary): Mapping of words to IDs.
            - corpus (List of List of tuples (int, int)): Bag-of-words representation of the corpus.
            - processed_texts (List[List[str]]): Preprocessed tokens.
            - top_words_per_topic (List[List[Tuple[str, float]]]): Top words for each topic.

    """
    processed_texts = [preprocess_text(text) for text in topic_texts]
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    lda = gensim.models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=42)
    
    # Get the top words for each topic based on num_words
    top_words_per_topic = []
    for topic_id in range(lda.num_topics):
        top_words_per_topic.append(lda.show_topic(topic_id, topn=num_words))

    return lda, dictionary, corpus, processed_texts, top_words_per_topic

def compute_coherence_score(lda: LdaModel, corpus: List[List[Tuple[int, int]]], dictionary: corpora.Dictionary, processed_texts: List[List[str]]) -> float:
    """
    Compute the coherence score c_v, for the LDA model.
    
    Args:
        lda (LdaModel): Trained LDA model.
        corpus (list of list of tuples): Bag-of-words representation of abstractss.
        dictionary (corpora.Dictionary): Dictionary of token-ID mappings.
        processed_texts (list of list of str): Tokenized abstracts.

    Returns:
        float: Coherence score of the LDA model.
    """
    coherence_model = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, texts=processed_texts, coherence='c_v')
    return coherence_model.get_coherence()

def compute_perplexity(lda: LdaModel, corpus: List[List[Tuple[int, int]]]) -> float:
    """
    Compute the perplexity score for the LDA model.

    Perplexity is a measure of how well the model predicts the corpus. A lower perplexity indicates a better model.

    Args:
        lda (LdaModel): Trained LDA model.
        corpus (list of list of tuples): Bag-of-words representation of the corpus.

    Returns:
        float: Perplexity score of the LDA model.

    """

    log_perplexity = lda.log_perplexity(corpus)
    return np.exp(-log_perplexity)  # Exponentiate the negative log perplexity


def compute_topic_variance(lda: LdaModel, corpus: List[List[Tuple[int, int]]]) -> float:
    """
    Compute the average variance of the topic distributions across abstracts.
    
    The variance measures the spread of topic distributions for each document. A higher variance indicates that documets exhibit more diverse topic distributions.

    Args:
        lda (LdaModel): Trained LDA model.
        corpus (list of list of tuples): Bag-of-words representation of the corpus.

    Returns:
        float: Mean variance of topic distributions.
        
    """
    topic_variances = []
    for doc in corpus:
        topic_distribution = lda.get_document_topics(doc)
        # Extract the distribution of topics (probabilities) for this document
        topic_probs = [prob for _, prob in topic_distribution]
        # Compute variance of topic probabilities for this document
        topic_variances.append(np.var(topic_probs))
    
    return np.mean(topic_variances)  # Return the mean variance over all documents

def save_topics_and_coherence_to_file(lda: LdaModel, coherence_score: float, perplexity_score: float, topic_variance_score: float, folder_path: str, num_words: int) -> None:
    """
    Save the topics, coherence score, preplexity score, and topic variance score to a text file.
    
    Args:
        lda (genism.models.LdaModel): Trained LDA model.
        coherence_score (float): Coherence score of the model.
        perplexity_score (float): Perplexity score of the model.
        topic_variance_score (float): Topic variance score of the model.
        folder_path (str): Directory where the results file will be saved.
        num_words (int): Number of top words to display per topic.

    Returns:
        None: Function writes to a file and does not return anything.
    
    """
    os.makedirs(folder_path, exist_ok=True)
    topic_file_path = os.path.join(folder_path, "standard_LDA_topic_modeling_results.txt")

    with open(topic_file_path, 'w') as file:
        file.write(f"Standard LDA Topic Modeling Results\n")
        file.write("=" * 50 + "\n")
        for i in range(lda.num_topics):
            file.write(f"Topic {i+1}:\n")
            file.write(f"{lda.print_topic(i, num_words)}\n")
            file.write("=" * 50 + "\n")
        file.write(f"\nCoherence Score: {coherence_score}\n")
        file.write(f"Perplexity Score: {perplexity_score}\n")
        file.write(f"Topic Variance Score: {topic_variance_score}\n")
    print(f"Topic modeling results saved to: {topic_file_path}")


def perform_topic_modeling_on_excel_data(excel_file: str, num_topics: int, num_words: int) -> Optional[Dict[str, Any]]:
    """
    Perform topic modeling a set of abstracts loaded from an Excel file.
    
    This function reads an Excel file, extracts the 'Abstracts' column, preprocesses the texts, and performs LDA topic modleing. It also computes coherence, perplexity, and topic variance scores.

    Args:
        excel_file (str): Path to the Excel file containing the data.
        num_topics (int): Number of topics to generate.
        num_words (int): Number of top words to display per topic.

    Returns:
        Optional[Dict[str, Any]: Dictionary containing the following keys and values.
            - 'lda': Trained LDA model.
            - 'dictionary': Dictionary mapping owrds to IDs.
            - 'corpus': Bag-of words representation of the corpus.
            - 'coherence_score': Coherence score of the model.
            - 'perplexity_score': Perplexity score of the model.
            - 'topic_variance_score': Topic variance score of the model.
            - 'topic_words_per_topic': Top words for each topic.
        None: If there is an error, returns None.
    
    """
    try:
        df = pd.read_excel(excel_file)

        if 'Abstracts' not in df.columns:
            print("Error: The Excel file must contain an 'Abstracts' column.")
            return None

        topic_texts = df['Abstracts'].dropna().tolist()

        if not topic_texts:
            print("Error: No text data found in the 'Abstracts' column.")
            return None

        print(f"Performing topic modeling on {len(topic_texts)} abstracts...")

        lda, dictionary, corpus, processed_texts, top_words_per_topic = perform_lda(topic_texts, num_topics, num_words)
        coherence_score = compute_coherence_score(lda, corpus, dictionary, processed_texts)
        perplexity_score = compute_perplexity(lda, corpus)
        topic_variance_score = compute_topic_variance(lda, corpus)

        return {
            'lda': lda,
            'dictionary': dictionary,
            'corpus': corpus,
            'coherence_score': coherence_score,
            'perplexity_score': perplexity_score,
            'topic_variance_score': topic_variance_score,
            'processed_texts': processed_texts,
            'top_words_per_topic': top_words_per_topic
        }

    except Exception as e:
        print(f"Error processing the Excel file: {e}")
        return None
