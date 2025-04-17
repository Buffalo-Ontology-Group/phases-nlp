import os
import pandas as pd
import gensim
from gensim import corpora
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import CoherenceModel, LdaModel
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess the text (tokenize, remove stopwords)
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

# Function to construct a custom eta matrix using seed words
def construct_eta(dictionary, seed_words_per_topic, num_topics, eta_value=0.9, default_eta=0.01):
    vocab_size = len(dictionary)
    eta = np.full((num_topics, vocab_size), default_eta)

    for topic_idx, seed_words in enumerate(seed_words_per_topic):
        for word in seed_words:
            if word in dictionary.token2id:
                word_id = dictionary.token2id[word]
                eta[topic_idx, word_id] = eta_value

    return eta

# Perform semi-guided LDA
def perform_guided_lda(topic_texts, num_topics, seed_words_per_topic):
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
def compute_coherence_score(lda, corpus, dictionary, processed_texts):
    coherence_model = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, texts=processed_texts, coherence='c_v')
    return coherence_model.get_coherence()

def compute_perplexity(lda, corpus):
    """Compute and return the perplexity score (exponentiated log perplexity)."""
    log_perplexity = lda.log_perplexity(corpus)
    return np.exp(-log_perplexity)  # Exponentiate the negative log perplexity

# Topic variance
def compute_topic_variance(lda):
    topic_word_distributions = lda.get_topics()
    variance = np.var(topic_word_distributions, axis=0).mean()
    return variance

# Save all metrics and topics
def save_topics_and_coherence_to_file(lda_model, coherence_score, perplexity_score, topic_variance_score, results_folder_path, num_words):
    os.makedirs(results_folder_path, exist_ok=True)
    topic_file_path = os.path.join(results_folder_path, "topic_modeling_results.txt")

    with open(topic_file_path, 'w') as file:
        file.write(f"Guided LDA Topic Modeling Results\n")
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
def perform_topic_modeling_on_excel_data_with_guided_lda(excel_file, num_topics, seed_words_per_topic, num_words):
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
