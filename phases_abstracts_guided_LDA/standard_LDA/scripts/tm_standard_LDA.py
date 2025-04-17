import os
import pandas as pd
import gensim
from gensim import corpora
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import CoherenceModel
import numpy as np

# Download NLTK resources if not already present
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Tokenize text and remove stopwords and non-alphabetic tokens."""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

def perform_lda(topic_texts, num_topics, num_words):
    """Train LDA model on processed texts and return model, dictionary, corpus."""
    processed_texts = [preprocess_text(text) for text in topic_texts]
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    lda = gensim.models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=42)
    
    # Get the top words for each topic based on num_words
    top_words_per_topic = []
    for topic_id in range(lda.num_topics):
        top_words_per_topic.append(lda.show_topic(topic_id, topn=num_words))

    return lda, dictionary, corpus, processed_texts, top_words_per_topic

def compute_coherence_score(lda, corpus, dictionary, processed_texts):
    """Compute and return coherence score (c_v)."""
    coherence_model = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, texts=processed_texts, coherence='c_v')
    return coherence_model.get_coherence()

def compute_perplexity(lda, corpus):
    """Compute and return the perplexity score (exponentiated log perplexity)."""
    log_perplexity = lda.log_perplexity(corpus)
    return np.exp(-log_perplexity)  # Exponentiate the negative log perplexity


def compute_topic_variance(lda, corpus):
    """Compute and return the variance of the topic distributions across documents."""
    topic_variances = []
    for doc in corpus:
        topic_distribution = lda.get_document_topics(doc)
        # Extract the distribution of topics (probabilities) for this document
        topic_probs = [prob for _, prob in topic_distribution]
        # Compute variance of topic probabilities for this document
        topic_variances.append(np.var(topic_probs))
    
    return np.mean(topic_variances)  # Return the mean variance over all documents

def save_topics_and_coherence_to_file(lda, coherence_score, perplexity_score, topic_variance_score, folder_path, num_words):
    """Save the topics and coherence score to a text file."""
    os.makedirs(folder_path, exist_ok=True)
    topic_file_path = os.path.join(folder_path, "topic_modeling_results.txt")
    with open(topic_file_path, 'w') as file:
        for i in range(lda.num_topics):
            file.write(f"Topic {i+1}:\n")
            file.write(f"{lda.print_topic(i, num_words)}\n")
            file.write("=" * 50 + "\n")
        file.write(f"\nCoherence Score: {coherence_score}\n")
        file.write(f"Perplexity Score: {perplexity_score}\n")
        file.write(f"Topic Variance Score: {topic_variance_score}\n")
    print(f"Topic modeling results saved to: {topic_file_path}")

def perform_topic_modeling_on_excel_data(excel_file, num_topics, num_words):
    """Main function to perform topic modeling from Excel."""
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
