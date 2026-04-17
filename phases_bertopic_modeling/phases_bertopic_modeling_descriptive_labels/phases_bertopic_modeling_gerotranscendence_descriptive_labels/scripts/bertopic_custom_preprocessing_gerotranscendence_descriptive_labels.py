import yaml
import spacy
import os
from docx import Document
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import hdbscan
import umap
from keybert import KeyBERT
from dotenv import load_dotenv
from transformers import pipeline
from collections import Counter
import plotly.io as pio
import re

# Suppress huggingface/tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# Load config
def load_config(config_path=None):
    """
    Load the YAML configuration file.

    Args:
        config_path (str, optional): Path to the configuration YAML file. If None, defaults to 'config_gerotranscendence.yaml' in the current script directory.
    
    Returns:
        dict: Parsed YAML configuration as a dictionary.
    
    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config_gerotranscendence.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Read .docx
def read_documents_from_docx(docx_file_path):
    """
    Read paragraphs from a .docx file.

    Args:
        docx_file_path (str): Path to the .docx file.

    Returns:
        List[str]: List of non-empty, stripped paragraph texts.

    Raises:
        FileNotFoundError: If the .docx file does not exist.

    """
    if not os.path.exists(docx_file_path):
        raise FileNotFoundError(f".docx file not found: {docx_file_path}")
    doc = Document(docx_file_path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]

# POS tokenizer using spaCy
def pos_tokenizer(text, nlp):
    """
    Tokenize text using spaCy, keeping only NOUN, VERB, ADJ, and ADV POS tags.

    Args:
        text (str): Input text to tokenize.
        nlp (spacy.Langugae): Loaded spaCy language model.

    Returns:
        List[str]: List of filtered token texts that are NOUN, VERB, ADJ, or ADV and not stopwords.

    """
    doc = nlp(text)
    return [t.text for t in doc if t.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not t.is_stop]

# Auto-generate topic labels using KeyBERT
def generate_auto_labels_with_keybert(topic_model, documents, kw_model):
    """
    Generate topic labels using representative documents and KeyBERT.

    Args:
        topic_model (BERTopic): Fitted BERTopic model.
        documents (List[str]): List of original documents.
        kw_model (KeyBERT): KeyBERT keyword extraction model.

    Returns:
        Dict[int, str]: Mapping from topic ID to generated label.
    """
    labels = {}
    rep_docs = topic_model.get_representative_docs()
    for topic_id, docs in rep_docs.items():
        if topic_id == -1:
            labels[topic_id] = "Noise / Outlier"
        else:
            doc_text = docs[0]
            keywords = kw_model.extract_keywords(doc_text, top_n=1)
            labels[topic_id] = keywords[0][0].title() if keywords else "Unnamed Topic"
    return labels

# Extract first two sentences
def get_first_two_sentences(text):
    """
    Extract the first two sentences from a text.

    Args:
        text (str): Input text.
    
    Returns:
        str: Concatenated first two sentences.

    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    return ' '.join(sentences[:2])

# Summarize documents assigned to each topic
def summarize_topic_documents(documents, topics):
    """
    Summarize combined documents for each topic using BART summarizer.

    Args:
        documents (List[str]): List of original text documents.
        topics (List[int]): Assigned topic IDs for each document.

    Returns:
        Dict[int, str]: Mapping from topic ID to generated summary text.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = {}

    topic_docs_map = {}
    for doc, topic_id in zip(documents, topics):
        if topic_id == -1:
            continue
        topic_docs_map.setdefault(topic_id, []).append(doc)

    for topic_id, docs in topic_docs_map.items():
        combined_text = " ".join(docs)
        input_len = len(combined_text.split())

        max_len = min(130, int(input_len * 0.6))
        min_len = min(30, int(input_len * 0.3))

        try:
            summary = summarizer(
                combined_text,
                max_length=max_len if max_len > min_len else min_len + 5,
                min_length=min_len,
                do_sample=False
            )[0]['summary_text']
            summary = get_first_two_sentences(summary)
        except Exception as e:
            summary = f"Summary generation failed: {e}"

        summaries[topic_id] = summary.strip()

    return summaries
# Generate more descriptive topic labels using summaries and keywords

def generate_specific_labels(topic_model, summaries, kw_model, num_keywords=3):
    """
    Generate specific topic labels using summary and topic keywords.

    Args:
        topic_model (BERTopic): Trained BERTopic model.
        summaries (Dict[int, str]): Topic ID -> summary.
        kw_model (KeyBERT): KeyBERT keyword extraction model.
        num_keywords (int): Number of top keywords to consider from BERTopic.

    Returns:
        Dict[int, str]: Topic ID -> specific label.
    """
    labels = {}
    for topic_id in topic_model.get_topic_info()['Topic']:
        if topic_id == -1:
            labels[topic_id] = "Noise / Outlier"
            continue

        # Get summary and extract a keyword
        summary = summaries.get(topic_id, "")
        summary_keywords = kw_model.extract_keywords(summary, top_n=1)
        summary_kw = summary_keywords[0][0].title() if summary_keywords else ""

        # Get top topic keyword from BERTopic model
        topic_keywords = topic_model.get_topic(topic_id)[:num_keywords]
        topic_kw = topic_keywords[0][0].title() if topic_keywords else ""

        # Prefer summary keyword, fallback to topic keyword
        label_keyword = summary_kw or topic_kw or "Unnamed"

        label = f"Gerotranscendence – {label_keyword}"
        labels[topic_id] = label

    return labels


# Run pipeline
def run_pipeline(config, documents, num_topics=5, num_keywords_per_topic=10):
    """
    Run the BERTopic pipeline with configurable preprocessing and modeling.

    Args:
        config (dict): Configuration dictionary loaded from YAML.
        documents (List[str]): Input list of documents.
        num_topics (int): Number of topics for visualization.
        num_keywords_per_topic (int): Number of keywords per topic.

    Returns:
        Tuple[List[int], List[float], BERTopic, Dict[int, str]]:
            topics (List[int]): Assigned topic ID for each document.
            probs (List[float]): Probability/confidence of each assignment.
            topic_model (BERTopic): Trained BERTopic model.
            topic_labels (Dict[int, str]): Mapping of topic ID to label.

    """
    nlp = spacy.load(config['spacy_model'])

    scheme = config['vectorizer_params']['scheme'].lower()
    vectorizer_class = TfidfVectorizer if scheme == "tfidf" else CountVectorizer

    vectorizer = vectorizer_class(
        tokenizer=lambda text: pos_tokenizer(text, nlp),
        stop_words=config['vectorizer_params']['stop_words'],
        ngram_range=tuple(config['vectorizer_params']['ngram_range'])
    )

    X = vectorizer.fit_transform(documents)
    sentence_model = SentenceTransformer(config['sentence_transformer_model'])
    embeddings = sentence_model.encode(documents)

    umap_model = umap.UMAP(**config['umap_params'])
    hdbscan_model = hdbscan.HDBSCAN(**config['hdbscan_params'])

    topic_model = BERTopic(
        vectorizer_model=vectorizer,
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
    )

    topics, probs = topic_model.fit_transform(documents)

    kw_model = KeyBERT(model=config['keybert_model'])
    topic_labels = generate_auto_labels_with_keybert(topic_model, documents, kw_model)

    return topics, probs, topic_model, topic_labels

# Visualizations
def visualize_all(topic_model, documents, topics, output_dir, num_topics, num_keywords_per_topic):
    """
    Generate and save multiple BERTopic visualizations.

    Args:
        topic_model (BERTopic): Trained BERTopic model.
        documents (List[str]): List of input documents.
        topics (List[int]): Topic assignments.
        output_dir (str): Directory to save visualization files.
        num_topics (int): Number of topics to visualize.
        num_keywords_per_topic (int): Number of top keywords to include per topic for barchart/heatmap.
    
    Notes: Saves HTML and PNG visualizations for UMAP, hierarchy, barchart, and heatmap.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        fig = topic_model.visualize_topics()
        fig.write_html(os.path.join(output_dir, "topics_umap_custom_preprocessing_gerotranscendence.html"))
        fig.write_image(os.path.join(output_dir, "topics_umap_custom_preprocessing_gerotranscendence.png"))
    except Exception as e:
        print(f"UMAP visualization error: {e}")

    try:
        fig = topic_model.visualize_hierarchy(top_n_topics=num_topics)
        fig.write_html(os.path.join(output_dir, "topic_hierarchy_custom_preprocessing_gerotranscendence.html"))
        fig.write_image(os.path.join(output_dir, "topic_hierarchy_custom_preprocessing_gerotranscendence.png"))
    except Exception as e:
        print(f"Hierarchy visualization error: {e}")

    try:
        fig = topic_model.visualize_barchart(top_n_topics=10)
        fig.write_html(os.path.join(output_dir, "topic_barchart_custom_preprocessing_gerotranscendence.html"))
        fig.write_image(os.path.join(output_dir, "topic_barchart_custom_preprocessing_gerotranscendence.png"))
    except Exception as e:
        print(f"Barchart error: {e}")

    try:
        fig = topic_model.visualize_heatmap(top_n_topics=num_topics)
        fig.write_html(os.path.join(output_dir, "topic_heatmap_custom_preprocessing_gerotranscendence.html"))
        fig.write_image(os.path.join(output_dir, "topic_heatmap_custom_preprocessing_gerotranscendence.png"))
    except Exception as e:
        print(f"Heatmap error: {e}")

# Save results
def save_results(documents, topics, probs, topic_model, topic_labels, output_dir, num_keywords_per_topic, summaries=None):
    """
    Save BERTopic results to TXT, CSV, and Excel formats.

    Args:
        documents (List[str]): List of original documents.
        topics (List[int]): Assigned topic IDs for each document.
        probs (List[float]): Probabilities/confidence scores for each assignment.
        topic_model (BERTopic): Trained BERTopic model.
        topic_labels (Dict[int, str]): Auto-generated topic labels.
        output_dir (str): Directory to save save result files.
        num_keywords_per_topic (int): Number of keywords to include per topic.
        summaries (Dict[int, str], optional): Optional topic summaries.

    Returns:
        None
    
    Notes:
        This function saves three files:
        - 'bertopic_results.txt': Human-readable topic keywords and document-topic assignments.
        - 'bertopic_results.csv': CSV file mapping documents to topics with probabilities.
        - 'bertopic_topic_summary.xlsx': Excel summary of topics with labels, keywords, and optional summaries.
    """
    txt_path = os.path.join(output_dir, "bertopic_results_custom_preprocessing_gerotranscendence.txt")
    csv_path = os.path.join(output_dir, "bertopic_results_custom_preprocessing_gerotranscendence.csv")
    excel_path = os.path.join(output_dir, "bertopic_topic_summary_custom_preprocessing_gerotranscendence.xlsx")

    # Step 1: Count auto label frequencies
    label_counts = Counter(topic_labels.values())

    # Step 2: Build descriptive labels for repeated auto labels
    adjusted_descriptive_labels = {}
    for topic_id, auto_label in topic_labels.items():
        if label_counts[auto_label] > 1:
            keywords = topic_model.get_topic(topic_id)
            top_keyword = keywords[0][0].title() if keywords else "Unspecified"
            descriptive_label = f"{auto_label} – {top_keyword}"
        else:
            descriptive_label = ""  # Leave blank if unique
        adjusted_descriptive_labels[topic_id] = descriptive_label

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("BERTopic custom preprocessing pipeline results gerotranscendence\n\n")
        for topic_id in topic_model.get_topic_info()['Topic']:
            if topic_id == -1:
                continue
            keywords = topic_model.get_topic(topic_id)
            keyword_list = [kw[0] for kw in keywords[:num_keywords_per_topic]]

            label_auto = topic_labels.get(topic_id, f"Topic {topic_id}")
            label_descriptive = adjusted_descriptive_labels.get(topic_id, label_auto)

            if label_descriptive:
                f.write(f"Topic {topic_id} ({label_descriptive})\n")
            else:
                f.write(f"Topic {topic_id}\n")

            f.write(f"Auto Label: {label_auto}\n")
            f.write(f"Keywords: {', '.join(keyword_list)}\n")

            if summaries and topic_id in summaries:
                f.write(f"Summary: {summaries[topic_id]}\n")
            f.write("\n")

        f.write("\nDocument-Level Assignments:\n")
        for i, prob in enumerate(probs):
            assigned_topic = topics[i]
            label_auto = topic_labels.get(assigned_topic, f"Topic {assigned_topic}")
            label_descriptive = adjusted_descriptive_labels.get(assigned_topic, "")
            label_text = f"{label_auto}" if not label_descriptive else f"{label_descriptive}"

            f.write(f"Doc {i + 1}: Topic {assigned_topic} ({label_text}), Auto: {label_auto}, Prob: {prob:.8f}\n")


    # Save to CSV
    rows = []
    for i, prob in enumerate(probs):
        assigned_topic = topics[i]
        label_auto = topic_labels.get(assigned_topic, f"Topic {assigned_topic}")
        label_descriptive = adjusted_descriptive_labels.get(assigned_topic, "")
        rows.append({
            "Document": documents[i],
            "Topic": assigned_topic,
            "Auto_Label": label_auto,
            "Descriptive_Label": label_descriptive,
            "Probability": prob
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    topic_data = []
    for topic_id in topic_model.get_topic_info()['Topic']:
        if topic_id == -1:
            continue
        label_auto = topic_labels.get(topic_id, f"Topic {topic_id}")
        label_descriptive = descriptive_labels.get(topic_id, label_auto)
        keywords = topic_model.get_topic(topic_id)
        keyword_list = [kw[0] for kw in keywords[:num_keywords_per_topic]]
        summary = summaries.get(topic_id, "") if summaries else ""
        topic_data.append({
            "Topic": topic_id,
            "Auto Label": label_auto,
            "Descriptive Label": label_descriptive,
            "Keywords": ", ".join(keyword_list),
            "Summary": summary
        })

    pd.DataFrame(topic_data).to_excel(excel_path, index=False)
    print(f"Results saved to:\nTXT: {txt_path}\nCSV: {csv_path}\nExcel: {excel_path}")
  

# Main
if __name__ == "__main__":
    config = load_config()
    docx_file_path = os.getenv("DOCX_FILE_PATH_GEROTRANSCENDENCE")
    if not docx_file_path or not os.path.exists(docx_file_path):
        raise ValueError("Check your .env: DOCX_FILE_PATH_GEROTRANSCENDENCE is missing or invalid.")

    documents = read_documents_from_docx(docx_file_path)
    print(f"Loaded {len(documents)} paragraphs from {docx_file_path}")

    num_topics = int(input("Enter number of topics to display in visualizations: "))
    num_keywords = int(input("Enter number of keywords per topic for heatmap: "))

    topics, probs, topic_model, topic_labels = run_pipeline(config, documents, num_topics, num_keywords)
    summaries = summarize_topic_documents(documents, topics)
    kw_model = KeyBERT(model=config['keybert_model'])
    descriptive_labels = generate_specific_labels(topic_model, summaries, kw_model, num_keywords)


    results_folder = os.getenv("RESULTS_DIRECTORY_CUSTOM_PREPROCESSING_GEROTRANSCENDENCE", os.path.dirname(docx_file_path))
    os.makedirs(results_folder, exist_ok=True)

    save_results(documents, topics, probs, topic_model, topic_labels, results_folder, num_keywords, summaries)
    visualize_all(topic_model, documents, topics, results_folder, num_topics, num_keywords)
