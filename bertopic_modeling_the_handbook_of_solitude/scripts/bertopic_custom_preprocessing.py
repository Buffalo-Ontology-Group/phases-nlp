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
import plotly.io as pio
import re

# Suppress huggingface/tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# Load config
def load_config(config_path=None):
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config_solitude.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Read .docx
def read_documents_from_docx(docx_file_path):
    if not os.path.exists(docx_file_path):
        raise FileNotFoundError(f".docx file not found: {docx_file_path}")
    doc = Document(docx_file_path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]

# POS tokenizer using spaCy
def pos_tokenizer(text, nlp):
    doc = nlp(text)
    return [t.text for t in doc if t.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not t.is_stop]

# Auto-generate topic labels using KeyBERT
def generate_auto_labels_with_keybert(topic_model, documents, kw_model):
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
    sentences = re.split(r'(?<=[.!?]) +', text)
    return ' '.join(sentences[:2])

# Summarize documents assigned to each topic
def summarize_topic_documents(documents, topics):
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

# Run pipeline
def run_pipeline(config, documents, num_topics=5, num_keywords_per_topic=10):
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
    os.makedirs(output_dir, exist_ok=True)

    try:
        fig = topic_model.visualize_topics()
        fig.write_html(os.path.join(output_dir, "topics_umap.html"))
        fig.write_image(os.path.join(output_dir, "topics_umap.png"))
    except Exception as e:
        print(f"UMAP visualization error: {e}")

    try:
        fig = topic_model.visualize_hierarchy(top_n_topics=num_topics)
        fig.write_html(os.path.join(output_dir, "topic_hierarchy.html"))
        fig.write_image(os.path.join(output_dir, "topic_hierarchy.png"))
    except Exception as e:
        print(f"Hierarchy visualization error: {e}")

    try:
        fig = topic_model.visualize_barchart(top_n_topics=10)
        fig.write_html(os.path.join(output_dir, "topic_barchart.html"))
        fig.write_image(os.path.join(output_dir, "topic_barchart.png"))
    except Exception as e:
        print(f"Barchart error: {e}")

    try:
        fig = topic_model.visualize_heatmap(top_n_topics=num_topics)
        fig.write_html(os.path.join(output_dir, "topic_heatmap.html"))
        fig.write_image(os.path.join(output_dir, "topic_heatmap.png"))
    except Exception as e:
        print(f"Heatmap error: {e}")

# Save results
def save_results(documents, topics, probs, topic_model, topic_labels, output_dir, num_keywords_per_topic, summaries=None):
    txt_path = os.path.join(output_dir, "bertopic_results.txt")
    csv_path = os.path.join(output_dir, "bertopic_results.csv")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("BERTopic Results\n\n")
        for topic_id in topic_model.get_topic_info()['Topic']:
            if topic_id == -1:
                continue
            keywords = topic_model.get_topic(topic_id)
            keyword_list = [kw[0] for kw in keywords[:num_keywords_per_topic]]
            label = topic_labels.get(topic_id, f"Topic {topic_id}")
            f.write(f"Topic {topic_id} ({label}): {', '.join(keyword_list)}\n")
            if summaries and topic_id in summaries:
                f.write(f"Summary: {summaries[topic_id]}\n")
            f.write("\n")

        f.write("\nDocument-Level Assignments:\n")
        for i, prob in enumerate(probs):
            assigned_topic = topics[i]
            label = topic_labels.get(assigned_topic, f"Topic {assigned_topic}")
            f.write(f"Doc {i + 1}: Topic {assigned_topic} ({label}), Prob: {prob:.8f}\n")

    # Save to CSV
    rows = []
    for i, prob in enumerate(probs):
        assigned_topic = topics[i]
        label = topic_labels.get(assigned_topic, f"Topic {assigned_topic}")
        rows.append({
            "Document": documents[i],
            "Topic": assigned_topic,
            "Topic_Label": label,
            "Probability": prob
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"Results saved to:\nTXT: {txt_path}\nCSV: {csv_path}")

    excel_path = os.path.join(output_dir, "bertopic_topic_summary.xlsx")
    topic_data = []
    for topic_id in topic_model.get_topic_info()['Topic']:
        if topic_id == -1:
            continue
        label = topic_labels.get(topic_id, f"Topic {topic_id}")
        keywords = topic_model.get_topic(topic_id)
        keyword_list = [kw[0] for kw in keywords[:num_keywords_per_topic]]
        summary = summaries.get(topic_id, "")
        topic_data.append({
            "Topic": topic_id,
            "Topic Label": label,
            "Keywords": ", ".join(keyword_list),
            "Summary": summary
        })

    pd.DataFrame(topic_data).to_excel(excel_path, index=False)
    print(f"Topic-level summary saved to:\nExcel: {excel_path}")


# Main
if __name__ == "__main__":
    config = load_config()
    docx_file_path = os.getenv("DOCX_FILE_PATH_SOLITUDE")
    if not docx_file_path or not os.path.exists(docx_file_path):
        raise ValueError("Check your .env: DOCX_FILE_PATH_SOLITUDE is missing or invalid.")

    documents = read_documents_from_docx(docx_file_path)
    print(f"Loaded {len(documents)} paragraphs from {docx_file_path}")

    num_topics = int(input("Enter number of topics to display in visualizations: "))
    num_keywords = int(input("Enter number of keywords per topic for heatmap: "))

    topics, probs, topic_model, topic_labels = run_pipeline(config, documents, num_topics, num_keywords)
    summaries = summarize_topic_documents(documents, topics)

    results_folder = os.getenv("RESULTS_DIRECTORY_CUSTOM_PREPROCESSING", os.path.dirname(docx_file_path))
    os.makedirs(results_folder, exist_ok=True)

    save_results(documents, topics, probs, topic_model, topic_labels, results_folder, num_keywords, summaries)
    visualize_all(topic_model, documents, topics, results_folder, num_topics, num_keywords)
