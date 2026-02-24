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
import re
from collections import Counter  # needed for generate_specific_labels

# Suppress huggingface/tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# -----------------------------------------------------------
# Vaccine sentence extraction (Approach 2) - PRE-BERTopic ONLY
# -----------------------------------------------------------
VACCINE_LEMMAS = {
    "vaccine", "vaccination", "adacel vaccine", "boostrix vaccine", "comvax vaccine",
    "DNA vaccine", "ETEC vaccine ACAM2010", "fluzone",
    "infanrix vaccine", "Menactra vaccine", "pentacel vaccine", 
    "pneumovax 23 vaccine", "rotaTeq vaccine"
}

def extract_vaccine_sentences(text, nlp, min_hits=1):
    """
    Keep only sentences that contain vaccine-related lemmas.
    Returns a list of kept sentences (strings).
    """
    doc = nlp(text)
    kept = []
    for sent in doc.sents:
        hits = sum(1 for tok in sent if tok.lemma_.lower() in VACCINE_LEMMAS)
        if hits >= min_hits:
            kept.append(sent.text.strip())
    return kept

def build_vaccine_corpus(documents, nlp, min_hits=1):
    """
    Convert each paragraph/document into vaccine-only text by keeping only
    vaccine-relevant sentences. Drops docs that contain no vaccine sentences.
    """
    out = []
    for d in documents:
        sents = extract_vaccine_sentences(d, nlp, min_hits=min_hits)
        if sents:
            out.append(" ".join(sents))
    return out

# Load config
def load_config(config_path=None):
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config_plotkins_vaccines.yaml")
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

# Auto-generate topic labels
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

# Patched: Generate descriptive labels (skips trivial qualifiers)
def generate_specific_labels(topic_model, summaries, kw_model, num_keywords=3):
    auto_labels = {}
    dup_counts = Counter()
    rep_docs = topic_model.get_representative_docs()

    for topic_id, docs in rep_docs.items():
        if topic_id == -1:
            auto_labels[topic_id] = "Noise / Outlier"
            continue
        base_text = docs[0] if docs else ""
        kws = kw_model.extract_keywords(base_text, top_n=3) if base_text else []
        base = kws[0][0].title() if kws else f"Topic {topic_id}"
        auto_labels[topic_id] = base
        dup_counts[base] += 1

    descriptive = {}
    used = set()
    for topic_id, base in auto_labels.items():
        if topic_id == -1 or dup_counts[base] == 1:
            label = base
        else:
            summary = summaries.get(topic_id, "")
            sum_terms = [k[0].title() for k in kw_model.extract_keywords(summary, top_n=5)] if summary else []
            topic_terms = [kw[0].title() for kw in (topic_model.get_topic(topic_id) or [])[:max(5, num_keywords)]]

            qualifier = ""
            base_norm = base.lower().rstrip('s')
            for cand in sum_terms + topic_terms:
                if not cand:
                    continue
                cand_norm = cand.lower().rstrip('s')
                if cand_norm == base_norm:
                    continue
                qualifier = cand
                break

            label = f"{base} – {qualifier}" if qualifier else f"{base} – T{topic_id}"

        if label in used:
            label = f"{label} – T{topic_id}"
        used.add(label)
        descriptive[topic_id] = label

    return descriptive

# Extract first two sentences
def get_first_two_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return ' '.join(sentences[:2])

# Summarize documents
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
def run_pipeline(config, documents, num_topics, num_keywords_per_topic):
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

# Save results
def save_results(
    documents,
    topics,
    probs,
    topic_model,
    topic_labels,
    output_dir,
    num_keywords_per_topic,
    summaries=None,
    descriptive_labels=None
):
    txt_path = os.path.join(output_dir, "bertopic_results_plotkins_vaccines.txt")
    csv_path = os.path.join(output_dir, "bertopic_results_plotkins_vaccines.csv")
    excel_path = os.path.join(output_dir, "bertopic_topic_summary_plotkins_vaccines.xlsx")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("BERTopic results plotkins vaccines\n\n")
        for topic_id in topic_model.get_topic_info()['Topic']:
            if topic_id == -1:
                continue
            keywords = topic_model.get_topic(topic_id) or []
            keyword_list = [kw[0] for kw in keywords[:num_keywords_per_topic]]

            auto_label = topic_labels.get(topic_id, f"Topic {topic_id}")
            desc_label = descriptive_labels.get(topic_id, auto_label) if descriptive_labels else auto_label

            f.write(f"Topic {topic_id}\n")
            f.write(f"  Auto Label:        {auto_label}\n")
            f.write(f"  Descriptive Label: {desc_label}\n")
            f.write(f"  Keywords:          {', '.join(keyword_list)}\n")
            if summaries and topic_id in summaries:
                f.write(f"  Summary:           {summaries[topic_id]}\n")
            f.write("\n")

        f.write("\nDocument-Level Assignments:\n")
        for i, prob in enumerate(probs):
            assigned_topic = topics[i]
            auto_label = topic_labels.get(assigned_topic, f"Topic {assigned_topic}")
            desc_label = descriptive_labels.get(assigned_topic, auto_label) if descriptive_labels else auto_label
            f.write(
                f"Doc {i + 1}: Topic {assigned_topic} "
                f"(Auto: {auto_label}; Descriptive: {desc_label}), "
                f"Prob: {prob:.8f}\n"
            )

    rows = []
    for i, prob in enumerate(probs):
        assigned_topic = topics[i]
        auto_label = topic_labels.get(assigned_topic, f"Topic {assigned_topic}")
        desc_label = descriptive_labels.get(assigned_topic, auto_label) if descriptive_labels else auto_label
        rows.append({
            "Document": documents[i],
            "Topic": assigned_topic,
            "Auto_Label": auto_label,
            "Descriptive_Label": desc_label,
            "Probability": prob
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    topic_data = []
    for topic_id in topic_model.get_topic_info()['Topic']:
        if topic_id == -1:
            continue
        auto_label = topic_labels.get(topic_id, f"Topic {topic_id}")
        desc_label = descriptive_labels.get(topic_id, auto_label) if descriptive_labels else auto_label
        keywords = topic_model.get_topic(topic_id) or []
        keyword_list = [kw[0] for kw in keywords[:num_keywords_per_topic]]
        summary = summaries.get(topic_id, "") if summaries else ""
        topic_data.append({
            "Topic": topic_id,
            "Auto Label": auto_label,
            "Descriptive Label": desc_label,
            "Keywords": ", ".join(keyword_list),
            "Summary": summary
        })

    pd.DataFrame(topic_data).to_excel(excel_path, index=False)
    print(f"Results saved to:\nTXT: {txt_path}\nCSV: {csv_path}\nExcel: {excel_path}")

# Hierarchical tree generation
def generate_hierarchical_tree(topic_model, documents, output_dir, topic_labels, top_n_topics=None):
    import re
    os.makedirs(output_dir, exist_ok=True)
    try:
        hierarchical_topics = topic_model.hierarchical_topics(documents)
        hier_csv = os.path.join(output_dir, "hierarchical_topics_plotkins_vaccines.csv")
        hierarchical_topics.to_csv(hier_csv, index=False)

        tree = topic_model.get_topic_tree(hierarchical_topics)
        tree_txt = os.path.join(output_dir, "topic_tree_plotkins_vaccines.txt")
        with open(tree_txt, "w", encoding="utf-8") as f:
            f.write(str(tree))

        allowed_ids = None
        if isinstance(top_n_topics, int) and top_n_topics > 0:
            info = topic_model.get_topic_info()
            allowed_ids = set(
                info[info['Topic'] != -1]
                .sort_values('Count', ascending=False)['Topic']
                .head(top_n_topics)
                .tolist()
            )

        cleaned_txt = os.path.join(output_dir, "topic_tree_ids_labels_plotkins_vaccines.txt")
        topic_re = re.compile(r"(.*?)(?:■──|├─|└─).*?Topic:\s*(\d+)\s*$")
        kept_lines = []
        with open(tree_txt, "r", encoding="utf-8") as f:
            for line in f:
                m = topic_re.match(line.rstrip("\n"))
                if not m:
                    continue
                prefix, tid = m.group(1), int(m.group(2))
                if allowed_ids is not None and tid not in allowed_ids:
                    continue
                label = topic_labels.get(tid, f"Topic {tid}")
                kept_lines.append(f"{prefix}■── Topic {tid} ({label})")

        with open(cleaned_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(kept_lines))

        print(
            "Hierarchical topic artifacts saved:\n"
            f" - Hierarchy CSV:                         {hier_csv}\n"
            f" - ASCII tree (original, full):           {tree_txt}\n"
            f" - ASCII tree (IDs+labels, filtered):     {cleaned_txt}"
        )
    except Exception as e:
        print(f"Hierarchical tree generation error: {e}")

# ---------- Visualizations: KEEP ONLY Hierarchy + Bar Chart ----------
def visualize_selected(topic_model, output_dir, num_topics, num_keywords_per_topic):
    """
    Generate and save only Hierarchical Clustering and Topic Bar Chart visualizations.

    Saves:
      - Hierarchical clustering (HTML + PNG)
      - Topic bar chart (HTML + PNG)
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        fig = topic_model.visualize_hierarchy(top_n_topics=num_topics)
        fig.write_html(os.path.join(output_dir, "topic_hierarchy_plotkins_vaccines.html"))
        fig.write_image(os.path.join(output_dir, "topic_hierarchy_plotkins_vaccines.png"))
        print("Saved hierarchical clustering visualization.")
    except Exception as e:
        print(f"Hierarchy visualization error: {e}")

    try:
        fig = topic_model.visualize_barchart(top_n_topics=num_topics, n_words=num_keywords_per_topic)
        fig.write_html(os.path.join(output_dir, "topic_barchart_plotkins_vaccines.html"))
        fig.write_image(os.path.join(output_dir, "topic_barchart_plotkins_vaccines.png"))
        print("Saved topic bar chart visualization.")
    except Exception as e:
        print(f"Bar chart error: {e}")

# -----------------------------------------------------------
# Main (single prompt for all chapters)
# -----------------------------------------------------------
if __name__ == "__main__":
    config = load_config()

    # Load spaCy ONCE for filtering (Approach 2)
    nlp_filter = spacy.load(config['spacy_model'])

    # Prompt ONCE
    num_topics = int(input("Enter number of topics to display (applies to all chapters): "))
    num_keywords = int(input("Enter number of keywords per topic (applies to all chapters): "))

    # Loop through CH1..CH9
    for ch in range(1, 10):
        docx_file_path = os.getenv(f"DOCX_FILE_PATH_PLOTKINS_VACCINES_CH{ch}")
        results_folder = os.getenv(f"RESULTS_PLOTKINS_VACCINES_CH{ch}")

        if not docx_file_path or not os.path.exists(docx_file_path):
            print(f"[Ch{ch}] Skipping: missing or invalid DOCX path: {docx_file_path}")
            continue

        if not results_folder:
            # Default results dir next to input if not set
            results_folder = os.path.join(
                os.path.dirname(docx_file_path),
                f"results_ch{ch}"
            )

        os.makedirs(results_folder, exist_ok=True)

        # Load documents
        documents = read_documents_from_docx(docx_file_path)
        print(f"[Ch{ch}] Loaded {len(documents)} paragraphs from {docx_file_path}")

        # --- Vaccine-only corpus (Approach 2) ---
        orig_n = len(documents)
        documents = build_vaccine_corpus(documents, nlp_filter, min_hits=1)
        print(f"[Ch{ch}] Kept {len(documents)} vaccine-relevant docs (from {orig_n})")

        if not documents:
            print(f"[Ch{ch}] Skipping: no vaccine-relevant text after filtering.")
            continue

        # Run pipeline (BERTopic unchanged)
        topics, probs, topic_model, topic_labels = run_pipeline(
            config, documents, num_topics, num_keywords
        )

        # Summaries + descriptive labels (shared params)
        summaries = summarize_topic_documents(documents, topics)
        kw_model = KeyBERT(model=config['keybert_model'])
        descriptive_labels = generate_specific_labels(
            topic_model, summaries, kw_model, num_keywords
        )

        # Save artifacts
        save_results(
            documents, topics, probs, topic_model,
            topic_labels, results_folder, num_keywords,
            summaries, descriptive_labels=descriptive_labels
        )

        # Trees + visuals
        generate_hierarchical_tree(
            topic_model, documents, results_folder,
            descriptive_labels, top_n_topics=num_topics
        )
        visualize_selected(topic_model, results_folder, num_topics, num_keywords)

        print(f"[Ch{ch}] Done. Artifacts in: {results_folder}")
