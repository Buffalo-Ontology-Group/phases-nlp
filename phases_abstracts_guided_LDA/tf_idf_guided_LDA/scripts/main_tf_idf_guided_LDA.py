import os
import click
from dotenv import load_dotenv
from tm_tf_idf_guided_LDA import perform_topic_modeling_on_excel_data_with_guided_lda, save_topics_and_coherence_to_file
from tf_idf import compute_tfidf_from_excel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

load_dotenv()

@click.command()
@click.option('--num_topics', prompt='Number of topics', type=int, help='Number of topics to generate in topic modeling')
@click.option('--num_words', prompt='Number of words per topic', type=int, help='Number of top words to display for each topic in the heatmap')

def perform_topic_modeling_from_excel(num_topics, num_words):
    # Step 0: Compute TF-IDF seed words
    print("\nStep 0: Computing TF-IDF seed words from Excel...")
    tfidf_folder = os.getenv('TF_IDF_RESULTS')
    compute_tfidf_from_excel(tfidf_folder, num_terms=num_topics * 10)  # Adjust multiplier as needed

    # Step 1: Load Excel data path
    excel_file = os.getenv('EXCEL_FILE_PATH')
    if not excel_file or not os.path.exists(excel_file):
        print(f"Error: The file {excel_file} does not exist or EXCEL_FILE_PATH is not set.")
        return

    # Step 2: Load TF-IDF seed words
    tfidf_seed_file = os.path.join(tfidf_folder, "tf_idf_results.txt")
    try:
        with open(tfidf_seed_file, 'r') as f:
            seed_words_list = [line.strip() for line in f.readlines() if line.strip()]
            total_seed_words = len(seed_words_list)

            if total_seed_words < num_topics:
                print(f"Error: You have {total_seed_words} TF-IDF seed words but requested {num_topics} topics.")
                return

            print(f"Distributing {total_seed_words} TF-IDF seed words across {num_topics} topics...")
            seed_words_per_topic = [[] for _ in range(num_topics)]
            for i, word in enumerate(seed_words_list):
                topic_idx = i % num_topics
                seed_words_per_topic[topic_idx].append(word)

            for idx, words in enumerate(seed_words_per_topic):
                print(f"Topic {idx + 1}: {len(words)} words")

    except FileNotFoundError:
        print(f"Error: TF-IDF seed file not found at {tfidf_seed_file}")
        return

    # Step 3: Run Guided LDA
    print(f"\nPerforming Guided LDA with {num_topics} topics...")
    lda_results = perform_topic_modeling_on_excel_data_with_guided_lda(
        excel_file, num_topics, seed_words_per_topic, num_words
    )

    if lda_results is None:
        print("Error: The LDA results are None. Please check the function implementation.")
        return

    coherence_score = lda_results['coherence_score']
    perplexity_score = lda_results['perplexity_score']
    topic_variance_score = lda_results['topic_variance_score']

    print(f"\nCoherence Score (Guided LDA): {coherence_score}")
    print(f"Perplexity Score (Guided LDA): {perplexity_score}")
    print(f"Topic Variance Score (Guided LDA): {topic_variance_score}")

    results_folder_path = os.getenv('RESULTS_DIRECTORY', os.path.dirname(excel_file))
    os.makedirs(results_folder_path, exist_ok=True)

    save_topics_and_coherence_to_file(
        lda_model=lda_results['lda'],
        coherence_score=coherence_score,
        perplexity_score=perplexity_score,
        topic_variance_score=topic_variance_score,
        results_folder_path=results_folder_path,
        num_words=num_words
    )

    # Step 4: Visualization
    print("\nGenerating pyLDAvis interactive visualization...")
    vis_data = gensimvis.prepare(
        lda_results['lda'],
        lda_results['corpus'],
        lda_results['dictionary']
    )
    vis_file_path = os.path.join(results_folder_path, "lda_visualization.html")
    pyLDAvis.save_html(vis_data, vis_file_path)
    print(f"pyLDAvis visualization saved to: {vis_file_path}")

    generate_heatmap(lda_results['lda'], num_words, results_folder_path)

def generate_heatmap(lda_model, num_words, results_folder_path):
    print("\nGenerating topic-word heatmap...")

    topic_word_data = {}
    vocab_set = set()
    for topic_id in range(lda_model.num_topics):
        topic_terms = lda_model.show_topic(topic_id, topn=num_words)
        for word, score in topic_terms:
            vocab_set.add(word)
            topic_word_data.setdefault(word, {})[f'Topic {topic_id+1}'] = score

    vocab = sorted(vocab_set)
    topics = [f'Topic {i+1}' for i in range(lda_model.num_topics)]
    data = pd.DataFrame(index=vocab, columns=topics)

    for word in vocab:
        for topic in topics:
            data.at[word, topic] = topic_word_data.get(word, {}).get(topic, 0)

    data = data.infer_objects().fillna(0).astype(float)

    plt.figure(figsize=(12, len(data) * 0.5))
    sns.heatmap(data, annot=True, cmap='viridis', cbar=True, linewidths=0.5)
    plt.title("Topic-Word Heatmap")
    plt.tight_layout()

    heatmap_path = os.path.join(results_folder_path, "lda_topic_word_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Heatmap saved to: {heatmap_path}")

# Entry point
if __name__ == '__main__':
    perform_topic_modeling_from_excel()
