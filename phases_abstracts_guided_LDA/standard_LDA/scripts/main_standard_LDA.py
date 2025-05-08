import os
import click
from dotenv import load_dotenv
from tm_standard_LDA import perform_topic_modeling_on_excel_data, save_topics_and_coherence_to_file
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional

# Load environment variables from the .env file
load_dotenv()

@click.command()
@click.option('--num_topics', prompt='Number of topics', type=int, help='Number of topics to generate in topic modeling')
@click.option('--num_words', prompt='Number of words per topic', type=int, help='Number of top words to display for each topic in the heatmap')

def perform_topic_modeling_from_excel(num_topics: int, num_words: int) -> None:
    """
    Perform topic modeling on text data from an Excel file using LDA.

    This function:
    - Loads the Excel file path from environment variables.
    - Applies topic modeling using LDA with user-specified number of topics.
    - Outputs coherence, perplexity, and topic variance scores.
    - Saves textual results and a pyLDAvis HTML visualization.
    - Generates and saves a heatmap of top words per topic.

    Parameters:
    - num_topics (int): Number of topics to generate.
    - num_words (int): Number of top words to display.

    """
    # Get the path to the Excel file from the environment variable
    excel_file = os.getenv('EXCEL_FILE_PATH')

    if not excel_file or not os.path.exists(excel_file):
        print(f"Error: The file {excel_file} does not exist or EXCEL_FILE_PATH is not set.")
        return

    # Perform topic modeling using standard LDA
    print(f"\nPerforming Standard LDA with {num_topics} topics...")
    lda_results = perform_topic_modeling_on_excel_data(excel_file, num_topics, num_words)

    if lda_results is None:
        print("Error: The LDA results are None. Please check the function implementation.")
        return

    coherence_score = lda_results['coherence_score']
    perplexity_score = lda_results['perplexity_score']
    topic_variance_score = lda_results['topic_variance_score']

    print(f"\nCoherence Score (Standard LDA): {coherence_score}")
    print(f"Perplexity Score (Standard LDA): {perplexity_score}")
    print(f"Topic Variance Score (Standard LDA): {topic_variance_score}")

    results_folder_path = os.getenv('RESULTS_DIRECTORY', os.path.dirname(excel_file))
    os.makedirs(results_folder_path, exist_ok=True)

    # Save textual results
    save_topics_and_coherence_to_file(lda_results['lda'], coherence_score, perplexity_score, topic_variance_score, results_folder_path, num_words)

    # Generate pyLDAvis HTML visualization
    print("\nGenerating pyLDAvis interactive visualization...")
    vis_data = gensimvis.prepare(
        lda_results['lda'],
        lda_results['corpus'],
        lda_results['dictionary']
    )

    vis_file_path = os.path.join(results_folder_path, "lda_visualization.html")
    pyLDAvis.save_html(vis_data, vis_file_path)
    print(f"pyLDAvis visualization saved to: {vis_file_path}")

    # Generate and save heatmap
    generate_heatmap(lda_results['lda'], num_words=num_words, results_folder_path=results_folder_path)

def generate_heatmap(lda_model, num_words: int, results_folder_path:str) -> None:
    """
    Generate and save a heatmap showing the top words for each topic.

    The heatmap displays topics as columns and words as rows, with intensity representing the word's weight in that topic.

    Parameters:
    -lda_model (LdaModel): Trained LDA model object from Gensim.
    - num_words (int): Number of top words to include per topic.
    - results_folder_path (str): Directory path to save the heatmap image.
    
    """
    print("\nGenerating topic-word heatmap...")

    # Get top words for each topic
    topic_word_data = {}
    vocab_set = set()
    for topic_id in range(lda_model.num_topics):
        topic_terms = lda_model.show_topic(topic_id, topn=num_words)
        for word, score in topic_terms:
            vocab_set.add(word)
            topic_word_data.setdefault(word, {})[f'Topic {topic_id+1}'] = score

    # Create DataFrame for heatmap
    vocab = sorted(vocab_set)
    topics = [f'Topic {i+1}' for i in range(lda_model.num_topics)]
    data = pd.DataFrame(index=vocab, columns=topics)

    for word in vocab:
        for topic in topics:
            data.at[word, topic] = topic_word_data.get(word, {}).get(topic, 0)

    # Ensure proper dtype inference before fillna and astype
    data = data.infer_objects()
    data = data.fillna(0).astype(float)

    # Plot heatmap
    plt.figure(figsize=(12, len(data) * 0.5))
    sns.heatmap(data, annot=True, cmap='viridis', cbar=True, linewidths=0.5)
    plt.title("Topic-Word Heatmap")
    plt.tight_layout()

    # Save heatmap
    heatmap_path = os.path.join(results_folder_path, "lda_topic_word_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Heatmap saved to: {heatmap_path}")

# Entry point to run the script
if __name__ == '__main__':
    perform_topic_modeling_from_excel()
