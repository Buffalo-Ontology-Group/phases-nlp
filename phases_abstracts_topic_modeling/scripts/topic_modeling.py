import os
import gensim
from gensim import corpora
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already present
nltk.download('punkt')
nltk.download('stopwords')


# Function to preprocess the text (tokenize, remove stopwords)
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())  # Tokenize the text and convert to lowercase
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stopwords and non-alphabetic words
    return tokens


# Function to perform topic modeling using LDA (Latent Dirichlet Allocation)
def perform_lda(topic_texts, num_topics):
    # Preprocess the texts (tokenization)
    processed_texts = [preprocess_text(text) for text in topic_texts]
    
    # Create a dictionary and corpus for topic modeling
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # Perform LDA (Latent Dirichlet Allocation) to generate topics
    lda = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
    return lda, dictionary, corpus


# Function to save the topics to a .txt file
def save_topics_to_file(lda, folder_path, titles):
    topic_file_path = os.path.join(folder_path, "topic_modeling_results.txt")

    with open(topic_file_path, 'w') as file:
        # Write each topic with the top 5 words
        for i in range(lda.num_topics):
            file.write(f"Topic {i+1}:\n")
            file.write(f"{lda.print_topic(i, 5)}\n")
            file.write("=" * 50 + "\n")

        file.write("\n\nTopic Assignments for Each Abstract:\n")
        
        # For each abstract, get the topic distribution and write the most relevant topic
        for i, title in enumerate(titles):
            topics = lda[corpora.Dictionary([preprocess_text(title)]).doc2bow(preprocess_text(title))]
            sorted_topics = sorted(topics, key=lambda x: x[1], reverse=True)  # Sort by relevance
            top_topic = sorted_topics[0][0] if sorted_topics else None
            file.write(f"Text: {title}\nAssigned Topic: {top_topic + 1 if top_topic is not None else 'N/A'}\n")
            file.write("=" * 50 + "\n")

    print(f"Topic modeling results saved to: {topic_file_path}")


# Function to process the downloaded text files and perform topic modeling
def perform_topic_modeling_on_downloaded_texts(folder_path, num_topics):
    topic_texts = []
    titles = []

    # Read the text files and collect their text for topic modeling
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Skip specific files like 'list.txt' or any other unwanted files
        if file_name == 'list.txt':
            print(f"Skipping file: {file_name}")
            continue

        # Only process .txt files (ignore others)
        if file_path.endswith(".txt"):
            with open(file_path, 'r') as file:
                content = file.read()

                # Debugging: Print content to inspect it
                # print(f"Reading file: {file_name}")
                print(content[:300])  # Print the first 300 characters to check file structure

                # Add entire content of the file to topic texts (no need for abstract extraction)
                if content.strip():  # Only add non-empty text
                    topic_texts.append(content.strip())
                    titles.append(file_name)
                else:
                    print(f"No content found in file {file_name}")

    # Perform topic modeling if texts are available
    if topic_texts:
        print(f"Performing topic modeling on the texts with {num_topics} topics...")
        lda, dictionary, corpus = perform_lda(topic_texts, num_topics)

        # Save topics to a .txt file
        save_topics_to_file(lda, folder_path, titles)
    else:
        print("No texts found to process for topic modeling.")
