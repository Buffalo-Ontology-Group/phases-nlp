# Bertopic Modeling on Plotkins Vaccines Book

This project is a Python application that builds a BERTopic-based pipeline for analyzing chapters from Plotkin’s Vaccines. It extracts and explores latent themes related to vaccines using BERTopic with a custom preprocessing workflow. The pipeline generates hierarchical topic trees that organize concepts found in the text, and these trees are then merged into a unified directed ontology graph. In addition, the project provides visual outputs such as keyword bar charts, hierarchical clustering dendrograms, and BART-based summaries to support deeper interpretation of the topics.



## Features
- Topic modeling with **BERTopic**, **UMAP**, and **HDBSCAN**  
- POS-filtered tokenization with **spaCy**
- Sentence level vaccine term filtering **Vaccine terms as seed words**
- Keyword labeling with **KeyBERT**  
- Summarization using **BART (transformers pipeline)**  
- Exports results to TXT, CSV, and Excel  
- Hierarchical tree building and visualization (ASCII + Graphviz diagrams)  
- Merging multiple topic trees into a consolidated ontology graph  


## Configuration

The number of topics and keywords to be generated is prompted at the command line. When running the script, the user will be asked to enter the desired number of topics and keywords.





 
