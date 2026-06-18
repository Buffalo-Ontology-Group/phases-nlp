import argparse
import os

from src.pubmed_search import search_pubmed
from src.get_pmc_articles import get_pmc_ids
from src.parse_fulltext import get_full_text_from_pmcid
from src.extract_definition_candidates import extract_definition_candidates
from src.rag_retrieval import retrieve_relevant_texts
from src.generate_definition import generate_definition
from src.synthesize_construct import synthesize_construct
from src.models import Article
from src.chunk_text import chunk_article
from src.retrieve_abstracts import fetch_pubmed_abstracts


def save_output(term, definition, evidence_summary, evidence, output_dir="data/outputs"):
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{term.replace(' ', '_')}_definition.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as file:
        file.write(f"# {term}\n\n")

        file.write("## Generated Ontology Definition\n\n")
        file.write(definition + "\n\n")

        file.write("## Evidence Summary\n\n")
        file.write(evidence_summary + "\n\n")

        file.write("## Retrieved Evidence\n\n")
        for item in evidence:
            file.write(f"- Score: {item['score']:.4f}\n")
            file.write(f"  Evidence: {item['text']}\n\n")

        file.write("## Curation Status\n\n")
        file.write("Needs expert review.\n")

    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Generate ontology-style definitions for psychological constructs using PubMed/PMC RAG."
    )

    parser.add_argument(
        "term",
        nargs="+",
        help="Psychological construct term, e.g. loneliness, social vulnerability"
    )

    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()
    term = " ".join(args.term)

    print(f"Searching PubMed for: {term}")

    pmids = search_pubmed(term, max_results=args.max_results)
    print(f"PubMed articles found: {len(pmids)}")

    pmid_to_pmcid = get_pmc_ids(pmids)
    print(f"PMC full-text articles found: {len(pmid_to_pmcid)}")

    definition_candidates = []
    full_text_chunks = []

    for pmid, pmcid in pmid_to_pmcid.items():
        print(f"Using full text: PMID {pmid}, {pmcid}")

        title, text = get_full_text_from_pmcid(pmcid)

        if pmid not in text and term.lower() not in text.lower():
            print(f"Skipping PMC article because it does not match PMID {pmid} or term '{term}'.")
            continue

        print(f"Article title: {title}")
        print(f"Article text length: {len(text)}")

        candidates = extract_definition_candidates(text, term)
        definition_candidates.extend(candidates)

        article = Article(
            pmid=pmid,
            pmcid=pmcid,
            title=title,
            source="PMC full text",
            text=text
        )

        chunks = chunk_article(article)
        print(f"Chunks created from article: {len(chunks)}")

        full_text_chunks.extend([chunk.text for chunk in chunks])

    print("Fetching PubMed abstracts as fallback evidence.")
    abstract_text = fetch_pubmed_abstracts(pmids)

    abstract_chunks = []
    if abstract_text:
        abstract_chunks = [
            chunk.strip()
            for chunk in abstract_text.split("\n\n")
            if term.lower() in chunk.lower()
        ]

    print(f"Abstract chunks found: {len(abstract_chunks)}")

    if definition_candidates:
        print(f"Found {len(definition_candidates)} definition-like candidate(s).")
        retrieval_texts = definition_candidates + full_text_chunks + abstract_chunks
    else:
        print("No explicit definition candidates found. Using full-text chunks + PubMed abstracts.")
        retrieval_texts = full_text_chunks + abstract_chunks

    print(f"Total retrieval texts: {len(retrieval_texts)}")

    if not retrieval_texts:
        print("No evidence retrieved from PMC full text or PubMed abstracts.")
        return

    retrieved = retrieve_relevant_texts(
        term,
        retrieval_texts,
        top_k=args.top_k
    )

    print(f"Retrieved evidence passages: {len(retrieved)}")

    evidence_summary = synthesize_construct(term, retrieved)
    definition = generate_definition(term, evidence_summary)

    output_path = save_output(
        term,
        definition,
        evidence_summary,
        retrieved
    )

    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
