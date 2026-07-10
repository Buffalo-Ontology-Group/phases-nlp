import click
import os

from psych_defgen_dummy.apa_dictionary import get_apa_dictionary_definition
from psych_defgen_dummy.select_definition_sentence import select_best_definition_sentence
from psych_defgen_dummy.pubmed_search import search_pubmed
from psych_defgen_dummy.get_pmc_articles import get_pmc_ids
from psych_defgen_dummy.parse_fulltext import get_full_text_from_pmcid
from psych_defgen_dummy.extract_definition_candidates import extract_definition_candidates
from psych_defgen_dummy.rag_retrieval import retrieve_relevant_texts
from psych_defgen_dummy.generate_definition import generate_definition
from psych_defgen_dummy.synthesize_construct import synthesize_construct
from psych_defgen_dummy.models import Article
from psych_defgen_dummy.chunk_text import chunk_article
from psych_defgen_dummy.retrieve_abstracts import fetch_pubmed_abstracts


def save_output(term, apa_entry, literature_derived_concept_summary, evidence_summary, evidence, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{term.replace(' ', '_')}_definition.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as file:
        file.write(f"# {term}\n\n")

        file.write("## APA Dictionary Definition\n\n")
        if apa_entry:
            file.write(apa_entry["definition"] + "\n\n")
            file.write(f"Source: {apa_entry['source']}\n\n")
            file.write(f"Reference: [APA Dictionary Entry]({apa_entry['url']})\n\n")
        else:
            file.write("No APA Dictionary definition was retrieved.\n\n")

        file.write("## Literature-derived Concept Summary\n\n")
        file.write(literature_derived_concept_summary + "\n\n")

        file.write("## Evidence Summary\n\n")
        file.write(evidence_summary + "\n\n")

        file.write("## Retrieved Evidence\n\n")
        for item in evidence:
            file.write(f"- Score: {item['score']:.4f}\n")
            file.write(f"  Evidence: {item['text']}\n\n")

        file.write("## Curation Status\n\n")
        file.write("Needs expert review.\n")

    return filepath


@click.command()
@click.argument("term", nargs=-1, required=True)
@click.option("--max-results", default=100, show_default=True, type=int)
@click.option("--top-k", default=10, show_default=True, type=int)
def main(term, max_results, top_k):
    term = " ".join(term)

    print(f"Searching APA Dictionary for: {term}")
    apa_entry = get_apa_dictionary_definition(term)

    if apa_entry:
        print("APA Dictionary definition found.")
    else:
        print("No APA Dictionary definition found.")

    print(f"Searching PubMed for: {term}")

    pmids = search_pubmed(term, max_results=max_results)
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

    retrieved = []
    if retrieval_texts:
        retrieved = retrieve_relevant_texts(
            term,
            retrieval_texts,
            top_k=top_k
        )

    print(f"Retrieved evidence passages: {len(retrieved)}")

    best_definition_sentence = select_best_definition_sentence(term, retrieved)

    if best_definition_sentence:
        evidence_summary = best_definition_sentence
    else:
        evidence_summary = synthesize_construct(term, retrieved)

    concept_summary = generate_definition(term, evidence_summary)

    output_path = save_output(
        term=term,
        apa_entry=apa_entry,
        literature_derived_concept_summary=concept_summary,
        evidence_summary=evidence_summary,
        evidence=retrieved
    )

    print(f"Saved output to: {output_path}")
    

if __name__ == "__main__":
    main()
