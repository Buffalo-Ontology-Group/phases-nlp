import os

import click

from psych_defgen_dummy.apa_dictionary import (
    get_apa_dictionary_definition,
)
from psych_defgen_dummy.chunk_text import chunk_article
from psych_defgen_dummy.extract_definition_candidates import (
    extract_definition_candidates,
)
from psych_defgen_dummy.generate_definition import generate_definition
from psych_defgen_dummy.get_pmc_articles import get_pmc_ids
from psych_defgen_dummy.models import Article
from psych_defgen_dummy.parse_fulltext import (
    get_full_text_from_pmcid,
)
from psych_defgen_dummy.pubmed_search import search_pubmed
from psych_defgen_dummy.rag_retrieval import (
    retrieve_relevant_texts,
)
from psych_defgen_dummy.retrieve_abstracts import (
    fetch_pubmed_abstracts,
)
from psych_defgen_dummy.select_definition_sentence import (
    select_best_definition_sentence,
)
from psych_defgen_dummy.synthesize_construct import (
    synthesize_construct,
)


def create_pmc_evidence_record(
    text,
    title,
    pmid,
    pmcid,
):
    """
    Create a PMC evidence record while preserving
    article metadata.
    """

    return {
        "text": text,
        "title": title,
        "authors": [],
        "journal": None,
        "year": None,
        "pmid": pmid,
        "pmcid": pmcid,
        "source": "PMC full text",
        "pubmed_url": (
            f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            if pmid
            else None
        ),
        "url": (
            f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
            if pmcid
            else None
        ),
    }


def create_abstract_evidence_record(record):
    """
    Convert a structured PubMed abstract record
    into the evidence format used by retrieval.
    """

    return {
        "text": record.get("abstract", ""),
        "title": record.get("title"),
        "authors": record.get("authors", []),
        "journal": record.get("journal"),
        "year": record.get("year"),
        "pmid": record.get("pmid"),
        "pmcid": None,
        "source": record.get(
            "source",
            "PubMed abstract",
        ),
        "pubmed_url": record.get("pubmed_url"),
        "url": None,
    }


def save_output(
    term,
    apa_entry,
    literature_derived_concept_summary,
    evidence_summary,
    evidence,
    output_dir="outputs",
):
    """
    Save the generated definition and retrieved
    evidence as a Markdown file.
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    filename = (
        f"{term.replace(' ', '_')}_definition.md"
    )

    filepath = os.path.join(
        output_dir,
        filename,
    )

    with open(
        filepath,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"# {term}\n\n"
        )

        file.write(
            "## APA Dictionary Definition\n\n"
        )

        if apa_entry:
            file.write(
                apa_entry["definition"] + "\n\n"
            )

            file.write(
                f"Source: {apa_entry['source']}\n\n"
            )

            file.write(
                "Reference: "
                f"[APA Dictionary Entry]"
                f"({apa_entry['url']})\n\n"
            )
        else:
            file.write(
                "No APA Dictionary definition was "
                "retrieved.\n\n"
            )

        file.write(
            "## Literature-derived Concept Summary\n\n"
        )

        file.write(
            literature_derived_concept_summary
            + "\n\n"
        )

        file.write(
            "## Evidence Summary\n\n"
        )

        file.write(
            evidence_summary + "\n\n"
        )

        file.write(
            "## Retrieved Evidence\n\n"
        )

        if not evidence:
            file.write(
                "No relevant evidence was "
                "retrieved.\n\n"
            )

        for index, item in enumerate(
            evidence,
            start=1,
        ):
            file.write(
                f"### Evidence {index}\n\n"
            )

            score = item.get("score")

            if score is not None:
                file.write(
                    f"**Score:** {score:.4f}\n\n"
                )

            title = item.get("title")

            if title:
                file.write(
                    f"**Article:** {title}\n\n"
                )

            authors = item.get("authors")

            if authors:
                if isinstance(authors, list):
                    authors = ", ".join(authors)

                file.write(
                    f"**Authors:** {authors}\n\n"
                )

            journal = item.get("journal")
            year = item.get("year")

            if journal and year:
                file.write(
                    f"**Journal:** "
                    f"{journal} ({year})\n\n"
                )
            elif journal:
                file.write(
                    f"**Journal:** {journal}\n\n"
                )
            elif year:
                file.write(
                    f"**Year:** {year}\n\n"
                )

            source = item.get("source")

            if source:
                file.write(
                    f"**Source:** {source}\n\n"
                )

            pmid = item.get("pmid")

            if pmid:
                file.write(
                    f"**PMID:** {pmid}\n\n"
                )

            pmcid = item.get("pmcid")

            if pmcid:
                file.write(
                    f"**PMCID:** {pmcid}\n\n"
                )

            pubmed_url = item.get(
                "pubmed_url"
            )

            if pubmed_url:
                file.write(
                    "**PubMed:** "
                    f"[View article]"
                    f"({pubmed_url})\n\n"
                )

            article_url = item.get("url")

            if article_url:
                file.write(
                    "**Full text:** "
                    f"[View article]"
                    f"({article_url})\n\n"
                )

            file.write(
                "**Evidence:**\n\n"
            )

            file.write(
                item["text"].strip()
                + "\n\n"
            )

        file.write(
            "## Curation Status\n\n"
        )

        file.write(
            "Needs expert review.\n"
        )

    return filepath


@click.command()
@click.argument(
    "term",
    nargs=-1,
    required=True,
)
@click.option(
    "--max-results",
    default=100,
    show_default=True,
    type=int,
)
@click.option(
    "--top-k",
    default=10,
    show_default=True,
    type=int,
)
def main(
    term,
    max_results,
    top_k,
):
    """
    Generate a literature-informed definition
    for a psychological construct.
    """

    term = " ".join(term)

    print(
        f"Searching APA Dictionary for: {term}"
    )

    apa_entry = (
        get_apa_dictionary_definition(term)
    )

    if apa_entry:
        print(
            "APA Dictionary definition found."
        )
    else:
        print(
            "No APA Dictionary definition found."
        )

    print(
        f"Searching PubMed for: {term}"
    )

    pmids = search_pubmed(
        term,
        max_results=max_results,
    )

    print(
        f"PubMed articles found: {len(pmids)}"
    )

    pmid_to_pmcid = get_pmc_ids(pmids)

    print(
        "PMC full-text articles found: "
        f"{len(pmid_to_pmcid)}"
    )

    definition_candidates = []
    full_text_chunks = []

    for pmid, pmcid in (
        pmid_to_pmcid.items()
    ):
        print(
            "Using full text: "
            f"PMID {pmid}, {pmcid}"
        )

        title, text = (
            get_full_text_from_pmcid(
                pmcid
            )
        )

        if not text:
            print(
                f"No text retrieved for {pmcid}. "
                "Skipping article."
            )
            continue

        if (
            str(pmid) not in text
            and term.lower()
            not in text.lower()
        ):
            print(
                "Skipping PMC article because it "
                f"does not match PMID {pmid} or "
                f"term '{term}'."
            )
            continue

        print(
            f"Article title: {title}"
        )

        print(
            "Article text length: "
            f"{len(text)}"
        )

        candidates = (
            extract_definition_candidates(
                text,
                term,
            )
        )

        for candidate in candidates:
            definition_candidates.append(
                create_pmc_evidence_record(
                    text=candidate,
                    title=title,
                    pmid=pmid,
                    pmcid=pmcid,
                )
            )

        article = Article(
            pmid=pmid,
            pmcid=pmcid,
            title=title,
            source="PMC full text",
            text=text,
        )

        chunks = chunk_article(
            article
        )

        print(
            "Chunks created from article: "
            f"{len(chunks)}"
        )

        for chunk in chunks:
            full_text_chunks.append(
                create_pmc_evidence_record(
                    text=chunk.text,
                    title=chunk.title,
                    pmid=chunk.pmid,
                    pmcid=chunk.pmcid,
                )
            )

    print(
        "Fetching PubMed abstracts as "
        "fallback evidence."
    )

    abstract_records = (
        fetch_pubmed_abstracts(pmids)
    )

    abstract_chunks = []

    for record in abstract_records:
        abstract_text = record.get(
            "abstract",
            "",
        ).strip()

        if (
            abstract_text
            and term.lower()
            in abstract_text.lower()
        ):
            abstract_chunks.append(
                create_abstract_evidence_record(
                    record
                )
            )

    print(
        "Abstract chunks found: "
        f"{len(abstract_chunks)}"
    )

    if definition_candidates:
        print(
            "Found "
            f"{len(definition_candidates)} "
            "definition-like candidate(s)."
        )

        retrieval_items = (
            definition_candidates
            + full_text_chunks
            + abstract_chunks
        )
    else:
        print(
            "No explicit definition candidates "
            "found. Using full-text chunks and "
            "PubMed abstracts."
        )

        retrieval_items = (
            full_text_chunks
            + abstract_chunks
        )

    print(
        "Total retrieval items: "
        f"{len(retrieval_items)}"
    )

    retrieved = []

    if retrieval_items:
        retrieved = (
            retrieve_relevant_texts(
                term,
                retrieval_items,
                top_k=top_k,
            )
        )

    print(
        "Retrieved evidence passages: "
        f"{len(retrieved)}"
    )

    best_definition_sentence = (
        select_best_definition_sentence(
            term,
            retrieved,
        )
    )

    if best_definition_sentence:
        evidence_summary = (
            best_definition_sentence
        )
    else:
        evidence_summary = (
            synthesize_construct(
                term,
                retrieved,
            )
        )

    concept_summary = generate_definition(
        term,
        evidence_summary,
    )

    output_path = save_output(
        term=term,
        apa_entry=apa_entry,
        literature_derived_concept_summary=(
            concept_summary
        ),
        evidence_summary=evidence_summary,
        evidence=retrieved,
    )

    print(
        f"Saved output to: {output_path}"
    )


if __name__ == "__main__":
    main()