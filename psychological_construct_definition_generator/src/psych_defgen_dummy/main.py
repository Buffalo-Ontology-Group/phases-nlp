import os

import click

from psych_defgen.apa_dictionary import (
    get_apa_dictionary_entry,
)
from psych_defgen.chunk_text import chunk_article
from psych_defgen.extract_definition_candidates import (
    extract_definition_candidates,
)
from psych_defgen.get_pmc_articles import get_pmc_ids
from psych_defgen.models import Article
from psych_defgen.parse_fulltext import (
    get_full_text_from_pmcid,
)
from psych_defgen.pubmed_search import search_pubmed
from psych_defgen.rag_retrieval import (
    retrieve_relevant_texts,
)
from psych_defgen.retrieve_abstracts import (
    fetch_pubmed_abstracts,
)
from psych_defgen.select_definition_sentence import (
    select_best_definition_sentence,
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
    Convert a structured PubMed abstract record into
    the evidence format used by retrieval.
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
    definition_from_literature,
    evidence,
    output_path=None,
    output_dir="outputs",
):
    """
    Save the definition found in the literature and the retrieved
    evidence as a Markdown file.
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    if output_path:
        filepath = output_path

        os.makedirs(
            os.path.dirname(filepath) or ".",
            exist_ok=True,
        )

    else:
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

        # -------------------------------------------------
        # APA Dictionary
        # -------------------------------------------------

        file.write(
            "## APA Dictionary\n\n"
        )

        apa_status = apa_entry.get("status")
        apa_url = apa_entry.get("url")

        if apa_status == "found":

            file.write(
                "An entry for this term was found in "
                "the APA Dictionary of Psychology.\n\n"
            )

            file.write(
                "The APA Dictionary is provided as an "
                "external reference only and is not used "
                "to identify definitions in the literature. \n\n"
            )

            file.write(
                f"Reference: "
                f"[APA Dictionary Entry]({apa_url})\n\n"
            )

        elif apa_status == "not_found":

            file.write(
                "No entry for this term was found in "
                "the APA Dictionary of Psychology.\n\n"
            )

            file.write(
                "This does not affect the search for "
                "a definition in the retrieved literature.\n\n"
            )

        elif apa_status == "unavailable":

            file.write(
                "The presence of an APA Dictionary entry "
                "could not be verified automatically.\n\n"
            )

            file.write(
                f"Reference: "
                f"[Search the APA Dictionary]({apa_url})"
                "\n\n"
            )

        else:

            file.write(
                "The APA Dictionary of Psychology could "
                "not be accessed automatically.\n\n"
            )

            file.write(
                f"Reference: "
                f"[APA Dictionary]({apa_url})\n\n"
            )

        # -------------------------------------------------
        # Definition found in literature
        # -------------------------------------------------

        file.write(
            "## Definition from Literature\n\n"
        )

        file.write(
            definition_from_literature.strip()
            + "\n\n"
        )

        # -------------------------------------------------
        # Retrieved Evidence
        # -------------------------------------------------

        file.write(
            "## Retrieved Evidence\n\n"
        )

        if not evidence:
            file.write(
                "No relevant evidence was retrieved.\n\n"
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

                if isinstance(
                    authors,
                    list,
                ):
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

        # -------------------------------------------------
        # Curation status
        # -------------------------------------------------

        file.write(
            "## Curation Status\n\n"
        )

        file.write(
            "Needs expert review.\n"
        )

    return filepath


@click.command()
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
@click.option(
    "--email",
    default=None,
    help=(
        "NCBI email address. Overrides the "
        "NCBI_EMAIL environment variable."
    ),
)
@click.option(
    "--api-key",
    default=None,
    help=(
        "NCBI API key. Overrides the "
        "NCBI_API_KEY environment variable."
    ),
)
@click.option(
    "--output",
    default=None,
    help=(
        "Output Markdown file. "
        "If omitted, a file is created in the "
        "outputs directory."
    ),
)
@click.argument(
    "term",
    nargs=-1,
    required=True,
)
def main(
    max_results,
    top_k,
    email,
    api_key,
    output,
    term,
):
    """
    Find an explicit definition of a psychological
    construct in retrieved PubMed abstracts and PMC
    full-text literature.

    The APA Dictionary is used only to verify and
    reference official entries and is not used as
    evidence when identifying definitions from the
    literature.
    """

    term = " ".join(term)

    email = (
        email
        or os.getenv("NCBI_EMAIL")
    )

    api_key = (
        api_key
        or os.getenv("NCBI_API_KEY")
    )

    if not email:
        raise click.ClickException(
            "An NCBI email address is required. "
            "Provide it using --email or set the "
            "NCBI_EMAIL environment variable."
        )

    # =====================================================
    # APA Dictionary
    # =====================================================

    print(
        f"Creating APA Dictionary reference for: "
        f"{term}"
    )

    apa_entry = (
        get_apa_dictionary_entry(
            term
        )
    )

    print(
        f"APA Dictionary URL: "
        f"{apa_entry['url']}"
    )

    # =====================================================
    # PubMed search
    # =====================================================

    print(
        f"Searching PubMed for: {term}"
    )

    pmids = search_pubmed(
        term,
        max_results=max_results,
        email=email,
        api_key=api_key,
    )

    print(
        f"PubMed articles found: {len(pmids)}"
    )

    # =====================================================
    # Find PMC full-text versions
    # =====================================================

    pmid_to_pmcid = get_pmc_ids(
        pmids,
        email=email,
        api_key=api_key,
    )

    print(
        "PMC full-text articles found: "
        f"{len(pmid_to_pmcid)}"
    )

    # Definition candidates specifically extracted
    # from PMC full text.
    definition_candidates = []

    # General chunks from PMC full text.
    full_text_chunks = []

    # =====================================================
    # Process PMC full text
    # =====================================================

    for pmid, pmcid in (
        pmid_to_pmcid.items()
    ):

        print(
            "Using full text: "
            f"PMID {pmid}, {pmcid}"
        )

        try:
            title, text = (
                get_full_text_from_pmcid(
                    pmcid,
                    email=email,
                    api_key=api_key,
                )
            )

        except Exception as error:
            print(
                f"Could not retrieve full text for "
                f"PMID {pmid}, {pmcid}: {error}"
            )

            print(
                "Skipping full text and continuing."
            )

            continue


        if not text:

            print(
                f"No text retrieved for {pmcid}. "
                "Skipping article."
            )

            continue

        # Basic relevance safeguard.
        if term.lower() not in text.lower():

            print(
                "Skipping PMC article because "
                f"term '{term}' was not found "
                "in the article text."
            )

            continue

        print(
            f"Article title: {title}"
        )

        print(
            "Article text length: "
            f"{len(text)}"
        )

        # -------------------------------------------------
        # Extract explicit definition candidates
        # from the full article.
        # -------------------------------------------------

        candidates = (
            extract_definition_candidates(
                text,
                term,
            )
        )

        print(
            "Definition candidates from "
            f"{pmcid}: {len(candidates)}"
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

        # -------------------------------------------------
        # Chunk complete article for broader RAG evidence.
        # -------------------------------------------------

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

    # =====================================================
    # PubMed abstracts
    #
    # IMPORTANT:
    # Abstracts are always processed, even when
    # PMC full text is available.
    # =====================================================

    print(
        "Fetching PubMed abstracts."
    )

    abstract_records = (
        fetch_pubmed_abstracts(
            pmids,
            email=email,
            api_key=api_key,
        )
    )

    abstract_chunks = []

    # Definition candidates specifically extracted
    # from abstracts.
    abstract_definition_candidates = []

    for record in abstract_records:

        abstract_text = record.get(
            "abstract",
            "",
        ).strip()

        if not abstract_text:
            continue

        # Only retain abstracts that mention
        # the requested construct.
        if (
            term.lower()
            not in abstract_text.lower()
        ):
            continue

        abstract_record = (
            create_abstract_evidence_record(
                record
            )
        )

        # Keep complete abstract for retrieval.
        abstract_chunks.append(
            abstract_record
        )

        # -------------------------------------------------
        # Search the abstract itself for explicit
        # definition statements.
        # -------------------------------------------------

        candidates = (
            extract_definition_candidates(
                abstract_text,
                term,
            )
        )

        for candidate in candidates:

            candidate_record = (
                create_abstract_evidence_record(
                    record
                )
            )

            candidate_record["text"] = (
                candidate
            )

            abstract_definition_candidates.append(
                candidate_record
            )

    print(
        "Abstracts retained: "
        f"{len(abstract_chunks)}"
    )

    print(
        "Definition candidates from abstracts: "
        f"{len(abstract_definition_candidates)}"
    )

    # =====================================================
    # Combine explicit definitions from BOTH sources
    # =====================================================

    all_definition_candidates = (
        definition_candidates
        + abstract_definition_candidates
    )

    print(
        "Total explicit definition candidates: "
        f"{len(all_definition_candidates)}"
    )

    # =====================================================
    # Create retrieval pool
    # =====================================================

    if all_definition_candidates:

        print(
            "Explicit definition candidates found. "
            "They will be considered together with "
            "full-text and abstract evidence."
        )

        retrieval_items = (
            all_definition_candidates
            + full_text_chunks
            + abstract_chunks
        )

    else:

        print(
            "No explicit definition candidates found. "
            "Using full-text chunks and PubMed "
            "abstracts for retrieval."
        )

        retrieval_items = (
            full_text_chunks
            + abstract_chunks
        )

    print(
        "Total retrieval items: "
        f"{len(retrieval_items)}"
    )

    # =====================================================
    # Semantic retrieval
    # =====================================================

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

    # =====================================================
    # Select Definition from Literature
    # =====================================================

    best_definition_sentence = None

    candidate_sources = (
        all_definition_candidates
        + retrieved
    )

    best_definition_sentence = (
        select_best_definition_sentence(
            term,
            candidate_sources,
        )
    )

    # =====================================================
    # Produce definition
    # =====================================================

    if best_definition_sentence:

        print(
            "Using an explicit definition found "
            "in the literature."
        )

        definition_from_literature = (
            best_definition_sentence.strip()
        )

        evidence_summary = (
            best_definition_sentence.strip()
        )

    else:

        print(
            "No explicit literature definition found. "
            "in the retrieved literature."
        )

        definition_from_literature = (
            "No explicit definition was found "
            "in the retrieved literature."
        )
     
    # =====================================================
    # Save output
    # =====================================================

    output_path = save_output(
        term=term,
        apa_entry=apa_entry,
        definition_from_literature=(
            definition_from_literature
        ),
        evidence=retrieved,
        output_path=output,
    )

    print(
        f"Saved output to: {output_path}"
    )


if __name__ == "__main__":
    main()