from Bio import Entrez

from .entrez_config import configure_entrez


def _extract_authors(author_list):
    """
    Convert a PubMed author list into readable names.
    """

    authors = []

    for author in author_list or []:
        collective_name = author.get("CollectiveName")
        if collective_name:
            authors.append(str(collective_name))
            continue

        last_name = author.get("LastName")
        initials = author.get("Initials")
        fore_name = author.get("ForeName")

        if last_name and initials:
            authors.append(f"{last_name} {initials}")
        elif fore_name and last_name:
            authors.append(f"{fore_name} {last_name}")
        elif last_name:
            authors.append(str(last_name))

    return authors


def _extract_abstract_text(abstract_sections):
    """
    Combine labeled PubMed abstract sections into one string.
    """

    if not abstract_sections:
        return ""

    sections = []

    for section in abstract_sections:
        text = str(section).strip()

        if not text:
            continue

        label = getattr(section, "attributes", {}).get("Label")

        if label:
            sections.append(f"{label}: {text}")
        else:
            sections.append(text)

    return "\n".join(sections)


def _extract_publication_year(article_record):
    """
    Extract the publication year from available PubMed date fields.
    """

    journal_issue = (
        article_record
        .get("MedlineCitation", {})
        .get("Article", {})
        .get("Journal", {})
        .get("JournalIssue", {})
    )

    pub_date = journal_issue.get("PubDate", {})

    year = pub_date.get("Year")
    if year:
        return str(year)

    medline_date = pub_date.get("MedlineDate")
    if medline_date:
        return str(medline_date)

    return None


def fetch_pubmed_abstracts(pmids):
    """
    Fetch PubMed abstracts and article metadata.

    Parameters
    ----------
    pmids : list
        List of PubMed IDs.

    Returns
    -------
    list[dict]
        Structured abstract records containing the PMID,
        title, authors, journal, publication year, URL,
        and abstract text.
    """

    configure_entrez()

    if not pmids:
        return []

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(str(pmid) for pmid in pmids),
        rettype="medline",
        retmode="xml",
    )

    try:
        records = Entrez.read(handle)
    finally:
        handle.close()

    abstract_records = []

    for pubmed_article in records.get("PubmedArticle", []):
        medline_citation = pubmed_article.get(
            "MedlineCitation",
            {},
        )

        article = medline_citation.get(
            "Article",
            {},
        )

        pmid = str(
            medline_citation.get("PMID", "")
        ).strip()

        title = str(
            article.get(
                "ArticleTitle",
                "Title unavailable",
            )
        ).strip()

        abstract_data = article.get(
            "Abstract",
            {},
        )

        abstract_sections = abstract_data.get(
            "AbstractText",
            [],
        )

        abstract_text = _extract_abstract_text(
            abstract_sections
        )

        if not abstract_text:
            continue

        authors = _extract_authors(
            article.get("AuthorList", [])
        )

        journal = str(
            article.get(
                "Journal",
                {},
            ).get(
                "Title",
                "",
            )
        ).strip()

        year = _extract_publication_year(
            pubmed_article
        )

        abstract_records.append(
            {
                "pmid": pmid or None,
                "title": title,
                "authors": authors,
                "journal": journal or None,
                "year": year,
                "abstract": abstract_text,
                "source": "PubMed abstract",
                "pubmed_url": (
                    f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    if pmid
                    else None
                ),
            }
        )

    return abstract_records