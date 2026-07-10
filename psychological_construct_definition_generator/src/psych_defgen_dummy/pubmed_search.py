from Bio import Entrez

Entrez.email = "your_email@example.com"


def search_pubmed(term, max_results=20):
    """
    Search PubMed for articles containing a term.

    Parameters
    ----------
    term : str
        Psychological construct term.

    max_results : int
        Number of articles to retrieve.

    Returns
    -------
    list
        PubMed IDs.
    """

    query = f'"{term}"[Title/Abstract]'

    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance"
    )

    record = Entrez.read(handle)
    handle.close()

    return record["IdList"]
