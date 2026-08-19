from Bio import Entrez

from .entrez_config import configure_entrez


def search_pubmed(
    term,
    max_results=100,
    email=None,
    api_key=None,
):
    """
    Search PubMed for articles containing a term.

    Parameters
    ----------
    term : str
        Psychological construct term.

    max_results : int
        Maximum number of articles to retrieve.

    email : str, optional
        NCBI email address. If not provided,
        NCBI_EMAIL may be read from the environment.

    api_key : str, optional
        NCBI API key. If not provided,
        NCBI_API_KEY may be read from the environment.

    Returns
    -------
    list
        PubMed IDs.
    """

    configure_entrez(
        email=email,
        api_key=api_key,
    )

    query = f'"{term}"[Title/Abstract]'

    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance",
    )

    record = Entrez.read(handle)
    handle.close()

    return record["IdList"]