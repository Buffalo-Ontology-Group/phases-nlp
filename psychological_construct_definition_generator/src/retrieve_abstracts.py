from Bio import Entrez

Entrez.email = "bdamayanthij@gmail.com"


def fetch_pubmed_abstracts(pmids):
    """
    Fetch PubMed abstracts for PMIDs.
    Used as fallback when PMC full text is unavailable or insufficient.
    """

    if not pmids:
        return ""

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="abstract",
        retmode="text"
    )

    abstracts = handle.read()
    handle.close()

    return abstracts
