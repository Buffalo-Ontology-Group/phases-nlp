from Bio import Entrez

from .entrez_config import configure_entrez


def get_pmc_ids(
    pmids,
    email=None,
    api_key=None,
):
    """
    Convert PubMed IDs to PMC IDs when full text is available.

    Parameters
    ----------
    pmids : list
        List of PubMed IDs.

    email : str, optional
        NCBI email address.

    api_key : str, optional
        NCBI API key.

    Returns
    -------
    dict
        Mapping of PMID to PMCID.
        Example: {"20652462": "PMC1234567"}
    """

    configure_entrez(
        email=email, 
        api_key=api_key,
    )

    if not pmids:
        return {}

    handle = Entrez.elink(
        dbfrom="pubmed",
        db="pmc",
        id=",".join(pmids),
        linkname="pubmed_pmc",
    )

    records = Entrez.read(handle)
    handle.close()

    pmid_to_pmcid = {}

    for record in records:
        pmid = record["IdList"][0]

        if "LinkSetDb" in record and record["LinkSetDb"]:
            links = record["LinkSetDb"][0]["Link"]

            if links:
                pmc_id_number = links[0]["Id"]
                pmcid = f"PMC{pmc_id_number}"
                pmid_to_pmcid[pmid] = pmcid

    return pmid_to_pmcid