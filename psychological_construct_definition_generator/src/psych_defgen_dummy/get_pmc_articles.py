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
        Example: {"20652462": "PMC3874845"}
    """

    configure_entrez(
        email=email,
        api_key=api_key,
    )

    if not pmids:
        return {}

    pmid_to_pmcid = {}

    for pmid in pmids:

        handle = Entrez.elink(
            dbfrom="pubmed",
            db="pmc",
            id=str(pmid),
            linkname="pubmed_pmc",
        )

        records = Entrez.read(handle)
        handle.close()

        if not records:
            continue

        record = records[0]

        link_sets = record.get(
            "LinkSetDb",
            [],
        )

        if not link_sets:
            continue

        links = link_sets[0].get(
            "Link",
            [],
        )

        if not links:
            continue

        pmc_id_number = links[0]["Id"]

        pmcid = f"PMC{pmc_id_number}"

        pmid_to_pmcid[str(pmid)] = pmcid

    return pmid_to_pmcid