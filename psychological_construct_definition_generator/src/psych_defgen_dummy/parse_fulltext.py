from Bio import Entrez
from bs4 import BeautifulSoup

from .entrez_config import configure_entrez


def fetch_pmc_full_text(pmcid):
    """Retrieve a PMC article as XML."""
    configure_entrez()

    handle = Entrez.efetch(
        db="pmc",
        id=pmcid.replace("PMC", ""),
        rettype="full",
        retmode="xml",
    )

    xml_data = handle.read()
    handle.close()

    return xml_data


def parse_pmc_xml(xml_data):
    """
    Parse PMC XML and return the article title and full text.

    Parameters
    ----------
    xml_data : str or bytes
        PMC article XML.

    Returns
    -------
    tuple
        Article title and extracted article text.
    """
    soup = BeautifulSoup(xml_data, "xml")

    title_tag = soup.find("article-title")
    title = (
        title_tag.get_text(" ", strip=True)
        if title_tag
        else "No title found"
    )

    body = soup.find("body")

    if body:
        text_parts = []

        for tag in body.find_all(["title", "p"]):
            text = tag.get_text(" ", strip=True)

            if text:
                text_parts.append(text)

        body_text = "\n".join(text_parts)

        if body_text.strip():
            return title, body_text

    # Fallback if the <body> element is absent or empty.
    article_text = soup.get_text(" ", strip=True)

    return title, article_text


def get_full_text_from_pmcid(pmcid):
    """
    Retrieve and parse the full text of a PMC article.

    Parameters
    ----------
    pmcid : str
        PubMed Central identifier, such as ``PMC1234567``.

    Returns
    -------
    tuple
        Article title and extracted article text.
    """
    xml_data = fetch_pmc_full_text(pmcid)
    title, text = parse_pmc_xml(xml_data)

    return title, text
