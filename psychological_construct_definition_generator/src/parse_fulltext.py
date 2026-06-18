from Bio import Entrez
from bs4 import BeautifulSoup

Entrez.email = "bdamayanthij@gmail.com"


def fetch_pmc_full_text(pmcid):
    handle = Entrez.efetch(
        db="pmc",
        id=pmcid.replace("PMC", ""),
        rettype="full",
        retmode="xml"
    )

    xml_data = handle.read()
    handle.close()

    return xml_data


def parse_pmc_xml(xml_data):
    soup = BeautifulSoup(xml_data, "xml")

    title_tag = soup.find("article-title")
    title = title_tag.get_text(" ", strip=True) if title_tag else "No title found"

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

    # fallback if <body> is absent or empty
    article_text = soup.get_text(" ", strip=True)

    return title, article_text


def get_full_text_from_pmcid(pmcid):
    xml_data = fetch_pmc_full_text(pmcid)
    title, text = parse_pmc_xml(xml_data)

    return title, text
