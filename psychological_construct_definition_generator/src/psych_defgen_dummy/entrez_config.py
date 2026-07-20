import os

from Bio import Entrez


def configure_entrez() -> None:
    """Configure Biopython Entrez using the user's NCBI email."""

    email = os.getenv("NCBI_EMAIL")

    if not email:
        raise RuntimeError(
            "NCBI_EMAIL environment variable is not set. "
            "Set it before using PubMed or PMC retrieval."
        )

    Entrez.email = email
