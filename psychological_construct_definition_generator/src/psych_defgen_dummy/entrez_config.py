import os

from Bio import Entrez


def configure_entrez(
    email=None,
    api_key=None,
) -> None:
    """
    Configure Biopython Entrez.

    Explicit parameters take priority over
    environment variables.
    """

    resolved_email = (
        email or os.getenv("NCBI_EMAIL")
    )

    resolved_api_key = (
        api_key or os.getenv("NCBI_API_KEY")
    )

    if not resolved_email:
        raise RuntimeError(
            "An NCBI email address is required. "
            "Provide email=... or set the "
            "NCBI_EMAIL environment variable."
        )

    Entrez.email = resolved_email

    if resolved_api_key:
        Entrez.api_key = resolved_api_key