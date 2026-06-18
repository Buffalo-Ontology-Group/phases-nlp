import re


def generate_definition(term, evidence_summary):
    if not evidence_summary or evidence_summary.startswith("No evidence"):
        return (
            f"{term.capitalize()} is a construct requiring expert review because "
            f"insufficient evidence was retrieved to generate a definition."
        )

    definition = re.sub(r'\s+', ' ', evidence_summary).strip()

    return definition
