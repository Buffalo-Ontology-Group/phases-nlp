import os
from docx import Document
from docx.text.run import Run
from dotenv import load_dotenv
from typing import List, Set

load_dotenv()

def is_highlighted(run: Run) -> bool:
    """
    Checks whether a given run of text in a Word document is highlighted.

    Args:
        run (Run): A segment of text from a Word document paragraph.

    Returns:
        bool: True if the run is highlighted, False otherwise.
    """
    return run.font.highlight_color is not None

def get_highlighted_text(doc_path: str) -> List[str]:
    """
    Extracts all highlighted text from a Word document.

    Args:
        doc_path (str): Path to the Word (.docx) document.

    Returns:
        List[str]: A list of highlighted text segments.
    """
    doc = Document(doc_path)
    highlighted_text = []

    for para in doc.paragraphs:
        for run in para.runs:
            if is_highlighted(run):
                highlighted_text.append(run.text)

    return highlighted_text

def save_highlighted_text_to_file(text: List[str], file_path: str) -> None:
    """
    Saves a list of text strings to a file, one per line.

    Args:
        text (List[str]): The list of text items to save.
        file_path (str): The path of the file to write to.
    """
    with open(file_path, 'w') as file:
        for item in text:
            file.write(item + '\n')

def get_common_highlighted_terms(highlighted_text_sme_B: List[str], highlighted_text_sme_H: List[str]) -> Set[str]:
    """
    Finds common highlighted terms between two lists.

    Args:
        highlighted_text_sme_B (List[str]): Highlighted text from SME_B.
        highlighted_text_sme_H (List[str]): Highlighted text from SME_H.

    Returns:
        Set[str]: A set of terms common to both inputs.
    """
    set_sme_B = set(highlighted_text_sme_B)
    set_sme_H = set(highlighted_text_sme_H)
    return set_sme_B.intersection(set_sme_H)

def seed_words() -> None:
    """
    Extracts highlighted terms from two Word documents (SME_B and SME_H),
    saves them to separate files, and also saves any common terms found
    between the two documents.
    """
    doc_path_sme_B = os.getenv('DOC_PATH_SME_B')
    doc_path_sme_H = os.getenv('DOC_PATH_SME_H')

    highlighted_text_sme_B = get_highlighted_text(doc_path_sme_B)
    highlighted_text_sme_H = get_highlighted_text(doc_path_sme_H)

    save_highlighted_text_to_file(highlighted_text_sme_B, os.getenv('HIGHLIGHTED_PATH_SME_B'))
    save_highlighted_text_to_file(highlighted_text_sme_H, os.getenv('HIGHLIGHTED_PATH_SME_H'))

    if highlighted_text_sme_B:
        print(f"Highlighted text from SME_B has been saved. Number of highlighted terms: {len(highlighted_text_sme_B)}")
    else:
        print("No highlighted text found in SME_B.")

    if highlighted_text_sme_H:
        print(f"Highlighted text from SME_H has been saved. Number of highlighted terms: {len(highlighted_text_sme_H)}")
    else:
        print("No highlighted text found in SME_H.")

    common_terms = get_common_highlighted_terms(highlighted_text_sme_B, highlighted_text_sme_H)

    if common_terms:
        save_highlighted_text_to_file(common_terms, os.getenv('COMMON_TERMS_FILE_PATH'))
        print(f"Common highlighted terms have been saved. Number of common terms: {len(common_terms)}")
    else:
        print("No common highlighted terms found.")

if __name__ == '__main__':
    seed_words()
