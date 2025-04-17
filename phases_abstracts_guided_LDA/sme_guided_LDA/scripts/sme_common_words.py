import os
from docx import Document
from dotenv import load_dotenv

load_dotenv()

def is_highlighted(run):
    return run.font.highlight_color is not None

def get_highlighted_text(doc_path):
    doc = Document(doc_path)
    highlighted_text = []

    for para in doc.paragraphs:
        for run in para.runs:
            if is_highlighted(run):
                highlighted_text.append(run.text)

    return highlighted_text

def save_highlighted_text_to_file(text, file_path):
    with open(file_path, 'w') as file:
        for item in text:
            file.write(item + '\n')

def get_common_highlighted_terms(highlighted_text_sme_B, highlighted_text_sme_H):
    set_sme_B = set(highlighted_text_sme_B)
    set_sme_H = set(highlighted_text_sme_H)
    return set_sme_B.intersection(set_sme_H)

# ✅ This is now the function you can import and call directly
def seed_words():
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

# Optional: allow this file to run standalone too
if __name__ == '__main__':
    seed_words()
