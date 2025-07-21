# src/extract_text.py

import pdfplumber
import re

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw = page.extract_text()
            if raw:
                text += clean_text(raw) + "\n\n"
    return text

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

if __name__ == "__main__":
    input_path = "data/HSC26-Bangla1st-Paper.pdf"
    output_path = "outputs/clean_bangla_corpus.txt"

    corpus = extract_text(input_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    print("✅ PDF text extracted and saved to outputs/clean_bangla_corpus.txt")
