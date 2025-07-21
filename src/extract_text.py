import pdfplumber
import re
import os

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            raw = page.extract_text()
            if raw:
                cleaned = clean_text(raw)
                text += cleaned + "\n\n"
            else:
                print(f"⚠️ No text extracted on page {page_num}")
    return text

def clean_text(text):
    # Fix broken Bangla ligatures, normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

if __name__ == "__main__":
    input_path = "data/HSC26-Bangla1st-Paper.pdf"
    output_path = "outputs/clean_bangla_corpus.txt"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    corpus = extract_text(input_path)

    # ✅ Debug Unicode characters:
    print("🔍 Debug: First 300 characters as UTF-8 bytes:\n")
    print(corpus[:300].encode('utf-8'))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    print("✅ PDF text extraction complete. Saved to:", output_path)
