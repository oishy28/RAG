import pdfplumber
import pytesseract
from PIL import Image
import os
import re

# Manually set Tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_entire_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            print(f"🔎 OCR processing page {i}...")
            image = page.to_image(resolution=300).original.convert("L")
            ocr_text = pytesseract.image_to_string(image, lang="ben")
            text += clean_text(ocr_text) + "\n\n"
    return text

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

if __name__ == "__main__":
    input_path = "data/HSC26-Bangla1st-Paper.pdf"
    output_path = "outputs/clean_bangla_corpus1.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    corpus = ocr_entire_pdf(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    print("✅ OCR extraction complete. Saved to:", output_path)
