# src/chunk_text.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def save_chunks(chunks, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"--- Chunk {i} ---\n")
            f.write(chunk.strip() + "\n\n")

if __name__ == "__main__":
    input_path = "outputs/clean_bangla_corpus1.txt"
    output_path = "outputs/chunks_bangla.txt"

    text = load_text(input_path)
    chunks = chunk_text(text)

    print(f"✅ Total chunks created: {len(chunks)}")
    save_chunks(chunks, output_path)
    print(f"💾 Chunks saved to: {output_path}")
