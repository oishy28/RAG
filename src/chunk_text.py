# src/chunk_text.py

from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

if __name__ == "__main__":
    input_path = "outputs/clean_bangla_corpus.txt"
    text = load_text(input_path)

    chunks = chunk_text(text)
    print(f"✅ Total chunks created: {len(chunks)}")
    print("\n🔹 First chunk preview:\n")
    print(chunks[0])
