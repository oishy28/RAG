# src/retrieve_with_mistral.py

import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# --------- Load FAISS and Chunks ----------
def load_index(index_path, chunks_path):
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# --------- Embed Query Using E5 Model ---------
def embed_query(query, model):
    # E5 expects query prefix
    query_with_prefix = f"query: {query}"
    return model.encode([query_with_prefix])

# --------- Retrieve Top-k Relevant Chunks ---------
def search_index(query_vec, index, k=5):
    D, I = index.search(np.array(query_vec), k)
    return I[0]

def retrieve_relevant_chunks(query, model, index, chunks, k=10):
    query_vec = embed_query(query, model)
    top_indices = search_index(query_vec, index, k)
    return [chunks[i] for i in top_indices]

# --------- Detect Language (Simple Heuristic) ---------
def detect_language(text):
    # Simple check: if contains Bangla Unicode block
    return "bn" if any('\u0980' <= c <= '\u09FF' for c in text) else "en"

# --------- Use Mistral via Ollama ---------
def generate_answer_with_mistral(query, relevant_chunks, language="bn"):
    context = "\n\n".join(f"[Chunk {i+1}] {chunk.strip()}" for i, chunk in enumerate(relevant_chunks))
    
    prompt = f"""
You are a helpful assistant answering based strictly on the retrieved document chunks.

QUESTION ({'Bangla' if language == 'bn' else 'English'}):
{query}

DOCUMENT CHUNKS:
{context}

INSTRUCTIONS:
- Use ONLY the information from the chunks above.
- If the answer is NOT found in the chunks, reply:
  - In Bangla: 'উত্তরের জন্য প্রাসঙ্গিক তথ্য পাওয়া যায়নি।'
  - In English: 'The answer is not found in the document.'
- Answer in the same language as the question in one line.
- DO NOT invent or add anything beyond the chunks.

ANSWER:
""".strip()

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# --------- Main CLI ---------
if __name__ == "__main__":
    index_path = "embeddings/faiss_index/index.faiss"
    chunks_path = "embeddings/faiss_index/chunks.pkl"
    model_name = "intfloat/multilingual-e5-base"

    print("🔄 Loading FAISS index and chunks...")
    index, chunks = load_index(index_path, chunks_path)
    model = SentenceTransformer(model_name)

    while True:
        query = input("\n❓ Enter your question (Bangla or English):\n> ")
        if query.strip().lower() in ["exit", "quit", "q"]:
            break

        language = detect_language(query)
        relevant_chunks = retrieve_relevant_chunks(query, model, index, chunks, k=5)

        print("\n🔍 Top Relevant Chunks:\n")
        for i, chunk in enumerate(relevant_chunks, 1):
            print(f"[Chunk {i}] {chunk.strip()}\n---")

        answer = generate_answer_with_mistral(query, relevant_chunks, language)
        print(f"\n🤖 Mistral Answer:\n➡️ {answer}")
