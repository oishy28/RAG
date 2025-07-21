# src/retrieve_with_llama3.py

import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# Load FAISS index and chunks
def load_index(index_path, chunks_path):
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# Embed query to vector
def embed_query(query, model):
    return model.encode([query])

# Search top-k results from FAISS index
def search_index(query_vec, index, k=10):
    D, I = index.search(np.array(query_vec), k)
    return I[0]

# Retrieve top-k chunks

def retrieve_relevant_chunks(query, model, index, chunks, k=10):
    query_vec = embed_query(query, model)
    top_indices = search_index(query_vec, index, k)
    return [chunks[i] for i in top_indices]

# Build a prompt for LLaMA3

def build_prompt(query, retrieved_chunks):
    joined_chunks = "\n\n".join([f"[Chunk {i+1}] {chunk}" for i, chunk in enumerate(retrieved_chunks)])
    prompt = f"""
প্রশ্ন: {query}

তথ্যসূত্র:
{joined_chunks}

উত্তরটি এক লাইনে দাও এবং শুধু উপরের অংশের তথ্য ব্যবহার করো। নিজের ধারণা থেকে কিছু যোগ করো না।
"""
    return prompt.strip()

if __name__ == "__main__":
    index_path = "embeddings/faiss_index/index.faiss"
    chunks_path = "embeddings/faiss_index/chunks.pkl"
    model_name = "intfloat/multilingual-e5-base"

    print("\n🔄 Loading FAISS index and chunks...")
    model = SentenceTransformer(model_name)
    index, chunks = load_index(index_path, chunks_path)

    while True:
        query = input("\n❓ Enter your question (Bangla or English):\n> ")
        if query.strip().lower() in ["exit", "quit", "q"]:
            break

        retrieved = retrieve_relevant_chunks(query, model, index, chunks, k=10)

        print("\n🔍 Top Relevant Chunks:\n")
        for i, chunk in enumerate(retrieved, 1):
            print(f"[Chunk {i}] {chunk[:400].strip()}\n---")

        prompt = build_prompt(query, retrieved)
        response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])

        print("\n🤖 LLaMA3 Answer:")
        print(f"➡️ {response['message']['content'].strip()}\n")
