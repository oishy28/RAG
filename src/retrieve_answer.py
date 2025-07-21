# src/retrieve_answer.py

import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_index(index_path, chunks_path):
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def embed_query(query, model):
    return model.encode([query])

def search_index(query_vec, index, k=3):
    D, I = index.search(np.array(query_vec), k)
    return I[0]

def retrieve_relevant_chunks(query, model, index, chunks, k=10):
    query_vec = embed_query(query, model)
    top_indices = search_index(query_vec, index, k)
    return [chunks[i] for i in top_indices]

if __name__ == "__main__":
    # Load FAISS and chunks
    index_path = "embeddings/faiss_index/index.faiss"
    chunks_path = "embeddings/faiss_index/chunks.pkl"
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    print("🔄 Loading model and index...")
    model = SentenceTransformer(model_name)
    index, chunks = load_index(index_path, chunks_path)

    # Sample queries
    while True:
        query = input("\n❓ Enter your question (Bangla or English):\n> ")
        if query.strip().lower() in ["exit", "quit", "q"]:
            break
        results = retrieve_relevant_chunks(query, model, index, chunks, k=10)

        print("\n🔍 Top Relevant Chunks:\n")
        for i, chunk in enumerate(results, 1):
            print(f"{i}. {chunk}\n---")
