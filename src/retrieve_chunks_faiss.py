

import pickle
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------- Helpers ----------
def normalize(text):
    return re.sub(r"[^\u0980-\u09FFa-zA-Z0-9]", "", text.lower())

def keyword_match_score(query, chunks):
    vectorizer = CountVectorizer().fit([query] + chunks)
    vectors = vectorizer.transform([query] + chunks)
    cosine_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return cosine_scores

def retrieve_relevant_chunks(query, model, index, chunks, k=20, hybrid_top_k=10):
    # Step 1: Semantic vector search
    query_vec = model.encode([f"query: {query}"])
    D, I = index.search(np.array(query_vec), k)
    semantic_chunks = [(i, chunks[i]) for i in I[0]]

    # Step 2: Keyword cosine similarity
    pattern_scores = keyword_match_score(query, chunks)
    pattern_top_indices = np.argsort(pattern_scores)[::-1][:k]
    pattern_chunks = [(i, chunks[i]) for i in pattern_top_indices]

    # Step 3: Token + phrase boost (no hardcoded terms)
    query_norm = normalize(query)
    query_tokens = set(query_norm.split())
    phrase_boost = {}

    for i, chunk in enumerate(chunks):
        chunk_norm = normalize(chunk)
        score = sum(1 for token in query_tokens if token in chunk_norm)
        if query_norm in chunk_norm:
            score += 3  # full string bonus
        if score > 0:
            phrase_boost[i] = score

    # Step 4: Merge with boosting
    combined = {}
    for i, text in semantic_chunks:
        combined[i] = (text, 1.0)
    for i, text in pattern_chunks:
        combined[i] = (text, combined.get(i, (text, 0))[1] + 1.0)
    for i, score in phrase_boost.items():
        combined[i] = (chunks[i], combined.get(i, (chunks[i], 0))[1] + score)

    # Sort and return
    ranked = sorted(combined.items(), key=lambda x: x[1][1], reverse=True)
    return [text for _, (text, _) in ranked[:hybrid_top_k]]


# --------- Main for CLI Testing ----------
if __name__ == "__main__":
    index_path = "embeddings/faiss_index/index.faiss"
    chunks_path = "embeddings/faiss_index/chunks.pkl"
    model_name = "intfloat/multilingual-e5-base"

    print("🔄 Loading FAISS index and chunks...")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index(index_path)
    model = SentenceTransformer(model_name)

    while True:
        query = input("\n❓ Enter your question (Bangla or English):\n> ")
        if query.strip().lower() in ["exit", "quit", "q"]:
            break
        top_chunks = retrieve_relevant_chunks(query, model, index, chunks)
        print("\n🔍 Top Relevant Chunks:\n")
        for i, chunk in enumerate(top_chunks, 1):
            print(f"[Chunk {i}] {chunk.strip()}\n---")
