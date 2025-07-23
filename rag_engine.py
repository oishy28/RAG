# rag_engine.py
import pickle
import faiss
import numpy as np
import re
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Normalize ----------
def normalize(text):
    return re.sub(r"[^\u0980-\u09FFa-zA-Z0-9]", "", text.lower())

def keyword_match_score(query, chunks):
    vectorizer = CountVectorizer().fit([query] + chunks)
    vectors = vectorizer.transform([query] + chunks)
    cosine_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return cosine_scores

def retrieve_relevant_chunks(query, model, index, chunks, k=20, hybrid_top_k=10):
    query_vec = model.encode([f"query: {query}"])
    D, I = index.search(np.array(query_vec), k)
    semantic_chunks = [(i, chunks[i]) for i in I[0]]
    pattern_scores = keyword_match_score(query, chunks)
    pattern_top_indices = np.argsort(pattern_scores)[::-1][:k]
    pattern_chunks = [(i, chunks[i]) for i in pattern_top_indices]
    query_norm = normalize(query)
    query_tokens = set(query_norm.split())
    phrase_boost = {}
    for i, chunk in enumerate(chunks):
        chunk_norm = normalize(chunk)
        score = sum(1 for token in query_tokens if token in chunk_norm)
        if query_norm in chunk_norm:
            score += 3
        if score > 0:
            phrase_boost[i] = score
    combined = {}
    for i, text in semantic_chunks:
        combined[i] = (text, 1.0)
    for i, text in pattern_chunks:
        combined[i] = (text, combined.get(i, (text, 0))[1] + 1.0)
    for i, score in phrase_boost.items():
        combined[i] = (chunks[i], combined.get(i, (chunks[i], 0))[1] + score)
    ranked = sorted(combined.items(), key=lambda x: x[1][1], reverse=True)
    return [text for _, (text, _) in ranked[:hybrid_top_k]]

def build_prompt(query, top_chunks):
    return f"""
### Source Chunks (may include MCQs or structured text):
{''.join(f"[Chunk {i+1}] {chunk.strip()}\n" for i, chunk in enumerate(top_chunks))}

### Question:
{query}

### Instructions:
- Analyze the source chunks and provide a concise answer from them.
- Respond **in the same language** as the question.
- Be **short and accurate**.
- If the answer is found as part of an MCQ or numbered entry, use it directly.
- If nothing relevant is found, give a precise answer based on your analysis.
""".strip()

def answer_with_mistral(prompt):
    response = ollama.chat(model='mistral', messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# Load once at the top level
print("🔄 Loading index and model...")
index = faiss.read_index("embeddings/faiss_index/index.faiss")
with open("embeddings/faiss_index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
model = SentenceTransformer("intfloat/multilingual-e5-base")

# Final function to use in API
def get_rag_answer(query):
    top_chunks = retrieve_relevant_chunks(query, model, index, chunks)
    prompt = build_prompt(query, top_chunks)
    answer = answer_with_mistral(prompt)
    return {
        "answer": answer,
        "top_chunks": top_chunks
    }
