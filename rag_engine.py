# # rag_engine.py
# import pickle
# import faiss
# import numpy as np
# import re
# import ollama
# from sentence_transformers import SentenceTransformer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # ---------- Normalize ----------
# def normalize(text):
#     return re.sub(r"[^\u0980-\u09FFa-zA-Z0-9]", "", text.lower())

# def keyword_match_score(query, chunks):
#     vectorizer = CountVectorizer().fit([query] + chunks)
#     vectors = vectorizer.transform([query] + chunks)
#     cosine_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
#     return cosine_scores

# def retrieve_relevant_chunks(query, model, index, chunks, k=20, hybrid_top_k=10):
#     query_vec = model.encode([f"query: {query}"])
#     D, I = index.search(np.array(query_vec), k)
#     semantic_chunks = [(i, chunks[i]) for i in I[0]]
#     pattern_scores = keyword_match_score(query, chunks)
#     pattern_top_indices = np.argsort(pattern_scores)[::-1][:k]
#     pattern_chunks = [(i, chunks[i]) for i in pattern_top_indices]
#     query_norm = normalize(query)
#     query_tokens = set(query_norm.split())
#     phrase_boost = {}
#     for i, chunk in enumerate(chunks):
#         chunk_norm = normalize(chunk)
#         score = sum(1 for token in query_tokens if token in chunk_norm)
#         if query_norm in chunk_norm:
#             score += 3
#         if score > 0:
#             phrase_boost[i] = score
#     combined = {}
#     for i, text in semantic_chunks:
#         combined[i] = (text, 1.0)
#     for i, text in pattern_chunks:
#         combined[i] = (text, combined.get(i, (text, 0))[1] + 1.0)
#     for i, score in phrase_boost.items():
#         combined[i] = (chunks[i], combined.get(i, (chunks[i], 0))[1] + score)
#     ranked = sorted(combined.items(), key=lambda x: x[1][1], reverse=True)
#     return [text for _, (text, _) in ranked[:hybrid_top_k]]

# def build_prompt(query, top_chunks):
#     return f"""
# ### Source Chunks (may include MCQs or structured text):
# {''.join(f"[Chunk {i+1}] {chunk.strip()}\n" for i, chunk in enumerate(top_chunks))}

# ### Question:
# {query}

# ### Instructions:
# - Read the bangla source chunks and provide a concise answer to the queston from them.
# - Respond **in the same language** as the question.
# - Be **short and accurate**.
# - If the answer is found as part of an MCQ or numbered entry, use it directly in one word if possible.
# - If nothing relevant is found, give a precise answer based on your analysis.
# """.strip()

# def answer_with_mistral(prompt):
#     response = ollama.chat(model='mistral', messages=[{"role": "user", "content": prompt}])
#     # response = ollama.chat(model='gemma:2b', messages=[{"role": "user", "content": prompt}])
#     return response['message']['content']

# # Load once at the top level
# print("🔄 Loading index and model...")
# index = faiss.read_index("embeddings/faiss_index/index.faiss")
# with open("embeddings/faiss_index/chunks.pkl", "rb") as f:
#     chunks = pickle.load(f)
# model = SentenceTransformer("intfloat/multilingual-e5-base")

# # Final function to use in API
# def get_rag_answer(query):
#     top_chunks = retrieve_relevant_chunks(query, model, index, chunks)
#     prompt = build_prompt(query, top_chunks)
#     answer = answer_with_mistral(prompt)
#     return {
#         "answer": answer,
#         "top_chunks": top_chunks
#     }
# rag_engine.py
import pickle
import faiss
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Load RAG components ---------
print("🔄 Loading index and models...")
index = faiss.read_index("embeddings/faiss_index/index.faiss")
with open("embeddings/faiss_index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

embed_model = SentenceTransformer("intfloat/multilingual-e5-base")

# Load your custom LLM
tokenizer = AutoTokenizer.from_pretrained("hassanaliemon/bn_rag_llama3-8b")
# model = AutoModelForCausalLM.from_pretrained(
#     "hassanaliemon/bn_rag_llama3-8b",
#     load_in_8bit=True,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
model = AutoModelForCausalLM.from_pretrained(
    "hassanaliemon/bn_rag_llama3-8b",
    torch_dtype=torch.float32,  # CPU float32
    device_map={"": "cpu"},
)


# --------- Preprocessing & retrieval ---------
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

# --------- Prompt construction ---------
def build_prompt(question, top_chunks):
    context = "\n".join(f"[{i+1}] {chunk.strip()}" for i, chunk in enumerate(top_chunks))
    return f"""এখানে একটি নির্দেশনা দেওয়া হলো, যা একটি কাজ সম্পন্ন করার উপায় বর্ণনা করে, এবং এর সাথে একটি ইনপুট দেওয়া হলো যা আরও প্রেক্ষাপট প্রদান করে। একটি উত্তর লিখুন যা অনুরোধটি সঠিকভাবে পূরণ করে। প্রসঙ্গ থেকে সুনির্দিষ্ট উত্তর দিন.

### Instruction:
{question}

### Input:
{context}

### Response:
"""

# --------- LLM Answer generation ---------
def answer_with_custom_llm(prompt):
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract response after "### Response:"
    return response.split("### Response:")[-1].strip()

# --------- API-ready function ---------
def get_rag_answer(query):
    top_chunks = retrieve_relevant_chunks(query, embed_model, index, chunks)
    prompt = build_prompt(query, top_chunks)
    answer = answer_with_custom_llm(prompt)
    return {
        "answer": answer,
        "top_chunks": top_chunks
    }
