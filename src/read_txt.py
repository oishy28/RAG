import pickle

chunks_path = "embeddings/faiss_index/chunks.pkl"
search_term = "শিক্ষার ব্রত গ্রহণ করে"

with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)

matches = [(i, chunk) for i, chunk in enumerate(chunks) if search_term in chunk]

if matches:
    print(f"✅ Found {len(matches)} chunk(s) containing '{search_term}':\n")
    for i, chunk in matches:
        print(f"[Chunk {i}]\n{chunk.strip()}\n{'-'*60}\n")
else:
    print(f"❌ No chunks found containing '{search_term}'")
