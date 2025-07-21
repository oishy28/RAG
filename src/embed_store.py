# # src/embed_store.py

# import os
# import pickle
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from chunk_text import load_text, chunk_text

# def embed_chunks(chunks, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
#     print("🔄 Loading embedding model...")
#     model = SentenceTransformer(model_name)
#     embeddings = model.encode(chunks, show_progress_bar=True)
#     return model, np.array(embeddings)

# def save_faiss_index(embeddings, chunks, index_path, chunks_path):
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)

#     os.makedirs(os.path.dirname(index_path), exist_ok=True)
#     faiss.write_index(index, index_path)

#     with open(chunks_path, "wb") as f:
#         pickle.dump(chunks, f)

#     print(f"✅ FAISS index saved to: {index_path}")
#     print(f"✅ Chunks saved to: {chunks_path}")

# if __name__ == "__main__":
#     input_path = "outputs/clean_bangla_corpus1.txt"
#     index_path = "embeddings/faiss_index/index.faiss"
#     chunks_path = "embeddings/faiss_index/chunks.pkl"

#     text = load_text(input_path)
#     chunks = chunk_text(text)

#     model, embeddings = embed_chunks(chunks)
#     save_faiss_index(embeddings, chunks, index_path, chunks_path)
# src/embed_store.py

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from chunk_text import load_text, chunk_text

def embed_chunks(chunks, model_name="intfloat/multilingual-e5-base"):
    print("🔄 Loading embedding model...")
    model = SentenceTransformer(model_name)

    # E5 model expects "passage: " prefix
    chunks_with_prefix = [f"passage: {chunk}" for chunk in chunks]
    embeddings = model.encode(chunks_with_prefix, show_progress_bar=True, normalize_embeddings=True)

    return model, np.array(embeddings)

def save_faiss_index(embeddings, chunks, index_path, chunks_path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ FAISS index saved to: {index_path}")
    print(f"✅ Chunks saved to: {chunks_path}")

if __name__ == "__main__":
    input_path = "outputs/clean_bangla_corpus1.txt"
    index_path = "embeddings/faiss_index/index.faiss"
    chunks_path = "embeddings/faiss_index/chunks.pkl"

    text = load_text(input_path)
    chunks = chunk_text(text)

    model, embeddings = embed_chunks(chunks)
    save_faiss_index(embeddings, chunks, index_path, chunks_path)
