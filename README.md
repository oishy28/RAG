# 📚 Multilingual RAG System – AI Engineer (Level-1) Assessment

## 🔍 Objective

This project is a **Retrieval-Augmented Generation (RAG)** system designed to answer **Bangla and English** questions from a Bangla literature book (HSC26 Bangla 1st Paper PDF). The system retrieves the most relevant document chunks and generates grounded answers using an LLM.

---

## ✅ Features

- Accepts **English and Bangla** queries
- Retrieves semantically and lexically relevant chunks from the document
- Uses **hybrid retrieval**: FAISS + keyword matching
- Generates concise answers using **LLaMA 3 via Ollama** or any LLM (e.g., GPT)
- Clearly highlights **source chunks**
- Lightweight and CLI-based (optional REST API bonus)

---

## ⚙️ Tools and Libraries Used

| Category              | Tool/Library                            |
|-----------------------|-----------------------------------------|
| Text Extraction       | `PyMuPDF`, `Tesseract` (for Bangla OCR) |
| Chunking              | `langchain.text_splitter`               |
| Embedding             | `intfloat/multilingual-e5-base` (via `sentence-transformers`) |
| Vector Store          | `FAISS`                                 |
| LLM                   | `LLaMA3` via `Ollama` OR `Mistral`/`GPT-4` |
| Retrieval Logic       | `numpy`, `scikit-learn` for hybrid search |
| Interface             | Command Line (CLI), optional REST API   |

---

## 🛠️ Setup Guide

1. **Install dependencies**:
   ```bash
   pip install faiss-cpu sentence-transformers scikit-learn ollama numpy
   ```

2. **Install Ollama** (for LLaMA/Mistral):
   [https://ollama.com/download](https://ollama.com/download)

3. **Run the following steps in order**:

### 1. Preprocess & Chunk

```bash
python src/chunk_text.py
```

### 2. Embed and Build Index

```bash
python src/embed_store.py
```

### 3. Query System

```bash
python src/retrieve_with_llama3.py  # or retrieve_with_mistral.py or retrieve_with_gpt.py
```

---

## ▶️ How to Run the Server

1. **Make sure your FAISS index and model are ready.**

2. **Start the FastAPI server**:

```bash
uvicorn main:app --reload
```

3. **Access the API docs in your browser**:

```
http://localhost:8000/docs
```

You can now test the `/query` endpoint with Bangla or English questions.


## 🧪 Sample Queries & Answers

| Query (Bangla)                                   | Expected Answer |
|--------------------------------------------------|-----------------|
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?         | শব্তুনাথ         |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামাকে           |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?        | ১৫ বছর          |

---

## 🧠 Q&A (as per submission guideline)

### 1. **What method or library did you use to extract the text, and why?**
Used `PyMuPDF` and `Tesseract` for Bangla OCR. `PyMuPDF` was selected for its accurate layout preservation. Challenges included line breaks, noise, and MCQ formatting, which were cleaned manually.

### 2. **What chunking strategy did you choose?**
Character-based chunking (via `RecursiveCharacterTextSplitter`, chunk size = 500, overlap = 50). It ensures semantically coherent chunks while avoiding long sequences.

### 3. **What embedding model did you use?**
`intfloat/multilingual-e5-base`. Chosen for its multilingual understanding and compatibility with FAISS. It captures semantic similarity better for Bangla queries than `MiniLM`.

### 4. **How are you comparing the query with stored chunks?**
Used a hybrid of:
- FAISS semantic vector search
- Cosine similarity over bag-of-words
- Token/phrase boosting

This balances contextual understanding with lexical matching for better retrieval.

### 5. **How do you ensure meaningful chunk-query comparison?**
- Queries are normalized before similarity scoring
- Embeddings include `"query: ..."` prefix for semantic alignment
- Hybrid logic helps if context is missing or vague

### 6. **Do the results seem relevant?**
Yes, but minor errors occur when chunks lack structure or are misaligned. Improvements include:
- Better OCR cleaning
- Sentence-level chunking
- Using larger LLMs (e.g., GPT-4 via API)

---

## 🚀 Bonus (Optional)

### API (Bonus)
Not implemented yet. Would include:
```bash
POST /query
{
  "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}
```

### Evaluation (Bonus)
Implemented:
- Cosine similarity + manual verification for chunk grounding
- Human evaluation confirms improved relevance after hybrid search

---

## 📂 File Structure

```
src/
│
├── chunk_text.py             # Loads and chunks PDF
├── embed_store.py            # Embeds and stores vectors in FAISS
├── retrieve_with_llama3.py   # Main CLI using Ollama + LLaMA
├── retrieve_with_mistral.py  # Alternate model
├── retrieve_with_gpt.py      # (Optional) GPT API
├── utils/                    # Contains text cleaning, OCR tools
embeddings/
outputs/
```

---

## 📌 Exit Commands

To quit CLI:
```bash
exit
quit
q
```

---

> 🎯 Good luck! Make sure to push your code to a **GitHub Public Repo** and attach this `README.md`.
