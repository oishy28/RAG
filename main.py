# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import get_rag_answer

app = FastAPI(title="Multilingual RAG API by O.Akhand")

class QueryInput(BaseModel):
    query: str

@app.post("/query")
async def query_rag(input: QueryInput):
    result = get_rag_answer(input.query)
    return {
        "query": input.query,
        "answer": result["answer"],
        "top_chunks": result["top_chunks"]
    }
