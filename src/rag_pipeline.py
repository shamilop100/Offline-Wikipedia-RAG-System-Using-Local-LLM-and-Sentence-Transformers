# src/rag_pipeline.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from src.retriever_chroma import query_collection
from src.generate_with_phi3 import answer_question  # Changed from generate_with_ollama
import numpy as np

EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def embed_query(q):
    v = EMB_MODEL.encode([q], convert_to_numpy=True)
    return v  # shape (1, dim)

def retrieve_texts(query, top_k=4):
    qemb = embed_query(query)
    res = query_collection(qemb, top_k=top_k)
    # res['documents'] etc depends on your Chroma query output shape
    # Chroma returns dict with 'ids', 'documents' etc.
    docs = res.get("documents", [[]])[0]
    # if docs empty, fallback to res['metadatas']
    return docs

def answer(query, top_k=4):
    retrieved = retrieve_texts(query, top_k=top_k)
    # retrieved is a list of text passages (strings)
    resp = answer_question(query, retrieved)
    return {"answer": resp, "context": retrieved}

if __name__ == "__main__":
    q = "Who was Nikola Tesla working with at the National Telephone Company, and what project did they collaborate on??"
    out = answer(q)
    print("Answer:\n", out["answer"])
    print("\nContext used:\n", out["context"])
