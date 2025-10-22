# src/embed_and_store.py
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os
from tqdm import tqdm

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "rag_text_corpus.jsonl")
EMB_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.npz")
MODEL_NAME = "all-MiniLM-L6-v2"

def load_documents(path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

def main():
    model = SentenceTransformer(MODEL_NAME)
    docs = load_documents(DATA_FILE)
    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]

    embeddings = []
    batch_size = 64
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)

    np.savez_compressed(EMB_FILE, embeddings=embeddings, ids=np.array(ids))
    print("Saved embeddings to", EMB_FILE)

if __name__ == "__main__":
    main()
