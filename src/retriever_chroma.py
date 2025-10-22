# src/retriever_chroma.py - UPDATED VERSION
import chromadb
import numpy as np
import json
import os

# Resolve project root so paths work regardless of CWD
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

# Candidate locations for embeddings (handles your src\data vs project\data situation)
EMB_CANDIDATES = [
    os.path.join(BASE_DIR, "data", "embeddings.npz"),
    os.path.join(BASE_DIR, "src", "data", "embeddings.npz"),
    os.path.join(BASE_DIR, "src", "embeddings.npz"),
    os.path.join(BASE_DIR, "embeddings.npz"),
]

def load_emb_npz(path=None):
    # try explicit path first, then candidates
    candidates = [path] if path else EMB_CANDIDATES
    checked = []
    for p in candidates:
        if not p:
            continue
        p = os.path.abspath(p)
        checked.append(p)
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            # expected arrays: "embeddings" and "ids"
            if "embeddings" not in d or "ids" not in d:
                raise ValueError(f"File {p} does not contain 'embeddings' and 'ids' arrays.")
            return d["embeddings"], d["ids"]
    raise FileNotFoundError(
        "Could not find embeddings.npz. Checked paths:\n" + "\n".join(checked) +
        "\n\nCreate embeddings.npz and place it in project data folder or pass the correct path to load_emb_npz(path=...)"
    )

def load_texts(path=None):
    path = path or os.path.join(BASE_DIR, "data", "rag_text_corpus.jsonl")
    if not os.path.exists(path):
        # also check src/data
        alt = os.path.join(BASE_DIR, "src", "data", "rag_text_corpus.jsonl")
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"Could not find rag_text_corpus.jsonl at {path} or {alt}")
    docs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            docs[j["id"]] = j["text"]
    return docs

def _decode_id(id_):
    # handle numpy bytes / numpy scalar / bytes / str
    if isinstance(id_, (bytes, bytearray)):
        return id_.decode()
    try:
        # numpy scalar with .item()
        val = id_.item() if hasattr(id_, "item") else id_
        if isinstance(val, (bytes, bytearray)):
            return val.decode()
        return str(val)
    except Exception:
        return str(id_)

def build_collection():
    embeddings, ids = load_emb_npz()
    docs = load_texts()
    
    # NEW ChromaDB client syntax (persistent)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Delete collection if exists
    try:
        client.delete_collection("rag_collection")
    except Exception:
        pass
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="rag_collection",
        metadata={"description": "RAG document collection"}
    )
    
    # Prepare data
    ids_list = [_decode_id(id_) for id_ in ids]
    texts = [docs.get(i, "") for i in ids_list]
    
    # Add to collection
    collection.add(
        documents=texts,
        metadatas=[{"source": i} for i in ids_list],
        ids=ids_list,
        embeddings=np.asarray(embeddings).tolist()
    )
    
    print("Built collection with", len(ids_list))

def query_collection(query_embedding, top_k=5):
    # Persistent client
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("rag_collection")
    
    # ensure query is list-like
    if isinstance(query_embedding, np.ndarray):
        q = query_embedding.tolist()
    else:
        q = query_embedding
    res = collection.query(
        query_embeddings=q,
        n_results=top_k
    )
    return res

if __name__ == "__main__":
    build_collection()