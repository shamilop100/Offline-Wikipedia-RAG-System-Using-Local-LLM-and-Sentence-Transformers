# src/build_index.py
from datasets import load_dataset
from tqdm import tqdm
import json
import re
import os

DATA_PATH = "data/rag_text_corpus.jsonl"

def normalize_text(s):
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def chunk_text(text, max_chars=1000):
    # simple whitespace chunker — replace with better token-based chunker if needed
    words = text.split()
    chunks=[]
    cur=[]
    cur_len=0
    for w in words:
        if cur_len + len(w) + 1 > max_chars:
            chunks.append(" ".join(cur))
            cur=[]
            cur_len=0
        cur.append(w)
        cur_len += len(w)+1
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def main():
    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")  # this loads the split
    # inspect
    print(ds)

    # choose split: prefer 'train' if present, otherwise use the first available split
    split_name = "train" if "train" in ds.keys() else next(iter(ds.keys()))
    print("Using split:", split_name)
    dataset = ds[split_name]

    # ensure output directory exists
    dirpath = os.path.dirname(DATA_PATH)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    out_fp = open(DATA_PATH, "w", encoding="utf-8")
    seen_texts = set()
    for idx, item in enumerate(tqdm(dataset)):
        # fields vary — try several common names
        text = item.get("text") or item.get("body") or item.get("passage") or item.get("document")
        uid = item.get("id") or item.get("document_id") or item.get("passage_id") or item.get("idx")
        if text is None:
            continue
        text = normalize_text(text)
        # chunk if long:
        for i, chunk in enumerate(chunk_text(text, max_chars=800)):
            if chunk in seen_texts:
                continue
            seen_texts.add(chunk)
            out_id = f"{uid}_{i}" if uid is not None else f"auto_{idx}_{i}"
            out = {"id": out_id, "text": chunk}
            out_fp.write(json.dumps(out, ensure_ascii=False) + "\n")
    out_fp.close()
    print("Saved to", DATA_PATH)

if __name__ == "__main__":
    main()
