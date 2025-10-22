🧠 Offline Wikipedia RAG System Using Local LLM (Phi-3) and Sentence Transformers
<p align="center"> <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" /> <img src="https://img.shields.io/badge/Ollama-Phi--3-green?logo=ollama" /> <img src="https://img.shields.io/badge/Sentence--Transformers-all--mpnet--base--v2-orange" /> <img src="https://img.shields.io/badge/License-MIT-lightgrey" /> <img src="https://img.shields.io/badge/Platform-VS%20Code-blueviolet" /> </p>

This project implements a Retrieval-Augmented Generation (RAG) pipeline that runs entirely offline using Ollama and Phi-3 as the local Large Language Model (LLM).
It retrieves context-rich information from Wikipedia, performs semantic search using Sentence Transformers, and generates factual, context-grounded answers via Phi-3.

🚀 Key Features

🔍 Semantic Retrieval: Embedding-based Wikipedia search using sentence-transformers.

🧩 Local LLM Generation: Uses Phi-3 through Ollama, ensuring privacy and offline capability.

🧠 Context-Aware Reasoning: Answers grounded in retrieved factual context.

⚙️ Modular Design: Easily swap out embedding models or LLMs.

💻 Runs Locally in Visual Studio Code — no cloud dependencies.

🧰 Tech Stack
Component	Tool / Library
Embeddings	sentence-transformers/all-mpnet-base-v2
Vector Store	chromadb
LLM Backend	Phi-3 via Ollama
Data Source	Wikipedia API
Environment	Python ≥ 3.10, Visual Studio Code
🗂️ Project Structure
rag_project/
│
├── data/                        # Optional cache or storage
├── src/
│   ├── build_index.py           # Build embeddings index from Wikipedia
│   ├── retrieve_and_generate.py # Retrieve context + generate answers
│   └── utils.py                 # Helper functions
│
├── requirements.txt
├── .gitignore
└── README.md

⚙️ Installation & Setup
🔸 1. Clone the Repository
git clone https://github.com/shamilop100/Offline-Wikipedia-RAG-System-Using-Local-LLM-and-Sentence-Transformers.git
cd Offline-Wikipedia-RAG-System-Using-Local-LLM-and-Sentence-Transformers

🔸 2. Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate     # On Windows
# source venv/bin/activate  # On Mac/Linux

🔸 3. Install Dependencies
pip install -r requirements.txt

🔸 4. Install and Run Ollama

Download Ollama from: https://ollama.ai

Then pull the Phi-3 model:

ollama pull phi3

📦 Code Overview
🔹 1. Build the Wikipedia Vector Index
# src/build_index.py
from sentence_transformers import SentenceTransformer
import chromadb
import wikipedia

def normalize_text(s):
    return ' '.join(s.strip().split())

def fetch_wikipedia_articles(topics, max_chars=2000):
    docs = []
    for topic in topics:
        try:
            content = wikipedia.page(topic).content[:max_chars]
            docs.append({"title": topic, "text": normalize_text(content)})
        except Exception:
            continue
    return docs

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("wiki_collection")

topics = ["Artificial intelligence", "Machine learning", "Neural networks", "Data science"]
docs = fetch_wikipedia_articles(topics)

for i, d in enumerate(docs):
    emb = model.encode(d["text"]).tolist()
    collection.add(
        ids=[str(i)],
        embeddings=[emb],
        metadatas=[{"title": d["title"]}],
        documents=[d["text"]]
    )

print("✅ Wikipedia index built successfully!")

🔹 2. Retrieve Context and Generate Answers with Phi-3 via Ollama
# src/retrieve_and_generate.py
from sentence_transformers import SentenceTransformer
import chromadb
import subprocess
import json

# Load retriever and vector DB
retriever = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_collection("wiki_collection")

def query_ollama_phi3(prompt):
    """Run a local Phi-3 generation through Ollama"""
    command = ["ollama", "run", "phi3", "--prompt", prompt]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip()

def rag_query(question, top_k=3):
    query_emb = retriever.encode(question).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    context = " ".join(results["documents"][0])
    prompt = f"Answer the following question using the given context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    return query_ollama_phi3(prompt)

print(rag_query("What is the role of neural networks in machine learning?"))

⚙️ Configurable Parameters
Parameter	Description	Default
Embedding Model	Sentence-transformers model for semantic search	all-mpnet-base-v2
LLM Backend	Local Phi-3 model via Ollama	phi3
Retrieval Count (top_k)	Number of retrieved chunks	3
Context Length (max_chars)	Truncate article text length	2000
🧩 How It Works

Retrieve: Encodes user query → retrieves top Wikipedia passages from ChromaDB.

Augment: Appends retrieved context to the query.

Generate: Sends the combined prompt to Phi-3 (via Ollama) for grounded answer generation.

Return: Produces human-like responses with factual context from Wikipedia.

🔮 Future Enhancements

🧮 Integrate hybrid retrieval (dense + sparse).

📊 Add answer evaluation metrics (BLEU, F1).

💬 Build a Streamlit UI for interactive chat.

⚡ Enable multi-model inference with Ollama (Phi-3 + Mistral).

🔗 Local caching of Wikipedia articles for fully offline operation.

📜 License

MIT License © 2025 Shamil Op

🤝 Acknowledgements

Ollama

Sentence Transformers

ChromaDB

Wikipedia API

Phi-3 Model (Microsoft)

⭐ If you found this project useful, please consider giving it a star on GitHub!
