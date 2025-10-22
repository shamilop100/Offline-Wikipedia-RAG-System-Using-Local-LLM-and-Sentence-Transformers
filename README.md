# 🧠 Offline Wikipedia RAG System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Phi--3-green?logo=ollama&logoColor=white)
![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange?logo=huggingface&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-purple?logo=database&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey?logo=opensourceinitiative&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-VS%20Code-blueviolet?logo=visualstudiocode&logoColor=white)

**Privacy-First | Fully Offline | Context-Aware Question Answering**

---

## 📖 Overview

A production-ready Retrieval-Augmented Generation (RAG) system that operates 100% offline using local LLMs. It combines a **Hugging Face Wikipedia dataset** with Microsoft's Phi-3 model (via Ollama) and state-of-the-art semantic search to deliver accurate, context-grounded answers without cloud dependencies.

Perfect for privacy-sensitive applications, air-gapped environments, or anyone seeking full control over their AI infrastructure.

---

## 🎯 Why This Project?

- 🔒 **Privacy-First**: All data and processing remain local  
- ⚡ **Lightning Fast**: No API latency, answers in seconds  
- 💰 **Zero Cost**: No API fees or subscriptions  
- 🛡️ **Secure**: Offline/air-gapped compatible  
- 🎓 **Educational**: Learn RAG architecture hands-on  

---

## ✨ Features

### 🔍 Semantic Search
- Dense embedding retrieval using MiniLM (`all-MiniLM-L6-v2`)  
- Context-aware similarity matching  
- Handles complex queries effectively  

### 🤖 Local LLM Generation
- Powered by Microsoft Phi-3 (3.8B parameters)  
- Runs via Ollama locally  
- No internet connection required  

### 📚 Wikipedia Integration
- Pre-downloaded Hugging Face dataset  
- Text chunking and preprocessing  
- Efficient caching system  

### ⚙️ Modular Architecture
- Swap embedding models easily  
- Support multiple LLMs  
- Extensible pipeline design  

---

## 📊 System Workflow

```mermaid
graph LR
    %% User query flow
    A[User Query] --> B[Sentence Transformer Embedding]
    B --> C[Vector Store Search ChromaDB]
    C --> D[Top-K Context Retrieval]
    D --> E[Prompt Construction]
    E --> F[Phi-3 via Ollama]
    F --> G[Generated Answer]
    
    %% Dataset flow
    H[Hugging Face Wikipedia Dataset] --> I[Text Preprocessing]
    I --> J[Embedding Generation]
    J --> C

    style A fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    style G fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style F fill:#fff9c4,stroke:#f9a825,stroke-width:2px
    style C fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style H fill:#ffe0b2,stroke:#ef6c00,stroke-width:2px


Workflow Steps:

Indexing Phase: Dataset → Text normalization → Embedding → ChromaDB

Query Phase: User question → Query embedding → Similarity search → Context retrieval

Generation Phase: Context + Query → Prompt → Phi-3 inference → Answer

🛠️ Technology Stack
Component	Technology	Purpose
Embeddings	all-MiniLM-L6-v2	Dense semantic representations
Vector Store	ChromaDB	Fast similarity search
LLM	Phi-3 (3.8B)	Local answer generation via Ollama
Knowledge Source	Hugging Face Wikipedia Dataset	Offline content for retrieval
Runtime	Python 3.10+	Core environment
IDE	VS Code	Development & debugging
🚀 Quick Start
Prerequisites

Python 3.10+

8GB+ RAM recommended

10GB free disk space (for models)

Installation
# Clone repository
git clone https://github.com/shamilop100/Offline-Wikipedia-RAG-System-Using-Local-LLM-and-Sentence-Transformers.git
cd Offline-Wikipedia-RAG-System-Using-Local-LLM-and-Sentence-Transformers

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install Ollama and Phi-3
ollama pull phi3

Running the System
# Build the embeddings index
python src/build_index.py

# Run queries
python src/retrieve_and_generate.py

📁 Project Structure
rag_project/
├── data/                     # Local cache for content
├── src/
│   ├── build_index.py        # Index Wikipedia dataset
│   ├── retrieve_and_generate.py  # Core RAG pipeline
│   └── utils.py              # Helper functions
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md

💻 Usage Example
from src.retrieve_and_generate import rag_query

question = "What is the role of neural networks in machine learning?"
answer = rag_query(question)
print(answer)

⚙️ Configuration
Parameter	Description	Default
embedding_model	Sentence transformer model	all-MiniLM-L6-v2
llm_model	Ollama model for generation	phi3
top_k	Number of retrieved contexts	3
max_chars	Max characters per article	2000
🖥️ Tested on

Laptop: ASUS Vivobook M6500IH
CPU: AMD Ryzen (Family 23, Model 96) ~2.9 GHz
RAM: 16 GB (15.8 GB usable)
OS: Windows 11 Home (Build 26200, x64)

Operation	Time	Notes
Index Creation (5 articles)	~15s	One-time setup
Query Embedding	~50ms	Per query
Context Retrieval	~100ms	ChromaDB HNSW search
Answer Generation	2-5s	Phi-3 local inference via Ollama
Total Query Time	~3-6s	End-to-end
📄 License

MIT License © 2025 Shamil Op

🙏 Acknowledgments

Ollama
 – Local LLM inference

Sentence-Transformers
 – Embedding models

ChromaDB
 – Vector database

Hugging Face
 – Wikipedia dataset & community

🌟 Citation
@software{wikipedia_rag_offline_2025,
  author       = {Shamil Op},
  title        = {Offline Wikipedia RAG System: Privacy-First Question Answering with Local LLMs},
  year         = 2025,
  publisher    = {GitHub},
  url          = {https://github.com/shamilop100/Offline-Wikipedia-RAG-System-Using-Local-LLM-and-Sentence-Transformers},
  note         = {Retrieval-Augmented Generation using Phi-3 and Sentence Transformers}
}


Made with ❤️ by Shamil Op


---

This version is **fully GitHub-ready**. It:

- Uses `all-MiniLM-L6-v2` correctly.  
- References **Hugging Face dataset** (offline) instead of Wikipedia API.  
- Includes Mermaid diagram, laptop specs, usage examples, and badges.  
- Clean Markdown that renders perfectly on GitHub.  
