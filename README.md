🧠 Offline Wikipedia RAG System
<p align="center"> <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/Ollama-Phi--3-green?logo=ollama&logoColor=white" /> <img src="https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange?logo=huggingface&logoColor=white" /> <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-purple?logo=database&logoColor=white" /> <img src="https://img.shields.io/badge/License-MIT-lightgrey?logo=opensourceinitiative&logoColor=white" /> <img src="https://img.shields.io/badge/Platform-VS%20Code-blueviolet?logo=visualstudiocode&logoColor=white" /> </p> <p align="center"> <b>Privacy-First | Fully Offline | Context-Aware Question Answering</b> </p>
📖 Overview

A production-ready Retrieval-Augmented Generation (RAG) system that operates 100% offline using local LLMs. It combines the Wikipedia knowledge base with Microsoft’s Phi-3 model (via Ollama) and state-of-the-art semantic search to deliver accurate, context-grounded answers without cloud dependencies.

Perfect for privacy-sensitive applications, air-gapped environments, or anyone seeking full control over their AI infrastructure.

🎯 Why This Project?

🔒 Privacy-First: All data and processing remain local

⚡ Lightning Fast: No API latency, answers in seconds

💰 Zero Cost: No API fees or subscriptions

🛡️ Secure: Offline/air-gapped compatible

🎓 Educational: Learn RAG architecture hands-on

✨ Features
🔍 Semantic Search

Dense embedding retrieval using MiniLM / MPNet

Context-aware similarity matching

Handles complex queries effectively

🤖 Local LLM Generation

Powered by Microsoft Phi-3 (3.8B parameters)

Runs via Ollama locally

No internet connection required

📚 Wikipedia Integration

Intelligent text chunking

Dynamic article fetching

Efficient caching system

⚙️ Modular Architecture

Swap embedding models easily

Support multiple LLMs

Extensible pipeline design

📊 System Workflow

graph LR
    %% User query flow
    A[User Query] --> B[Sentence Transformer Embedding]
    B --> C[Vector Store Search (ChromaDB)]
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

Indexing Phase: Wikipedia articles → Text normalization → Embedding generation → ChromaDB storage

Query Phase: User question → Query embedding → Similarity search → Context retrieval

Generation Phase: Context + Query → Prompt → Phi-3 inference → Answer
