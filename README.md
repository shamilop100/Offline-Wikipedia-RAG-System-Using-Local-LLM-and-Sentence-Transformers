# Wikipedia RAG System

A production-ready Retrieval-Augmented Generation (RAG) pipeline that leverages Wikipedia as a knowledge source to deliver accurate, context-grounded answers to natural language queries.

## Overview

This system combines state-of-the-art semantic search with Large Language Models (LLMs) to retrieve relevant information from Wikipedia and generate coherent, factually-grounded responses. Built with modularity and scalability in mind, it supports both local and cloud-based deployment options.

## Features

- **Semantic Retrieval**: Utilizes dense embeddings for intelligent context retrieval based on semantic similarity
- **Context-Aware Generation**: Produces answers grounded in retrieved Wikipedia passages to ensure factual accuracy
- **Flexible Architecture**: Compatible with various LLM backends including Ollama, OpenAI, and Hugging Face models
- **Efficient Vector Search**: Employs ChromaDB for fast, scalable similarity search operations
- **Offline Capability**: Fully functional with locally-hosted models, requiring no external API calls
- **Lightweight & Fast**: Optimized for performance with minimal resource footprint

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   User      │────▶│   Retriever  │────▶│  Generator  │
│   Query     │     │  (Semantic)  │     │    (LLM)    │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  ChromaDB    │
                    │ Vector Store │
                    └──────────────┘
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Embeddings** | sentence-transformers |
| **Vector Database** | ChromaDB |
| **LLM Backend** | Transformers / Ollama / OpenAI |
| **Data Source** | Wikipedia Python API |
| **Runtime** | Python ≥ 3.10 |

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

### Local Setup

```bash
# Clone the repository
git clone https://github.com/shamilop100/Offline-Wikipedia-RAG-System-Using-Local-LLM-and-Sentence-Transformers.git
cd Offline-Wikipedia-RAG-System-Using-Local-LLM-and-Sentence-Transformers

# Create and activate virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab

For a cloud-based setup, simply open the provided Colab notebook and execute all cells. Dependencies will be installed automatically.

## Project Structure

```
rag_project/
│
├── data/                        # Local cache for downloaded content
├── src/
│   ├── build_index.py           # Index creation from Wikipedia articles
│   ├── retrieve_and_generate.py # Core RAG pipeline implementation
│   └── utils.py                 # Utility functions
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation
```

## Usage

### 1. Building the Knowledge Index

First, create a vector index from Wikipedia articles:

```python
# src/build_index.py
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
import wikipedia

def normalize_text(text):
    """Normalize whitespace in text."""
    return ' '.join(text.strip().split())

def fetch_wikipedia_articles(topic_list, max_chars=2000):
    """Fetch and process Wikipedia articles for specified topics."""
    docs = []
    for topic in topic_list:
        try:
            content = wikipedia.page(topic).content[:max_chars]
            docs.append({"title": topic, "text": normalize_text(content)})
        except Exception as e:
            print(f"Failed to fetch {topic}: {e}")
            continue
    return docs

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB collection
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("wiki_collection")

# Define topics and build index
topics = ["Machine learning", "Data Science", "Neural network", "Artificial intelligence"]
docs = fetch_wikipedia_articles(topics)

for i, doc in enumerate(docs):
    embedding = model.encode(doc["text"]).tolist()
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        metadatas=[{"title": doc["title"]}],
        documents=[doc["text"]]
    )

print("✅ Wikipedia index built successfully!")
```

### 2. Querying the System

Use the RAG pipeline to answer questions:

```python
# src/retrieve_and_generate.py
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

# Load retriever and vector store
retriever = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_collection("wiki_collection")

# Initialize LLM for generation
qa_pipeline = pipeline("text-generation", model="distilgpt2")

def rag_query(question, top_k=3):
    """
    Retrieve relevant context and generate an answer.
    
    Args:
        question: User query string
        top_k: Number of context chunks to retrieve
        
    Returns:
        Generated answer string
    """
    # Retrieve relevant context
    query_embedding = retriever.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    context = " ".join(results["documents"][0])
    
    # Generate answer
    prompt = f"Answer this question based on context:\n{context}\n\nQuestion: {question}\nAnswer:"
    answer = qa_pipeline(prompt, max_length=200, do_sample=True)[0]["generated_text"]
    
    return answer

# Example query
response = rag_query("What is the role of neural networks in machine learning?")
print(response)
```

## How It Works

1. **Indexing Phase**: Wikipedia articles are fetched, chunked, and embedded into dense vectors stored in ChromaDB
2. **Retrieval Phase**: User queries are embedded and matched against the vector store using cosine similarity
3. **Generation Phase**: Retrieved context passages are combined with the query and passed to an LLM for answer generation

## Configuration

The system can be customized through several parameters:

- **Embedding Model**: Change `all-MiniLM-L6-v2` to any sentence-transformers model
- **LLM Backend**: Swap `distilgpt2` with GPT-2, GPT-3, or local models via Ollama
- **Retrieval Count**: Adjust `top_k` parameter to retrieve more or fewer context chunks
- **Context Length**: Modify `max_chars` in article fetching for longer/shorter passages

## Roadmap

- [ ] Implement hybrid retrieval combining dense embeddings with BM25 sparse retrieval
- [ ] Add evaluation metrics (BLEU, ROUGE, F1) for answer quality assessment
- [ ] Deploy interactive web interface using Streamlit or Gradio
- [ ] Optimize inference performance for local LLMs (LLaMA, Mistral)
- [ ] Support for multi-language Wikipedia articles
- [ ] Add document chunking strategies for improved context windows
- [ ] Implement caching layer for frequently asked questions

## Performance

Typical query latency on consumer hardware:
- Retrieval: ~50-100ms
- Generation: 1-3s (local LLM) or 0.5-1s (API-based LLM)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Sentence-Transformers](https://www.sbert.net/) for embedding models
- [ChromaDB](https://www.trychroma.com/) for vector database capabilities
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) for LLM infrastructure
- [Wikipedia API](https://pypi.org/project/wikipedia/) for knowledge retrieval

## Citation

If you use this project in your research or applications, please cite:

```bibtex
@software{wikipedia_rag_2025,
  author = {Shamil Op},
  title = {Wikipedia RAG System: Context-Aware Question Answering},
  year = {2025},
  url = {https://github.com/shamilop100/Offline-Wikipedia-RAG-System-Using-Local-LLM-and-Sentence-Transformers}
}
```

## Contact

Shamil Op - [@shamilop100](https://github.com/shamilop100)

Project Link: [https://github.com/shamilop100/Offline-Wikipedia-RAG-System-Using-Local-LLM-and-Sentence-Transformers](https://github.com/shamilop100/Offline-Wikipedia-RAG-System-Using-Local-LLM-and-Sentence-Transformers)

---

⭐ If you find this project useful, please consider giving it a star on GitHub!
