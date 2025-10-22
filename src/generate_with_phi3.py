import requests
import json
from ollama import generate as ollama_generate  # optional; if you have ollama package installed

OLLAMA_URL = "http://localhost:11434"  # change if your Ollama server is at a different address
MODEL = "phi3"  # switched from "llama3:latest" to a smaller model

def call_ollama_http(prompt, model=MODEL, timeout=30):
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    try:
        j = r.json()
        return j.get("response") or j.get("text") or r.text
    except ValueError:
        return r.text

def call_ollama_client(prompt, model=MODEL):
    response = ollama_generate(model=model, prompt=prompt)
    # ollama.generate may return a dict or other object depending on client version
    if isinstance(response, dict):
        return response.get("response") or response.get("text") or str(response)
    # fallback to string representation
    return str(response)

def build_prompt(question, retrieved_texts):
    system = (
        "You are a helpful AI assistant. Answer concisely and use only the provided context. "
        "If the answer isn't in the context, say \"I don't know.\""
    )
    context =  "\n\n".join([f"CONTEXT: {t}" for t in retrieved_texts])
    return f"{system}\n\n{context}\n\nUSER: {question}"

def answer_question(question, retrieved_texts):
    prompt = build_prompt(question, retrieved_texts)
    try:
        return call_ollama_http(prompt)
    except Exception as e:
        print("HTTP call failed, trying ollama client:", e)
        return call_ollama_client(prompt)
