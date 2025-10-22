# app_streamlit.py
import streamlit as st
from src.rag_pipeline import answer

st.set_page_config(page_title="RAG + Ollama Demo")
st.title("RAG + Ollama — Mini Demo")

query = st.text_input("Ask a question:", value="Who invented the telephone?")
if st.button("Ask"):
    with st.spinner("Retrieving and generating..."):
        out = answer(query, top_k=4)
    st.subheader("Answer")
    st.write(out["answer"])
    st.subheader("Context (retrieved passages)")
    for i, c in enumerate(out["context"]):
        st.markdown(f"**Passage {i+1}:** {c}")
