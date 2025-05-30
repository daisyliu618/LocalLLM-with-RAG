import streamlit as st
import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
from rank_bm25 import BM25Okapi
import nltk
import tiktoken

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

FAISS_INDEX_FILE = Path("output/chunk_index.faiss")
MAPPING_FILE = Path("output/chunk_index_mapping.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_SEMANTIC = 5
TOP_K_KEYWORD = 5
CONTEXT_TOKEN_LIMIT = 2048
ENCODING_NAME = "cl100k_base"
OLLAMA_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"

@st.cache_resource
def load_faiss_index():
    return faiss.read_index(str(FAISS_INDEX_FILE))

@st.cache_resource
def load_mapping():
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def build_bm25_index(mapping):
    chunk_texts = [chunk['chunk_text'] for chunk in mapping]
    tokenized_corpus = [nltk.word_tokenize(text.lower()) for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunk_texts

def count_tokens(text, encoding_name=ENCODING_NAME):
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))

def build_context(chunks, token_limit):
    context_chunks = []
    total_tokens = 0
    for chunk in chunks:
        chunk_tokens = count_tokens(chunk['chunk_text'])
        if total_tokens + chunk_tokens > token_limit:
            break
        context_chunks.append(chunk['chunk_text'])
        total_tokens += chunk_tokens
    return "\n\n".join(context_chunks)

def get_ollama_answer(context, query, model=OLLAMA_MODEL):
    prompt = f"""You are an expert assistant. Use the following context to answer the user's question.

Context:
{context}

Question: {query}
Answer (please cite source files when possible and explain your reasoning):"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120
        )
        if response.ok:
            return response.json().get("response", "").strip(), None
        else:
            return None, f"[Ollama API Error] {response.text}"
    except Exception as e:
        return None, f"[Ollama API Error] {e}"

def main():
    st.title("Local RAG with Ollama")
    st.write("Enter your ask below. The app will retrieve relevant chunks using both semantic (FAISS) and keyword (BM25) search, then generate an answer using Ollama. The LLM will only answer from the provided context.")

    # Reload Data Button
    if st.button("Reload Data (Clear Cache)"):
        st.cache_resource.clear()
        st.success("Cache cleared. Data and models will be reloaded on next query.")

    # Use a form to handle Enter key presses
    with st.form("search_form"):
        query = st.text_input("Your query:", key="query_input")
        search_clicked = st.form_submit_button("Search")
    
    if search_clicked and query:
        try:
            with st.spinner("Loading models and indexes..."):
                index = load_faiss_index()
                mapping = load_mapping()
                embedder = load_embedder()
                bm25, chunk_texts = build_bm25_index(mapping)
            with st.spinner("Retrieving chunks..."):
                # --- FAISS Semantic Search ---
                query_emb = embedder.encode([query], convert_to_numpy=True).astype('float32')
                D, I = index.search(query_emb, TOP_K_SEMANTIC)
                faiss_indices = list(I[0])
                faiss_results = [mapping[idx] for idx in faiss_indices if 0 <= idx < len(mapping)]
                # --- BM25 Keyword Search ---
                tokenized_query = nltk.word_tokenize(query.lower())
                bm25_scores = bm25.get_scores(tokenized_query)
                bm25_indices = np.argsort(bm25_scores)[::-1][:TOP_K_KEYWORD]
                bm25_results = [mapping[idx] for idx in bm25_indices if bm25_scores[idx] > 0]
                # --- Merge Results ---
                seen = set()
                merged_results = []
                for res in faiss_results + bm25_results:
                    key = (res['file_path'], res['chunk_index'])
                    if key not in seen:
                        merged_results.append(res)
                        seen.add(key)
            st.subheader(f"Top {TOP_K_SEMANTIC} Semantic (FAISS) Results:")
            for rank, idx in enumerate(faiss_indices):
                if idx < 0 or idx >= len(mapping):
                    continue
                chunk = mapping[idx]
                with st.expander(f"[FAISS #{rank+1}] File: {chunk['file_path']} | Chunk Index: {chunk['chunk_index']}"):
                    st.write(chunk["chunk_text"])
            st.subheader(f"Top {TOP_K_KEYWORD} Keyword (BM25) Results:")
            for rank, idx in enumerate(bm25_indices):
                if idx < 0 or idx >= len(mapping) or bm25_scores[idx] <= 0:
                    continue
                chunk = mapping[idx]
                with st.expander(f"[BM25 #{rank+1}] Score: {bm25_scores[idx]:.4f} | File: {chunk['file_path']} | Chunk Index: {chunk['chunk_index']}"):
                    st.write(chunk["chunk_text"])
            st.subheader(f"Combined Results ({len(merged_results)} unique chunks):")
            for i, chunk in enumerate(merged_results):
                with st.expander(f"[Combined #{i+1}] File: {chunk['file_path']} | Chunk Index: {chunk['chunk_index']}"):
                    st.write(chunk["chunk_text"])
            # --- Prepare context for LLM (merged results, token-limited) ---
            context = build_context(merged_results, CONTEXT_TOKEN_LIMIT)
            st.subheader("Ollama LLM Answer:")
            with st.spinner("Generating answer with Ollama..."):
                answer, error = get_ollama_answer(context, query)
            if error:
                st.error(error)
            else:
                st.success(answer)
        except Exception as e:
            st.error(f"[Error] {e}")

if __name__ == "__main__":
    main() 