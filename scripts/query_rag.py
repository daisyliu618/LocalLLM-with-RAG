import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import google.generativeai as genai
from rank_bm25 import BM25Okapi
import nltk
from sentence_transformers import CrossEncoder
import tiktoken

FAISS_INDEX_FILE = Path("output/chunk_index.faiss")
MAPPING_FILE = Path("output/chunk_index_mapping.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 8
BM25_K = 3
RERANK_K = 5
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CONTEXT_TOKEN_LIMIT = 3072
ENCODING_NAME = "cl100k_base"



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

def get_gemini_answer(context, query):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY environment variable not set.")
        return None
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
    prompt = f"""
You are an expert assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say 'I don't know.'

Context:
{context}

Question: {query}
Answer (cite the source file if possible):
"""
    response = model.generate_content(prompt)
    return response.text.strip() if hasattr(response, 'text') else str(response)

def build_bm25_index(mapping):
    chunk_texts = [chunk['chunk_text'] for chunk in mapping]
    tokenized_corpus = [nltk.word_tokenize(text.lower()) for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunk_texts

def rerank_chunks(query, chunks):
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    pairs = [(query, chunk['chunk_text']) for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks], [score for chunk, score in scored_chunks]

def main():
    # Load FAISS index
    index = faiss.read_index(str(FAISS_INDEX_FILE))
    # Load mapping
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    # Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)
    # Build BM25 index
    bm25, chunk_texts = build_bm25_index(mapping)

    # Get user query
    query = input("Enter your query: ").strip()
    if not query:
        print("No query entered.")
        return

    # --- FAISS Semantic Search ---
    query_emb = model.encode([query], convert_to_numpy=True).astype('float32')
    D, I = index.search(query_emb, TOP_K)
    faiss_indices = list(I[0])
    faiss_results = [mapping[idx] for idx in faiss_indices if 0 <= idx < len(mapping)]

    # --- BM25 Keyword Search ---
    tokenized_query = nltk.word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Only include BM25 results above a minimum threshold
    min_bm25_score = np.mean(bm25_scores) + 0.5 * np.std(bm25_scores)
    bm25_indices = np.argsort(bm25_scores)[::-1][:BM25_K]
    bm25_results = [mapping[idx] for idx in bm25_indices if bm25_scores[idx] > min_bm25_score]

    # --- Merge Results with Priority ---
    seen = set()
    merged_results = []
    
    # Add FAISS results first (higher priority)
    for res in faiss_results:
        key = (res['file_path'], res['chunk_index'])
        if key not in seen:
            merged_results.append(res)
            seen.add(key)
    
    # Add BM25 results that aren't already included
    for res in bm25_results:
        key = (res['file_path'], res['chunk_index'])
        if key not in seen:
            merged_results.append(res)
            seen.add(key)

    # --- Rerank with Cross-Encoder ---
    reranked_chunks, rerank_scores = rerank_chunks(query, merged_results)

    print(f"\nTop {TOP_K} Semantic (FAISS) Results:")
    for rank, idx in enumerate(faiss_indices):
        if idx < 0 or idx >= len(mapping):
            continue
        chunk = mapping[idx]
        print(f"[FAISS #{rank+1}] File: {chunk['file_path']} | Chunk Index: {chunk['chunk_index']}")
        print(f"Chunk Text:\n{chunk['chunk_text'][:500]}\n{'-'*40}")

    print(f"\nTop {BM25_K} Keyword (BM25) Results:")
    for rank, idx in enumerate(bm25_indices):
        if idx < 0 or idx >= len(mapping):
            continue
        chunk = mapping[idx]
        print(f"[BM25 #{rank+1}] File: {chunk['file_path']} | Chunk Index: {chunk['chunk_index']}")
        print(f"Chunk Text:\n{chunk['chunk_text'][:500]}\n{'-'*40}")

    print(f"\nTop {RERANK_K} Reranked Results (Cross-Encoder):")
    for i, (chunk, score) in enumerate(zip(reranked_chunks[:RERANK_K], rerank_scores[:RERANK_K])):
        print(f"[Rerank #{i+1}] Score: {score:.4f} | File: {chunk['file_path']} | Chunk Index: {chunk['chunk_index']}")
        print(f"Chunk Text:\n{chunk['chunk_text'][:500]}\n{'-'*40}")

    # --- Prepare context for LLM (reranked top results, token-limited) ---
    context = build_context(reranked_chunks, CONTEXT_TOKEN_LIMIT)
    print(f"\n[Gemini LLM Answer]\n-------------------")
    answer = get_gemini_answer(context, query)
    print(answer or "[No answer generated]")

if __name__ == "__main__":
    main() 