#!/usr/bin/env python3
"""
Evaluate search quality with test queries.
This script runs several test queries and shows the results
to help assess search quality improvements.
"""

import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from sentence_transformers import CrossEncoder

# Import the search functions from query_rag
import sys
sys.path.append('scripts')

FAISS_INDEX_FILE = Path("output/chunk_index.faiss")
MAPPING_FILE = Path("output/chunk_index_mapping.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 8
BM25_K = 3
RERANK_K = 5
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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

def search_query(query, index, mapping, model, bm25):
    """Perform search and return results."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    # FAISS Semantic Search
    query_emb = model.encode([query], convert_to_numpy=True).astype('float32')
    D, I = index.search(query_emb, TOP_K)
    faiss_indices = list(I[0])
    faiss_results = [mapping[idx] for idx in faiss_indices if 0 <= idx < len(mapping)]

    # BM25 Keyword Search
    tokenized_query = nltk.word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Apply minimum threshold
    min_bm25_score = np.mean(bm25_scores) + 0.5 * np.std(bm25_scores)
    bm25_indices = np.argsort(bm25_scores)[::-1][:BM25_K]
    bm25_results = [mapping[idx] for idx in bm25_indices if bm25_scores[idx] > min_bm25_score]

    # Merge Results with Priority
    seen = set()
    merged_results = []
    
    # Add FAISS results first
    for res in faiss_results:
        key = (res['file_path'], res['chunk_index'])
        if key not in seen:
            merged_results.append(res)
            seen.add(key)
    
    # Add BM25 results
    for res in bm25_results:
        key = (res['file_path'], res['chunk_index'])
        if key not in seen:
            merged_results.append(res)
            seen.add(key)

    # Rerank with Cross-Encoder
    if merged_results:
        reranked_chunks, rerank_scores = rerank_chunks(query, merged_results)
    else:
        reranked_chunks, rerank_scores = [], []

    # Show results summary
    print(f"Found {len(faiss_results)} semantic results")
    print(f"Found {len(bm25_results)} keyword results (above threshold)")
    print(f"Total merged: {len(merged_results)} unique results")
    
    print(f"\nTop {min(3, len(reranked_chunks))} Results:")
    for i, (chunk, score) in enumerate(zip(reranked_chunks[:3], rerank_scores[:3])):
        print(f"\n[{i+1}] Score: {score:.4f}")
        print(f"File: {chunk['file_path']}")
        print(f"Text: {chunk['chunk_text'][:200]}...")
    
    return len(merged_results), rerank_scores[:5] if rerank_scores else []

def main():
    # Load index and mapping
    if not FAISS_INDEX_FILE.exists() or not MAPPING_FILE.exists():
        print("Error: Index files not found. Please run the pipeline first.")
        return

    index = faiss.read_index(str(FAISS_INDEX_FILE))
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    model = SentenceTransformer(EMBEDDING_MODEL)
    bm25, chunk_texts = build_bm25_index(mapping)

    # Test queries - customize these based on your documents
    test_queries = [
        "How to implement user authentication?",
        "What are the main features of the system?",
        "Database configuration and setup",
        "API endpoints and routes",
        "Error handling and logging",
        "Security best practices",
        "Performance optimization techniques",
        "Testing and quality assurance"
    ]
    
    print("RAG Search Quality Evaluation")
    print(f"Total chunks in index: {len(mapping)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    
    results_summary = []
    
    for query in test_queries:
        try:
            num_results, scores = search_query(query, index, mapping, model, bm25)
            avg_score = np.mean(scores) if scores else 0.0
            results_summary.append({
                'query': query,
                'num_results': num_results,
                'avg_rerank_score': avg_score,
                'max_score': max(scores) if scores else 0.0
            })
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            results_summary.append({
                'query': query,
                'num_results': 0,
                'avg_rerank_score': 0.0,
                'max_score': 0.0
            })
    
    # Summary report
    print(f"\n{'='*60}")
    print("SEARCH QUALITY SUMMARY")
    print(f"{'='*60}")
    
    total_queries = len(results_summary)
    avg_results_per_query = np.mean([r['num_results'] for r in results_summary])
    avg_rerank_score = np.mean([r['avg_rerank_score'] for r in results_summary])
    
    print(f"Queries tested: {total_queries}")
    print(f"Average results per query: {avg_results_per_query:.1f}")
    print(f"Average rerank score: {avg_rerank_score:.4f}")
    
    print(f"\nPer-query breakdown:")
    for result in results_summary:
        print(f"'{result['query'][:40]}...': {result['num_results']} results, "
              f"score: {result['avg_rerank_score']:.4f}")

if __name__ == "__main__":
    main() 