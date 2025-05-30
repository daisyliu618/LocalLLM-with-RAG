# Makefile for Local RAG Pipeline

.PHONY: all clean web web-ollama

all:
	python3 scripts/load_documents.py
	python3 scripts/chunk_documents.py
	python3 scripts/embed_chunks.py
	python3 scripts/build_faiss_index.py

clean:
	rm -rf output/*.json output/*.faiss

web:
	streamlit run scripts/web_rag.py

web-ollama:
	streamlit run scripts/web_rag_ollama.py 