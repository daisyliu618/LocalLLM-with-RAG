# Makefile for Local RAG Pipeline

.PHONY: all clean web web-local init

init:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

all: init
	./venv/bin/python scripts/load_documents.py
	./venv/bin/python scripts/chunk_documents.py
	./venv/bin/python scripts/embed_chunks.py
	./venv/bin/python scripts/build_faiss_index.py

clean:
	rm -rf output/*.json output/*.faiss

web: init
	./venv/bin/streamlit run scripts/web_rag.py

web-local: init
	./venv/bin/streamlit run scripts/web_rag_ollama.py
