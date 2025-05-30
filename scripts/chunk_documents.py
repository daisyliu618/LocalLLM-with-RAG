import json
from pathlib import Path
from typing import List, Dict
import nltk
import hashlib
import os

CHUNK_SENTENCES = 8
CHUNK_OVERLAP = 3
INPUT_FILE = Path("output/parsed_documents.json")
OUTPUT_FILE = Path("output/chunks.json")
CACHE_FILE = Path("output/chunks_cache.json")

# Ensure NLTK punkt is downloaded
venv_nltk_path = os.path.join(os.getcwd(), 'venv', 'nltk_data')
if venv_nltk_path not in nltk.data.path:
    nltk.data.path.append(venv_nltk_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt...")
    nltk.download('punkt', download_dir=venv_nltk_path)
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab...")
    nltk.download('punkt_tab', download_dir=venv_nltk_path)

def sentence_chunk_text(text: str, chunk_sentences: int, overlap: int) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    chunks = []
    start = 0
    while start < len(sentences):
        end = min(start + chunk_sentences, len(sentences))
        chunk = " ".join(sentences[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(sentences):
            break
        start += chunk_sentences - overlap
    return chunks

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    # Load cache if exists
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    else:
        cache = {}
    cached_chunks = cache.get("chunks", {})  # file_path -> {"hash": ..., "chunks": [...]}

    all_chunks = []
    new_cache = {}
    for doc in documents:
        file_path = doc.get('file_path')
        text = doc.get('text', '')
        if not text.strip():
            continue
        text_hash = hash_text(text)
        cache_entry = cached_chunks.get(file_path)
        if cache_entry and cache_entry.get("hash") == text_hash:
            # Reuse cached chunks
            chunks = cache_entry["chunks"]
        else:
            try:
                chunks = sentence_chunk_text(text, CHUNK_SENTENCES, CHUNK_OVERLAP)
            except Exception as e:
                print(f"NLTK sentence chunking failed for {file_path}, falling back to char chunking: {e}")
                chunk_size = 1000
                overlap = 200
                chunks = []
                start = 0
                text_length = len(text)
                while start < text_length:
                    end = min(start + chunk_size, text_length)
                    chunk = text[start:end]
                    if chunk.strip():
                        chunks.append(chunk)
                    if end == text_length:
                        break
                    start += chunk_size - overlap
        # Save to cache
        new_cache[file_path] = {"hash": text_hash, "chunks": chunks}
        for idx, chunk in enumerate(chunks):
            chunk_info = {
                "file_path": file_path,
                "file_type": doc.get("file_type"),
                "metadata": doc.get("metadata", {}),
                "chunk_index": idx,
                "chunk_text": chunk
            }
            all_chunks.append(chunk_info)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump({"chunks": new_cache}, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_chunks)} chunks to {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 