import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import hashlib

INPUT_FILE = Path("output/chunks.json")
OUTPUT_FILE = Path("output/chunk_embeddings.json")
CACHE_FILE = Path("output/chunk_embeddings_cache.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Load cache if exists
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    else:
        cache = {}
    cached_embeddings = cache.get("embeddings", {})  # key: (file_path, chunk_index) -> {"hash": ..., "embedding": ...}

    model = SentenceTransformer(EMBEDDING_MODEL)

    results = []
    new_cache = {}
    to_embed = []
    to_embed_keys = []
    # Prepare for incremental embedding
    for chunk in chunks:
        key = (chunk["file_path"], chunk["chunk_index"])
        chunk_hash = hash_text(chunk["chunk_text"])
        cache_entry = cached_embeddings.get(str(key))
        if cache_entry and cache_entry.get("hash") == chunk_hash:
            # Reuse cached embedding
            embedding = cache_entry["embedding"]
            entry = {
                "embedding": embedding,
                "chunk_text": chunk["chunk_text"],
                "file_path": chunk["file_path"],
                "file_type": chunk["file_type"],
                "metadata": chunk["metadata"],
                "chunk_index": chunk["chunk_index"]
            }
            results.append(entry)
            new_cache[str(key)] = {"hash": chunk_hash, "embedding": embedding}
        else:
            to_embed.append(chunk["chunk_text"])
            to_embed_keys.append(key)

    # Embed new/changed chunks
    if to_embed:
        print(f"Embedding {len(to_embed)} new/changed chunks...")
        embeddings = model.encode(to_embed, show_progress_bar=True, convert_to_numpy=True)
        for chunk, key, emb in zip([chunks[i] for i, k in enumerate([c for c in chunks if (c["file_path"], c["chunk_index"]) in to_embed_keys])], to_embed_keys, embeddings):
            entry = {
                "embedding": emb.tolist(),
                "chunk_text": chunk["chunk_text"],
                "file_path": chunk["file_path"],
                "file_type": chunk["file_type"],
                "metadata": chunk["metadata"],
                "chunk_index": chunk["chunk_index"]
            }
            results.append(entry)
            chunk_hash = hash_text(chunk["chunk_text"])
            new_cache[str(key)] = {"hash": chunk_hash, "embedding": emb.tolist()}

    # Save results and update cache
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump({"embeddings": new_cache}, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} embeddings to {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 