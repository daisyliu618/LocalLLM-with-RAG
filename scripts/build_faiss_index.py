import json
from pathlib import Path
import numpy as np
import faiss

INPUT_FILE = Path("output/chunk_embeddings.json")
INDEX_FILE = Path("output/chunk_index.faiss")
MAPPING_FILE = Path("output/chunk_index_mapping.json")

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Please run embed_chunks.py first.")
        return

    print("Loading chunk embeddings...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        chunk_data = json.load(f)

    if not chunk_data:
        print("No chunk data found.")
        return

    # Extract embeddings and create mapping
    embeddings = []
    mapping = []
    
    for item in chunk_data:
        embeddings.append(item["embedding"])
        # Create mapping entry without the embedding to save space
        mapping_entry = {
            "chunk_text": item["chunk_text"],
            "file_path": item["file_path"],
            "file_type": item["file_type"],
            "metadata": item["metadata"],
            "chunk_index": item["chunk_index"]
        }
        mapping.append(mapping_entry)

    # Convert to numpy array
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Building FAISS index with {len(embeddings)} vectors of dimension {embeddings.shape[1]}...")

    # Create FAISS index (using IndexFlatIP for inner product/cosine similarity)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    
    # Add embeddings to index
    index.add(embeddings)

    # Save index and mapping
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"FAISS index saved to {INDEX_FILE}")
    print(f"Index mapping saved to {MAPPING_FILE}")
    print(f"Index contains {index.ntotal} vectors")

if __name__ == "__main__":
    main() 