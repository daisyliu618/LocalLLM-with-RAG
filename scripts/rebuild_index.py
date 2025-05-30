#!/usr/bin/env python3
"""
Rebuild the entire search index with improved settings.
This script will:
1. Clear old cache files
2. Re-chunk documents with better parameters
3. Re-embed chunks 
4. Rebuild FAISS index
"""

import os
import subprocess
from pathlib import Path

def run_script(script_name):
    """Run a Python script and handle errors."""
    script_path = Path("scripts") / script_name
    print(f"\n{'='*50}")
    print(f"Running: {script_name}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(
            ["python", str(script_path)], 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

def clear_cache_files():
    """Clear cache files to force regeneration."""
    cache_files = [
        "output/chunks_cache.json",
        "output/chunk_embeddings_cache.json"
    ]
    
    print("Clearing cache files...")
    for cache_file in cache_files:
        cache_path = Path(cache_file)
        if cache_path.exists():
            cache_path.unlink()
            print(f"Deleted: {cache_file}")
        else:
            print(f"Not found: {cache_file}")

def main():
    print("Rebuilding RAG index with improved settings...")
    
    # Step 1: Clear cache to force regeneration
    clear_cache_files()
    
    # Step 2: Re-chunk documents with new parameters
    run_script("chunk_documents.py")
    
    # Step 3: Re-embed chunks
    run_script("embed_chunks.py")
    
    # Step 4: Rebuild FAISS index
    run_script("build_faiss_index.py")
    
    print("\n" + "="*50)
    print("Index rebuild complete!")
    print("="*50)
    print("\nYou can now test the improved search with:")
    print("python scripts/query_rag.py")

if __name__ == "__main__":
    main() 