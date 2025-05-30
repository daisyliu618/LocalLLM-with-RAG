import os
from pathlib import Path
from unstructured.partition.auto import partition
from docx import Document
import PyPDF2
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = Path("data")
OUTPUT_FILE = Path("output/parsed_documents.json")
CACHE_FILE = Path("output/parsed_documents_cache.json")


def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text


def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text


def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        return ""


def extract_text_with_unstructured(file_path):
    try:
        elements = partition(filename=str(file_path))
        text = "\n".join([el.text for el in elements if hasattr(el, 'text') and el.text])
        metadata = [el.metadata.to_dict() if hasattr(el, 'metadata') and el.metadata else {} for el in elements]
        return text, metadata
    except Exception as e:
        print(f"Unstructured failed for {file_path}: {e}")
        return "", []


def extract_text_and_metadata(file_path):
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
        return text, {}
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
        return text, {}
    elif ext == ".txt":
        text = extract_text_from_txt(file_path)
        return text, {}
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]:
        # Use unstructured for images (OCR)
        text, metadata = extract_text_with_unstructured(file_path)
        return text, {"unstructured_metadata": metadata}
    else:
        # fallback to unstructured for other types
        text, metadata = extract_text_with_unstructured(file_path)
        return text, {"unstructured_metadata": metadata}


def process_file(file_path):
    ext = file_path.suffix.lower()
    text, metadata = extract_text_and_metadata(file_path)
    doc_info = {
        "file_path": str(file_path),
        "file_type": ext,
        "text": text,
        "metadata": metadata
    }
    print(f"Processed: {file_path.name} | Type: {ext} | Text length: {len(text)}")
    return doc_info


def main():
    file_paths = []
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            file_path = Path(root) / fname
            file_paths.append(file_path)

    # Load cache if exists
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    else:
        cache = {}

    # Build a dict: file_path -> (mtime, doc_info)
    cached_results = {}
    for entry in cache.get("results", []):
        cached_results[entry["file_path"]] = entry
    cached_mtimes = cache.get("mtimes", {})

    results = []
    new_cache_results = []
    new_cache_mtimes = {}
    files_to_process = []
    for file_path in file_paths:
        file_path_str = str(file_path)
        mtime = os.path.getmtime(file_path)
        if file_path_str in cached_mtimes and cached_mtimes[file_path_str] == mtime:
            # Unchanged, reuse cached result
            doc_info = cached_results[file_path_str]
            results.append(doc_info)
            new_cache_results.append(doc_info)
            new_cache_mtimes[file_path_str] = mtime
        else:
            files_to_process.append(file_path)

    # Process new/changed files in parallel
    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, fp): fp for fp in files_to_process}
        for future in as_completed(future_to_file):
            doc_info = future.result()
            results.append(doc_info)
            new_cache_results.append(doc_info)
            file_path_str = doc_info["file_path"]
            new_cache_mtimes[file_path_str] = os.path.getmtime(file_path_str)

    # Save results and update cache
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump({"results": new_cache_results, "mtimes": new_cache_mtimes}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved parsed documents to {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 