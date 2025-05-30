import os
from pathlib import Path
from unstructured.partition.auto import partition
from docx import Document
import PyPDF2
import json
import pandas as pd
from datetime import datetime

DATA_DIR = Path("data")
OUTPUT_FILE = Path("output/parsed_documents.json")

def extract_text_and_metadata(file_path: Path):
    """Extract text content and metadata from various file types."""
    file_ext = file_path.suffix.lower()
    
    # Special handling for CSV files - create individual chunks per row
    if file_ext == '.csv':
        try:
            df = pd.read_csv(file_path)
            chunks = []
            
            for index, row in df.iterrows():
                # Create a text representation of each row
                row_text = f"File: {file_path.name}\n"
                row_text += f"Row {index + 1}:\n"
                
                for column, value in row.items():
                    if pd.notna(value):  # Skip NaN values
                        row_text += f"{column}: {value}\n"
                
                chunk_metadata = {
                    "unstructured_metadata": [{
                        "file_directory": str(file_path.parent),
                        "filename": file_path.name,
                        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        "csv_row_index": index,
                        "filetype": "text/csv"
                    }]
                }
                
                chunks.append({
                    "text": row_text.strip(),
                    "metadata": chunk_metadata
                })
            
            return chunks
            
        except Exception as e:
            print(f"Error processing CSV file {file_path}: {e}")
            # Fall back to treating it as plain text
            pass
    
    try:
        # Existing code for other file types using unstructured
        elements = partition(str(file_path))
        
        # Combine all text
        text_parts = []
        for element in elements:
            if hasattr(element, 'text') and element.text.strip():
                text_parts.append(element.text.strip())
        
        combined_text = '\n'.join(text_parts)
        
        # Get metadata from the first element if available
        metadata = {}
        if elements:
            first_element = elements[0]
            if hasattr(first_element, 'metadata'):
                metadata = first_element.metadata.to_dict()
        
        # For non-CSV files, return single chunk
        return [{
            "text": combined_text,
            "metadata": {"unstructured_metadata": [metadata]}
        }]
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def main():
    INPUT_DIR = Path("data")
    OUTPUT_FILE = Path("output/parsed_documents.json")
    
    all_documents = []
    
    for file_path in INPUT_DIR.glob("*"):
        if file_path.is_file() and not file_path.name.startswith('.'):
            print(f"Processing: {file_path}")
            
            chunks = extract_text_and_metadata(file_path)
            
            for i, chunk in enumerate(chunks):
                doc_info = {
                    "file_path": str(file_path),
                    "file_type": file_path.suffix.lower(),
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "chunk_id": f"{file_path.name}_{i}" if len(chunks) > 1 else file_path.name
                }
                all_documents.append(doc_info)
                
            print(f"  → Created {len(chunks)} chunks")
    
    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(all_documents)} documents to {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 