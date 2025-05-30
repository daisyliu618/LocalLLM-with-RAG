import streamlit as st
import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import google.generativeai as genai
import requests
from rank_bm25 import BM25Okapi
import nltk
import tiktoken
import hashlib
import time
from datetime import datetime
import subprocess

# Ensure NLTK punkt is available with explicit path setup
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
nltk_data_path = os.path.join(project_root, 'venv', 'nltk_data')

# Add the path if it exists and is not already in the list
if os.path.exists(nltk_data_path) and nltk_data_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_path)

# Try to download and set up punkt tokenizer with fallbacks
def setup_nltk_punkt():
    try:
        # Try to find existing punkt data
        nltk.data.find('tokenizers/punkt')
        return True
    except LookupError:
        pass
    
    # Try downloading both versions
    for punkt_version in ['punkt_tab', 'punkt']:
        try:
            if os.path.exists(nltk_data_path):
                nltk.download(punkt_version, download_dir=nltk_data_path, quiet=True)
            else:
                nltk.download(punkt_version, quiet=True)
            # Test if it works
            nltk.data.find('tokenizers/punkt')
            return True
        except:
            continue
    
    return False

# Set up punkt with error handling
punkt_available = setup_nltk_punkt()
if not punkt_available:
    st.warning("NLTK punkt tokenizer could not be set up. Word tokenization may fail.")

# Alternative tokenization function as fallback
def safe_word_tokenize(text):
    try:
        return nltk.word_tokenize(text.lower())
    except:
        # Simple fallback tokenization
        import re
        return re.findall(r'\b\w+\b', text.lower())

FAISS_INDEX_FILE = Path("output/chunk_index.faiss")
MAPPING_FILE = Path("output/chunk_index_mapping.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_SEMANTIC = 5
TOP_K_KEYWORD = 5
DEFAULT_CONTEXT_TOKEN_LIMIT = 4096
ENCODING_NAME = "cl100k_base"
OLLAMA_URL = "http://localhost:11434/api/generate"
CACHE_FILE_GEMINI = Path("output/query_cache_gemini.json")
CACHE_FILE_OLLAMA = Path("output/query_cache_ollama.json")

# === QUERY CACHING SYSTEM ===
def normalize_query(query):
    """Normalize query for caching purposes"""
    return query.lower().strip()

def get_query_hash(query):
    """Get a hash of the normalized query for cache key"""
    normalized = normalize_query(query)
    return hashlib.md5(normalized.encode()).hexdigest()

def get_cache_file(provider):
    """Get the appropriate cache file based on provider"""
    return CACHE_FILE_GEMINI if provider == "Gemini" else CACHE_FILE_OLLAMA

def load_cache(provider):
    """Load query cache from file"""
    cache_file = get_cache_file(provider)
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache, provider):
    """Save query cache to file"""
    try:
        cache_file = get_cache_file(provider)
        cache_file.parent.mkdir(exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.warning(f"Failed to save cache: {e}")

def get_cached_answer(query, provider):
    """Get cached answer for a query if it exists"""
    cache = load_cache(provider)
    query_hash = get_query_hash(query)
    
    if query_hash in cache:
        cached_entry = cache[query_hash]
        return cached_entry.get('answer'), cached_entry.get('timestamp')
    return None, None

def cache_answer(query, answer, provider, context_info=None):
    """Cache an answer for a query"""
    cache = load_cache(provider)
    query_hash = get_query_hash(query)
    
    cache[query_hash] = {
        'query': query,
        'answer': answer,
        'timestamp': datetime.now().isoformat(),
        'provider': provider,
        'context_info': context_info or {}
    }
    
    # Keep only last 100 cached queries to prevent unlimited growth
    if len(cache) > 100:
        # Remove oldest entries
        sorted_entries = sorted(cache.items(), key=lambda x: x[1]['timestamp'])
        cache = dict(sorted_entries[-100:])
    
    save_cache(cache, provider)

def clear_cache(provider):
    """Clear the query cache for a specific provider"""
    cache_file = get_cache_file(provider)
    if cache_file.exists():
        cache_file.unlink()
    st.session_state.pop('cache_cleared', None)

def clear_all_caches():
    """Clear all caches"""
    for provider in ["Gemini", "Ollama"]:
        clear_cache(provider)

@st.cache_resource
def load_faiss_index():
    return faiss.read_index(str(FAISS_INDEX_FILE))

@st.cache_resource
def load_mapping():
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def build_bm25_index(mapping):
    chunk_texts = [chunk['chunk_text'] for chunk in mapping]
    tokenized_corpus = [safe_word_tokenize(text) for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunk_texts

def count_tokens(text, encoding_name=ENCODING_NAME):
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))

def build_context(chunks, token_limit):
    """Enhanced context building with better token management"""
    context_chunks = []
    total_tokens = 0
    chunks_included = 0
    
    for chunk in chunks:
        chunk_text = chunk['chunk_text']
        chunk_tokens = count_tokens(chunk_text)
        
        if total_tokens + chunk_tokens > token_limit:
            # Try to fit partial chunk if we have significant remaining space
            remaining_tokens = token_limit - total_tokens
            if remaining_tokens > 200:  # Only if we have meaningful space
                # Truncate at sentence boundary
                sentences = chunk_text.split('.')
                partial_text = ""
                for sentence in sentences:
                    test_text = partial_text + sentence + "."
                    if count_tokens(test_text) <= remaining_tokens:
                        partial_text = test_text
                    else:
                        break
                
                if partial_text and len(partial_text) > 50:  # Only add if meaningful
                    context_chunks.append(partial_text + "...")
                    total_tokens += count_tokens(partial_text)
                    chunks_included += 1
            break
        
        context_chunks.append(chunk_text)
        total_tokens += chunk_tokens
        chunks_included += 1
    
    context_info = {
        'chunks_included': chunks_included,
        'total_chunks_available': len(chunks),
        'tokens_used': total_tokens,
        'token_limit': token_limit
    }
    
    return "\n\n".join(context_chunks), context_info

def get_gemini_answer(context, query, model_name="models/gemini-1.5-pro-latest"):
    """Get answer from Gemini API"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None, "[ERROR] GEMINI_API_KEY environment variable not set."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"""
    You are an expert assistant. Use the following context to answer the user's question.
    IMPORTANT: Only use information from the provided context. If the context doesn't contain enough information to answer the question, say so explicitly.

Context:
{context}

Question: {query}
Answer (please cite source files when possible and explain your reasoning):
"""
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip() if hasattr(response, 'text') else str(response)
        return answer, None
    except Exception as e:
        return None, f"[Gemini API Error] {e}"

def get_ollama_answer(context, query, model_name="llama3"):
    """Get answer from Ollama API"""
    prompt = f"""You are an expert assistant. Use the following context to answer the user's question.
IMPORTANT: Only use information from the provided context. If the context doesn't contain enough information to answer the question, say so explicitly.

Context:
{context}

Question: {query}
Answer (please cite source files when possible and explain your reasoning):"""
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=120
        )
        if response.ok:
            return response.json().get("response", "").strip(), None
        else:
            return None, f"[Ollama API Error] {response.text}"
    except Exception as e:
        return None, f"[Ollama API Error] {e}"

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.ok
    except:
        return False

def main():
    st.title("🔍 Local & Cloud RAG")
    st.write("Enhanced RAG with query caching, configurable context window, and support for both local (Ollama) and cloud (Gemini) models")

    # === SIDEBAR CONFIGURATION ===
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Provider Selection
        st.subheader("🤖 Model Provider")
        provider = st.selectbox(
            "Select Provider",
            ["Ollama (Local)", "Gemini (Cloud)"],
            help="Choose between Ollama (local) or Google Gemini (cloud) models"
        )
        
        # Model-specific settings
        if provider == "Ollama (Local)": # Ollama
            st.subheader("🏠 Ollama Settings")
            ollama_model = st.selectbox(
                "Ollama Model",
                ["llama3", "mistral", "codellama", "vicuna", "orca-mini", "phi3", "gemma"],
                help="Select local Ollama model"
            )
            
            # Check Ollama connection
            if test_ollama_connection():
                st.success("✅ Ollama server running")
            else:
                st.error("❌ Ollama server not accessible")
                st.info("Start Ollama: `ollama serve`")
        
        else:
            st.subheader("☁️ Gemini Settings")
            gemini_model = st.selectbox(
                "Gemini Model",
                ["models/gemini-1.5-pro-latest", "models/gemini-1.5-flash-latest", "models/gemini-pro"],
                help="Select Gemini model variant"
            )
            
            # Check API key
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                st.success("✅ Gemini API key found")
            else:
                st.error("❌ GEMINI_API_KEY not set")
        
        # Context window settings
        st.subheader("📄 Context Window")
        context_limit = st.slider(
            "Max Context Tokens", 
            min_value=1024, 
            max_value=16384, 
            value=DEFAULT_CONTEXT_TOKEN_LIMIT, 
            step=512,
            help="Maximum tokens to send to the LLM"
        )
        
        # Search settings
        st.subheader("🔍 Search Settings")
        semantic_k = st.slider("Semantic Results", 1, 15, TOP_K_SEMANTIC)
        keyword_k = st.slider("Keyword Results", 1, 15, TOP_K_KEYWORD)
        
        # Cache settings
        st.subheader("💾 Cache Management")
        current_provider = "Gemini" if provider == "Gemini (Cloud)" else "Ollama"
        cache = load_cache(current_provider)
        cache_size = len(cache)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cached Queries", cache_size)
        with col2:
            st.metric("Provider", current_provider)
        
        if st.button("🗑️ Clear Current Cache"):
            clear_cache(current_provider)
            st.success(f"{current_provider} cache cleared!")
            st.rerun()
            
        if st.button("🧹 Clear All Caches"):
            clear_all_caches()
            st.success("All caches cleared!")
            st.rerun()
        
        # Enable/disable cache
        use_cache = st.checkbox("Use Query Cache", value=True, 
                               help="Cache answers to avoid recomputing similar queries")
        
        # Show recent cached queries
        if cache_size > 0 and st.button("📋 Show Recent Queries"):
            st.subheader(f"Recent {current_provider} Queries")
            sorted_cache = sorted(cache.items(), 
                                key=lambda x: x[1]['timestamp'], 
                                reverse=True)
            for i, (_, entry) in enumerate(sorted_cache[:5]):
                with st.expander(f"Query {i+1}: {entry['query'][:50]}..."):
                    st.write(f"**Timestamp:** {entry['timestamp']}")
                    st.write(f"**Provider:** {entry.get('provider', 'Unknown')}")
                    st.write(f"**Answer:** {entry['answer'][:200]}...")


    # Use a form to handle Enter key presses
    with st.form("search_form"):
        query = st.text_input("Your question:", key="query_input", 
                            placeholder="Ask anything about your documents...")
        col_search, col_rebuild = st.columns([1,1], gap="small")
        with col_search:
            search_clicked = st.form_submit_button("🔍 Search")
        with col_rebuild:
            rebuild_clicked = st.form_submit_button("📚 Rebuild Data")

    if rebuild_clicked:
        with st.spinner("Running 'make all' to rebuild data pipeline. This may take a while..."):
            try:
                result = subprocess.run(["make", "all"], capture_output=True, text=True, check=True)
                st.success("✅ Data pipeline rebuilt successfully!")
                st.text_area("make all output", result.stdout, height=200)
                st.cache_resource.clear()
            except subprocess.CalledProcessError as e:
                st.error(f"[make all failed] {e}")
                st.text_area("make all error output", e.stderr, height=200)
        # Don't run search logic if rebuild was clicked
        return
    
    if search_clicked and query:
        start_time = time.time()
        current_provider = "Gemini" if provider == "Gemini (Cloud)" else "Ollama"
        
        # Check cache first if enabled
        if use_cache:
            cached_answer, cache_timestamp = get_cached_answer(query, current_provider)
            if cached_answer:
                st.info(f"📋 Found cached {current_provider} answer from {cache_timestamp}")
                st.success(f"### 🎯 Cached Answer:\n\n{cached_answer}")
                # Show cache performance
                response_time = time.time() - start_time
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Response Time", f"{response_time:.3f}s")
                with col2:
                    st.metric("Cache Hit", "✅ Yes")
                with col3:
                    st.metric("Provider", current_provider)
                return
        
        # If not cached, proceed with normal retrieval
        try:
            with st.spinner("Loading models and indexes..."):
                index = load_faiss_index()
                mapping = load_mapping()
                embedder = load_embedder()
                bm25, chunk_texts = build_bm25_index(mapping)
                
            with st.spinner("Retrieving chunks..."):
                # --- FAISS Semantic Search ---
                query_emb = embedder.encode([query], convert_to_numpy=True).astype('float32')
                D, I = index.search(query_emb, semantic_k)
                faiss_indices = list(I[0])
                faiss_results = [mapping[idx] for idx in faiss_indices if 0 <= idx < len(mapping)]
                
                # --- BM25 Keyword Search ---
                tokenized_query = safe_word_tokenize(query)
                bm25_scores = bm25.get_scores(tokenized_query)
                bm25_indices = np.argsort(bm25_scores)[::-1][:keyword_k]
                bm25_results = [mapping[idx] for idx in bm25_indices if bm25_scores[idx] > 0]
                
                # --- Merge Results with Smart Prioritization ---
                seen = set()
                merged_results = []
                
                # First, add high-scoring BM25 results (exact matches get priority)
                high_score_threshold = 2.0  # Adjust this threshold as needed
                for idx in bm25_indices:
                    if idx < 0 or idx >= len(mapping) or bm25_scores[idx] <= 0:
                        continue
                    if bm25_scores[idx] >= high_score_threshold:  # High-scoring keyword matches first
                        res = mapping[idx]
                        key = (res['file_path'], res['chunk_index'])
                        if key not in seen:
                            merged_results.append(res)
                            seen.add(key)
                
                # Then add remaining FAISS semantic results
                for res in faiss_results:
                    key = (res['file_path'], res['chunk_index'])
                    if key not in seen:
                        merged_results.append(res)
                        seen.add(key)
                
                # Finally, add remaining lower-scoring BM25 results
                for idx in bm25_indices:
                    if idx < 0 or idx >= len(mapping) or bm25_scores[idx] <= 0:
                        continue
                    if bm25_scores[idx] < high_score_threshold:  # Lower-scoring keyword matches last
                        res = mapping[idx]
                        key = (res['file_path'], res['chunk_index'])
                        if key not in seen:
                            merged_results.append(res)
                            seen.add(key)

            # --- Prepare context for LLM ---
            with st.spinner(f"Building context and generating answer with {current_provider}..."):
                context, context_info = build_context(merged_results, context_limit)
                
                # Generate answer based on provider
                if provider == "Gemini (Cloud)":
                    answer, error = get_gemini_answer(context, query, gemini_model)
                else:
                    answer, error = get_ollama_answer(context, query, ollama_model)
            
            # === DISPLAY ANSWER FIRST ===
            if error:
                st.error(error)
            else:
                st.success(f"### 🎯 {current_provider} Generated Answer:\n\n{answer}")
                
                # Cache the answer if caching is enabled
                if use_cache and answer:
                    cache_answer(query, answer, current_provider, context_info)
                    st.info(f"💾 Answer cached for {current_provider}")

            # Show search results in expandable sections
            st.subheader(f"🔍 Search Results ({len(merged_results)} chunks found)")
            tab1, tab2, tab3 = st.tabs(["Combined", "Semantic", "Keyword"])

            with tab1:
                st.write(f"**Combined Results ({len(merged_results)} unique chunks):**")
                for i, chunk in enumerate(merged_results):
                    st.markdown(f"**[#{i+1}] {chunk['file_path']} | Chunk {chunk['chunk_index']}**")
                    with st.container():
                        st.text_area(
                            f"Content #{i+1}", 
                            chunk["chunk_text"], 
                            height=100, 
                            key=f"combined_{i}",
                            disabled=True
                        )
                    st.markdown("---")

            with tab2:
                st.write(f"**Top {semantic_k} Semantic (FAISS) Results:**")
                for rank, idx in enumerate(faiss_indices):
                    if idx < 0 or idx >= len(mapping):
                        continue
                    chunk = mapping[idx]
                    st.markdown(f"**[FAISS #{rank+1}] {chunk['file_path']} | Chunk {chunk['chunk_index']}**")
                    with st.container():
                        st.text_area(
                            f"FAISS Content #{rank+1}", 
                            chunk["chunk_text"], 
                            height=100, 
                            key=f"faiss_{rank}",
                            disabled=True
                        )
                    st.markdown("---")

            with tab3:
                st.write(f"**Top {keyword_k} Keyword (BM25) Results:**")
                for rank, idx in enumerate(bm25_indices):
                    if idx < 0 or idx >= len(mapping) or bm25_scores[idx] <= 0:
                        continue
                    chunk = mapping[idx]
                    st.markdown(f"**[BM25 #{rank+1}] Score: {bm25_scores[idx]:.4f} | {chunk['file_path']} | Chunk {chunk['chunk_index']}**")
                    with st.container():
                        st.text_area(
                            f"BM25 Content #{rank+1}", 
                            chunk["chunk_text"], 
                            height=100, 
                            key=f"bm25_{rank}",
                            disabled=True
                        )
                    st.markdown("---")

            # === SHOW PERFORMANCE METRICS AT BOTTOM ===
            st.subheader("📊 Performance Metrics")
            
            # Show context stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Chunks Included", f"{context_info['chunks_included']}/{context_info['total_chunks_available']}")
            with col2:
                st.metric("Tokens Used", f"{context_info['tokens_used']}/{context_info['token_limit']}")
            with col3:
                efficiency = (context_info['tokens_used'] / context_info['token_limit']) * 100
                st.metric("Token Efficiency", f"{efficiency:.1f}%")
            with col4:
                response_time = time.time() - start_time
                st.metric("Total Response Time", f"{response_time:.2f}s")
                
        except Exception as e:
            st.error(f"[Error] {e}")

if __name__ == "__main__":
    main()