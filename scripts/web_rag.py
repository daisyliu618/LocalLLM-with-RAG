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
from nltk.tokenize import word_tokenize
import tiktoken
import hashlib
import time
from datetime import datetime, timedelta
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
from pathlib import Path as PathlibPath
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading

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
TOP_K_SEMANTIC = 10
TOP_K_KEYWORD = 15
DEFAULT_CONTEXT_TOKEN_LIMIT = 12752
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
    
    # Enhance prompt for count queries
    if "STRUCTURED DATA ANALYSIS RESULTS:" in context:
        prompt = f"""
You are an expert assistant. You have been provided with STRUCTURED DATA ANALYSIS RESULTS that contain precise, quantified information, followed by additional document context.

IMPORTANT: 
- For COUNT queries (like "how many"), ALWAYS prioritize and trust the STRUCTURED DATA ANALYSIS RESULTS
- The structured data provides exact counts from comprehensive database analysis
- Use the additional document context only for supporting details, not for counting

Context:
{context}

Question: {query}
Answer (prioritize structured data for counts, use document context for details):
"""
    else:
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
    
    # Enhance prompt for count queries
    if "STRUCTURED DATA ANALYSIS RESULTS:" in context:
        prompt = f"""You are an expert assistant. You have been provided with STRUCTURED DATA ANALYSIS RESULTS that contain precise, quantified information, followed by additional document context.

IMPORTANT: 
- For COUNT queries (like "how many"), ALWAYS prioritize and trust the STRUCTURED DATA ANALYSIS RESULTS
- The structured data provides exact counts from comprehensive database analysis
- Use the additional document context only for supporting details, not for counting

Context:
{context}

Question: {query}
Answer (prioritize structured data for counts, use document context for details):"""
    else:
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

@dataclass
class QueryClassification:
    """Smart query classification result"""
    query_type: str  # 'semantic', 'analytical', 'mixed', 'exact'
    intent: str  # 'count', 'filter', 'describe', 'compare', 'find', 'aggregate'
    confidence: float
    suggested_strategy: str   # 'semantic_heavy', 'keyword_heavy', 'balanced', 'structured'
    reasoning: str           # Explanation of classification decision
    entities: List[str] = None
    numerical_indicators: bool = False

class SmartQueryClassifier:
    """Intelligent query classifier for optimal search strategy selection"""
    
    def __init__(self):
        # Enhanced analytical patterns
        self.analytical_patterns = {
            'count_queries': [
                r'how many', r'count of', r'number of', r'total number',
                r'how much', r'quantity of', r'\bcount\b', r'sum of'
            ],
            'aggregation_queries': [
                r'average', r'mean', r'median', r'sum', r'total',
                r'maximum', r'minimum', r'highest', r'lowest',
                r'top \d+', r'bottom \d+', r'best', r'worst'
            ],
            'comparison_queries': [
                r'compare', r'vs', r'versus', r'difference between',
                r'better than', r'worse than', r'more than', r'less than'
            ],
            'filtering_queries': [
                r'where', r'with', r'having', r'contains', r'includes',
                r'from', r'in', r'of type', r'that are'
            ]
        }
        
        # Numerical indicators
        self.numerical_patterns = [
            r'\d+(?:\.\d+)?', r'over \d+', r'under \d+', r'above \d+', r'below \d+',
            r'between \d+ and \d+', r'more than \d+', r'less than \d+'
        ]
        
        # Semantic indicators
        self.semantic_indicators = [
            r'like', r'similar to', r'resembles', r'kind of', r'type of',
            r'style', r'flavor', r'taste', r'notes', r'characteristics',
            r'quality', r'feels like', r'reminds me of', r'comparable to',
            r'concept', r'idea', r'theme', r'meaning'
        ]
        
        # Exact match indicators (favor keyword search)
        self.exact_match_indicators = [
            r'exactly', r'precisely', r'specific', r'named', r'called',
            r'titled', r'brand', r'model', r'id', r'code', r'quote'
        ]
    
    def classify_query(self, query: str) -> QueryClassification:
        """Advanced query classification with smart strategy selection"""
        query_lower = query.lower().strip()
        
        # Initialize scores
        analytical_score = 0
        semantic_score = 0
        exact_match_score = 0
        
        # Count analytical patterns
        for pattern_type, patterns in self.analytical_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    analytical_score += 2 if pattern_type == 'count_queries' else 1
        
        # Count semantic indicators
        for pattern in self.semantic_indicators:
            if re.search(pattern, query_lower):
                semantic_score += 1
        
        # Count exact match indicators
        for pattern in self.exact_match_indicators:
            if re.search(pattern, query_lower):
                exact_match_score += 1
        
        # Check for numerical indicators
        numerical_indicators = any(re.search(pattern, query_lower) for pattern in self.numerical_patterns)
        if numerical_indicators:
            analytical_score += 1
        
        # Extract entities (capitalized words)
        entities = self._extract_entities(query)
        
        # Determine intent
        intent = self._determine_intent(query_lower, analytical_score)
        
        # Smart classification logic
        query_type, suggested_strategy, reasoning = self._smart_classify(
            query_lower, analytical_score, semantic_score, exact_match_score,
            numerical_indicators, entities
        )
        
        # Calculate confidence
        total_signals = analytical_score + semantic_score + exact_match_score
        confidence = min(total_signals / 5.0, 1.0) if total_signals > 0 else 0.4
        
        return QueryClassification(
            query_type=query_type,
            intent=intent,
            confidence=confidence,
            suggested_strategy=suggested_strategy,
            reasoning=reasoning,
            entities=entities,
            numerical_indicators=numerical_indicators
        )
    
    def _smart_classify(self, query: str, analytical_score: int, semantic_score: int, 
                       exact_match_score: int, numerical_indicators: bool, 
                       entities: List[str]) -> Tuple[str, str, str]:
        """Intelligent classification with strategy recommendation"""
        
        # Handle contact information queries first (phone, address, email, etc.)
        if re.search(r'phone number|address|email|website|location|contact', query):
            return ('exact', 'keyword_heavy',
                   f"Contact information query - keyword search is optimal for exact matches")
        
        # Detect CSV/structured data queries
        csv_indicators = ['how many', 'count', 'total number', 'number of']
        needs_csv = any(indicator in query for indicator in csv_indicators)
        
        # Pure analytical queries - especially count queries
        if analytical_score >= 3 and semantic_score == 0:
            if needs_csv:
                return ('analytical', 'structured', 
                       f"COUNT QUERY detected ({analytical_score}) - requires CSV/structured data for accurate results")
            else:
                return ('analytical', 'structured', 
                       f"Strong analytical signals ({analytical_score}) suggest structured search")
        
        # Count queries with entities (like "how many people went to Stanford")
        elif 'how many' in query and len(entities) >= 1:
            return ('analytical', 'structured',
                   f"COUNT QUERY with entities - needs structured data analysis, not document search")
        
        # Pure semantic queries - favor semantic search
        elif semantic_score >= 2 and analytical_score == 0 and not numerical_indicators:
            return ('semantic', 'semantic_heavy',
                   f"Strong semantic indicators ({semantic_score}) suggest conceptual search")
        
        # Exact match queries - favor keyword search
        elif exact_match_score >= 1 and analytical_score == 0:
            return ('exact', 'keyword_heavy',
                   f"Exact match indicators suggest keyword-focused search")
        
        # Mixed analytical with descriptive terms
        elif analytical_score >= 1 and (semantic_score > 0 or len(entities) > 0):
            if needs_csv:
                return ('mixed', 'structured',
                       f"Mixed analytical query - may need CSV data for counting, document search for context")
            else:
                return ('mixed', 'balanced',
                       f"Mixed query with analytical ({analytical_score}) and semantic elements")
        
        # Entity-rich queries
        elif len(entities) >= 2:
            return ('semantic', 'semantic_heavy',
                   f"Multiple entities ({len(entities)}) suggest semantic search")
        
        # Single entity with context
        elif len(entities) == 1 and len(query.split()) > 3:
            return ('semantic', 'balanced',
                   f"Single entity with descriptive context")
        
        # Short, specific queries
        elif len(query.split()) <= 3 and len(entities) > 0:
            return ('exact', 'keyword_heavy',
                   f"Short query with entities suggests keyword search")
        
        # Questions starting with "what", "why", "how" (but not "how many")
        elif re.search(r'^(what|why|how(?! many))', query):
            return ('semantic', 'semantic_heavy',
                   f"Explanatory question suggests semantic search")
        
        # Default to balanced approach
        else:
            return ('mixed', 'balanced',
                   f"Mixed signals, using balanced hybrid approach")
    
    def _determine_intent(self, query: str, analytical_score: int) -> str:
        """Determine the specific intent of the query"""
        # Check for specific information lookups first (phone, address, email, etc.)
        if re.search(r'phone number|address|email|website|location', query):
            return 'find'
        # Then check for counting queries (but be more specific)
        elif re.search(r'how many|count of|total number of|number of.*(?:people|items|things|students|companies)', query):
            return 'count'
        elif re.search(r'compare|vs|versus|difference', query):
            return 'compare'
        elif re.search(r'average|mean|sum|maximum|minimum|highest|lowest', query):
            return 'aggregate'
        elif re.search(r'where|with|having|filter', query):
            return 'filter'
        elif re.search(r'find|show|list|get', query):
            return 'find'
        elif re.search(r'what|why|how|explain|describe', query):
            return 'describe'
        elif analytical_score > 0:
            return 'analyze'
        else:
            return 'describe'
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entity names from query using dynamic pattern detection"""
        words = query.split()
        entities = []
        
        # Define words to exclude from entity extraction
        stop_words = {
            'how', 'what', 'where', 'when', 'why', 'who', 'which', 'whose',
            'many', 'much', 'some', 'any', 'all', 'every', 'each', 'both',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'among',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'shall', 'ought', 'need', 'dare',
            'people', 'person', 'number', 'count', 'total', 'went', 'go', 'goes',
            'going', 'attended', 'attend', 'attends', 'attending', 'studied',
            'study', 'studies', 'studying', 'graduated', 'graduate', 'graduates',
            'graduating', 'phone', 'address', 'email', 'website', 'contact'
        }
        
        # Common institutional/organizational suffixes that indicate entities
        institutional_suffixes = [
            'university', 'college', 'institute', 'school', 'academy',
            'company', 'corporation', 'corp', 'inc', 'llc', 'ltd',
            'restaurant', 'cafe', 'bar', 'pub', 'grill', 'kitchen',
            'hospital', 'clinic', 'center', 'centre', 'foundation',
            'organization', 'group', 'association', 'society'
        ]
        
        # Multi-word entities (quoted strings)
        quoted_entities = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted_entities)
        
        # Dynamic pattern detection for institutional entities
        # Pattern: [Proper Noun(s)] + [Institutional Suffix]
        for suffix in institutional_suffixes:
            pattern = rf'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+{re.escape(suffix)}\b'
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                # Clean up and validate the entity
                entity_words = match.split()
                if all(word.lower() not in stop_words for word in entity_words):
                    entities.append(match.title())
        
        # Pattern: Detect sequences of capitalized words (proper noun phrases)
        # This catches multi-word proper nouns like "Rangoon Ruby", "New York", etc.
        capitalized_sequences = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', query)
        for sequence in capitalized_sequences:
            words_in_sequence = sequence.split()
            # Filter out sequences that contain stop words or common non-entity words
            if (len(words_in_sequence) >= 2 and
                all(word.lower() not in stop_words for word in words_in_sequence) and
                not any(word.lower() in ['many', 'went', 'what', 'phone', 'number'] for word in words_in_sequence)):
                entities.append(sequence)
        
        # Single capitalized words (proper nouns)
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Skip if word is too short or is a stop word
            if len(clean_word) <= 2 or clean_word.lower() in stop_words:
                continue
            
            # Check if it's capitalized
            if clean_word.istitle():
                # Skip first word unless it's clearly not a question word
                if i == 0 and clean_word.lower() in {'how', 'what', 'where', 'when', 'why', 'who', 'which'}:
                    continue
                
                # Additional filtering for quality
                if (len(clean_word) >= 3 and
                    not clean_word.lower().endswith('ing') and
                    not clean_word.lower().endswith('ed') and
                    clean_word.lower() not in {'many', 'what', 'phone', 'went', 'this', 'that', 'they', 'them'}):
                    entities.append(clean_word)
        
        # Detect potential acronyms (2-5 uppercase letters)
        acronyms = re.findall(r'\b([A-Z]{2,5})\b', query)
        for acronym in acronyms:
            if acronym.lower() not in stop_words:
                entities.append(acronym)
        
        # Remove duplicates and filter final results
        entities = list(set(entities))
        
        # Prioritize longer entities and remove shorter fragments
        # Sort by length (longest first) to prioritize complete entity names
        entities.sort(key=len, reverse=True)
        
        filtered_entities = []
        for entity in entities:
            # Skip if it's just a stop word or very short
            if (entity.lower() in stop_words or 
                len(entity) <= 2 or
                entity.lower() in {'how', 'what', 'many', 'went', 'phone', 'number'} or
                all(c.lower() in 'aeiou' for c in entity)):  # Skip vowel-only "words"
                continue
            
            # Skip if this entity is a substring of a longer entity already added
            is_substring = False
            for existing_entity in filtered_entities:
                if entity != existing_entity and entity.lower() in existing_entity.lower():
                    is_substring = True
                    break
            
            if not is_substring:
                filtered_entities.append(entity)
        
        return filtered_entities

# Initialize global classifier
smart_classifier = SmartQueryClassifier()

def get_smart_search_weights(query: str) -> Tuple[float, float, str]:
    """
    Get intelligent search weights based on query classification
    Returns: (semantic_weight, keyword_weight, strategy_info)
    """
    classification = smart_classifier.classify_query(query)
    
    # Map strategy to weights
    if classification.suggested_strategy == 'semantic_heavy':
        semantic_weight, keyword_weight = 0.8, 0.2
    elif classification.suggested_strategy == 'keyword_heavy':
        semantic_weight, keyword_weight = 0.2, 0.8
    elif classification.suggested_strategy == 'structured':
        semantic_weight, keyword_weight = 0.3, 0.7  # Favor keywords for structured
    else:  # balanced
        semantic_weight, keyword_weight = 0.6, 0.4
    
    strategy_info = f"{classification.query_type.title()} query - {classification.reasoning}"
    
    return semantic_weight, keyword_weight, strategy_info

def perform_analytical_query_processing(query: str, merged_results: list, classification) -> str:
    """
    Process analytical queries to provide structured answers
    """
    if classification.intent == 'count' and 'how many' in query.lower():
        # Extract the entity being counted
        entities = classification.entities if classification.entities else []
        
        # Count relevant chunks and extract information
        relevant_chunks = []
        people_mentioned = set()
        locations_mentioned = set()
        
        for chunk in merged_results[:10]:  # Analyze top chunks
            chunk_text = chunk.get('chunk_text', '').lower()
            
            # Look for mentions of the entities in question
            for entity in entities:
                if entity.lower() in chunk_text:
                    relevant_chunks.append(chunk)
                    
                    # Extract potential people names (capitalized words)
                    import re
                    names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', chunk.get('chunk_text', ''))
                    people_mentioned.update(names)
                    
                    # Look for graduation/attendance indicators
                    if any(word in chunk_text for word in ['graduated', 'attended', 'studied', 'degree', 'university']):
                        # This chunk likely contains educational information
                        pass
        
        # Generate analytical summary
        analytical_summary = f"""
🔍 **Analytical Query Processing Results:**

**Query Intent:** Count query - "{query}"

**Entities Found:** {', '.join(entities) if entities else 'None detected'}

**Analysis Results:**
• **Relevant Chunks Found:** {len(relevant_chunks)}
• **People Mentioned:** {len(people_mentioned)} unique individuals
• **Names Identified:** {', '.join(list(people_mentioned)[:5]) if people_mentioned else 'None clearly identified'}

**Raw Data Analysis:**
Based on the document chunks analyzed, I found {len(relevant_chunks)} relevant sections. 
However, this appears to be a structured counting query that would benefit from 
CSV data analysis rather than unstructured document search.

**Recommendation:** For accurate counting queries like "how many people...", 
the system would need access to structured data (CSV files) rather than 
unstructured document chunks.
"""
        return analytical_summary
    
    return None

def discover_all_files() -> Dict[str, List[str]]:
    """
    Automatically discover all supported files in the data directory
    Returns a dictionary with file types as keys and file paths as values
    """
    file_types = {
        'csv': [],
        'pdf': [],
        'txt': [],
        'md': [],
        'docx': [],
        'xlsx': [],
        'json': []
    }
    
    data_dir = Path('data')
    
    if data_dir.exists():
        for file_path in data_dir.rglob('*'):  # Recursive search
            if file_path.is_file():
                extension = file_path.suffix.lower().lstrip('.')
                if extension in file_types:
                    file_types[extension].append(str(file_path))
    
    # Remove empty categories
    return {k: v for k, v in file_types.items() if v}

def analyze_all_files_for_count_query(query: str, classification) -> str:
    """
    Analyze all available files (CSV, PDF, TXT, etc.) for accurate counting queries
    """
    # Discover all files by type
    all_files = discover_all_files()
    
    if not all_files:
        return "❌ No supported files found in data directory for analysis."
    
    results = {}
    query_lower = query.lower()
    entities = classification.entities if classification.entities else []
    
    # Extract search terms from query patterns
    search_terms = entities.copy()
    patterns = [
        r'went to (\w+)', r'attended (\w+)', r'graduated from (\w+)', r'from (\w+)',
        r'about (\w+)', r'mentions (\w+)', r'contains (\w+)', r'discussing (\w+)',
        r'scored over (\d+)', r'score above (\d+)', r'rated over (\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            search_terms.append(match.group(1))
    
    # Remove duplicates and normalize case
    search_terms = list(set([term.lower() for term in search_terms]))
    
    # Dynamic file filtering based on query context and file content analysis
    relevant_files = {}
    
    # For people/entity counting queries, analyze file structure to determine relevance
    if any(indicator in query_lower for indicator in ['how many', 'count', 'people', 'person', 'students', 'individuals']):
        for file_type, file_paths in all_files.items():
            if file_type == 'csv':
                relevant_paths = []
                for path in file_paths:
                    try:
                        # Quick analysis of CSV structure to determine if it contains people/entity data
                        df_sample = pd.read_csv(path, nrows=5)  # Read just first 5 rows for analysis
                        column_names = [col.lower() for col in df_sample.columns]
                        
                        # Check if CSV likely contains people/entity data based on column patterns
                        people_indicators = 0
                        non_people_indicators = 0
                        
                        # Look for column patterns that suggest people data
                        for col in column_names:
                            if any(pattern in col for pattern in ['name', 'person', 'individual', 'student', 'employee', 'country', 'nationality', 'education', 'degree', 'profession', 'job']):
                                people_indicators += 1
                            elif any(pattern in col for pattern in ['price', 'cost', 'score', 'rating', 'product', 'item', 'roaster', 'origin', 'flavor']):
                                non_people_indicators += 1
                        
                        # Include file if it shows signs of containing people data
                        if people_indicators > non_people_indicators:
                            relevant_paths.append(path)
                        elif people_indicators > 0 and non_people_indicators == 0:
                            relevant_paths.append(path)
                        # If uncertain, include for broader search
                        elif people_indicators == 0 and non_people_indicators == 0:
                            relevant_paths.append(path)
                            
                    except Exception:
                        # If can't analyze, include by default
                        relevant_paths.append(path)
                
                if relevant_paths:
                    relevant_files[file_type] = relevant_paths
            else:
                # For non-CSV files, include all by default
                relevant_files[file_type] = file_paths
    else:
        # For non-counting queries, use all files
        relevant_files = all_files

    try:
        # Process each file type
        for file_type, file_paths in relevant_files.items():
            if file_type == 'csv':
                csv_results = analyze_csv_files(file_paths, query_lower, entities, search_terms)
                if csv_results:
                    results.update(csv_results)
            
            elif file_type in ['pdf', 'txt', 'md', 'docx']:
                text_results = analyze_text_files(file_paths, file_type, query_lower, entities, search_terms)
                if text_results:
                    results.update(text_results)
            
            elif file_type == 'json':
                json_results = analyze_json_files(file_paths, query_lower, entities, search_terms)
                if json_results:
                    results.update(json_results)
        
        # Format comprehensive results
        if results:
            output = "📊 **Multi-Format File Analysis Results:**\n\n"
            
            total_matches = 0
            for search_term, data in results.items():
                output += f"**'{search_term}' Analysis:**\n"
                output += f"📁 **Files Searched:** {', '.join(data['file_types'])}\n"
                output += f"🔢 **Total Matches:** {data['total_count']}\n\n"
                
                total_matches += data['total_count']
                
                # Show breakdown by file type
                for file_type, type_data in data['by_file_type'].items():
                    if type_data['count'] > 0:
                        output += f"  **{file_type.upper()} Files ({type_data['count']} matches):**\n"
                        
                        for detail in type_data['details'][:5]:  # Show first 5
                            if file_type == 'csv':
                                # Structured data format
                                record_info = []
                                for key, value in detail.items():
                                    if pd.notna(value) and str(value) != 'nan':
                                        record_info.append(f"{key}: {value}")
                                if record_info:
                                    output += f"    • {' | '.join(record_info)}\n"
                            else:
                                # Text/document format
                                source = detail.get('source', 'Unknown')
                                preview = detail.get('context', '')[:100]
                                output += f"    • {source}: ...{preview}...\n"
                        
                        if len(type_data['details']) > 5:
                            output += f"    • ... and {len(type_data['details']) - 5} more matches\n"
                        output += "\n"
                
                output += f"**✅ Subtotal for '{search_term}': {data['total_count']}**\n"
                output += "---\n\n"
            
            # Overall summary
            output += f"🎯 **GRAND TOTAL ACROSS ALL FILES: {total_matches}**\n\n"
            
            # Show file discovery summary
            file_summary = []
            for file_type, paths in relevant_files.items():
                file_summary.append(f"{len(paths)} {file_type.upper()}")
            output += f"📂 **Files Analyzed:** {', '.join(file_summary)}\n"
            
            return output
        
        else:
            file_summary = ', '.join([f"{len(paths)} {ftype}" for ftype, paths in relevant_files.items()])
            return f"❌ No matching records found in available files ({file_summary}) for this query."
            
    except Exception as e:
        return f"❌ Error analyzing files: {str(e)}"

def analyze_csv_files(file_paths: List[str], query_lower: str, entities: List[str], search_terms: List[str]) -> Dict:
    """Analyze CSV files for count queries"""
    results = {}
    
    for csv_path in file_paths:
        try:
            df = pd.read_csv(csv_path)
            filename = Path(csv_path).stem
            
            for term in search_terms:
                if term not in results:
                    results[term] = {
                        'total_count': 0,
                        'file_types': set(),
                        'by_file_type': {}
                    }
                
                if 'csv' not in results[term]['by_file_type']:
                    results[term]['by_file_type']['csv'] = {'count': 0, 'details': []}
                
                # Search in all text columns
                matches = pd.DataFrame()
                for col in df.select_dtypes(include=['object']).columns:
                    col_matches = df[df[col].astype(str).str.contains(term, case=False, na=False)]
                    matches = pd.concat([matches, col_matches]).drop_duplicates()
                
                if len(matches) > 0:
                    results[term]['total_count'] += len(matches)
                    results[term]['file_types'].add(f"{filename}.csv")
                    results[term]['by_file_type']['csv']['count'] += len(matches)
                    
                    # Add structured details
                    for _, row in matches.head(10).iterrows():
                        detail = {col: row[col] for col in df.columns if pd.notna(row[col])}
                        detail['source_file'] = csv_path
                        results[term]['by_file_type']['csv']['details'].append(detail)
        
        except Exception as e:
            continue
    
    # Convert sets to lists for JSON serialization
    for term in results:
        results[term]['file_types'] = list(results[term]['file_types'])
    
    return results

def analyze_text_files(file_paths: List[str], file_type: str, query_lower: str, entities: List[str], search_terms: List[str]) -> Dict:
    """Analyze text-based files (PDF, TXT, MD, DOCX) for count queries"""
    results = {}
    
    for file_path in file_paths:
        try:
            # Extract text based on file type
            if file_type == 'txt' or file_type == 'md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            
            elif file_type == 'pdf':
                # Would need PDF extraction library like PyPDF2 or pdfplumber
                # For now, skip PDF processing or use existing chunk data
                continue
            
            elif file_type == 'docx':
                # Would need python-docx library
                # For now, skip DOCX processing
                continue
            
            else:
                continue
            
            filename = Path(file_path).stem
            
            # Search for terms in text
            for term in search_terms:
                if term not in results:
                    results[term] = {
                        'total_count': 0,
                        'file_types': set(),
                        'by_file_type': {}
                    }
                
                if file_type not in results[term]['by_file_type']:
                    results[term]['by_file_type'][file_type] = {'count': 0, 'details': []}
                
                # Count occurrences (case-insensitive)
                import re
                matches = re.findall(rf'\b{re.escape(term)}\b', text_content, re.IGNORECASE)
                
                if matches:
                    count = len(matches)
                    results[term]['total_count'] += count
                    results[term]['file_types'].add(f"{filename}.{file_type}")
                    results[term]['by_file_type'][file_type]['count'] += count
                    
                    # Extract context around matches
                    for match in re.finditer(rf'\b{re.escape(term)}\b', text_content, re.IGNORECASE):
                        start = max(0, match.start() - 50)
                        end = min(len(text_content), match.end() + 50)
                        context = text_content[start:end].strip()
                        
                        results[term]['by_file_type'][file_type]['details'].append({
                            'source': f"{filename}.{file_type}",
                            'context': context,
                            'position': match.start()
                        })
                        
                        # Limit to 10 contexts per file
                        if len(results[term]['by_file_type'][file_type]['details']) >= 10:
                            break
        
        except Exception as e:
            continue
    
    # Convert sets to lists
    for term in results:
        results[term]['file_types'] = list(results[term]['file_types'])
    
    return results

def analyze_json_files(file_paths: List[str], query_lower: str, entities: List[str], search_terms: List[str]) -> Dict:
    """Analyze JSON files for count queries"""
    results = {}
    
    for json_path in file_paths:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            filename = Path(json_path).stem
            
            # Convert JSON to searchable text
            json_text = json.dumps(json_data, indent=2).lower()
            
            for term in search_terms:
                if term not in results:
                    results[term] = {
                        'total_count': 0,
                        'file_types': set(),
                        'by_file_type': {}
                    }
                
                if 'json' not in results[term]['by_file_type']:
                    results[term]['by_file_type']['json'] = {'count': 0, 'details': []}
                
                # Count occurrences in JSON text
                import re
                matches = re.findall(rf'\b{re.escape(term.lower())}\b', json_text)
                
                if matches:
                    count = len(matches)
                    results[term]['total_count'] += count
                    results[term]['file_types'].add(f"{filename}.json")
                    results[term]['by_file_type']['json']['count'] += count
                    
                    # Extract relevant JSON paths/values
                    def find_in_json(obj, search_term, path=""):
                        findings = []
                        if isinstance(obj, dict):
                            for key, value in obj.items():
                                new_path = f"{path}.{key}" if path else key
                                if isinstance(value, str) and search_term.lower() in value.lower():
                                    findings.append({'path': new_path, 'value': value})
                                findings.extend(find_in_json(value, search_term, new_path))
                        elif isinstance(obj, list):
                            for i, item in enumerate(obj):
                                new_path = f"{path}[{i}]"
                                findings.extend(find_in_json(item, search_term, new_path))
                        elif isinstance(obj, str) and search_term.lower() in obj.lower():
                            findings.append({'path': path, 'value': obj})
                        return findings
                    
                    findings = find_in_json(json_data, term)
                    for finding in findings[:10]:  # Limit to 10
                        results[term]['by_file_type']['json']['details'].append({
                            'source': f"{filename}.json",
                            'json_path': finding['path'],
                            'context': finding['value'][:100]
                        })
        
        except Exception as e:
            continue
    
    # Convert sets to lists
    for term in results:
        results[term]['file_types'] = list(results[term]['file_types'])
    
    return results

# === ADVANCED COUNT QUERY SYSTEM ===

# File metadata cache
FILE_METADATA_CACHE = Path("output/file_metadata_cache.json")
CACHE_LOCK = threading.Lock()

@dataclass
class FileMetadata:
    """Metadata about a file for intelligent counting"""
    path: str
    size_bytes: int
    row_count: int
    columns: List[str]
    column_types: Dict[str, str]
    people_score: float  # 0-1 score indicating likelihood of containing people data
    last_modified: datetime
    sample_data: Dict[str, Any]
    created_at: datetime

@dataclass
class CountResult:
    """Enhanced result for count queries"""
    total_count: int
    confidence_level: float  # 0-1 confidence in accuracy
    method_used: str  # 'exact', 'sampled', 'cached'
    files_analyzed: List[str]
    breakdown_by_file: Dict[str, int]
    execution_time: float
    sample_size: Optional[int] = None
    estimated_margin_error: Optional[float] = None

class AdvancedCountQuerySystem:
    """Advanced system for fast, accurate count queries with intelligent caching"""
    
    def __init__(self):
        self.metadata_cache = self._load_metadata_cache()
        
    def _load_metadata_cache(self) -> Dict[str, FileMetadata]:
        """Load file metadata cache"""
        if FILE_METADATA_CACHE.exists():
            try:
                with open(FILE_METADATA_CACHE, 'r') as f:
                    cache_data = json.load(f)
                    metadata_dict = {}
                    for path, data in cache_data.items():
                        # Convert datetime strings back to datetime objects
                        if isinstance(data['last_modified'], str):
                            data['last_modified'] = datetime.fromisoformat(data['last_modified'])
                        if isinstance(data['created_at'], str):
                            data['created_at'] = datetime.fromisoformat(data['created_at'])
                        metadata_dict[path] = FileMetadata(**data)
                    return metadata_dict
            except Exception:
                return {}
        return {}
    
    def _save_metadata_cache(self):
        """Save metadata cache to disk"""
        with CACHE_LOCK:
            try:
                FILE_METADATA_CACHE.parent.mkdir(exist_ok=True)
                cache_data = {}
                for path, metadata in self.metadata_cache.items():
                    cache_data[path] = {
                        'path': metadata.path,
                        'size_bytes': metadata.size_bytes,
                        'row_count': metadata.row_count,
                        'columns': metadata.columns,
                        'column_types': metadata.column_types,
                        'people_score': metadata.people_score,
                        'last_modified': metadata.last_modified.isoformat(),
                        'sample_data': metadata.sample_data,
                        'created_at': metadata.created_at.isoformat()
                    }
                
                with open(FILE_METADATA_CACHE, 'w') as f:
                    json.dump(cache_data, f, indent=2, default=str)
            except Exception as e:
                print(f"Warning: Could not save metadata cache: {e}")
    
    def _analyze_file_metadata(self, file_path: str) -> FileMetadata:
        """Analyze a file and create metadata"""
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        stat_info = path_obj.stat()
        last_modified = datetime.fromtimestamp(stat_info.st_mtime)
        
        if file_path.endswith('.csv'):
            return self._analyze_csv_metadata(file_path, stat_info.st_size, last_modified)
        else:
            # For non-CSV files, create basic metadata
            return FileMetadata(
                path=file_path,
                size_bytes=stat_info.st_size,
                row_count=0,
                columns=[],
                column_types={},
                people_score=0.0,
                last_modified=last_modified,
                sample_data={},
                created_at=datetime.now()
            )
    
    def _analyze_csv_metadata(self, file_path: str, size_bytes: int, last_modified: datetime) -> FileMetadata:
        """Analyze CSV file metadata with intelligent scoring"""
        try:
            # Read sample for analysis
            sample_df = pd.read_csv(file_path, nrows=100)
            full_row_count = len(pd.read_csv(file_path))
            
            columns = list(sample_df.columns)
            column_types = {col: str(sample_df[col].dtype) for col in columns}
            
            # Calculate people score using intelligent heuristics
            people_score = self._calculate_people_score(sample_df, columns)
            
            # Extract sample data for context
            sample_data = {
                'first_5_rows': sample_df.head(5).to_dict('records'),
                'column_stats': self._get_column_stats(sample_df)
            }
            
            return FileMetadata(
                path=file_path,
                size_bytes=size_bytes,
                row_count=full_row_count,
                columns=columns,
                column_types=column_types,
                people_score=people_score,
                last_modified=last_modified,
                sample_data=sample_data,
                created_at=datetime.now()
            )
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return FileMetadata(
                path=file_path,
                size_bytes=size_bytes,
                row_count=0,
                columns=[],
                column_types={},
                people_score=0.0,
                last_modified=last_modified,
                sample_data={},
                created_at=datetime.now()
            )
    
    def _calculate_people_score(self, df: pd.DataFrame, columns: List[str]) -> float:
        """Calculate likelihood that this file contains people/entity data"""
        score = 0.0
        total_indicators = 0
        
        # Column name analysis
        people_column_patterns = [
            'name', 'person', 'individual', 'student', 'employee', 'user',
            'first_name', 'last_name', 'full_name', 'given_name', 'surname',
            'country', 'nationality', 'citizenship', 'location', 'address',
            'education', 'degree', 'qualification', 'university', 'college',
            'profession', 'job', 'occupation', 'role', 'title', 'position',
            'age', 'birth', 'gender', 'email', 'phone', 'contact'
        ]
        
        non_people_patterns = [
            'price', 'cost', 'amount', 'value', 'money', 'currency',
            'product', 'item', 'inventory', 'stock', 'catalog',
            'score', 'rating', 'review', 'feedback', 'quality',
            'roaster', 'origin', 'flavor', 'taste', 'aroma',
            'weight', 'size', 'dimension', 'measurement'
        ]
        
        for col in columns:
            col_lower = col.lower()
            total_indicators += 1
            
            # Check for people indicators
            people_matches = sum(1 for pattern in people_column_patterns if pattern in col_lower)
            non_people_matches = sum(1 for pattern in non_people_patterns if pattern in col_lower)
            
            if people_matches > 0:
                score += people_matches * 2  # Weight people indicators more
            if non_people_matches > 0:
                score -= non_people_matches
        
        # Content analysis
        for col in df.select_dtypes(include=['object']).columns:
            try:
                sample_values = df[col].dropna().astype(str).head(20)
                
                # Check for name patterns
                name_patterns = sum(1 for val in sample_values 
                                  if len(val.split()) >= 2 and val.istitle())
                if name_patterns > len(sample_values) * 0.3:  # 30% threshold
                    score += 3
                
                # Check for country/location patterns
                location_indicators = ['university', 'college', 'usa', 'japan', 'china', 
                                     'uk', 'canada', 'germany', 'france', 'australia']
                location_matches = sum(1 for val in sample_values 
                                     for indicator in location_indicators 
                                     if indicator.lower() in val.lower())
                if location_matches > 0:
                    score += 2
                    
            except Exception:
                continue
        
        # Normalize score
        max_possible_score = len(columns) * 2 + 10  # Rough estimate
        normalized_score = min(score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
        
        return max(0.0, normalized_score)
    
    def _get_column_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistical information about columns"""
        stats = {}
        for col in df.columns:
            try:
                if df[col].dtype in ['object']:
                    stats[col] = {
                        'type': 'text',
                        'unique_count': df[col].nunique(),
                        'null_count': df[col].isnull().sum(),
                        'sample_values': df[col].dropna().head(3).tolist()
                    }
                else:
                    stats[col] = {
                        'type': 'numeric',
                        'unique_count': df[col].nunique(),
                        'null_count': df[col].isnull().sum(),
                        'min': float(df[col].min()) if pd.notna(df[col].min()) else None,
                        'max': float(df[col].max()) if pd.notna(df[col].max()) else None
                    }
            except Exception:
                stats[col] = {'type': 'unknown', 'error': True}
        
        return stats
    
    def get_file_metadata(self, file_path: str, force_refresh: bool = False) -> FileMetadata:
        """Get or update file metadata"""
        abs_path = str(Path(file_path).absolute())
        
        # Check if we need to refresh metadata
        needs_refresh = (
            force_refresh or 
            abs_path not in self.metadata_cache or
            not Path(abs_path).exists()
        )
        
        if not needs_refresh:
            cached_metadata = self.metadata_cache[abs_path]
            file_modified = datetime.fromtimestamp(Path(abs_path).stat().st_mtime)
            if file_modified > cached_metadata.last_modified:
                needs_refresh = True
        
        if needs_refresh:
            metadata = self._analyze_file_metadata(abs_path)
            self.metadata_cache[abs_path] = metadata
            self._save_metadata_cache()
            return metadata
        
        return self.metadata_cache[abs_path]
    
    def execute_count_query(self, query: str, classification, max_execution_time: float = 30.0) -> CountResult:
        """Execute an optimized count query with multiple strategies"""
        start_time = time.time()
        
        # Discover and analyze files
        all_files = discover_all_files()
        relevant_files = self._filter_relevant_files(query, classification, all_files)
        
        # Get metadata for all relevant files
        file_metadata = {}
        for file_type, file_paths in relevant_files.items():
            if file_type == 'csv':  # Focus on CSV files for structured counting
                for file_path in file_paths:
                    try:
                        metadata = self.get_file_metadata(file_path)
                        file_metadata[file_path] = metadata
                    except Exception as e:
                        print(f"Warning: Could not analyze {file_path}: {e}")
        
        # Extract search terms
        search_terms = self._extract_search_terms(query, classification)
        
        # Choose execution strategy based on file sizes and time constraints
        strategy = self._choose_execution_strategy(file_metadata, max_execution_time)
        
        # Execute count with chosen strategy
        if strategy == 'exact':
            return self._execute_exact_count(search_terms, file_metadata, start_time)
        elif strategy == 'sampled':
            return self._execute_sampled_count(search_terms, file_metadata, start_time)
        else:  # cached/approximate
            return self._execute_cached_count(search_terms, file_metadata, start_time)
    
    def _filter_relevant_files(self, query: str, classification, all_files: Dict) -> Dict:
        """Filter files based on query context and metadata"""
        query_lower = query.lower()
        relevant_files = {}
        
        # Determine if this is a people-related query
        is_people_query = any(indicator in query_lower for indicator in 
                            ['people', 'person', 'students', 'individuals', 'employees', 'users'])
        
        for file_type, file_paths in all_files.items():
            if file_type == 'csv':
                relevant_paths = []
                for file_path in file_paths:
                    try:
                        metadata = self.get_file_metadata(file_path)
                        
                        if is_people_query:
                            # For people queries, be more selective - only include files with strong people indicators
                            # Higher threshold and must have more people indicators than non-people indicators
                            if metadata.people_score > 0.6:  # Higher threshold for people-related content
                                relevant_paths.append(file_path)
                            elif metadata.people_score > 0.4:
                                # Additional check: look at column names for strong people indicators
                                # Exclude product-related terms that might be confused with people terms
                                strong_people_indicators = [
                                    col for col in metadata.columns 
                                    if any(indicator in col.lower() for indicator in 
                                          ['first_name', 'last_name', 'full_name', 'student', 'employee', 
                                           'education', 'degree', 'profession', 'job', 'birth', 'age'])
                                    and not any(product_term in col.lower() for product_term in 
                                              ['coffee', 'product', 'item', 'price', 'score', 'rating'])
                                ]
                                if len(strong_people_indicators) >= 1:  # Must have at least 1 strong people indicator
                                    relevant_paths.append(file_path)
                        else:
                            # For other queries, include all files but rank by relevance
                            relevant_paths.append(file_path)
                            
                    except Exception:
                        # If analysis fails, include by default
                        relevant_paths.append(file_path)
                
                if relevant_paths:
                    # Sort by people score if it's a people query
                    if is_people_query:
                        relevant_paths.sort(key=lambda p: self.metadata_cache.get(
                            str(Path(p).absolute()), FileMetadata('', 0, 0, [], {}, 0.0, datetime.now(), {}, datetime.now())
                        ).people_score, reverse=True)
                    
                    relevant_files[file_type] = relevant_paths
            else:
                relevant_files[file_type] = file_paths
        
        return relevant_files
    
    def _extract_search_terms(self, query: str, classification) -> List[str]:
        """Extract and normalize search terms"""
        search_terms = []
        
        # Get entities from classification
        if classification.entities:
            search_terms.extend(classification.entities)
        
        # Extract terms from common patterns
        patterns = [
            r'from (\w+)', r'in (\w+)', r'at (\w+)', r'of (\w+)',
            r'went to ([\w\s]+)', r'attended ([\w\s]+)', r'graduated from ([\w\s]+)'
        ]
        
        query_lower = query.lower()
        for pattern in patterns:
            matches = re.findall(pattern, query_lower)
            search_terms.extend(matches)
        
        # Clean and normalize
        search_terms = [term.strip().lower() for term in search_terms if term.strip()]
        return list(set(search_terms))  # Remove duplicates
    
    def _choose_execution_strategy(self, file_metadata: Dict, max_time: float) -> str:
        """Choose execution strategy based on data size and time constraints"""
        total_rows = sum(metadata.row_count for metadata in file_metadata.values())
        total_size = sum(metadata.size_bytes for metadata in file_metadata.values())
        
        # Simple heuristics for strategy selection
        if total_rows < 100000:  # Small dataset
            return 'exact'
        elif total_rows < 1000000 and total_size < 100 * 1024 * 1024:  # Medium dataset
            return 'exact'
        elif max_time > 10.0:  # Have time for sampling
            return 'sampled'
        else:  # Use cached/approximate
            return 'cached'
    
    def _execute_exact_count(self, search_terms: List[str], file_metadata: Dict, start_time: float) -> CountResult:
        """Execute exact count on all data"""
        total_count = 0
        files_analyzed = []
        breakdown = {}
        
        for file_path, metadata in file_metadata.items():
            try:
                df = pd.read_csv(file_path)
                file_count = 0
                
                for term in search_terms:
                    # Search across all text columns
                    for col in df.select_dtypes(include=['object']).columns:
                        matches = df[df[col].astype(str).str.contains(term, case=False, na=False)]
                        file_count += len(matches.drop_duplicates())
                
                total_count += file_count
                breakdown[file_path] = file_count
                files_analyzed.append(file_path)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return CountResult(
            total_count=total_count,
            confidence_level=1.0,
            method_used='exact',
            files_analyzed=files_analyzed,
            breakdown_by_file=breakdown,
            execution_time=time.time() - start_time
        )
    
    def _execute_sampled_count(self, search_terms: List[str], file_metadata: Dict, start_time: float) -> CountResult:
        """Execute count using statistical sampling"""
        total_count = 0
        files_analyzed = []
        breakdown = {}
        total_sample_size = 0
        
        for file_path, metadata in file_metadata.items():
            try:
                # Calculate sample size (minimum 1000 rows or 10% of data)
                sample_size = min(max(1000, metadata.row_count // 10), metadata.row_count)
                total_sample_size += sample_size
                
                # Read sample
                df_sample = pd.read_csv(file_path, nrows=sample_size)
                sample_count = 0
                
                for term in search_terms:
                    for col in df_sample.select_dtypes(include=['object']).columns:
                        matches = df_sample[df_sample[col].astype(str).str.contains(term, case=False, na=False)]
                        sample_count += len(matches.drop_duplicates())
                
                # Extrapolate to full dataset
                extrapolated_count = int(sample_count * (metadata.row_count / sample_size))
                total_count += extrapolated_count
                breakdown[file_path] = extrapolated_count
                files_analyzed.append(file_path)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Calculate confidence and margin of error
        confidence = 0.85  # 85% confidence for sampling
        margin_error = 0.1   # ±10% estimated margin of error
        
        return CountResult(
            total_count=total_count,
            confidence_level=confidence,
            method_used='sampled',
            files_analyzed=files_analyzed,
            breakdown_by_file=breakdown,
            execution_time=time.time() - start_time,
            sample_size=total_sample_size,
            estimated_margin_error=margin_error
        )
    
    def _execute_cached_count(self, search_terms: List[str], file_metadata: Dict, start_time: float) -> CountResult:
        """Execute approximate count using cached metadata"""
        total_count = 0
        files_analyzed = []
        breakdown = {}
        
        for file_path, metadata in file_metadata.items():
            # Use sample data from metadata to estimate
            estimated_count = 0
            if metadata.sample_data and 'first_5_rows' in metadata.sample_data:
                sample_rows = metadata.sample_data['first_5_rows']
                sample_matches = 0
                
                for row in sample_rows:
                    for term in search_terms:
                        if any(term.lower() in str(value).lower() for value in row.values()):
                            sample_matches += 1
                            break  # Count each row only once
                
                # Rough extrapolation
                if len(sample_rows) > 0:
                    estimated_count = int(sample_matches * (metadata.row_count / len(sample_rows)))
            
            total_count += estimated_count
            breakdown[file_path] = estimated_count
            files_analyzed.append(file_path)
        
        return CountResult(
            total_count=total_count,
            confidence_level=0.6,  # Lower confidence for cached estimates
            method_used='cached',
            files_analyzed=files_analyzed,
            breakdown_by_file=breakdown,
            execution_time=time.time() - start_time
        )

# Initialize global advanced count system
advanced_count_system = AdvancedCountQuerySystem()

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
                # === SMART QUERY CLASSIFICATION ===
                classification = smart_classifier.classify_query(query)
                semantic_weight, keyword_weight, strategy_info = get_smart_search_weights(query)
                
                # === SPECIAL HANDLING FOR COUNT QUERIES ===
                csv_analysis_result = None  # Initialize variable for later use
                if classification.intent == 'count' and 'how many' in query.lower():
                    with st.spinner("🔍 Running advanced count analysis..."):
                        # Use the new advanced count system
                        count_result = advanced_count_system.execute_count_query(query, classification)
                        
                        # Format results for display
                        if count_result.total_count > 0:
                            analysis_text = f"""📊 **Advanced Count Analysis Results:**

**🎯 Total Count: {count_result.total_count}**

**📈 Analysis Details:**
• **Method Used:** {count_result.method_used.title()}
• **Confidence Level:** {count_result.confidence_level*100:.1f}%
• **Execution Time:** {count_result.execution_time:.2f} seconds
• **Files Analyzed:** {len(count_result.files_analyzed)}

**📋 Breakdown by File:**"""

                            for file_path, count in count_result.breakdown_by_file.items():
                                filename = Path(file_path).name
                                analysis_text += f"\n• **{filename}:** {count} matches"
                            
                            if count_result.method_used == 'sampled' and count_result.sample_size:
                                analysis_text += f"\n\n**📊 Sampling Details:**"
                                analysis_text += f"\n• Sample Size: {count_result.sample_size:,} rows"
                                if count_result.estimated_margin_error:
                                    analysis_text += f"\n• Estimated Margin of Error: ±{count_result.estimated_margin_error*100:.1f}%"
                            
                            # Show results
                            st.success(f"### 📊 Structured Data Analysis:\n\n{analysis_text}")
                            st.info("**Additional context from document search below...**")
                            
                            # Store for LLM context
                            csv_analysis_result = analysis_text
                
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
                
                # --- Smart Result Merging Based on Query Classification ---
                seen = set()
                merged_results = []
                
                # Apply intelligent weighting based on query classification
                if semantic_weight >= 0.7:  # Semantic-heavy queries
                    # Prioritize semantic results, add keyword as backup
                    for res in faiss_results:
                        key = (res['file_path'], res['chunk_index'])
                        if key not in seen:
                            merged_results.append(res)
                            seen.add(key)
                    
                    # Add high-scoring keyword results as backup
                    high_score_threshold = 1.5
                    for idx in bm25_indices:
                        if idx < 0 or idx >= len(mapping) or bm25_scores[idx] <= 0:
                            continue
                        if bm25_scores[idx] >= high_score_threshold:
                            res = mapping[idx]
                            key = (res['file_path'], res['chunk_index'])
                            if key not in seen:
                                merged_results.append(res)
                                seen.add(key)
                
                elif keyword_weight >= 0.7:  # Keyword-heavy queries
                    # Prioritize keyword results, add semantic as backup
                    for idx in bm25_indices:
                        if idx < 0 or idx >= len(mapping) or bm25_scores[idx] <= 0:
                            continue
                        res = mapping[idx]
                        key = (res['file_path'], res['chunk_index'])
                        if key not in seen:
                            merged_results.append(res)
                            seen.add(key)
                    
                    # Add semantic results as backup
                    for res in faiss_results:
                        key = (res['file_path'], res['chunk_index'])
                        if key not in seen and len(merged_results) < semantic_k + keyword_k:
                            merged_results.append(res)
                            seen.add(key)
                
                else:  # Balanced approach
                    # Interleave results based on relative weights
                    semantic_ratio = semantic_weight / (semantic_weight + keyword_weight)
                    
                    # First, add very high-scoring BM25 results (exact matches)
                    high_score_threshold = 3.0
                    for idx in bm25_indices:
                        if idx < 0 or idx >= len(mapping) or bm25_scores[idx] <= 0:
                            continue
                        if bm25_scores[idx] >= high_score_threshold:
                            res = mapping[idx]
                            key = (res['file_path'], res['chunk_index'])
                            if key not in seen:
                                merged_results.append(res)
                                seen.add(key)
                    
                    # Then interleave based on weights
                    semantic_idx = 0
                    keyword_idx = 0
                    total_added = len(merged_results)
                    
                    while total_added < semantic_k + keyword_k and (semantic_idx < len(faiss_results) or keyword_idx < len(bm25_results)):
                        # Decide whether to add semantic or keyword result
                        should_add_semantic = (
                            semantic_idx < len(faiss_results) and 
                            (len([r for r in merged_results if r in faiss_results]) / max(len(merged_results), 1)) < semantic_ratio
                        ) or keyword_idx >= len(bm25_results)
                        
                        if should_add_semantic and semantic_idx < len(faiss_results):
                            res = faiss_results[semantic_idx]
                            key = (res['file_path'], res['chunk_index'])
                            if key not in seen:
                                merged_results.append(res)
                                seen.add(key)
                                total_added += 1
                            semantic_idx += 1
                        elif keyword_idx < len(bm25_results):
                            idx = bm25_indices[keyword_idx]
                            if idx >= 0 and idx < len(mapping) and bm25_scores[idx] > 0:
                                res = mapping[idx]
                                key = (res['file_path'], res['chunk_index'])
                                if key not in seen:
                                    merged_results.append(res)
                                    seen.add(key)
                                    total_added += 1
                            keyword_idx += 1
                        else:
                            break

            # --- Prepare context for LLM ---
            with st.spinner(f"Building context and generating answer with {current_provider}..."):
                context, context_info = build_context(merged_results, context_limit)
                
                # Add advanced count analysis results to context for count queries if available
                if csv_analysis_result is not None:
                    context = f"STRUCTURED DATA ANALYSIS RESULTS:\n{csv_analysis_result}\n\nADDITIONAL DOCUMENT CONTEXT:\n{context}"
                
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