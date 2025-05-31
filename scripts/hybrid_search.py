import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import faiss
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import tiktoken

# Ensure NLTK punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

@dataclass
class SearchResult:
    """Unified search result structure"""
    content: str
    score: float
    source: str
    row_data: Dict[str, Any]
    search_type: str  # 'semantic', 'keyword', 'structured', 'hybrid'
    metadata: Dict[str, Any] = None

@dataclass
class QueryClassification:
    """Enhanced query classification result"""
    query_type: str  # 'semantic', 'analytical', 'mixed'
    intent: str  # 'count', 'filter', 'describe', 'compare', 'find', 'aggregate'
    entities: List[str]
    numerical_filters: Dict[str, Any]
    confidence: float
    suggested_alpha: float  # Recommended alpha for hybrid search
    preferred_method: str   # 'semantic', 'keyword', 'structured', 'hybrid'
    reasoning: str         # Explanation of classification decision

class SmartQueryClassifier:
    """Advanced query classifier for intelligent search strategy selection"""
    
    def __init__(self, csv_columns: List[str]):
        self.csv_columns = csv_columns
        
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
        
        # Numerical filter patterns
        self.numerical_patterns = [
            (r'over (\d+(?:\.\d+)?)', 'gt'),
            (r'above (\d+(?:\.\d+)?)', 'gt'),
            (r'more than (\d+(?:\.\d+)?)', 'gt'),
            (r'greater than (\d+(?:\.\d+)?)', 'gt'),
            (r'> *(\d+(?:\.\d+)?)', 'gt'),
            (r'under (\d+(?:\.\d+)?)', 'lt'),
            (r'below (\d+(?:\.\d+)?)', 'lt'),
            (r'less than (\d+(?:\.\d+)?)', 'lt'),
            (r'< *(\d+(?:\.\d+)?)', 'lt'),
            (r'between (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?)', 'between'),
            (r'from (\d+(?:\.\d+)?) to (\d+(?:\.\d+)?)', 'between'),
            (r'(\d+(?:\.\d+)?) to (\d+(?:\.\d+)?)', 'between'),
            (r'equals? (\d+(?:\.\d+)?)', 'eq'),
            (r'= *(\d+(?:\.\d+)?)', 'eq'),
        ]
        
        # Semantic indicators
        self.semantic_indicators = [
            r'like', r'similar to', r'resembles', r'kind of', r'type of',
            r'style', r'flavor', r'taste', r'notes', r'characteristics',
            r'quality', r'feels like', r'reminds me of', r'comparable to'
        ]
        
        # Exact match indicators (favor keyword search)
        self.exact_match_indicators = [
            r'exactly', r'precisely', r'specific', r'named', r'called',
            r'titled', r'brand', r'model', r'id', r'code'
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
        
        # Extract numerical filters
        numerical_filters = self._extract_numerical_filters(query_lower)
        if numerical_filters:
            analytical_score += 2
        
        # Extract entities and check against column names
        entities = self._extract_entities(query)
        column_matches = self._find_column_matches(query_lower)
        
        # Determine intent with higher granularity
        intent = self._determine_intent(query_lower, analytical_score)
        
        # Smart classification logic
        query_type, preferred_method, suggested_alpha, reasoning = self._smart_classify(
            query_lower, analytical_score, semantic_score, exact_match_score,
            numerical_filters, entities, column_matches
        )
        
        # Calculate confidence
        total_signals = analytical_score + semantic_score + exact_match_score
        confidence = min(total_signals / 5.0, 1.0) if total_signals > 0 else 0.3
        
        return QueryClassification(
            query_type=query_type,
            intent=intent,
            entities=entities,
            numerical_filters=numerical_filters,
            confidence=confidence,
            suggested_alpha=suggested_alpha,
            preferred_method=preferred_method,
            reasoning=reasoning
        )
    
    def _smart_classify(self, query: str, analytical_score: int, semantic_score: int, 
                       exact_match_score: int, numerical_filters: Dict, 
                       entities: List[str], column_matches: List[str]) -> Tuple[str, str, float, str]:
        """Intelligent classification with reasoning"""
        
        # Pure analytical queries
        if analytical_score >= 3 and semantic_score == 0:
            return ('analytical', 'structured', 0.2, 
                   f"Strong analytical signals ({analytical_score}) with numerical filters")
        
        # Pure semantic queries
        elif semantic_score >= 2 and analytical_score == 0 and not numerical_filters:
            return ('semantic', 'semantic', 0.9,
                   f"Strong semantic indicators ({semantic_score}) suggesting conceptual search")
        
        # Exact match queries
        elif exact_match_score >= 1 and analytical_score == 0:
            return ('semantic', 'keyword', 0.1,
                   f"Exact match indicators suggest keyword search")
        
        # Count with descriptive terms (e.g., "how many Ethiopian coffee")
        elif analytical_score >= 1 and (semantic_score > 0 or len(entities) > 0):
            return ('mixed', 'hybrid', 0.4,
                   f"Analytical query ({analytical_score}) with descriptive elements")
        
        # Column name matching with values
        elif len(column_matches) > 0 and numerical_filters:
            return ('analytical', 'structured', 0.3,
                   f"Direct column references with numerical constraints")
        
        # Entity-rich queries
        elif len(entities) >= 2:
            return ('semantic', 'hybrid', 0.6,
                   f"Multiple entities detected, favoring semantic with keyword backup")
        
        # Single entity with descriptive terms
        elif len(entities) == 1 and len(query.split()) > 3:
            return ('semantic', 'hybrid', 0.7,
                   f"Single entity with descriptive context")
        
        # Short, specific queries
        elif len(query.split()) <= 3 and len(entities) > 0:
            return ('semantic', 'keyword', 0.3,
                   f"Short query with entities suggests keyword search")
        
        # Default to balanced hybrid
        else:
            return ('mixed', 'hybrid', 0.6,
                   f"Mixed signals, using balanced hybrid approach")
    
    def _determine_intent(self, query: str, analytical_score: int) -> str:
        """Determine the specific intent of the query"""
        if re.search(r'how many|count|number of|total number', query):
            return 'count'
        elif re.search(r'compare|vs|versus|difference', query):
            return 'compare'
        elif re.search(r'average|mean|sum|maximum|minimum|highest|lowest', query):
            return 'aggregate'
        elif re.search(r'where|with|having|filter', query):
            return 'filter'
        elif re.search(r'find|show|list|get', query):
            return 'find'
        elif analytical_score > 0:
            return 'analyze'
        else:
            return 'describe'
    
    def _find_column_matches(self, query: str) -> List[str]:
        """Find column names mentioned in the query"""
        matches = []
        for col in self.csv_columns:
            col_variations = [
                col.lower(),
                col.lower().replace('_', ' '),
                col.lower().replace('-', ' '),
                ''.join(col.lower().split())
            ]
            
            for variation in col_variations:
                if variation in query and len(variation) > 2:
                    matches.append(col)
                    break
        
        return matches
    
    def _extract_numerical_filters(self, query: str) -> Dict[str, Any]:
        """Enhanced numerical filter extraction"""
        filters = {}
        
        for pattern, op in self.numerical_patterns:
            matches = re.findall(pattern, query)
            if matches:
                if op == 'between':
                    if isinstance(matches[0], tuple) and len(matches[0]) == 2:
                        filters[op] = [float(matches[0][0]), float(matches[0][1])]
                    else:
                        continue
                else:
                    filters[op] = float(matches[0])
                break
        
        return filters
    
    def _extract_entities(self, query: str) -> List[str]:
        """Enhanced entity extraction"""
        words = query.split()
        entities = []
        
        # Capitalized words (proper nouns)
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word.istitle() and len(clean_word) > 2:
                entities.append(clean_word)
        
        # Multi-word entities (quoted strings)
        quoted_entities = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted_entities)
        
        # Common location patterns
        location_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+([A-Z][a-z]+)\b',  # City, Country
            r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'       # Three-word places
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                entities.append(' '.join(match))
        
        return list(set(entities))  # Remove duplicates

class CSVHybridSearchEngine:
    """
    Enhanced hybrid search engine with smart query classification
    """
    
    def __init__(self, csv_file_path: str, text_columns: List[str] = None, 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.csv_path = Path(csv_file_path)
        self.df = pd.read_csv(csv_file_path)
        self.text_columns = text_columns or self._auto_detect_text_columns()
        self.embedding_model_name = embedding_model
        self.embedder = SentenceTransformer(embedding_model)
        
        # Initialize smart classifier
        self.classifier = SmartQueryClassifier(self.df.columns.tolist())
        
        # Search indices
        self.vector_index = None
        self.bm25_index = None
        self.text_chunks = []
        self.chunk_metadata = []
        
        self.setup_indices()
    
    def _auto_detect_text_columns(self) -> List[str]:
        """Auto-detect text columns for semantic search"""
        text_cols = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Check if column contains meaningful text (not just categories)
                sample_vals = self.df[col].dropna().head(10)
                avg_length = sample_vals.astype(str).str.len().mean()
                if avg_length > 10:  # Assume columns with avg length > 10 chars are text
                    text_cols.append(col)
        return text_cols
    
    def setup_indices(self):
        """Set up all search indices"""
        self._create_text_chunks()
        self._build_vector_index()
        self._build_bm25_index()
    
    def _create_text_chunks(self):
        """Create searchable text chunks from CSV rows"""
        self.text_chunks = []
        self.chunk_metadata = []
        
        for idx, row in self.df.iterrows():
            # Create comprehensive text representation
            text_parts = []
            
            # Add text columns
            for col in self.text_columns:
                if pd.notna(row[col]):
                    text_parts.append(f"{col}: {row[col]}")
            
            # Add numerical columns with context
            for col in self.df.columns:
                if col not in self.text_columns and pd.notna(row[col]):
                    text_parts.append(f"{col}: {row[col]}")
            
            chunk_text = " | ".join(text_parts)
            self.text_chunks.append(chunk_text)
            self.chunk_metadata.append({
                'row_index': idx,
                'row_data': row.to_dict()
            })
    
    def _build_vector_index(self):
        """Build FAISS vector index for semantic search"""
        if not self.text_chunks:
            return
        
        # Generate embeddings
        embeddings = self.embedder.encode(self.text_chunks, show_progress_bar=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.vector_index.add(embeddings.astype('float32'))
    
    def _build_bm25_index(self):
        """Build BM25 index for keyword search"""
        tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in self.text_chunks]
        self.bm25_index = BM25Okapi(tokenized_chunks)
    
    def classify_query(self, query: str) -> QueryClassification:
        """Use the enhanced smart classifier"""
        return self.classifier.classify_query(query)
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Perform semantic search using vector similarity"""
        if self.vector_index is None:
            return []
        
        # Encode query
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.vector_index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                metadata = self.chunk_metadata[idx]
                results.append(SearchResult(
                    content=self.text_chunks[idx],
                    score=float(score),
                    source=f"Row {metadata['row_index']}",
                    row_data=metadata['row_data'],
                    search_type='semantic'
                ))
        
        return results
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Perform keyword search using BM25"""
        if self.bm25_index is None:
            return []
        
        # Tokenize query
        query_tokens = word_tokenize(query.lower())
        
        # Get scores for all documents
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include relevant results
                metadata = self.chunk_metadata[idx]
                results.append(SearchResult(
                    content=self.text_chunks[idx],
                    score=float(scores[idx]),
                    source=f"Row {metadata['row_index']}",
                    row_data=metadata['row_data'],
                    search_type='keyword'
                ))
        
        return results
    
    def structured_search(self, query: str, classification: QueryClassification) -> List[SearchResult]:
        """Perform structured search using pandas operations"""
        results = []
        df_filtered = self.df.copy()
        
        # Apply numerical filters
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_lower = col.lower()
                if any(term in query.lower() for term in [col_lower, col_lower.replace('_', ' ')]):
                    # Apply numerical filters to this column
                    if 'gt' in classification.numerical_filters:
                        df_filtered = df_filtered[df_filtered[col] > classification.numerical_filters['gt']]
                    elif 'lt' in classification.numerical_filters:
                        df_filtered = df_filtered[df_filtered[col] < classification.numerical_filters['lt']]
                    elif 'between' in classification.numerical_filters:
                        bounds = classification.numerical_filters['between']
                        df_filtered = df_filtered[
                            (df_filtered[col] >= bounds[0]) & (df_filtered[col] <= bounds[1])
                        ]
        
        # Apply text filters for entities
        for entity in classification.entities:
            for col in self.text_columns:
                if self.df[col].dtype == 'object':
                    mask = df_filtered[col].str.contains(entity, case=False, na=False)
                    if mask.any():
                        df_filtered = df_filtered[mask]
                        break
        
        # Handle count queries
        if classification.intent == 'count':
            count = len(df_filtered)
            results.append(SearchResult(
                content=f"Found {count} records matching the criteria",
                score=1.0,
                source="Structured Query",
                row_data={'count': count, 'query': query},
                search_type='structured'
            ))
        else:
            # Return filtered rows
            for idx, row in df_filtered.head(10).iterrows():
                text_repr = " | ".join([f"{col}: {row[col]}" for col in self.df.columns if pd.notna(row[col])])
                results.append(SearchResult(
                    content=text_repr,
                    score=1.0,
                    source=f"Row {idx}",
                    row_data=row.to_dict(),
                    search_type='structured'
                ))
        
        return results
    
    def rerank_results(self, results: List[SearchResult], query: str, alpha: float = 0.7) -> List[SearchResult]:
        """Rerank combined results using Reciprocal Rank Fusion (RRF)"""
        if not results:
            return results
        
        # Group results by search type
        semantic_results = [r for r in results if r.search_type == 'semantic']
        keyword_results = [r for r in results if r.search_type == 'keyword']
        structured_results = [r for r in results if r.search_type == 'structured']
        
        # Apply RRF formula: score = 1 / (k + rank) where k=60 is standard
        k = 60
        final_scores = {}
        
        # Process each result type
        for result_list, weight in [(semantic_results, alpha), 
                                   (keyword_results, 1-alpha), 
                                   (structured_results, 1.0)]:
            for rank, result in enumerate(result_list):
                result_id = id(result)
                if result_id not in final_scores:
                    final_scores[result_id] = {
                        'result': result,
                        'rrf_score': 0.0
                    }
                final_scores[result_id]['rrf_score'] += weight * (1.0 / (k + rank + 1))
        
        # Sort by RRF score
        reranked = sorted(final_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
        
        # Update scores and return
        final_results = []
        for item in reranked:
            result = item['result']
            result.score = item['rrf_score']
            final_results.append(result)
        
        return final_results
    
    def hybrid_search(self, query: str, top_k: int = 10, alpha: float = None) -> List[SearchResult]:
        """
        Enhanced hybrid search with smart strategy selection
        
        Args:
            query: Search query
            top_k: Number of results to return  
            alpha: Weight for semantic vs keyword search (auto-determined if None)
        """
        # Classify query and get smart recommendations
        classification = self.classify_query(query)
        
        # Use suggested alpha if not provided
        if alpha is None:
            alpha = classification.suggested_alpha
        
        all_results = []
        
        # Choose search strategy based on classification
        if classification.preferred_method == 'semantic':
            # Pure semantic search
            semantic_results = self.semantic_search(query, top_k)
            all_results.extend(semantic_results)
            
        elif classification.preferred_method == 'keyword':
            # Pure keyword search
            keyword_results = self.keyword_search(query, top_k)
            all_results.extend(keyword_results)
            
        elif classification.preferred_method == 'structured':
            # Structured search with semantic backup
            structured_results = self.structured_search(query, classification)
            all_results.extend(structured_results)
            
            # Add semantic results if structured doesn't find enough
            if len(structured_results) < 3:
                semantic_results = self.semantic_search(query, top_k)
                all_results.extend(semantic_results)
                
        else:  # hybrid
            # Full hybrid search
            semantic_results = self.semantic_search(query, top_k)
            keyword_results = self.keyword_search(query, top_k)
            all_results.extend(semantic_results)
            all_results.extend(keyword_results)
            
            # Add structured search for analytical queries
            if classification.query_type in ['analytical', 'mixed']:
                structured_results = self.structured_search(query, classification)
                all_results.extend(structured_results)
        
        # Rerank and deduplicate
        reranked_results = self.rerank_results(all_results, query, alpha)
        
        # Remove duplicates based on row_index (prefer higher scored duplicates)
        seen_rows = set()
        final_results = []
        
        for result in reranked_results:
            # Always include count/aggregate results
            if result.search_type == 'structured' and 'count' in result.row_data:
                final_results.append(result)
            else:
                row_idx = result.row_data.get('row_index', id(result))
                if row_idx not in seen_rows:
                    seen_rows.add(row_idx)
                    final_results.append(result)
                    
                    # Add classification info to result metadata
                    if result.metadata is None:
                        result.metadata = {}
                    result.metadata.update({
                        'query_classification': classification.query_type,
                        'search_strategy': classification.preferred_method,
                        'confidence': classification.confidence,
                        'reasoning': classification.reasoning
                    })
        
        return final_results[:top_k]

def demo_coffee_search():
    """Demo function for coffee data"""
    coffee_engine = CSVHybridSearchEngine(
        "data/top-rated-coffee-clean.csv",
        text_columns=['coffee_name', 'roaster_location', 'coffee_origin']
    )
    
    # Test queries
    test_queries = [
        "how many coffee scored over 95",  # Analytical
        "Ethiopian coffee with floral notes",  # Semantic  
        "coffee from Colombia",  # Mixed
        "highest rated Geisha variety",  # Mixed
        "total count of light roast coffee"  # Analytical
    ]
    
    print("🔍 Coffee Hybrid Search Demo\n")
    
    for query in test_queries:
        print(f"Query: '{query}'")
        results = coffee_engine.hybrid_search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result.search_type}] {result.content[:100]}...")
            print(f"     Score: {result.score:.3f}")
        print()

if __name__ == "__main__":
    demo_coffee_search() 