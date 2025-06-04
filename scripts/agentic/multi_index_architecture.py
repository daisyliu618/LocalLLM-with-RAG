"""
Multi-Index Architecture System for Agentic RAG

This module implements the architecture for transforming from a single unified index
to multiple specialized indices organized by document type and content domain.

The system includes:
- Document type classification and routing
- Specialized index management for different content types
- Query routing and cross-index result synthesis
- Metadata structures and cross-index relationships
- Performance optimization and monitoring

Architecture Components:
1. IndexTypologyManager - Manages index schemas and organization
2. DocumentTypeClassifier - Routes documents to appropriate indices
3. IndexRouter - Routes queries to relevant indices
4. ResultComposer - Synthesizes results from multiple indices
5. MetadataManager - Handles cross-index relationships and metadata
6. PerformanceMonitor - Tracks multi-index performance metrics
"""

import json
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from enum import Enum
from datetime import datetime
from pathlib import Path


class DocumentType(Enum):
    """Document type classification for specialized indexing"""
    TECHNICAL_DOC = "technical_doc"        # API docs, technical specifications
    BUSINESS_DOC = "business_doc"          # Business reports, presentations
    REFERENCE_DATA = "reference_data"      # CSV, structured data, tables
    ACADEMIC_PAPER = "academic_paper"      # Research papers, academic content
    LEGAL_DOC = "legal_doc"               # Contracts, legal documents
    CONVERSATIONAL = "conversational"     # Chat logs, Q&A, forums
    CODE_DOC = "code_doc"                 # Code documentation, README files
    GENERAL_TEXT = "general_text"         # General purpose text documents
    MIXED_CONTENT = "mixed_content"       # Documents with multiple content types


class IndexType(Enum):
    """Types of specialized indices"""
    SEMANTIC_DENSE = "semantic_dense"     # High-quality embeddings for semantic search
    KEYWORD_SPARSE = "keyword_sparse"     # BM25/TF-IDF for keyword matching
    STRUCTURED_DATA = "structured_data"   # Specialized for CSV/table data
    CODE_SEARCH = "code_search"          # Optimized for code and technical docs
    HIERARCHICAL = "hierarchical"        # Document structure-aware indexing
    TEMPORAL = "temporal"                # Time-aware indexing for versioned docs
    METADATA_RICH = "metadata_rich"      # Heavy metadata with lightweight content


class QueryComplexity(Enum):
    """Query complexity classification for routing decisions"""
    SIMPLE = "simple"                    # Single concept, direct lookup
    MODERATE = "moderate"                # Multiple concepts, some complexity
    COMPLEX = "complex"                  # Multi-faceted, requires synthesis
    ANALYTICAL = "analytical"           # Requires data aggregation/analysis


@dataclass
class IndexSchema:
    """Schema definition for a specialized index"""
    index_id: str
    index_type: IndexType
    document_types: List[DocumentType]
    
    # Index configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Metadata schema
    required_metadata: List[str] = None
    optional_metadata: List[str] = None
    
    # Performance tuning
    max_chunks_per_doc: int = 100
    similarity_threshold: float = 0.7
    
    # Index-specific settings
    custom_settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.required_metadata is None:
            self.required_metadata = ["doc_id", "doc_type", "created_at"]
        if self.optional_metadata is None:
            self.optional_metadata = ["author", "tags", "version"]
        if self.custom_settings is None:
            self.custom_settings = {}


@dataclass
class DocumentMetadata:
    """Rich metadata for cross-index document tracking"""
    # Required fields (no defaults)
    doc_id: str
    original_filename: str
    file_type: str
    document_type: DocumentType
    word_count: int
    estimated_reading_time: int
    processed_at: datetime
    chunk_count: int
    index_assignments: List[str]  # List of index_ids containing this document
    
    # Optional fields (with defaults)
    language: str = "en"
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    version: Optional[str] = None
    tags: List[str] = None
    content_quality_score: float = 0.0
    indexing_confidence: float = 1.0
    related_documents: List[str] = None  # Related doc_ids
    parent_document: Optional[str] = None  # For hierarchical docs
    child_documents: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.related_documents is None:
            self.related_documents = []
        if self.child_documents is None:
            self.child_documents = []


@dataclass
class QueryRoutingDecision:
    """Decision result for query routing"""
    query: str
    complexity: QueryComplexity
    target_indices: List[str]
    
    # Routing reasoning
    reasoning: str
    confidence: float
    
    # Index-specific parameters
    index_weights: Dict[str, float]  # Weight for each target index
    search_parameters: Dict[str, Dict[str, Any]]  # Per-index search params
    
    # Performance expectations
    expected_response_time: float
    estimated_cost: float
    
    # Fallback strategy
    fallback_indices: List[str]
    
    # Metadata
    routing_time: datetime
    router_version: str


@dataclass
class CrossIndexResult:
    """Result from multiple index search with synthesis metadata"""
    query: str
    individual_results: Dict[str, List[Dict[str, Any]]]  # Results per index
    
    # Synthesis metadata
    total_chunks_found: int
    indices_queried: List[str]
    synthesis_strategy: str
    
    # Quality metrics
    result_diversity: float  # Measure of result variety
    cross_index_overlap: float  # How much results overlap between indices
    synthesis_confidence: float
    
    # Performance metrics
    total_query_time: float
    per_index_times: Dict[str, float]
    synthesis_time: float
    
    # Final synthesized results
    synthesized_results: List[Dict[str, Any]]
    result_count: int
    
    # Metadata
    timestamp: datetime
    composer_version: str


class IndexTypologyManager:
    """
    Manages the overall typology and organization of specialized indices
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.schemas: Dict[str, IndexSchema] = {}
        self.document_type_mapping: Dict[DocumentType, List[str]] = {}
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize default schemas
        self._initialize_default_schemas()
    
    def _initialize_default_schemas(self):
        """Initialize default index schemas for common document types"""
        
        # Technical documentation index
        tech_schema = IndexSchema(
            index_id="technical_docs",
            index_type=IndexType.CODE_SEARCH,
            document_types=[DocumentType.TECHNICAL_DOC, DocumentType.CODE_DOC],
            chunk_size=1024,  # Larger chunks for technical content
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            required_metadata=["doc_id", "doc_type", "api_version", "technology"],
            custom_settings={
                "code_aware_chunking": True,
                "preserve_code_blocks": True,
                "extract_api_signatures": True
            }
        )
        
        # Business documents index
        business_schema = IndexSchema(
            index_id="business_docs",
            index_type=IndexType.SEMANTIC_DENSE,
            document_types=[DocumentType.BUSINESS_DOC],
            chunk_size=512,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            required_metadata=["doc_id", "doc_type", "department", "quarter"],
            custom_settings={
                "extract_metrics": True,
                "identify_kpis": True,
                "preserve_tables": True
            }
        )
        
        # Structured data index
        structured_schema = IndexSchema(
            index_id="structured_data",
            index_type=IndexType.STRUCTURED_DATA,
            document_types=[DocumentType.REFERENCE_DATA],
            chunk_size=256,  # Smaller chunks for structured data
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            required_metadata=["doc_id", "doc_type", "schema_version", "data_format"],
            custom_settings={
                "preserve_structure": True,
                "extract_schema": True,
                "enable_sql_queries": True
            }
        )
        
        # Academic papers index
        academic_schema = IndexSchema(
            index_id="academic_papers",
            index_type=IndexType.HIERARCHICAL,
            document_types=[DocumentType.ACADEMIC_PAPER],
            chunk_size=768,  # Medium chunks for academic content
            embedding_model="sentence-transformers/allenai-specter",
            required_metadata=["doc_id", "doc_type", "authors", "publication_year", "journal"],
            custom_settings={
                "preserve_citations": True,
                "extract_abstracts": True,
                "identify_methodology": True,
                "track_references": True
            }
        )
        
        # General purpose index (fallback)
        general_schema = IndexSchema(
            index_id="general_content",
            index_type=IndexType.SEMANTIC_DENSE,
            document_types=[DocumentType.GENERAL_TEXT, DocumentType.MIXED_CONTENT],
            chunk_size=512,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            required_metadata=["doc_id", "doc_type", "language"],
            custom_settings={
                "balanced_approach": True,
                "auto_language_detection": True
            }
        )
        
        # Register all schemas
        for schema in [tech_schema, business_schema, structured_schema, academic_schema, general_schema]:
            self.register_schema(schema)
    
    def register_schema(self, schema: IndexSchema) -> bool:
        """Register a new index schema"""
        try:
            self.schemas[schema.index_id] = schema
            
            # Update document type mapping
            for doc_type in schema.document_types:
                if doc_type not in self.document_type_mapping:
                    self.document_type_mapping[doc_type] = []
                self.document_type_mapping[doc_type].append(schema.index_id)
            
            self.logger.info(f"Registered index schema: {schema.index_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register schema {schema.index_id}: {e}")
            return False
    
    def get_indices_for_document_type(self, doc_type: DocumentType) -> List[str]:
        """Get list of index IDs that handle a specific document type"""
        return self.document_type_mapping.get(doc_type, ["general_content"])
    
    def get_schema(self, index_id: str) -> Optional[IndexSchema]:
        """Get schema for a specific index"""
        return self.schemas.get(index_id)
    
    def get_all_schemas(self) -> Dict[str, IndexSchema]:
        """Get all registered schemas"""
        return self.schemas.copy()
    
    def validate_schema(self, schema: IndexSchema) -> Tuple[bool, List[str]]:
        """Validate a schema configuration"""
        errors = []
        
        if not schema.index_id:
            errors.append("Index ID is required")
        
        if schema.chunk_size <= 0:
            errors.append("Chunk size must be positive")
        
        if schema.chunk_overlap >= schema.chunk_size:
            errors.append("Chunk overlap must be less than chunk size")
        
        if not schema.document_types:
            errors.append("At least one document type must be specified")
        
        if not (0.0 <= schema.similarity_threshold <= 1.0):
            errors.append("Similarity threshold must be between 0 and 1")
        
        return len(errors) == 0, errors
    
    def save_configuration(self, path: Optional[str] = None) -> bool:
        """Save current configuration to file"""
        config_path = path or self.config_path or "multi_index_config.json"
        
        try:
            config = {
                "schemas": {
                    schema_id: {
                        "index_type": schema.index_type.value,
                        "document_types": [dt.value for dt in schema.document_types],
                        "chunk_size": schema.chunk_size,
                        "chunk_overlap": schema.chunk_overlap,
                        "embedding_model": schema.embedding_model,
                        "required_metadata": schema.required_metadata,
                        "optional_metadata": schema.optional_metadata,
                        "max_chunks_per_doc": schema.max_chunks_per_doc,
                        "similarity_threshold": schema.similarity_threshold,
                        "custom_settings": schema.custom_settings
                    }
                    for schema_id, schema in self.schemas.items()
                },
                "document_type_mapping": {
                    doc_type.value: indices 
                    for doc_type, indices in self.document_type_mapping.items()
                },
                "saved_at": datetime.now().isoformat()
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_configuration(self, path: Optional[str] = None) -> bool:
        """Load configuration from file"""
        config_path = path or self.config_path or "multi_index_config.json"
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Clear existing configuration
            self.schemas.clear()
            self.document_type_mapping.clear()
            
            # Load schemas
            for schema_id, schema_data in config["schemas"].items():
                schema = IndexSchema(
                    index_id=schema_id,
                    index_type=IndexType(schema_data["index_type"]),
                    document_types=[DocumentType(dt) for dt in schema_data["document_types"]],
                    chunk_size=schema_data["chunk_size"],
                    chunk_overlap=schema_data["chunk_overlap"],
                    embedding_model=schema_data["embedding_model"],
                    required_metadata=schema_data["required_metadata"],
                    optional_metadata=schema_data["optional_metadata"],
                    max_chunks_per_doc=schema_data["max_chunks_per_doc"],
                    similarity_threshold=schema_data["similarity_threshold"],
                    custom_settings=schema_data["custom_settings"]
                )
                self.register_schema(schema)
            
            self.logger.info(f"Configuration loaded from {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return False
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Analyze current configuration and provide optimization recommendations"""
        recommendations = []
        
        # Check for document types without dedicated indices
        all_doc_types = set(DocumentType)
        covered_types = set(self.document_type_mapping.keys())
        uncovered_types = all_doc_types - covered_types
        
        if uncovered_types:
            recommendations.append({
                "type": "missing_coverage",
                "priority": "medium",
                "description": f"Document types without dedicated indices: {[dt.value for dt in uncovered_types]}",
                "suggestion": "Consider creating specialized indices for frequently used document types"
            })
        
        # Check for overlapping document types
        type_to_indices = {}
        for doc_type, indices in self.document_type_mapping.items():
            if len(indices) > 2:  # More than primary + fallback
                recommendations.append({
                    "type": "excessive_overlap",
                    "priority": "low",
                    "description": f"Document type {doc_type.value} is handled by {len(indices)} indices",
                    "suggestion": "Consider consolidating indices to reduce overlap and improve performance"
                })
        
        # Check for performance anti-patterns
        for schema_id, schema in self.schemas.items():
            if schema.chunk_size > 1024 and schema.index_type == IndexType.SEMANTIC_DENSE:
                recommendations.append({
                    "type": "performance_issue",
                    "priority": "medium",
                    "description": f"Index {schema_id} has large chunk size ({schema.chunk_size}) for dense semantic search",
                    "suggestion": "Consider reducing chunk size for better semantic similarity accuracy"
                })
        
        return recommendations


class MetadataManager:
    """
    Manages document metadata and cross-index relationships
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.metadata_store: Dict[str, DocumentMetadata] = {}
        self.storage_path = storage_path or "document_metadata.json"
        self.logger = logging.getLogger(__name__)
        
        # Index tracking
        self.documents_by_index: Dict[str, Set[str]] = {}  # index_id -> set of doc_ids
        self.index_assignments: Dict[str, Set[str]] = {}   # doc_id -> set of index_ids
    
    def add_document_metadata(self, metadata: DocumentMetadata) -> bool:
        """Add or update document metadata"""
        try:
            self.metadata_store[metadata.doc_id] = metadata
            
            # Update index tracking
            self.index_assignments[metadata.doc_id] = set(metadata.index_assignments)
            
            for index_id in metadata.index_assignments:
                if index_id not in self.documents_by_index:
                    self.documents_by_index[index_id] = set()
                self.documents_by_index[index_id].add(metadata.doc_id)
            
            self.logger.debug(f"Added metadata for document: {metadata.doc_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add metadata for {metadata.doc_id}: {e}")
            return False
    
    def get_document_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Get metadata for a specific document"""
        return self.metadata_store.get(doc_id)
    
    def get_documents_by_type(self, doc_type: DocumentType) -> List[DocumentMetadata]:
        """Get all documents of a specific type"""
        return [
            metadata for metadata in self.metadata_store.values()
            if metadata.document_type == doc_type
        ]
    
    def get_documents_in_index(self, index_id: str) -> List[str]:
        """Get list of document IDs in a specific index"""
        return list(self.documents_by_index.get(index_id, set()))
    
    def get_related_documents(self, doc_id: str, max_results: int = 10) -> List[Tuple[str, float]]:
        """Get related documents based on metadata similarity"""
        source_doc = self.get_document_metadata(doc_id)
        if not source_doc:
            return []
        
        related_scores = []
        
        for other_id, other_doc in self.metadata_store.items():
            if other_id == doc_id:
                continue
            
            similarity_score = self._calculate_document_similarity(source_doc, other_doc)
            if similarity_score > 0.3:  # Threshold for relatedness
                related_scores.append((other_id, similarity_score))
        
        # Sort by similarity and return top results
        related_scores.sort(key=lambda x: x[1], reverse=True)
        return related_scores[:max_results]
    
    def _calculate_document_similarity(self, doc1: DocumentMetadata, doc2: DocumentMetadata) -> float:
        """Calculate similarity between two documents based on metadata"""
        score = 0.0
        
        # Document type similarity
        if doc1.document_type == doc2.document_type:
            score += 0.3
        
        # Author similarity
        if doc1.author and doc2.author and doc1.author == doc2.author:
            score += 0.2
        
        # Tag overlap
        if doc1.tags and doc2.tags:
            common_tags = set(doc1.tags) & set(doc2.tags)
            tag_similarity = len(common_tags) / max(len(doc1.tags), len(doc2.tags))
            score += tag_similarity * 0.3
        
        # Temporal proximity (if both have creation dates)
        if doc1.created_at and doc2.created_at:
            time_diff = abs((doc1.created_at - doc2.created_at).days)
            if time_diff < 30:  # Within 30 days
                temporal_score = max(0, (30 - time_diff) / 30) * 0.2
                score += temporal_score
        
        return min(score, 1.0)
    
    def get_index_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about document distribution across indices"""
        stats = {}
        
        for index_id, doc_ids in self.documents_by_index.items():
            docs = [self.metadata_store[doc_id] for doc_id in doc_ids]
            
            # Document type distribution
            type_counts = {}
            for doc in docs:
                doc_type = doc.document_type.value
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            # Size statistics
            word_counts = [doc.word_count for doc in docs if doc.word_count > 0]
            chunk_counts = [doc.chunk_count for doc in docs if doc.chunk_count > 0]
            
            stats[index_id] = {
                "document_count": len(doc_ids),
                "document_types": type_counts,
                "avg_word_count": sum(word_counts) / len(word_counts) if word_counts else 0,
                "avg_chunk_count": sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0,
                "total_chunks": sum(chunk_counts) if chunk_counts else 0
            }
        
        return stats
    
    def save_metadata(self, path: Optional[str] = None) -> bool:
        """Save metadata to persistent storage"""
        storage_path = path or self.storage_path
        
        try:
            # Convert to serializable format
            serializable_data = {
                "metadata": {
                    doc_id: {
                        "doc_id": meta.doc_id,
                        "original_filename": meta.original_filename,
                        "file_type": meta.file_type,
                        "document_type": meta.document_type.value,
                        "word_count": meta.word_count,
                        "estimated_reading_time": meta.estimated_reading_time,
                        "language": meta.language,
                        "processed_at": meta.processed_at.isoformat(),
                        "chunk_count": meta.chunk_count,
                        "index_assignments": meta.index_assignments,
                        "title": meta.title,
                        "author": meta.author,
                        "created_at": meta.created_at.isoformat() if meta.created_at else None,
                        "modified_at": meta.modified_at.isoformat() if meta.modified_at else None,
                        "version": meta.version,
                        "tags": meta.tags,
                        "content_quality_score": meta.content_quality_score,
                        "indexing_confidence": meta.indexing_confidence,
                        "related_documents": meta.related_documents,
                        "parent_document": meta.parent_document,
                        "child_documents": meta.child_documents
                    }
                    for doc_id, meta in self.metadata_store.items()
                },
                "saved_at": datetime.now().isoformat()
            }
            
            with open(storage_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            self.logger.info(f"Metadata saved to {storage_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
            return False
    
    def load_metadata(self, path: Optional[str] = None) -> bool:
        """Load metadata from persistent storage"""
        storage_path = path or self.storage_path
        
        try:
            with open(storage_path, 'r') as f:
                data = json.load(f)
            
            # Clear existing data
            self.metadata_store.clear()
            self.documents_by_index.clear()
            self.index_assignments.clear()
            
            # Load metadata
            for doc_id, meta_data in data["metadata"].items():
                metadata = DocumentMetadata(
                    doc_id=meta_data["doc_id"],
                    original_filename=meta_data["original_filename"],
                    file_type=meta_data["file_type"],
                    document_type=DocumentType(meta_data["document_type"]),
                    word_count=meta_data["word_count"],
                    estimated_reading_time=meta_data["estimated_reading_time"],
                    processed_at=datetime.fromisoformat(meta_data["processed_at"]),
                    chunk_count=meta_data["chunk_count"],
                    index_assignments=meta_data["index_assignments"],
                    language=meta_data["language"],
                    title=meta_data["title"],
                    author=meta_data["author"],
                    created_at=datetime.fromisoformat(meta_data["created_at"]) if meta_data["created_at"] else None,
                    modified_at=datetime.fromisoformat(meta_data["modified_at"]) if meta_data["modified_at"] else None,
                    version=meta_data["version"],
                    tags=meta_data["tags"],
                    content_quality_score=meta_data["content_quality_score"],
                    indexing_confidence=meta_data["indexing_confidence"],
                    related_documents=meta_data["related_documents"],
                    parent_document=meta_data["parent_document"],
                    child_documents=meta_data["child_documents"]
                )
                self.add_document_metadata(metadata)
            
            self.logger.info(f"Metadata loaded from {storage_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return False


# Example usage and testing functions
def create_test_architecture() -> Tuple[IndexTypologyManager, MetadataManager]:
    """Create a test multi-index architecture setup"""
    
    # Initialize managers
    topology_manager = IndexTypologyManager()
    metadata_manager = MetadataManager()
    
    # Add some test document metadata
    test_docs = [
        DocumentMetadata(
            doc_id="tech_doc_001",
            original_filename="api_documentation.md",
            file_type="markdown",
            document_type=DocumentType.TECHNICAL_DOC,
            word_count=2500,
            estimated_reading_time=10,
            processed_at=datetime.now(),
            chunk_count=15,
            index_assignments=["technical_docs", "general_content"],
            title="REST API Documentation",
            author="Engineering Team",
            tags=["api", "rest", "documentation"]
        ),
        DocumentMetadata(
            doc_id="business_report_001",
            original_filename="q3_financial_report.pdf",
            file_type="pdf",
            document_type=DocumentType.BUSINESS_DOC,
            word_count=5000,
            estimated_reading_time=20,
            processed_at=datetime.now(),
            chunk_count=25,
            index_assignments=["business_docs"],
            title="Q3 2024 Financial Report",
            author="Finance Team",
            tags=["finance", "quarterly", "report", "2024"]
        ),
        DocumentMetadata(
            doc_id="research_paper_001",
            original_filename="machine_learning_paper.pdf",
            file_type="pdf",
            document_type=DocumentType.ACADEMIC_PAPER,
            word_count=8000,
            estimated_reading_time=32,
            processed_at=datetime.now(),
            chunk_count=40,
            index_assignments=["academic_papers"],
            title="Advanced Machine Learning Techniques",
            author="Dr. Sarah Johnson",
            tags=["machine learning", "AI", "research", "algorithms"]
        )
    ]
    
    for doc_metadata in test_docs:
        metadata_manager.add_document_metadata(doc_metadata)
    
    return topology_manager, metadata_manager


def validate_architecture_design(topology_manager: IndexTypologyManager, 
                                metadata_manager: MetadataManager) -> Dict[str, Any]:
    """Validate the architecture design and return analysis results"""
    
    validation_results = {
        "schema_validation": {},
        "metadata_statistics": {},
        "optimization_recommendations": [],
        "architecture_health": "healthy"
    }
    
    # Validate all schemas
    for schema_id, schema in topology_manager.get_all_schemas().items():
        is_valid, errors = topology_manager.validate_schema(schema)
        validation_results["schema_validation"][schema_id] = {
            "valid": is_valid,
            "errors": errors
        }
        if not is_valid:
            validation_results["architecture_health"] = "needs_attention"
    
    # Get metadata statistics
    validation_results["metadata_statistics"] = metadata_manager.get_index_statistics()
    
    # Get optimization recommendations
    validation_results["optimization_recommendations"] = topology_manager.get_optimization_recommendations()
    
    return validation_results


if __name__ == "__main__":
    # Example usage
    print("Multi-Index Architecture System")
    print("=" * 50)
    
    # Create test architecture
    topology_mgr, metadata_mgr = create_test_architecture()
    
    # Validate architecture
    validation = validate_architecture_design(topology_mgr, metadata_mgr)
    
    print(f"Architecture Health: {validation['architecture_health']}")
    print(f"Schemas Registered: {len(topology_mgr.get_all_schemas())}")
    print(f"Documents Tracked: {len(metadata_mgr.metadata_store)}")
    
    # Show index statistics
    for index_id, stats in validation["metadata_statistics"].items():
        print(f"\nIndex: {index_id}")
        print(f"  Documents: {stats['document_count']}")
        print(f"  Total Chunks: {stats['total_chunks']}")
        print(f"  Avg Words: {stats['avg_word_count']:.0f}")
    
    # Show optimization recommendations
    if validation["optimization_recommendations"]:
        print("\nOptimization Recommendations:")
        for rec in validation["optimization_recommendations"]:
            print(f"  - {rec['type']}: {rec['description']}") 