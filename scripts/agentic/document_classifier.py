"""
Document Type Classifier for Agentic RAG Multi-Index System

This module implements intelligent document type classification to route documents 
to appropriate specialized indices during ingestion. The classifier analyzes:
- Document format and file types
- Content patterns and structure
- Metadata and linguistic features
- LLM-powered content understanding

Key Features:
- Multi-modal classification (rule-based + LLM-powered)
- Confidence scoring and manual override capabilities
- Support for PDF, DOCX, TXT, CSV, image formats
- Content-aware classification for technical docs, business documents, reference data
- Integration with multi-index architecture system
"""

import json
import logging
import asyncio
import re
import mimetypes
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from enum import Enum
from datetime import datetime
from pathlib import Path
import hashlib

# Import from our multi-index architecture
from multi_index_architecture import (
    DocumentType, IndexType, DocumentMetadata, IndexTypologyManager
)


class ClassificationMethod(Enum):
    """Methods used for document classification"""
    RULE_BASED = "rule_based"           # Pattern matching and heuristics
    LLM_ANALYSIS = "llm_analysis"       # LLM-powered content understanding
    HYBRID = "hybrid"                   # Combination of rule-based and LLM
    MANUAL_OVERRIDE = "manual_override" # User-specified classification


class ConfidenceLevel(Enum):
    """Confidence levels for classification results"""
    VERY_HIGH = "very_high"     # 0.9-1.0
    HIGH = "high"               # 0.7-0.9
    MEDIUM = "medium"           # 0.5-0.7
    LOW = "low"                 # 0.3-0.5
    VERY_LOW = "very_low"       # 0.0-0.3


@dataclass
class DocumentFeatures:
    """Extracted features from document analysis"""
    # File characteristics
    filename: str
    file_extension: str
    file_size: int
    mime_type: str
    
    # Content characteristics
    text_content_sample: str  # First 1000 chars
    word_count: int
    line_count: int
    avg_line_length: float
    
    # Structure indicators
    has_code_blocks: bool
    has_tables: bool
    has_headers: bool
    has_citations: bool
    has_equations: bool
    has_metadata: bool
    
    # Language and style
    language: str
    technical_term_ratio: float
    business_term_ratio: float
    academic_term_ratio: float
    code_ratio: float
    
    # Format-specific features
    pdf_metadata: Optional[Dict[str, Any]] = None
    docx_styles: Optional[List[str]] = None
    csv_schema: Optional[Dict[str, str]] = None
    
    # Content patterns
    url_count: int = 0
    email_count: int = 0
    phone_count: int = 0
    date_count: int = 0
    number_count: int = 0


@dataclass
class ClassificationResult:
    """Result of document classification"""
    # Primary classification (required fields first)
    document_type: DocumentType
    confidence: float
    classification_method: ClassificationMethod
    alternative_types: List[Tuple[DocumentType, float]]  # (type, confidence)
    reasoning: str
    key_features: List[str]
    classification_time: datetime
    primary_index: str
    fallback_indices: List[str]
    can_override: bool
    feature_extraction_quality: float
    content_analysis_quality: float
    
    # Override capabilities (fields with defaults)
    manual_override_applied: bool = False
    override_reason: Optional[str] = None
    
    def to_confidence_level(self) -> ConfidenceLevel:
        """Convert confidence score to confidence level"""
        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class ContentAnalyzer:
    """Base class for content analyzers"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Technical terms for pattern matching
        self.technical_terms = {
            'api', 'endpoint', 'json', 'xml', 'http', 'https', 'rest', 'soap',
            'database', 'sql', 'query', 'schema', 'table', 'index', 'primary key',
            'function', 'method', 'class', 'object', 'variable', 'parameter',
            'algorithm', 'implementation', 'framework', 'library', 'module',
            'configuration', 'deployment', 'server', 'client', 'authentication',
            'authorization', 'security', 'encryption', 'protocol', 'interface',
            'python', 'print', 'def', 'import', 'return', 'code', 'programming'
        }
        
        # Business terms
        self.business_terms = {
            'revenue', 'profit', 'loss', 'quarter', 'annual', 'financial',
            'budget', 'forecast', 'strategy', 'market', 'customer', 'client',
            'stakeholder', 'roi', 'kpi', 'metric', 'performance', 'analysis',
            'report', 'dashboard', 'executive', 'management', 'board',
            'compliance', 'audit', 'risk', 'investment', 'portfolio',
            'sales', 'marketing', 'operations', 'human resources', 'hr'
        }
        
        # Academic terms
        self.academic_terms = {
            'abstract', 'introduction', 'methodology', 'results', 'conclusion',
            'literature', 'review', 'hypothesis', 'experiment', 'analysis',
            'research', 'study', 'paper', 'journal', 'publication', 'citation',
            'reference', 'bibliography', 'doi', 'isbn', 'university', 'college',
            'professor', 'phd', 'masters', 'undergraduate', 'graduate',
            'thesis', 'dissertation', 'conference', 'proceedings', 'symposium'
        }
    
    def extract_features(self, file_path: str, content: str) -> DocumentFeatures:
        """Extract comprehensive features from document"""
        path_obj = Path(file_path)
        
        # Handle edge cases
        if not content or not content.strip():
            return self._create_empty_content_features(path_obj)
        
        # Basic file info
        file_size = path_obj.stat().st_size if path_obj.exists() else len(content)
        mime_type, _ = mimetypes.guess_type(file_path)
        
        # Content analysis
        lines = content.split('\n')
        words = content.split()
        
        # Structure detection
        has_code_blocks = self._detect_code_blocks(content)
        has_tables = self._detect_tables(content)
        has_headers = self._detect_headers(content)
        has_citations = self._detect_citations(content)
        has_equations = self._detect_equations(content)
        has_metadata = self._detect_metadata(content)
        
        # Term analysis
        content_lower = content.lower()
        technical_ratio = self._calculate_term_ratio(content_lower, self.technical_terms)
        business_ratio = self._calculate_term_ratio(content_lower, self.business_terms)
        academic_ratio = self._calculate_term_ratio(content_lower, self.academic_terms)
        code_ratio = self._calculate_code_ratio(content)
        
        # Pattern counting (improved patterns)
        url_count = len(re.findall(r'https?://\S+', content))
        email_count = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content))
        phone_count = len(re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', content))  # More flexible phone pattern
        date_count = len(re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', content))
        number_count = len(re.findall(r'\b\d+\.?\d*\b', content))
        
        return DocumentFeatures(
            filename=path_obj.name,
            file_extension=path_obj.suffix.lower(),
            file_size=file_size,
            mime_type=mime_type or "unknown",
            text_content_sample=content[:1000],
            word_count=len(words),
            line_count=len(lines),
            avg_line_length=sum(len(line) for line in lines) / len(lines) if lines else 0,
            has_code_blocks=has_code_blocks,
            has_tables=has_tables,
            has_headers=has_headers,
            has_citations=has_citations,
            has_equations=has_equations,
            has_metadata=has_metadata,
            language="en",  # TODO: Implement language detection
            technical_term_ratio=technical_ratio,
            business_term_ratio=business_ratio,
            academic_term_ratio=academic_ratio,
            code_ratio=code_ratio,
            url_count=url_count,
            email_count=email_count,
            phone_count=phone_count,
            date_count=date_count,
            number_count=number_count
        )
    
    def _create_empty_content_features(self, path_obj: Path) -> DocumentFeatures:
        """Create features for empty content"""
        return DocumentFeatures(
            filename=path_obj.name,
            file_extension=path_obj.suffix.lower(),
            file_size=0,
            mime_type="unknown",
            text_content_sample="",
            word_count=0,
            line_count=0,
            avg_line_length=0.0,
            has_code_blocks=False,
            has_tables=False,
            has_headers=False,
            has_citations=False,
            has_equations=False,
            has_metadata=False,
            language="en",
            technical_term_ratio=0.0,
            business_term_ratio=0.0,
            academic_term_ratio=0.0,
            code_ratio=0.0,
            url_count=0,
            email_count=0,
            phone_count=0,
            date_count=0,
            number_count=0
        )
    
    def _detect_code_blocks(self, content: str) -> bool:
        """Detect code blocks in content"""
        code_patterns = [
            r'```[\s\S]*?```',  # Markdown code blocks
            r'`[^`\n]+`',       # Inline code
            r'<code>[\s\S]*?</code>',  # HTML code tags
            r'def\s+\w+\s*\(',  # Python function definitions
            r'function\s+\w+\s*\(',  # JavaScript function definitions
            r'class\s+\w+\s*[{:]',   # Class definitions
            r'\w+\.\w+\(',      # Method calls
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                return True
        return False
    
    def _detect_tables(self, content: str) -> bool:
        """Detect tables in content"""
        table_patterns = [
            r'\|.*\|.*\|',      # Markdown tables
            r'<table[\s\S]*?</table>',  # HTML tables
            r'\t.*\t.*\t',      # Tab-separated values
            r',.*,.*,',         # CSV-like content
        ]
        
        for pattern in table_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                return True
        return False
    
    def _detect_headers(self, content: str) -> bool:
        """Detect headers/headings in content"""
        header_patterns = [
            r'^#{1,6}\s+.+$',   # Markdown headers
            r'<h[1-6]>.*</h[1-6]>',  # HTML headers
            r'^[A-Z][A-Z\s]+$', # ALL CAPS headers
        ]
        
        for pattern in header_patterns:
            if re.search(pattern, content, re.MULTILINE):
                return True
        return False
    
    def _detect_citations(self, content: str) -> bool:
        """Detect academic citations"""
        citation_patterns = [
            r'\[\d+\]',         # Numbered citations
            r'\([A-Za-z]+\s+\d{4}\)',  # Author-year citations
            r'doi:\s*10\.\d+',  # DOI citations
            r'arXiv:\d+\.\d+',  # arXiv citations
            r'References?\s*$', # References section
            r'Bibliography\s*$',  # Bibliography section
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                return True
        return False
    
    def _detect_equations(self, content: str) -> bool:
        """Detect mathematical equations"""
        equation_patterns = [
            r'\$.*\$',          # LaTeX inline math
            r'\$\$[\s\S]*?\$\$',  # LaTeX display math
            r'\\begin\{equation\}',  # LaTeX equations
            r'\\frac\{.*\}\{.*\}',   # Fractions
            r'\\sum_{.*}',      # Summations
            r'\\int_{.*}',      # Integrals
        ]
        
        for pattern in equation_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
    
    def _detect_metadata(self, content: str) -> bool:
        """Detect structured metadata"""
        metadata_patterns = [
            r'---\s*\n[\s\S]*?\n---',  # YAML frontmatter
            r'\{[\s\S]*".*":.*[\s\S]*\}',  # JSON-like structure
            r'<meta\s+.*>',     # HTML meta tags
            r'Author:\s*.+',    # Key-value metadata
            r'Title:\s*.+',
            r'Date:\s*.+',
        ]
        
        for pattern in metadata_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                return True
        return False
    
    def _calculate_term_ratio(self, content: str, terms: Set[str]) -> float:
        """Calculate ratio of domain-specific terms in content"""
        words = content.split()
        if not words:
            return 0.0
        
        term_count = sum(1 for word in words if word.lower() in terms)
        return term_count / len(words)
    
    def _calculate_code_ratio(self, content: str) -> float:
        """Calculate ratio of code-like content"""
        lines = content.split('\n')
        if not lines:
            return 0.0
        
        code_indicators = [
            r'^\s*def\s+',      # Python functions
            r'^\s*class\s+',    # Class definitions
            r'^\s*import\s+',   # Imports
            r'^\s*from\s+.*import',  # From imports
            r'^\s*if\s+.*:',    # If statements
            r'^\s*for\s+.*:',   # For loops
            r'^\s*while\s+.*:', # While loops
            r'^\s*#.*',         # Comments
            r'^\s*//.*',        # C-style comments
            r'^\s*/\*.*\*/',    # Block comments
        ]
        
        code_line_count = 0
        for line in lines:
            for pattern in code_indicators:
                if re.search(pattern, line):
                    code_line_count += 1
                    break
        
        return code_line_count / len(lines)


class RuleBasedClassifier:
    """Rule-based document classifier using heuristics and patterns"""
    
    def __init__(self, topology_manager: IndexTypologyManager):
        self.topology_manager = topology_manager
        self.content_analyzer = ContentAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def classify(self, file_path: str, content: str) -> ClassificationResult:
        """Classify document using rule-based approach"""
        features = self.content_analyzer.extract_features(file_path, content)
        
        # Handle empty or problematic content
        if features.word_count == 0:
            return self._create_low_confidence_result(file_path, features, "Empty content")
        
        # Check for binary or problematic content
        if self._is_binary_like_content(content):
            return self._create_low_confidence_result(file_path, features, "Binary-like content detected")
        
        # Classification logic based on features
        scores = self._calculate_type_scores(features)
        
        # Get best classification
        best_type = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_type]
        
        # Adjust confidence for low-quality content
        if features.word_count < 10:
            confidence *= 0.5  # Reduce confidence for very short content
        
        # Get alternatives (top 3 excluding best)
        alternatives = sorted(
            [(t, s) for t, s in scores.items() if t != best_type],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(features, best_type, scores)
        key_features = self._extract_key_features(features, best_type)
        
        # Get index assignments
        indices = self.topology_manager.get_indices_for_document_type(best_type)
        primary_index = indices[0] if indices else "general_content"
        fallback_indices = indices[1:] if len(indices) > 1 else ["general_content"]
        
        return ClassificationResult(
            document_type=best_type,
            confidence=confidence,
            classification_method=ClassificationMethod.RULE_BASED,
            alternative_types=alternatives,
            reasoning=reasoning,
            key_features=key_features,
            classification_time=datetime.now(),
            primary_index=primary_index,
            fallback_indices=fallback_indices,
            can_override=True,
            feature_extraction_quality=0.85,  # High for rule-based
            content_analysis_quality=0.75     # Good for heuristics
        )
    
    def _is_binary_like_content(self, content: str) -> bool:
        """Check if content appears to be binary or encoded"""
        if not content:
            return False
        
        # Check for high ratio of non-printable characters
        printable_chars = sum(1 for c in content if c.isprintable() or c.isspace())
        if len(content) > 10 and printable_chars / len(content) < 0.7:
            return True
        
        # Check for null bytes
        if '\x00' in content:
            return True
        
        return False
    
    def _create_low_confidence_result(self, file_path: str, features: DocumentFeatures, reason: str) -> ClassificationResult:
        """Create low-confidence classification result for problematic content"""
        return ClassificationResult(
            document_type=DocumentType.GENERAL_TEXT,
            confidence=0.1,  # Very low confidence
            classification_method=ClassificationMethod.RULE_BASED,
            alternative_types=[],
            reasoning=f"Low confidence classification: {reason}",
            key_features=[f"File: {features.filename}", f"Size: {features.word_count} words"],
            classification_time=datetime.now(),
            primary_index="general_content",
            fallback_indices=[],
            can_override=True,
            feature_extraction_quality=0.3,  # Low for problematic content
            content_analysis_quality=0.2     # Low for heuristics on bad content
        )
    
    def _calculate_type_scores(self, features: DocumentFeatures) -> Dict[DocumentType, float]:
        """Calculate scores for each document type"""
        scores = {doc_type: 0.0 for doc_type in DocumentType}
        
        # File extension indicators
        ext_scores = self._score_by_extension(features.file_extension)
        for doc_type, score in ext_scores.items():
            scores[doc_type] += score * 0.3  # 30% weight
        
        # Content pattern scores
        content_scores = self._score_by_content_patterns(features)
        for doc_type, score in content_scores.items():
            scores[doc_type] += score * 0.4  # 40% weight
        
        # Term frequency scores
        term_scores = self._score_by_term_frequency(features)
        for doc_type, score in term_scores.items():
            scores[doc_type] += score * 0.3  # 30% weight
        
        # Normalize scores to 0-1 range
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
    
    def _score_by_extension(self, extension: str) -> Dict[DocumentType, float]:
        """Score document types based on file extension"""
        ext_mapping = {
            '.py': {DocumentType.TECHNICAL_DOC: 0.8, DocumentType.CODE_DOC: 0.9},
            '.js': {DocumentType.TECHNICAL_DOC: 0.8, DocumentType.CODE_DOC: 0.9},
            '.java': {DocumentType.TECHNICAL_DOC: 0.8, DocumentType.CODE_DOC: 0.9},
            '.cpp': {DocumentType.TECHNICAL_DOC: 0.8, DocumentType.CODE_DOC: 0.9},
            '.md': {DocumentType.TECHNICAL_DOC: 0.6, DocumentType.CODE_DOC: 0.7, DocumentType.GENERAL_TEXT: 0.5},
            '.rst': {DocumentType.TECHNICAL_DOC: 0.7, DocumentType.CODE_DOC: 0.6},
            '.pdf': {DocumentType.ACADEMIC_PAPER: 0.6, DocumentType.BUSINESS_DOC: 0.5, DocumentType.TECHNICAL_DOC: 0.4},
            '.docx': {DocumentType.BUSINESS_DOC: 0.7, DocumentType.GENERAL_TEXT: 0.5},
            '.doc': {DocumentType.BUSINESS_DOC: 0.7, DocumentType.GENERAL_TEXT: 0.5},
            '.csv': {DocumentType.REFERENCE_DATA: 0.9},
            '.xlsx': {DocumentType.REFERENCE_DATA: 0.8, DocumentType.BUSINESS_DOC: 0.6},
            '.xls': {DocumentType.REFERENCE_DATA: 0.8, DocumentType.BUSINESS_DOC: 0.6},
            '.json': {DocumentType.REFERENCE_DATA: 0.7, DocumentType.TECHNICAL_DOC: 0.6},
            '.xml': {DocumentType.REFERENCE_DATA: 0.6, DocumentType.TECHNICAL_DOC: 0.5},
            '.txt': {DocumentType.GENERAL_TEXT: 0.8},
            '.log': {DocumentType.TECHNICAL_DOC: 0.5, DocumentType.CONVERSATIONAL: 0.4},
        }
        
        return ext_mapping.get(extension, {DocumentType.GENERAL_TEXT: 0.3})
    
    def _score_by_content_patterns(self, features: DocumentFeatures) -> Dict[DocumentType, float]:
        """Score document types based on content patterns"""
        scores = {doc_type: 0.0 for doc_type in DocumentType}
        
        # Technical documentation indicators
        if features.has_code_blocks or features.code_ratio > 0.1:
            scores[DocumentType.TECHNICAL_DOC] += 0.6
            scores[DocumentType.CODE_DOC] += 0.8
        
        # Academic paper indicators
        if features.has_citations:
            scores[DocumentType.ACADEMIC_PAPER] += 0.7
        if features.has_equations:
            scores[DocumentType.ACADEMIC_PAPER] += 0.5
        
        # Business document indicators
        if features.has_tables and not features.has_code_blocks:
            scores[DocumentType.BUSINESS_DOC] += 0.5
            scores[DocumentType.REFERENCE_DATA] += 0.3
        
        # Structured data indicators
        if features.file_extension in ['.csv', '.json', '.xml']:
            scores[DocumentType.REFERENCE_DATA] += 0.8
        
        # Conversational indicators
        if features.email_count > 0 or 'conversation' in features.filename.lower():
            scores[DocumentType.CONVERSATIONAL] += 0.6
        
        return scores
    
    def _score_by_term_frequency(self, features: DocumentFeatures) -> Dict[DocumentType, float]:
        """Score document types based on domain-specific term frequency"""
        scores = {doc_type: 0.0 for doc_type in DocumentType}
        
        # Technical content
        if features.technical_term_ratio > 0.02:  # 2% technical terms
            scores[DocumentType.TECHNICAL_DOC] += min(features.technical_term_ratio * 10, 1.0)
            scores[DocumentType.CODE_DOC] += min(features.technical_term_ratio * 8, 1.0)
        
        # Business content
        if features.business_term_ratio > 0.01:  # 1% business terms
            scores[DocumentType.BUSINESS_DOC] += min(features.business_term_ratio * 15, 1.0)
        
        # Academic content
        if features.academic_term_ratio > 0.01:  # 1% academic terms
            scores[DocumentType.ACADEMIC_PAPER] += min(features.academic_term_ratio * 12, 1.0)
        
        return scores
    
    def _generate_reasoning(self, features: DocumentFeatures, doc_type: DocumentType, scores: Dict[DocumentType, float]) -> str:
        """Generate human-readable reasoning for classification"""
        reasons = []
        
        # File extension reasoning
        if features.file_extension in ['.py', '.js', '.java', '.cpp']:
            reasons.append(f"File extension '{features.file_extension}' indicates code documentation")
        elif features.file_extension == '.csv':
            reasons.append("CSV format indicates structured reference data")
        elif features.file_extension in ['.pdf', '.docx']:
            reasons.append(f"'{features.file_extension}' format commonly used for {doc_type.value}")
        
        # Content pattern reasoning
        if features.has_code_blocks:
            reasons.append("Contains code blocks indicating technical documentation")
        if features.has_citations:
            reasons.append("Contains citations indicating academic content")
        if features.has_tables and not features.has_code_blocks:
            reasons.append("Contains tables suggesting business or reference data")
        
        # Term frequency reasoning
        if features.technical_term_ratio > 0.02:
            reasons.append(f"High technical term frequency ({features.technical_term_ratio:.1%})")
        if features.business_term_ratio > 0.01:
            reasons.append(f"Contains business terminology ({features.business_term_ratio:.1%})")
        if features.academic_term_ratio > 0.01:
            reasons.append(f"Contains academic terminology ({features.academic_term_ratio:.1%})")
        
        # Confidence reasoning
        confidence = scores[doc_type]
        if confidence > 0.8:
            reasons.append("Classification confidence is very high")
        elif confidence > 0.6:
            reasons.append("Classification confidence is high")
        elif confidence > 0.4:
            reasons.append("Classification confidence is moderate")
        else:
            reasons.append("Classification confidence is low - consider manual review")
        
        return "; ".join(reasons) if reasons else "Classification based on general content patterns"
    
    def _extract_key_features(self, features: DocumentFeatures, doc_type: DocumentType) -> List[str]:
        """Extract key features that influenced classification"""
        key_features = []
        
        key_features.append(f"File extension: {features.file_extension}")
        key_features.append(f"Word count: {features.word_count}")
        
        if features.has_code_blocks:
            key_features.append("Code blocks present")
        if features.has_tables:
            key_features.append("Tables present")
        if features.has_citations:
            key_features.append("Citations present")
        if features.has_equations:
            key_features.append("Equations present")
        
        if features.technical_term_ratio > 0.01:
            key_features.append(f"Technical terms: {features.technical_term_ratio:.1%}")
        if features.business_term_ratio > 0.01:
            key_features.append(f"Business terms: {features.business_term_ratio:.1%}")
        if features.academic_term_ratio > 0.01:
            key_features.append(f"Academic terms: {features.academic_term_ratio:.1%}")
        
        return key_features


class DocumentTypeClassifier:
    """
    Main document type classifier that integrates multiple classification methods
    """
    
    def __init__(self, topology_manager: IndexTypologyManager, config: Optional[Dict[str, Any]] = None):
        self.topology_manager = topology_manager
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize classifiers
        self.rule_based_classifier = RuleBasedClassifier(topology_manager)
        
        # Classification history for learning
        self.classification_history: List[ClassificationResult] = []
        self.override_history: List[Tuple[str, DocumentType, str]] = []  # (doc_id, type, reason)
        
        # Performance metrics
        self.metrics = {
            "total_classifications": 0,
            "rule_based_count": 0,
            "llm_count": 0,
            "hybrid_count": 0,
            "override_count": 0,
            "avg_confidence": 0.0,
            "avg_processing_time": 0.0
        }
    
    def classify_document(self, file_path: str, content: str, 
                         method: ClassificationMethod = ClassificationMethod.RULE_BASED,
                         manual_override: Optional[DocumentType] = None) -> ClassificationResult:
        """
        Classify a document and return classification result
        
        Args:
            file_path: Path to the document file
            content: Text content of the document
            method: Classification method to use
            manual_override: Manual classification override
            
        Returns:
            ClassificationResult with document type and metadata
        """
        start_time = datetime.now()
        
        try:
            # Handle manual override
            if manual_override:
                result = self._create_manual_override_result(file_path, content, manual_override)
            elif method == ClassificationMethod.RULE_BASED:
                result = self.rule_based_classifier.classify(file_path, content)
            else:
                # For now, fallback to rule-based (LLM implementation would go here)
                result = self.rule_based_classifier.classify(file_path, content)
                self.logger.warning(f"Method {method} not fully implemented, using rule-based fallback")
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(result, processing_time)
            
            # Store in history
            self.classification_history.append(result)
            
            self.logger.info(f"Classified {file_path} as {result.document_type.value} "
                           f"(confidence: {result.confidence:.2f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Classification failed for {file_path}: {e}")
            # Return fallback classification
            return self._create_fallback_result(file_path, content, str(e))
    
    def classify_batch(self, documents: List[Tuple[str, str]], 
                      method: ClassificationMethod = ClassificationMethod.RULE_BASED) -> List[ClassificationResult]:
        """
        Classify multiple documents in batch
        
        Args:
            documents: List of (file_path, content) tuples
            method: Classification method to use
            
        Returns:
            List of ClassificationResult objects
        """
        results = []
        
        for file_path, content in documents:
            try:
                result = self.classify_document(file_path, content, method)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch classification failed for {file_path}: {e}")
                fallback_result = self._create_fallback_result(file_path, content, str(e))
                results.append(fallback_result)
        
        self.logger.info(f"Batch classified {len(results)} documents")
        return results
    
    def apply_manual_override(self, doc_id: str, new_type: DocumentType, reason: str) -> bool:
        """
        Apply manual override to a previously classified document
        
        Args:
            doc_id: Document identifier (usually file path hash)
            new_type: New document type
            reason: Reason for override
            
        Returns:
            True if override was successful
        """
        try:
            # Find the classification result
            for result in self.classification_history:
                result_doc_id = self._generate_doc_id(result)
                if result_doc_id == doc_id:
                    # Update the result
                    result.document_type = new_type
                    result.manual_override_applied = True
                    result.override_reason = reason
                    result.can_override = False  # Already overridden
                    
                    # Update index assignments
                    indices = self.topology_manager.get_indices_for_document_type(new_type)
                    result.primary_index = indices[0] if indices else "general_content"
                    result.fallback_indices = indices[1:] if len(indices) > 1 else ["general_content"]
                    
                    # Store override history
                    self.override_history.append((doc_id, new_type, reason))
                    self.metrics["override_count"] += 1
                    
                    self.logger.info(f"Applied manual override for {doc_id}: {new_type.value}")
                    return True
            
            self.logger.warning(f"Document {doc_id} not found for override")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to apply override for {doc_id}: {e}")
            return False
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics and performance metrics"""
        if not self.classification_history:
            return {"message": "No classifications performed yet"}
        
        # Calculate confidence distribution
        confidences = [r.confidence for r in self.classification_history]
        confidence_levels = [r.to_confidence_level().value for r in self.classification_history]
        
        # Document type distribution
        type_counts = {}
        for result in self.classification_history:
            doc_type = result.document_type.value
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        # Method distribution
        method_counts = {}
        for result in self.classification_history:
            method = result.classification_method.value
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            "total_documents": len(self.classification_history),
            "document_type_distribution": type_counts,
            "classification_method_distribution": method_counts,
            "confidence_stats": {
                "average": sum(confidences) / len(confidences),
                "min": min(confidences),
                "max": max(confidences),
                "distribution": {level: confidence_levels.count(level) for level in set(confidence_levels)}
            },
            "override_stats": {
                "total_overrides": len(self.override_history),
                "override_rate": len(self.override_history) / len(self.classification_history)
            },
            "performance_metrics": self.metrics
        }
    
    def _create_manual_override_result(self, file_path: str, content: str, doc_type: DocumentType) -> ClassificationResult:
        """Create classification result for manual override"""
        indices = self.topology_manager.get_indices_for_document_type(doc_type)
        
        return ClassificationResult(
            document_type=doc_type,
            confidence=1.0,  # Perfect confidence for manual override
            classification_method=ClassificationMethod.MANUAL_OVERRIDE,
            alternative_types=[],
            reasoning="Manual override by user",
            key_features=["User specified classification"],
            classification_time=datetime.now(),
            primary_index=indices[0] if indices else "general_content",
            fallback_indices=indices[1:] if len(indices) > 1 else ["general_content"],
            can_override=False,  # Already overridden
            manual_override_applied=True,
            feature_extraction_quality=1.0,
            content_analysis_quality=1.0
        )
    
    def _create_fallback_result(self, file_path: str, content: str, error: str) -> ClassificationResult:
        """Create fallback classification result for errors"""
        return ClassificationResult(
            document_type=DocumentType.GENERAL_TEXT,
            confidence=0.1,  # Very low confidence
            classification_method=ClassificationMethod.RULE_BASED,
            alternative_types=[],
            reasoning=f"Fallback classification due to error: {error}",
            key_features=["Error fallback"],
            classification_time=datetime.now(),
            primary_index="general_content",
            fallback_indices=[],
            can_override=True,
            feature_extraction_quality=0.1,
            content_analysis_quality=0.1
        )
    
    def _generate_doc_id(self, result: ClassificationResult) -> str:
        """Generate unique document ID for tracking"""
        # Use a combination of filename and content hash
        content_sample = result.key_features[0] if result.key_features else ""
        unique_string = f"{content_sample}_{result.classification_time.isoformat()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]
    
    def _update_metrics(self, result: ClassificationResult, processing_time: float):
        """Update performance metrics"""
        self.metrics["total_classifications"] += 1
        
        if result.classification_method == ClassificationMethod.RULE_BASED:
            self.metrics["rule_based_count"] += 1
        elif result.classification_method == ClassificationMethod.LLM_ANALYSIS:
            self.metrics["llm_count"] += 1
        elif result.classification_method == ClassificationMethod.HYBRID:
            self.metrics["hybrid_count"] += 1
        elif result.classification_method == ClassificationMethod.MANUAL_OVERRIDE:
            self.metrics["override_count"] += 1
        
        # Update running averages
        total = self.metrics["total_classifications"]
        current_avg_conf = self.metrics["avg_confidence"]
        current_avg_time = self.metrics["avg_processing_time"]
        
        self.metrics["avg_confidence"] = ((current_avg_conf * (total - 1)) + result.confidence) / total
        self.metrics["avg_processing_time"] = ((current_avg_time * (total - 1)) + processing_time) / total


# Example usage and testing functions
def create_test_classifier() -> DocumentTypeClassifier:
    """Create a test document type classifier"""
    from multi_index_architecture import IndexTypologyManager
    
    topology_manager = IndexTypologyManager()
    classifier = DocumentTypeClassifier(topology_manager)
    
    return classifier


def test_classification_examples():
    """Test the classifier with example documents"""
    classifier = create_test_classifier()
    
    # Test documents
    test_docs = [
        ("api_documentation.md", """
        # REST API Documentation
        
        ## Authentication Endpoint
        
        ```python
        def authenticate(username, password):
            return jwt.encode(payload, secret_key)
        ```
        
        The API uses JWT tokens for authentication. Send requests to `/api/auth`.
        """),
        
        ("financial_report.docx", """
        Q3 2024 Financial Report
        
        Revenue: $2.5M
        Profit Margin: 15.2%
        
        Key Performance Indicators:
        - Customer acquisition cost: $150
        - ROI: 24%
        - Market share: 12%
        
        Executive Summary: Our quarterly performance shows strong growth...
        """),
        
        ("research_paper.pdf", """
        Abstract: This paper presents a novel approach to machine learning...
        
        1. Introduction
        The field of artificial intelligence has seen remarkable advances...
        
        2. Methodology
        We employed a randomized controlled experiment...
        
        References:
        [1] Smith, J. (2023). AI Advances. Journal of ML, 15(3), 45-67.
        [2] Doe, A. et al. (2024). Deep Learning Methods. arXiv:2024.1234
        """),
        
        ("data.csv", """
        id,name,category,value
        1,Product A,Electronics,299.99
        2,Product B,Books,19.99
        3,Product C,Clothing,49.99
        """)
    ]
    
    print("Document Classification Test Results")
    print("=" * 50)
    
    for file_path, content in test_docs:
        result = classifier.classify_document(file_path, content)
        
        print(f"\nFile: {file_path}")
        print(f"Classification: {result.document_type.value}")
        print(f"Confidence: {result.confidence:.2f} ({result.to_confidence_level().value})")
        print(f"Primary Index: {result.primary_index}")
        print(f"Method: {result.classification_method.value}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Key Features: {', '.join(result.key_features[:3])}")
        
        if result.alternative_types:
            alternatives = ', '.join([f"{t.value}({s:.2f})" for t, s in result.alternative_types[:2]])
            print(f"Alternatives: {alternatives}")
    
    # Print overall statistics
    print(f"\n{'-' * 50}")
    print("Classification Statistics:")
    stats = classifier.get_classification_stats()
    print(f"Total documents: {stats['total_documents']}")
    print(f"Average confidence: {stats['confidence_stats']['average']:.2f}")
    print("Document types:", ", ".join(f"{k}({v})" for k, v in stats['document_type_distribution'].items()))


if __name__ == "__main__":
    # Run the test examples
    test_classification_examples() 