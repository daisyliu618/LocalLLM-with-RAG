"""
LLM-Powered Agentic Query Classification System

This module implements an intelligent query classification system that uses
Large Language Models to understand query intent and recommend optimal
retrieval strategies, replacing the rule-based SmartQueryClassifier.

Architecture:
- LLM-powered query analysis with structured output
- Dynamic strategy selection based on query characteristics  
- Confidence scoring and reasoning transparency
- Fallback mechanisms for reliability
- Prompt engineering framework for optimization
"""

import json
import re
import logging
import os
import httpx
import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime

# LLM Provider imports (will be implemented with actual providers)
from typing import Protocol


class QueryType(Enum):
    """Enhanced query type classification"""
    SEMANTIC = "semantic"           # Conceptual, similarity-based queries
    ANALYTICAL = "analytical"      # Count, aggregate, numerical analysis
    EXACT_MATCH = "exact_match"    # Precise keyword matches
    MIXED = "mixed"                # Hybrid queries needing multiple strategies
    CONVERSATIONAL = "conversational"  # Natural dialog queries
    STRUCTURED = "structured"      # SQL-like, data queries


class SearchStrategy(Enum):
    """Recommended search strategies"""
    SEMANTIC_HEAVY = "semantic_heavy"      # 80% semantic, 20% keyword
    KEYWORD_HEAVY = "keyword_heavy"        # 20% semantic, 80% keyword  
    BALANCED = "balanced"                  # 60% semantic, 40% keyword
    STRUCTURED = "structured"             # Direct data querying
    AUTO_ROUTED = "auto_routed"           # LLM selects strategy dynamically
    MULTI_INDEX = "multi_index"           # Query multiple specialized indices


class QueryIntent(Enum):
    """Detailed query intent classification"""
    FIND = "find"                    # Find specific information
    COUNT = "count"                  # Count items/entities  
    AGGREGATE = "aggregate"          # Sum, average, min/max operations
    COMPARE = "compare"              # Compare entities or concepts
    FILTER = "filter"               # Filter data by criteria
    DESCRIBE = "describe"           # Explain or describe concepts
    ANALYZE = "analyze"             # Deep analytical queries
    NAVIGATE = "navigate"           # Browse or explore content


@dataclass
class AgenticQueryClassification:
    """
    Comprehensive query classification result from LLM analysis
    """
    # Core classification
    query_type: QueryType
    intent: QueryIntent
    strategy: SearchStrategy
    
    # Confidence and reasoning
    confidence: float  # 0.0 to 1.0
    reasoning: str     # LLM explanation of classification
    
    # Extracted elements
    entities: List[str]           # Named entities (people, places, organizations)
    concepts: List[str]           # Abstract concepts and themes
    keywords: List[str]           # Important keywords for search
    
    # Strategy parameters
    semantic_weight: float        # Recommended semantic search weight
    keyword_weight: float         # Recommended keyword search weight
    
    # Advanced features
    complexity_score: float       # Query complexity (1-10 scale)
    index_routing: List[str]      # Recommended indices to query
    requires_aggregation: bool    # Needs data aggregation
    temporal_aspects: Optional[Dict[str, Any]]  # Time-based query elements
    
    # Metadata
    classification_time: datetime
    llm_provider: str
    prompt_version: str
    fallback_used: bool = False


class LLMProvider(Protocol):
    """Protocol for LLM provider implementations"""
    
    async def generate_completion(
        self, 
        prompt: str, 
        temperature: float = 0.1,
        max_tokens: int = 1000
    ) -> str:
        """Generate completion from LLM"""
        ...
        
    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        """Estimate API call cost"""
        ...


@dataclass
class PromptTemplate:
    """Structured prompt template for query classification"""
    name: str
    version: str
    system_prompt: str
    user_prompt_template: str
    expected_output_schema: Dict[str, Any]
    temperature: float = 0.1
    max_tokens: int = 1000


class PromptEngineeringFramework:
    """
    Framework for managing, testing, and optimizing classification prompts
    """
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.performance_metrics: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_template(self, template: PromptTemplate):
        """Register a new prompt template"""
        self.templates[template.name] = template
        self.logger.info(f"Registered prompt template: {template.name} v{template.version}")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get prompt template by name"""
        return self.templates.get(name)
    
    def create_classification_prompt(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Create a structured prompt for query classification
        
        Args:
            query: User query to classify
            context: Additional context (available indices, document types, etc.)
        """
        template = self.get_template("query_classification_v1")
        if not template:
            raise ValueError("Query classification template not found")
        
        context = context or {}
        
        return template.user_prompt_template.format(
            query=query,
            available_indices=context.get('available_indices', ['semantic', 'keyword', 'structured']),
            document_types=context.get('document_types', ['pdf', 'txt', 'csv']),
            timestamp=datetime.now().isoformat(),
            llm_provider=context.get('llm_provider', 'MockProvider')
        )
    
    def validate_output_schema(self, llm_output: str, template_name: str) -> bool:
        """Validate LLM output against expected schema"""
        template = self.get_template(template_name)
        if not template:
            return False
            
        try:
            parsed_output = json.loads(llm_output)
            # Add JSON schema validation logic here
            return True
        except json.JSONDecodeError:
            return False


class LLMResponseParser:
    """
    Robust parser for structured LLM responses with error handling
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_classification_response(
        self, 
        llm_output: str, 
        fallback_classification: Optional[AgenticQueryClassification] = None
    ) -> AgenticQueryClassification:
        """
        Parse LLM response into structured classification
        
        Args:
            llm_output: Raw LLM response
            fallback_classification: Fallback if parsing fails
        """
        try:
            # Try to parse JSON directly
            if llm_output.strip().startswith('{'):
                data = json.loads(llm_output.strip())
                return self._create_classification_from_dict(data, llm_output)
            
            # Try to extract JSON from mixed content
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                return self._create_classification_from_dict(data, llm_output)
            
            # Try structured text parsing
            return self._parse_structured_text(llm_output)
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            if fallback_classification:
                return fallback_classification
            else:
                return self._create_default_classification(llm_output)
    
    def _create_classification_from_dict(
        self, 
        data: Dict[str, Any], 
        raw_response: str
    ) -> AgenticQueryClassification:
        """Create classification from parsed dictionary"""
        return AgenticQueryClassification(
            query_type=QueryType(data.get('query_type', 'mixed')),
            intent=QueryIntent(data.get('intent', 'find')),
            strategy=SearchStrategy(data.get('strategy', 'balanced')),
            confidence=float(data.get('confidence', 0.5)),
            reasoning=data.get('reasoning', 'Parsed from LLM response'),
            entities=data.get('entities', []),
            concepts=data.get('concepts', []),
            keywords=data.get('keywords', []),
            semantic_weight=float(data.get('semantic_weight', 0.6)),
            keyword_weight=float(data.get('keyword_weight', 0.4)),
            complexity_score=float(data.get('complexity_score', 5.0)),
            index_routing=data.get('index_routing', ['semantic', 'keyword']),
            requires_aggregation=bool(data.get('requires_aggregation', False)),
            temporal_aspects=data.get('temporal_aspects'),
            classification_time=datetime.now(),
            llm_provider=data.get('llm_provider', 'unknown'),
            prompt_version=data.get('prompt_version', 'v1'),
            fallback_used=False
        )
    
    def _parse_structured_text(self, text: str) -> AgenticQueryClassification:
        """Parse structured text response"""
        # Implementation for parsing non-JSON structured responses
        return self._create_default_classification(text)
    
    def _create_default_classification(self, original_query: str) -> AgenticQueryClassification:
        """Create default classification when parsing fails"""
        return AgenticQueryClassification(
            query_type=QueryType.MIXED,
            intent=QueryIntent.FIND,
            strategy=SearchStrategy.BALANCED,
            confidence=0.3,
            reasoning="Failed to parse LLM response, using default classification",
            entities=[],
            concepts=[],
            keywords=original_query.split()[:5],
            semantic_weight=0.6,
            keyword_weight=0.4,
            complexity_score=5.0,
            index_routing=['semantic', 'keyword'],
            requires_aggregation=False,
            temporal_aspects=None,
            classification_time=datetime.now(),
            llm_provider='parser_fallback',
            prompt_version='fallback',
            fallback_used=True
        )


class AgenticQueryClassifier:
    """
    Main LLM-powered query classifier that replaces SmartQueryClassifier
    """
    
    def __init__(
        self, 
        llm_provider: LLMProvider,
        prompt_framework: PromptEngineeringFramework,
        response_parser: LLMResponseParser,
        fallback_classifier: Optional[Any] = None  # Legacy SmartQueryClassifier
    ):
        self.llm_provider = llm_provider
        self.prompt_framework = prompt_framework
        self.response_parser = response_parser
        self.fallback_classifier = fallback_classifier
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.classification_count = 0
        self.fallback_count = 0
        self.average_response_time = 0.0
        
        # Initialize prompt templates
        self._setup_default_prompts()
    
    def _setup_default_prompts(self):
        """Setup default prompt templates"""
        classification_template = PromptTemplate(
            name="query_classification_v1",
            version="1.0",
            system_prompt="""
            You are an expert query classifier for a Retrieval Augmented Generation (RAG) system.
            Your job is to analyze user queries and recommend optimal retrieval strategies.
            
            Provide structured analysis with reasoning for each decision.
            Consider query complexity, intent, entities, and optimal search strategies.
            
            Always respond with valid JSON matching the expected schema.
            """,
            user_prompt_template="""
            QUERY TO CLASSIFY: "{query}"
            
            AVAILABLE SEARCH INDICES: {available_indices}
            DOCUMENT TYPES: {document_types}
            TIMESTAMP: {timestamp}
            
            Please analyze this query and provide a comprehensive classification.
            
            Consider:
            1. What type of query is this? (semantic, analytical, exact_match, mixed, conversational, structured)
            2. What is the user's intent? (find, count, aggregate, compare, filter, describe, analyze, navigate)
            3. What retrieval strategy would work best?
            4. What entities, concepts, and keywords are important?
            5. How complex is this query on a scale of 1-10?
            6. Does this require data aggregation or special processing?
            
            Respond with JSON in this exact format:
            {{
                "query_type": "semantic|analytical|exact_match|mixed|conversational|structured",
                "intent": "find|count|aggregate|compare|filter|describe|analyze|navigate",
                "strategy": "semantic_heavy|keyword_heavy|balanced|structured|auto_routed|multi_index",
                "confidence": 0.0-1.0,
                "reasoning": "Detailed explanation of classification decision",
                "entities": ["list", "of", "named", "entities"],
                "concepts": ["list", "of", "abstract", "concepts"],
                "keywords": ["important", "keywords", "for", "search"],
                "semantic_weight": 0.0-1.0,
                "keyword_weight": 0.0-1.0,
                "complexity_score": 1-10,
                "index_routing": ["recommended", "indices"],
                "requires_aggregation": true|false,
                "temporal_aspects": {{"time_period": "if relevant", "temporal_operators": "before/after/during"}},
                "llm_provider": "{llm_provider}",
                "prompt_version": "v1"
            }}
            """,
            expected_output_schema={
                "type": "object",
                "required": ["query_type", "intent", "strategy", "confidence", "reasoning"],
                "properties": {
                    "query_type": {"type": "string", "enum": ["semantic", "analytical", "exact_match", "mixed", "conversational", "structured"]},
                    "intent": {"type": "string", "enum": ["find", "count", "aggregate", "compare", "filter", "describe", "analyze", "navigate"]},
                    "strategy": {"type": "string", "enum": ["semantic_heavy", "keyword_heavy", "balanced", "structured", "auto_routed", "multi_index"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                }
            }
        )
        
        self.prompt_framework.register_template(classification_template)
    
    async def classify_query_async(
        self, 
        query: str, 
        context: Dict[str, Any] = None
    ) -> AgenticQueryClassification:
        """
        Asynchronously classify query using LLM
        
        Args:
            query: User query to classify
            context: Additional context for classification
        """
        start_time = datetime.now()
        
        try:
            # Create prompt
            prompt = self.prompt_framework.create_classification_prompt(query, context)
            
            # Get LLM response
            llm_response = await self.llm_provider.generate_completion(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse response
            classification = self.response_parser.parse_classification_response(llm_response)
            
            # Update performance metrics
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(response_time, classification.fallback_used)
            
            self.logger.info(f"Query classified as {classification.query_type.value} "
                           f"with {classification.confidence:.2f} confidence")
            
            return classification
            
        except Exception as e:
            self.logger.error(f"LLM classification failed: {e}")
            return await self._fallback_classification(query, context)
    
    def classify_query(
        self, 
        query: str, 
        context: Dict[str, Any] = None
    ) -> AgenticQueryClassification:
        """
        Synchronous wrapper for query classification
        """
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, create a new one in a thread
                import threading
                
                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self.classify_query_async(query, context))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    return future.result(timeout=30)
                    
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                return asyncio.run(self.classify_query_async(query, context))
                
        except Exception as e:
            self.logger.error(f"Synchronous classification failed: {e}")
            # Fallback to sync-only classification
            return asyncio.run(self._fallback_classification(query, context))
    
    async def _fallback_classification(
        self, 
        query: str, 
        context: Dict[str, Any] = None
    ) -> AgenticQueryClassification:
        """
        Fallback to rule-based classification when LLM fails
        """
        self.fallback_count += 1
        
        if self.fallback_classifier:
            # Use legacy SmartQueryClassifier
            legacy_result = self.fallback_classifier.classify_query(query)
            
            # Convert to new format
            return self._convert_legacy_classification(legacy_result, query)
        else:
            # Create basic rule-based classification
            return self._basic_rule_classification(query)
    
    def _convert_legacy_classification(self, legacy_result, query: str) -> AgenticQueryClassification:
        """Convert legacy SmartQueryClassifier result to new format"""
        # Map legacy types to new enum values
        type_mapping = {
            'semantic': QueryType.SEMANTIC,
            'analytical': QueryType.ANALYTICAL,
            'exact': QueryType.EXACT_MATCH,
            'mixed': QueryType.MIXED
        }
        
        strategy_mapping = {
            'semantic_heavy': SearchStrategy.SEMANTIC_HEAVY,
            'keyword_heavy': SearchStrategy.KEYWORD_HEAVY,
            'balanced': SearchStrategy.BALANCED,
            'structured': SearchStrategy.STRUCTURED
        }
        
        return AgenticQueryClassification(
            query_type=type_mapping.get(legacy_result.query_type, QueryType.MIXED),
            intent=QueryIntent(legacy_result.intent) if hasattr(legacy_result, 'intent') else QueryIntent.FIND,
            strategy=strategy_mapping.get(legacy_result.suggested_strategy, SearchStrategy.BALANCED),
            confidence=legacy_result.confidence,
            reasoning=f"Fallback to rule-based: {legacy_result.reasoning}",
            entities=legacy_result.entities or [],
            concepts=[],  # Legacy doesn't extract concepts
            keywords=query.split(),
            semantic_weight=0.6 if legacy_result.suggested_strategy == 'semantic_heavy' else 0.4,
            keyword_weight=0.4 if legacy_result.suggested_strategy == 'semantic_heavy' else 0.6,
            complexity_score=5.0,  # Default complexity
            index_routing=['semantic', 'keyword'],
            requires_aggregation=legacy_result.numerical_indicators if hasattr(legacy_result, 'numerical_indicators') else False,
            temporal_aspects=None,
            classification_time=datetime.now(),
            llm_provider='fallback_rule_based',
            prompt_version='legacy',
            fallback_used=True
        )
    
    def _basic_rule_classification(self, query: str) -> AgenticQueryClassification:
        """Basic rule-based classification when no fallback is available"""
        query_lower = query.lower()
        
        # Simple rule-based classification
        if any(word in query_lower for word in ['how many', 'count', 'number of']):
            query_type = QueryType.ANALYTICAL
            intent = QueryIntent.COUNT
            strategy = SearchStrategy.STRUCTURED
        elif any(word in query_lower for word in ['like', 'similar', 'resembles']):
            query_type = QueryType.SEMANTIC
            intent = QueryIntent.FIND
            strategy = SearchStrategy.SEMANTIC_HEAVY
        else:
            query_type = QueryType.MIXED
            intent = QueryIntent.FIND  
            strategy = SearchStrategy.BALANCED
        
        return AgenticQueryClassification(
            query_type=query_type,
            intent=intent,
            strategy=strategy,
            confidence=0.4,
            reasoning="Basic rule-based fallback classification",
            entities=[],
            concepts=[],
            keywords=query.split(),
            semantic_weight=0.6,
            keyword_weight=0.4,
            complexity_score=5.0,
            index_routing=['semantic', 'keyword'],
            requires_aggregation=False,
            temporal_aspects=None,
            classification_time=datetime.now(),
            llm_provider='basic_rule_fallback',
            prompt_version='basic',
            fallback_used=True
        )
    
    def _update_metrics(self, response_time: float, fallback_used: bool):
        """Update performance metrics"""
        self.classification_count += 1
        if fallback_used:
            self.fallback_count += 1
        
        # Update average response time
        self.average_response_time = (
            (self.average_response_time * (self.classification_count - 1) + response_time) 
            / self.classification_count
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get classifier performance metrics"""
        return {
            'total_classifications': self.classification_count,
            'fallback_count': self.fallback_count,
            'fallback_rate': self.fallback_count / max(self.classification_count, 1),
            'average_response_time': self.average_response_time,
            'success_rate': (self.classification_count - self.fallback_count) / max(self.classification_count, 1)
        }


# Production-ready LLM provider implementations
class GeminiProvider:
    """Google Gemini LLM provider implementation"""
    
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.logger = logging.getLogger(__name__)
    
    async def generate_completion(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """Generate completion from Gemini API"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{self.base_url}/models/{self.model}:generateContent"
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens,
                        "candidateCount": 1
                    }
                }
                
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key
                }
                
                response = await client.post(url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    self.logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                    raise Exception(f"Gemini API error: {response.status_code}")
                
                data = response.json()
                
                if 'candidates' in data and len(data['candidates']) > 0:
                    content = data['candidates'][0]['content']['parts'][0]['text']
                    return content.strip()
                else:
                    raise Exception("No valid response from Gemini API")
                    
        except httpx.TimeoutException:
            self.logger.error("Gemini API request timed out")
            raise Exception("Gemini API timeout")
        except Exception as e:
            self.logger.error(f"Gemini API error: {e}")
            raise
    
    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        """Estimate API call cost for Gemini"""
        # Gemini 1.5 Flash pricing (approximate)
        input_tokens = len(prompt.split()) * 1.3  # Rough token estimation
        output_tokens = max_tokens
        
        # Pricing per 1M tokens (as of 2024)
        input_cost_per_million = 0.075  # $0.075 per 1M input tokens
        output_cost_per_million = 0.30   # $0.30 per 1M output tokens
        
        input_cost = (input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * output_cost_per_million
        
        return input_cost + output_cost


class OllamaProvider:
    """Ollama local LLM provider implementation"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    async def generate_completion(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """Generate completion from Ollama API"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                url = f"{self.base_url}/api/generate"
                
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
                
                response = await client.post(url, json=payload)
                
                if response.status_code != 200:
                    self.logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    raise Exception(f"Ollama API error: {response.status_code}")
                
                data = response.json()
                
                if 'response' in data:
                    return data['response'].strip()
                else:
                    raise Exception("No valid response from Ollama API")
                    
        except httpx.ConnectError:
            self.logger.error("Could not connect to Ollama. Make sure Ollama is running.")
            raise Exception("Ollama connection failed - is Ollama running?")
        except httpx.TimeoutException:
            self.logger.error("Ollama API request timed out")
            raise Exception("Ollama API timeout")
        except Exception as e:
            self.logger.error(f"Ollama API error: {e}")
            raise
    
    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        """Estimate API call cost for Ollama (local models have no cost)"""
        return 0.0


# Configuration and Factory classes
@dataclass
class AgenticClassifierConfig:
    """Configuration for agentic query classifier"""
    llm_provider_type: str = "gemini"  # "gemini", "ollama", "openai"
    model_name: str = None  # Model-specific name
    api_key: str = None     # API key for external providers
    base_url: str = None    # Base URL for custom endpoints
    enable_fallback: bool = True
    max_retry_attempts: int = 3
    timeout_seconds: float = 30.0
    cache_classifications: bool = True
    log_level: str = "INFO"


class AgenticClassifierFactory:
    """Factory for creating configured agentic classifiers"""
    
    @staticmethod
    def create_classifier(
        config: AgenticClassifierConfig,
        fallback_classifier: Optional[Any] = None
    ) -> AgenticQueryClassifier:
        """Create and configure an agentic query classifier"""
        
        # Create LLM provider
        if config.llm_provider_type == "gemini":
            model = config.model_name or "gemini-1.5-flash"
            llm_provider = GeminiProvider(
                api_key=config.api_key,
                model=model
            )
        elif config.llm_provider_type == "ollama":
            base_url = config.base_url or "http://localhost:11434"
            model = config.model_name or "llama3.2"
            llm_provider = OllamaProvider(
                base_url=base_url,
                model=model
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {config.llm_provider_type}")
        
        # Create framework components
        prompt_framework = PromptEngineeringFramework()
        response_parser = LLMResponseParser()
        
        # Create classifier
        classifier = AgenticQueryClassifier(
            llm_provider=llm_provider,
            prompt_framework=prompt_framework,
            response_parser=response_parser,
            fallback_classifier=fallback_classifier if config.enable_fallback else None
        )
        
        return classifier


# Backward compatibility wrapper
class SmartQueryClassifierWrapper:
    """
    Wrapper to provide backward compatibility with existing SmartQueryClassifier interface
    """
    
    def __init__(self, agentic_classifier: AgenticQueryClassifier):
        self.agentic_classifier = agentic_classifier
    
    def classify_query(self, query: str):
        """Legacy interface method"""
        agentic_result = self.agentic_classifier.classify_query(query)
        
        # Create legacy-compatible result object
        result = type('QueryClassification', (), {})()
        result.query_type = agentic_result.query_type.value
        result.intent = agentic_result.intent.value
        result.confidence = agentic_result.confidence
        result.suggested_strategy = agentic_result.strategy.value
        result.reasoning = agentic_result.reasoning
        result.entities = agentic_result.entities
        result.numerical_indicators = agentic_result.requires_aggregation
        
        return result


# Example usage and testing functions
def create_test_examples() -> List[Tuple[str, Dict[str, Any]]]:
    """Create test examples for validating the classification system"""
    return [
        (
            "How many people went to Stanford University?",
            {
                "expected_type": QueryType.ANALYTICAL,
                "expected_intent": QueryIntent.COUNT,
                "expected_strategy": SearchStrategy.STRUCTURED
            }
        ),
        (
            "What are some restaurants similar to Ethiopian cuisine?",
            {
                "expected_type": QueryType.SEMANTIC,
                "expected_intent": QueryIntent.FIND,
                "expected_strategy": SearchStrategy.SEMANTIC_HEAVY
            }
        ),
        (
            "Find the exact phone number for John Smith",
            {
                "expected_type": QueryType.EXACT_MATCH,
                "expected_intent": QueryIntent.FIND,
                "expected_strategy": SearchStrategy.KEYWORD_HEAVY
            }
        ),
        (
            "Compare the performance of Tesla vs BMW in the luxury car market",
            {
                "expected_type": QueryType.MIXED,
                "expected_intent": QueryIntent.COMPARE,
                "expected_strategy": SearchStrategy.BALANCED
            }
        )
    ]


if __name__ == "__main__":
    # Example usage with fallback to mock for testing
    try:
        config = AgenticClassifierConfig(
            llm_provider_type="gemini",
            enable_fallback=True
        )
        
        classifier = AgenticClassifierFactory.create_classifier(config)
        
        # Test classification
        test_query = "How many Ethiopian restaurants are there in San Francisco?"
        result = classifier.classify_query(test_query)
        
        print(f"Query: {test_query}")
        print(f"Classification: {result.query_type.value}")
        print(f"Intent: {result.intent.value}")
        print(f"Strategy: {result.strategy.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.reasoning}")
        
    except Exception as e:
        print(f"Note: {e}")
        print("For testing without API keys, use the test suite with MockLLMProvider") 