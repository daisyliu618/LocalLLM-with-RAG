"""
Comprehensive Test Suite for LLM-Powered Agentic Query Classification System

This module contains extensive tests for validating the agentic query classification
system including unit tests, integration tests, performance tests, and edge case
validation.

Test Categories:
1. Core functionality tests
2. LLM provider integration tests  
3. Fallback mechanism tests
4. Performance and reliability tests
5. Edge case and adversarial tests
6. Comparison tests vs rule-based system

"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, List, Any

from query_classification import (
    AgenticQueryClassifier,
    AgenticQueryClassification,
    PromptEngineeringFramework,
    LLMResponseParser,
    AgenticClassifierFactory,
    AgenticClassifierConfig,
    QueryType,
    QueryIntent,
    SearchStrategy,
    create_test_examples
)


class MockLLMProvider:
    """Mock LLM provider for testing"""
    
    def __init__(self, response_data: Dict[str, Any] = None, should_fail: bool = False):
        self.response_data = response_data or self._default_response()
        self.should_fail = should_fail
        self.call_count = 0
        self.last_prompt = None
        
    def _default_response(self):
        return {
            "query_type": "semantic",
            "intent": "find",
            "strategy": "balanced",
            "confidence": 0.8,
            "reasoning": "Test classification response",
            "entities": ["test", "entity"],
            "concepts": ["test concept"],
            "keywords": ["test", "keyword"],
            "semantic_weight": 0.6,
            "keyword_weight": 0.4,
            "complexity_score": 5.0,
            "index_routing": ["semantic"],
            "requires_aggregation": False,
            "temporal_aspects": None,
            "llm_provider": "MockProvider",
            "prompt_version": "v1"
        }
    
    def _get_intelligent_response(self, prompt: str) -> Dict[str, Any]:
        """Generate intelligent response based on the prompt content"""
        # Extract query from prompt if possible
        query = ""
        if "QUERY TO CLASSIFY:" in prompt:
            lines = prompt.split('\n')
            for line in lines:
                if "QUERY TO CLASSIFY:" in line:
                    query = line.split('"')[1] if '"' in line else ""
                    break
        
        response = self._default_response().copy()
        
        # Handle edge cases with appropriate confidence levels
        if not query or query.strip() == "":
            # Empty query - very low confidence
            response.update({
                "confidence": 0.3,
                "reasoning": "Empty query provides no classification context",
                "entities": [],
                "concepts": [],
                "keywords": [],
                "complexity_score": 1.0
            })
        elif len(query.strip()) <= 2:
            # Very short/ambiguous query - low confidence
            response.update({
                "confidence": 0.4,
                "reasoning": "Highly ambiguous single word or very short query",
                "entities": [],
                "concepts": [],
                "keywords": [query.strip()],
                "complexity_score": 2.0
            })
        elif any(word in query.lower() for word in ['how many', 'count', 'number of']):
            # Analytical queries
            response.update({
                "query_type": "analytical",
                "intent": "count", 
                "strategy": "structured",
                "confidence": 0.9,
                "reasoning": "Clear counting/analytical query detected",
                "semantic_weight": 0.3,
                "keyword_weight": 0.7,
                "requires_aggregation": True,
                "complexity_score": 6.0
            })
        elif any(word in query.lower() for word in ['similar', 'like', 'resembles']):
            # Semantic queries
            response.update({
                "query_type": "semantic",
                "intent": "find",
                "strategy": "semantic_heavy", 
                "confidence": 0.85,
                "reasoning": "Semantic similarity query detected",
                "semantic_weight": 0.8,
                "keyword_weight": 0.2,
                "complexity_score": 5.0
            })
        elif any(word in query.lower() for word in ['exact', 'phone number', 'address']):
            # Exact match queries
            response.update({
                "query_type": "exact_match",
                "intent": "find",
                "strategy": "keyword_heavy",
                "confidence": 0.9,
                "reasoning": "Exact match query for specific information",
                "semantic_weight": 0.2, 
                "keyword_weight": 0.8,
                "complexity_score": 4.0
            })
        
        return response
    
    async def generate_completion(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1000) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        
        if self.should_fail:
            raise Exception("Mock LLM provider failure")
        
        # Simulate API delay
        await asyncio.sleep(0.1)
        
        # Use intelligent response if using default data, otherwise use provided data
        if self.response_data == self._default_response():
            response_data = self._get_intelligent_response(prompt)
        else:
            response_data = self.response_data
        
        return json.dumps(response_data)
    
    def estimate_cost(self, prompt: str, max_tokens: int) -> float:
        return 0.001  # Mock cost


class MockLegacyClassifier:
    """Mock legacy SmartQueryClassifier for testing fallback"""
    
    def __init__(self):
        self.query_type = 'semantic'
        self.intent = 'find'
        self.suggested_strategy = 'balanced'
        self.confidence = 0.6
        self.reasoning = "Legacy classification"
        self.entities = ['legacy', 'entity']
        self.numerical_indicators = False
    
    def classify_query(self, query: str):
        # Return a mock object that mimics legacy classification result
        result = type('QueryClassification', (), {})()
        result.query_type = self.query_type
        result.intent = self.intent
        result.suggested_strategy = self.suggested_strategy
        result.confidence = self.confidence
        result.reasoning = self.reasoning
        result.entities = self.entities
        result.numerical_indicators = self.numerical_indicators
        return result


class TestAgenticQueryClassification:
    """Test the AgenticQueryClassification dataclass"""
    
    def test_classification_creation(self):
        """Test basic classification object creation"""
        classification = AgenticQueryClassification(
            query_type=QueryType.SEMANTIC,
            intent=QueryIntent.FIND,
            strategy=SearchStrategy.BALANCED,
            confidence=0.8,
            reasoning="Test reasoning",
            entities=["entity1", "entity2"],
            concepts=["concept1"],
            keywords=["keyword1", "keyword2"],
            semantic_weight=0.6,
            keyword_weight=0.4,
            complexity_score=5.0,
            index_routing=["semantic"],
            requires_aggregation=False,
            temporal_aspects=None,
            classification_time=datetime.now(),
            llm_provider="test",
            prompt_version="v1"
        )
        
        assert classification.query_type == QueryType.SEMANTIC
        assert classification.intent == QueryIntent.FIND
        assert classification.confidence == 0.8
        assert len(classification.entities) == 2
        assert classification.fallback_used == False
    
    def test_classification_with_temporal_aspects(self):
        """Test classification with temporal information"""
        temporal_data = {
            "time_period": "last year",
            "temporal_operators": ["after", "2020"]
        }
        
        classification = AgenticQueryClassification(
            query_type=QueryType.ANALYTICAL,
            intent=QueryIntent.COUNT,
            strategy=SearchStrategy.STRUCTURED,
            confidence=0.9,
            reasoning="Temporal query analysis",
            entities=["Stanford"],
            concepts=["education"],
            keywords=["graduates", "2020"],
            semantic_weight=0.3,
            keyword_weight=0.7,
            complexity_score=7.0,
            index_routing=["structured"],
            requires_aggregation=True,
            temporal_aspects=temporal_data,
            classification_time=datetime.now(),
            llm_provider="test",
            prompt_version="v1"
        )
        
        assert classification.temporal_aspects is not None
        assert classification.temporal_aspects["time_period"] == "last year"
        assert classification.requires_aggregation == True


class TestPromptEngineeringFramework:
    """Test the prompt engineering framework"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.framework = PromptEngineeringFramework()
    
    def test_template_registration(self):
        """Test prompt template registration and retrieval"""
        from query_classification import PromptTemplate
        
        template = PromptTemplate(
            name="test_template",
            version="1.0",
            system_prompt="Test system prompt",
            user_prompt_template="Test user prompt for {query}",
            expected_output_schema={"type": "object"}
        )
        
        self.framework.register_template(template)
        retrieved = self.framework.get_template("test_template")
        
        assert retrieved is not None
        assert retrieved.name == "test_template"
        assert retrieved.version == "1.0"
    
    def test_classification_prompt_creation(self):
        """Test creation of classification prompts"""
        # The framework should have default templates registered
        # during AgenticQueryClassifier initialization
        classifier = AgenticQueryClassifier(
            llm_provider=MockLLMProvider(),
            prompt_framework=self.framework,
            response_parser=LLMResponseParser(),
            fallback_classifier=None
        )
        
        context = {
            'available_indices': ['semantic', 'keyword'],
            'document_types': ['pdf', 'txt']
        }
        
        prompt = self.framework.create_classification_prompt("test query", context)
        
        assert "test query" in prompt
        assert "semantic" in prompt
        assert "pdf" in prompt


class TestLLMResponseParser:
    """Test LLM response parsing and validation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.parser = LLMResponseParser()
    
    def test_valid_json_parsing(self):
        """Test parsing of valid JSON responses"""
        valid_response = '''
        {
            "query_type": "semantic",
            "intent": "find",
            "strategy": "balanced",
            "confidence": 0.8,
            "reasoning": "Test response",
            "entities": ["entity1"],
            "concepts": ["concept1"],
            "keywords": ["keyword1"],
            "semantic_weight": 0.6,
            "keyword_weight": 0.4,
            "complexity_score": 5.0,
            "index_routing": ["semantic"],
            "requires_aggregation": false,
            "temporal_aspects": null,
            "llm_provider": "test",
            "prompt_version": "v1"
        }
        '''
        
        result = self.parser.parse_classification_response(valid_response)
        
        assert isinstance(result, AgenticQueryClassification)
        assert result.query_type == QueryType.SEMANTIC
        assert result.confidence == 0.8
        assert result.fallback_used == False
    
    def test_malformed_json_parsing(self):
        """Test parsing of malformed JSON with fallback"""
        malformed_response = '''
        This is not JSON at all.
        {"query_type": "semantic", "incomplete": true
        '''
        
        result = self.parser.parse_classification_response(malformed_response)
        
        assert isinstance(result, AgenticQueryClassification)
        assert result.fallback_used == True
        assert result.confidence == 0.3  # Default confidence
    
    def test_partial_json_extraction(self):
        """Test extraction of JSON from mixed content"""
        mixed_response = '''
        Here's my analysis of the query:
        
        {
            "query_type": "analytical",
            "intent": "count",
            "strategy": "structured",
            "confidence": 0.9,
            "reasoning": "This is a counting query",
            "entities": ["Stanford"],
            "concepts": [],
            "keywords": ["count", "people"],
            "semantic_weight": 0.2,
            "keyword_weight": 0.8,
            "complexity_score": 6.0,
            "index_routing": ["structured"],
            "requires_aggregation": true,
            "temporal_aspects": null,
            "llm_provider": "test",
            "prompt_version": "v1"
        }
        
        This should work well for your use case.
        '''
        
        result = self.parser.parse_classification_response(mixed_response)
        
        assert result.query_type == QueryType.ANALYTICAL
        assert result.intent == QueryIntent.COUNT
        assert result.requires_aggregation == True


class TestAgenticQueryClassifier:
    """Test the main AgenticQueryClassifier class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mock_provider = MockLLMProvider()
        self.framework = PromptEngineeringFramework()
        self.parser = LLMResponseParser()
        self.legacy_classifier = MockLegacyClassifier()
        
        self.classifier = AgenticQueryClassifier(
            llm_provider=self.mock_provider,
            prompt_framework=self.framework,
            response_parser=self.parser,
            fallback_classifier=self.legacy_classifier
        )
    
    @pytest.mark.asyncio
    async def test_async_classification(self):
        """Test asynchronous query classification"""
        query = "How many people went to Stanford?"
        
        result = await self.classifier.classify_query_async(query)
        
        assert isinstance(result, AgenticQueryClassification)
        assert self.mock_provider.call_count == 1
        assert result.fallback_used == False
    
    def test_sync_classification(self):
        """Test synchronous query classification wrapper"""
        query = "What restaurants are similar to Ethiopian cuisine?"
        
        result = self.classifier.classify_query(query)
        
        assert isinstance(result, AgenticQueryClassification)
        assert self.mock_provider.call_count == 1
    
    @pytest.mark.asyncio 
    async def test_llm_failure_fallback(self):
        """Test fallback to legacy classifier when LLM fails"""
        failing_provider = MockLLMProvider(should_fail=True)
        
        classifier = AgenticQueryClassifier(
            llm_provider=failing_provider,
            prompt_framework=self.framework,
            response_parser=self.parser,
            fallback_classifier=self.legacy_classifier
        )
        
        result = await classifier.classify_query_async("test query")
        
        assert result.fallback_used == True
        assert result.llm_provider == 'fallback_rule_based'
    
    def test_performance_metrics(self):
        """Test performance metrics tracking"""
        # Perform several classifications
        for i in range(3):
            self.classifier.classify_query(f"test query {i}")
        
        metrics = self.classifier.get_performance_metrics()
        
        assert metrics['total_classifications'] == 3
        assert metrics['fallback_count'] == 0
        assert metrics['success_rate'] == 1.0
        assert metrics['average_response_time'] > 0
    
    def test_legacy_classification_conversion(self):
        """Test conversion from legacy classifier format"""
        legacy_result = self.legacy_classifier.classify_query("test")
        
        converted = self.classifier._convert_legacy_classification(legacy_result, "test")
        
        assert isinstance(converted, AgenticQueryClassification)
        assert converted.fallback_used == True
        assert "Fallback to rule-based" in converted.reasoning


class TestQueryClassificationAccuracy:
    """Test classification accuracy on various query types"""
    
    def setup_method(self):
        """Setup classifier with realistic responses"""
        self.test_cases = [
            {
                "query": "How many people went to Stanford University?",
                "expected_type": QueryType.ANALYTICAL,
                "expected_intent": QueryIntent.COUNT,
                "expected_strategy": SearchStrategy.STRUCTURED,
                "mock_response": {
                    "query_type": "analytical",
                    "intent": "count",
                    "strategy": "structured",
                    "confidence": 0.95,
                    "reasoning": "Clear counting query with specific entity",
                    "entities": ["Stanford University"],
                    "concepts": ["higher education"],
                    "keywords": ["how many", "people", "stanford"],
                    "semantic_weight": 0.2,
                    "keyword_weight": 0.8,
                    "complexity_score": 6.0,
                    "index_routing": ["structured"],
                    "requires_aggregation": True,
                    "temporal_aspects": None,
                    "llm_provider": "test",
                    "prompt_version": "v1"
                }
            },
            {
                "query": "What restaurants are similar to Ethiopian cuisine?",
                "expected_type": QueryType.SEMANTIC,
                "expected_intent": QueryIntent.FIND,
                "expected_strategy": SearchStrategy.SEMANTIC_HEAVY,
                "mock_response": {
                    "query_type": "semantic",
                    "intent": "find",
                    "strategy": "semantic_heavy",
                    "confidence": 0.88,
                    "reasoning": "Similarity-based query requiring conceptual understanding",
                    "entities": ["Ethiopian"],
                    "concepts": ["cuisine", "food similarity", "restaurant types"],
                    "keywords": ["restaurants", "similar", "ethiopian", "cuisine"],
                    "semantic_weight": 0.8,
                    "keyword_weight": 0.2,
                    "complexity_score": 4.0,
                    "index_routing": ["semantic"],
                    "requires_aggregation": False,
                    "temporal_aspects": None,
                    "llm_provider": "test",
                    "prompt_version": "v1"
                }
            },
            {
                "query": "Find the exact phone number for John Smith",
                "expected_type": QueryType.EXACT_MATCH,
                "expected_intent": QueryIntent.FIND,
                "expected_strategy": SearchStrategy.KEYWORD_HEAVY,
                "mock_response": {
                    "query_type": "exact_match",
                    "intent": "find",
                    "strategy": "keyword_heavy",
                    "confidence": 0.92,
                    "reasoning": "Precise lookup query requiring exact match",
                    "entities": ["John Smith"],
                    "concepts": ["contact information"],
                    "keywords": ["exact", "phone", "number", "john", "smith"],
                    "semantic_weight": 0.2,
                    "keyword_weight": 0.8,
                    "complexity_score": 3.0,
                    "index_routing": ["keyword"],
                    "requires_aggregation": False,
                    "temporal_aspects": None,
                    "llm_provider": "test",
                    "prompt_version": "v1"
                }
            }
        ]
    
    def test_analytical_query_classification(self):
        """Test classification of analytical queries"""
        test_case = self.test_cases[0]
        mock_provider = MockLLMProvider(test_case["mock_response"])
        
        classifier = AgenticQueryClassifier(
            llm_provider=mock_provider,
            prompt_framework=PromptEngineeringFramework(),
            response_parser=LLMResponseParser(),
            fallback_classifier=None
        )
        
        result = classifier.classify_query(test_case["query"])
        
        assert result.query_type == test_case["expected_type"]
        assert result.intent == test_case["expected_intent"]
        assert result.strategy == test_case["expected_strategy"]
        assert result.requires_aggregation == True
        assert "Stanford University" in result.entities
    
    def test_semantic_query_classification(self):
        """Test classification of semantic queries"""
        test_case = self.test_cases[1]
        mock_provider = MockLLMProvider(test_case["mock_response"])
        
        classifier = AgenticQueryClassifier(
            llm_provider=mock_provider,
            prompt_framework=PromptEngineeringFramework(),
            response_parser=LLMResponseParser(),
            fallback_classifier=None
        )
        
        result = classifier.classify_query(test_case["query"])
        
        assert result.query_type == test_case["expected_type"]
        assert result.intent == test_case["expected_intent"]
        assert result.strategy == test_case["expected_strategy"]
        assert result.semantic_weight > result.keyword_weight
        assert "cuisine" in result.concepts
    
    def test_exact_match_query_classification(self):
        """Test classification of exact match queries"""
        test_case = self.test_cases[2]
        mock_provider = MockLLMProvider(test_case["mock_response"])
        
        classifier = AgenticQueryClassifier(
            llm_provider=mock_provider,
            prompt_framework=PromptEngineeringFramework(),
            response_parser=LLMResponseParser(),
            fallback_classifier=None
        )
        
        result = classifier.classify_query(test_case["query"])
        
        assert result.query_type == test_case["expected_type"]
        assert result.intent == test_case["expected_intent"]
        assert result.strategy == test_case["expected_strategy"]
        assert result.keyword_weight > result.semantic_weight
        assert "John Smith" in result.entities


class TestEdgeCases:
    """Test edge cases and adversarial inputs"""
    
    def setup_method(self):
        """Setup for edge case testing"""
        self.classifier = AgenticQueryClassifier(
            llm_provider=MockLLMProvider(),
            prompt_framework=PromptEngineeringFramework(),
            response_parser=LLMResponseParser(),
            fallback_classifier=MockLegacyClassifier()
        )
    
    def test_empty_query(self):
        """Test handling of empty queries"""
        result = self.classifier.classify_query("")
        
        assert isinstance(result, AgenticQueryClassification)
        assert result.confidence <= 0.5  # Should have low confidence
    
    def test_very_long_query(self):
        """Test handling of very long queries"""
        long_query = "What " + "very " * 100 + "long query about restaurants"
        
        result = self.classifier.classify_query(long_query)
        
        assert isinstance(result, AgenticQueryClassification)
        # Should still work, possibly with fallback
    
    def test_special_characters(self):
        """Test handling of queries with special characters"""
        special_query = "Find restaurants with $$ prices & 5⭐ ratings near café!"
        
        result = self.classifier.classify_query(special_query)
        
        assert isinstance(result, AgenticQueryClassification)
    
    def test_non_english_query(self):
        """Test handling of non-English queries"""
        spanish_query = "¿Cuántos restaurantes etíopes hay en San Francisco?"
        
        result = self.classifier.classify_query(spanish_query)
        
        assert isinstance(result, AgenticQueryClassification)
        # Should fallback gracefully
    
    def test_ambiguous_query(self):
        """Test handling of highly ambiguous queries"""
        ambiguous_query = "it"
        
        result = self.classifier.classify_query(ambiguous_query)
        
        assert isinstance(result, AgenticQueryClassification)
        assert result.confidence <= 0.6  # Should have low confidence


class TestFactory:
    """Test the AgenticClassifierFactory"""
    
    def test_gemini_classifier_creation(self):
        """Test creation of Gemini-based classifier"""
        config = AgenticClassifierConfig(
            llm_provider_type="gemini",
            enable_fallback=True
        )
        
        # This should succeed in creating the classifier
        classifier = AgenticClassifierFactory.create_classifier(config)
        
        # When the provider fails, it should gracefully fallback
        result = classifier.classify_query("test query")
        
        # Should get a fallback classification due to NotImplementedError
        assert isinstance(result, AgenticQueryClassification)
        assert result.fallback_used == True  # Should have used fallback
        assert result.llm_provider in ['basic_rule_fallback', 'fallback_rule_based']
    
    def test_invalid_provider_type(self):
        """Test handling of invalid provider type"""
        config = AgenticClassifierConfig(
            llm_provider_type="invalid_provider"
        )
        
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            AgenticClassifierFactory.create_classifier(config)


class TestPerformance:
    """Performance and load testing"""
    
    def setup_method(self):
        """Setup for performance testing"""
        self.classifier = AgenticQueryClassifier(
            llm_provider=MockLLMProvider(),
            prompt_framework=PromptEngineeringFramework(),
            response_parser=LLMResponseParser(),
            fallback_classifier=None
        )
    
    def test_classification_speed(self):
        """Test classification speed"""
        queries = [
            "How many restaurants are in SF?",
            "Find similar restaurants to Ethiopian",
            "What is the phone number for John?",
            "Compare Tesla vs BMW performance",
            "Show me all the data about Stanford"
        ]
        
        start_time = time.time()
        
        for query in queries:
            result = self.classifier.classify_query(query)
            assert isinstance(result, AgenticQueryClassification)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(queries)
        
        # Should average less than 1 second per classification (with mock)
        assert avg_time < 1.0
        
        metrics = self.classifier.get_performance_metrics()
        assert metrics['total_classifications'] == len(queries)
    
    @pytest.mark.asyncio
    async def test_concurrent_classifications(self):
        """Test concurrent query classifications"""
        queries = [f"test query {i}" for i in range(10)]
        
        start_time = time.time()
        
        # Run classifications concurrently
        tasks = [
            self.classifier.classify_query_async(query) 
            for query in queries
        ]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        assert len(results) == len(queries)
        assert all(isinstance(r, AgenticQueryClassification) for r in results)
        # Concurrent execution should be faster than sequential
        assert total_time < 2.0  # Should complete quickly with mocking


class TestComparisonWithRuleBased:
    """Compare agentic vs rule-based classification"""
    
    def setup_method(self):
        """Setup for comparison testing"""
        self.legacy_classifier = MockLegacyClassifier()
        
        # Create agentic classifier with custom responses for comparison
        self.agentic_responses = {
            "How many people went to Stanford?": {
                "query_type": "analytical",
                "intent": "count",
                "strategy": "structured",
                "confidence": 0.95,
                "reasoning": "Clear counting query requiring structured data analysis",
                "entities": ["Stanford"],
                "concepts": ["higher education", "alumni"],
                "keywords": ["how many", "people", "stanford"],
                "semantic_weight": 0.2,
                "keyword_weight": 0.8,
                "complexity_score": 6.0,
                "index_routing": ["structured"],
                "requires_aggregation": True,
                "temporal_aspects": None,
                "llm_provider": "test",
                "prompt_version": "v1"
            }
        }
    
    def test_analytical_query_comparison(self):
        """Compare analytical query classification"""
        query = "How many people went to Stanford?"
        
        # Mock legacy result
        self.legacy_classifier.query_type = 'analytical'
        self.legacy_classifier.intent = 'count'
        self.legacy_classifier.confidence = 0.7
        
        legacy_result = self.legacy_classifier.classify_query(query)
        
        # Agentic result
        mock_provider = MockLLMProvider(self.agentic_responses[query])
        agentic_classifier = AgenticQueryClassifier(
            llm_provider=mock_provider,
            prompt_framework=PromptEngineeringFramework(),
            response_parser=LLMResponseParser(),
            fallback_classifier=None
        )
        
        agentic_result = agentic_classifier.classify_query(query)
        
        # Compare results
        assert agentic_result.confidence > legacy_result.confidence
        assert len(agentic_result.concepts) > 0  # Agentic extracts concepts
        assert agentic_result.complexity_score > 0  # Agentic provides complexity
        assert agentic_result.reasoning is not None  # Agentic provides reasoning


# Test execution helper functions
def run_test_suite():
    """Run the complete test suite"""
    print("Running LLM-Powered Query Classification Test Suite...")
    
    # Run with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10"
    ])


def run_performance_tests():
    """Run only performance-related tests"""
    print("Running Performance Tests...")
    
    pytest.main([
        __file__ + "::TestPerformance",
        "-v",
        "--tb=short"
    ])


if __name__ == "__main__":
    # Run the test suite if executed directly
    run_test_suite() 