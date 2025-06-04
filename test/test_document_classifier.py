"""
Comprehensive Test Suite for Document Type Classifier

Tests cover:
- Basic classification functionality
- Feature extraction accuracy
- Edge cases and error handling
- Performance and accuracy metrics
- Integration with multi-index architecture
- Manual override capabilities
- Batch processing
"""

import unittest
import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'agentic'))

from document_classifier import (
    DocumentTypeClassifier, RuleBasedClassifier, ContentAnalyzer,
    ClassificationResult, DocumentFeatures, ClassificationMethod,
    ConfidenceLevel, DocumentType
)
from multi_index_architecture import IndexTypologyManager


class TestContentAnalyzer(unittest.TestCase):
    """Test the ContentAnalyzer component"""
    
    def setUp(self):
        self.analyzer = ContentAnalyzer()
    
    def test_feature_extraction_basic(self):
        """Test basic feature extraction"""
        content = "This is a simple test document with some basic content."
        features = self.analyzer.extract_features("test.txt", content)
        
        self.assertEqual(features.filename, "test.txt")
        self.assertEqual(features.file_extension, ".txt")
        self.assertEqual(features.word_count, 10)
        self.assertFalse(features.has_code_blocks)
        self.assertFalse(features.has_tables)
        self.assertEqual(features.language, "en")
    
    def test_code_detection(self):
        """Test code block detection"""
        code_content = """
        Here's some Python code:
        
        ```python
        def hello_world():
            print("Hello, World!")
            return True
        ```
        
        And some inline `code` here.
        """
        
        features = self.analyzer.extract_features("readme.md", code_content)
        
        self.assertTrue(features.has_code_blocks)
        self.assertGreater(features.code_ratio, 0.05)
        self.assertGreater(features.technical_term_ratio, 0.0)
    
    def test_table_detection(self):
        """Test table detection"""
        table_content = """
        | Column 1 | Column 2 | Column 3 |
        |----------|----------|----------|
        | Data A   | Data B   | Data C   |
        | Value 1  | Value 2  | Value 3  |
        """
        
        features = self.analyzer.extract_features("data.md", table_content)
        
        self.assertTrue(features.has_tables)
    
    def test_citation_detection(self):
        """Test academic citation detection"""
        academic_content = """
        This paper builds on previous work [1] and extends the methodology
        described by Smith et al. (2023). The approach follows Johnson (2024)
        with modifications based on doi:10.1234/example.
        
        References:
        [1] Author, A. (2023). Paper Title. Journal Name.
        """
        
        features = self.analyzer.extract_features("paper.txt", academic_content)
        
        self.assertTrue(features.has_citations)
        self.assertGreater(features.academic_term_ratio, 0.0)
    
    def test_business_content_detection(self):
        """Test business content detection"""
        business_content = """
        Q3 Financial Report
        
        Revenue increased by 15% this quarter, with profit margins
        improving to 12.5%. Our ROI on marketing campaigns was 8.2%.
        
        Key Performance Indicators:
        - Customer acquisition cost: $125
        - Market share: 18%
        - Employee satisfaction: 87%
        """
        
        features = self.analyzer.extract_features("report.txt", business_content)
        
        self.assertGreater(features.business_term_ratio, 0.02)
        self.assertGreater(features.number_count, 5)
    
    def test_pattern_counting(self):
        """Test URL, email, and other pattern counting"""
        pattern_content = """
        Contact us at info@example.com or visit https://www.example.com
        You can also call (555) 123-4567 or email support@test.org
        
        Meeting scheduled for 12/15/2024 at 2:30 PM.
        Reference numbers: 12345, 67890, 11111
        """
        
        features = self.analyzer.extract_features("contact.txt", pattern_content)
        
        self.assertEqual(features.email_count, 2)
        self.assertEqual(features.url_count, 1)
        self.assertEqual(features.phone_count, 1)
        self.assertGreaterEqual(features.date_count, 1)
        self.assertGreater(features.number_count, 5)


class TestRuleBasedClassifier(unittest.TestCase):
    """Test the RuleBasedClassifier component"""
    
    def setUp(self):
        self.topology_manager = IndexTypologyManager()
        self.classifier = RuleBasedClassifier(self.topology_manager)
    
    def test_python_file_classification(self):
        """Test classification of Python code files"""
        python_content = """
        #!/usr/bin/env python3
        \"\"\"
        API Documentation Generator
        
        This module generates API documentation from code.
        \"\"\"
        
        import json
        import requests
        
        def generate_docs(api_endpoint):
            \"\"\"Generate documentation for API endpoint\"\"\"
            response = requests.get(f"{api_endpoint}/docs")
            return response.json()
        
        class APIDocGenerator:
            def __init__(self, base_url):
                self.base_url = base_url
        """
        
        result = self.classifier.classify("api_generator.py", python_content)
        
        self.assertIn(result.document_type, [DocumentType.TECHNICAL_DOC, DocumentType.CODE_DOC])
        self.assertGreater(result.confidence, 0.6)
        self.assertIn("technical_docs", result.primary_index)
    
    def test_csv_file_classification(self):
        """Test classification of CSV data files"""
        csv_content = """
        id,product_name,category,price,stock_quantity
        1,Laptop Pro,Electronics,1299.99,45
        2,Office Chair,Furniture,299.99,120
        3,Coffee Maker,Appliances,89.99,67
        4,Wireless Mouse,Electronics,29.99,200
        5,Desk Lamp,Furniture,45.99,80
        """
        
        result = self.classifier.classify("products.csv", csv_content)
        
        self.assertEqual(result.document_type, DocumentType.REFERENCE_DATA)
        self.assertGreater(result.confidence, 0.7)
        self.assertEqual(result.primary_index, "structured_data")
    
    def test_business_document_classification(self):
        """Test classification of business documents"""
        business_content = """
        QUARTERLY BUSINESS REVIEW - Q4 2024
        
        Executive Summary
        
        This quarter demonstrated exceptional growth across all key performance 
        indicators. Revenue reached $4.2M, representing a 23% increase over Q3.
        
        Financial Highlights:
        - Total Revenue: $4,200,000
        - Gross Profit Margin: 67%
        - Operating Expenses: $1,800,000
        - Net Income: $980,000
        - ROI: 18.5%
        
        Market Performance:
        - Customer acquisition cost decreased to $145
        - Customer lifetime value increased to $2,800
        - Market share grew to 15.2%
        
        Strategic Initiatives:
        The board approved three major investments for 2025...
        """
        
        result = self.classifier.classify("q4_review.docx", business_content)
        
        self.assertEqual(result.document_type, DocumentType.BUSINESS_DOC)
        self.assertGreater(result.confidence, 0.6)
        self.assertEqual(result.primary_index, "business_docs")
    
    def test_academic_paper_classification(self):
        """Test classification of academic papers"""
        academic_content = """
        Machine Learning Approaches to Natural Language Processing:
        A Comparative Study
        
        Abstract
        
        This paper presents a comprehensive comparison of machine learning 
        approaches for natural language processing tasks. We evaluate deep 
        learning models including transformers, LSTM networks, and traditional 
        methods using standardized datasets.
        
        1. Introduction
        
        Natural language processing (NLP) has experienced significant advances
        in recent years, primarily driven by deep learning methodologies.
        Previous research by Smith et al. (2023) and Johnson & Davis (2024)
        has established baseline performance metrics...
        
        2. Methodology
        
        We conducted experiments using three distinct approaches:
        - Transformer-based models (BERT, GPT)
        - Recurrent neural networks (LSTM, GRU)
        - Traditional machine learning (SVM, Random Forest)
        
        The experimental design follows established protocols...
        
        References
        
        [1] Smith, A., Johnson, B., & Wilson, C. (2023). Deep Learning for NLP.
            Journal of Machine Learning Research, 24(5), 112-134.
        [2] Davis, M. (2024). Transformer Architectures. arXiv:2024.1234.
        [3] Brown, K. et al. (2023). Language Models at Scale. Nature AI, 8, 45-67.
        """
        
        result = self.classifier.classify("ml_nlp_paper.pdf", academic_content)
        
        self.assertEqual(result.document_type, DocumentType.ACADEMIC_PAPER)
        self.assertGreater(result.confidence, 0.7)
        self.assertEqual(result.primary_index, "academic_papers")
    
    def test_mixed_content_classification(self):
        """Test classification of mixed content documents"""
        mixed_content = """
        Project Documentation
        
        This document contains various types of content including code,
        business information, and technical details.
        
        ```python
        # Some code example
        def process_data(input_file):
            return pd.read_csv(input_file)
        ```
        
        Financial Impact:
        - Development cost: $50,000
        - Expected ROI: 15%
        - Market opportunity: $2M
        
        Technical specifications and user requirements...
        """
        
        result = self.classifier.classify("project_doc.md", mixed_content)
        
        # Should classify as technical doc, code doc, or general text due to mixed nature
        self.assertIn(result.document_type, [
            DocumentType.TECHNICAL_DOC, 
            DocumentType.CODE_DOC,      # Added CODE_DOC as acceptable
            DocumentType.GENERAL_TEXT,
            DocumentType.MIXED_CONTENT
        ])
        # Confidence might be lower due to mixed signals
        self.assertGreater(result.confidence, 0.3)
    
    def test_reasoning_generation(self):
        """Test that classification includes proper reasoning"""
        code_content = """
        # API Configuration
        API_BASE_URL = "https://api.example.com/v1"
        
        def authenticate(api_key):
            headers = {"Authorization": f"Bearer {api_key}"}
            return headers
        """
        
        result = self.classifier.classify("config.py", code_content)
        
        self.assertIsNotNone(result.reasoning)
        self.assertIn("code", result.reasoning.lower())
        self.assertTrue(len(result.key_features) > 0)
        self.assertIn("File extension: .py", result.key_features)


class TestDocumentTypeClassifier(unittest.TestCase):
    """Test the main DocumentTypeClassifier"""
    
    def setUp(self):
        self.topology_manager = IndexTypologyManager()
        self.classifier = DocumentTypeClassifier(self.topology_manager)
    
    def test_basic_classification(self):
        """Test basic classification functionality"""
        content = "This is a test document for classification."
        result = self.classifier.classify_document("test.txt", content)
        
        self.assertIsInstance(result, ClassificationResult)
        self.assertIsInstance(result.document_type, DocumentType)
        self.assertIsInstance(result.confidence, float)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertIsInstance(result.classification_time, datetime)
    
    def test_manual_override(self):
        """Test manual classification override"""
        content = "Some content that might be misclassified."
        
        # First classify normally
        result = self.classifier.classify_document("test.txt", content)
        original_type = result.document_type
        
        # Apply manual override
        override_type = DocumentType.TECHNICAL_DOC
        if original_type == DocumentType.TECHNICAL_DOC:
            override_type = DocumentType.BUSINESS_DOC
        
        override_result = self.classifier.classify_document(
            "test.txt", content, manual_override=override_type
        )
        
        self.assertEqual(override_result.document_type, override_type)
        self.assertEqual(override_result.confidence, 1.0)
        self.assertEqual(override_result.classification_method, ClassificationMethod.MANUAL_OVERRIDE)
        self.assertTrue(override_result.manual_override_applied)
    
    def test_batch_classification(self):
        """Test batch classification"""
        documents = [
            ("doc1.py", "def hello(): pass"),
            ("doc2.csv", "id,name\n1,test"),
            ("doc3.txt", "Regular text content here")
        ]
        
        results = self.classifier.classify_batch(documents)
        
        self.assertEqual(len(results), 3)
        
        # Check that each result is valid
        for result in results:
            self.assertIsInstance(result, ClassificationResult)
            self.assertIsInstance(result.document_type, DocumentType)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
    
    def test_confidence_levels(self):
        """Test confidence level conversion"""
        # High confidence case
        high_conf_content = """
        import pandas as pd
        import numpy as np
        
        def process_data(filename):
            \"\"\"Process CSV data file\"\"\"
            df = pd.read_csv(filename)
            return df.describe()
        """
        
        result = self.classifier.classify_document("analysis.py", high_conf_content)
        confidence_level = result.to_confidence_level()
        
        # Should be high confidence for clear code file
        self.assertIn(confidence_level, [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH])
    
    def test_index_assignment(self):
        """Test that documents are assigned to correct indices"""
        # Technical document
        tech_result = self.classifier.classify_document(
            "api.md", 
            "# API Documentation\n\n```python\napi.get('/users')\n```"
        )
        
        self.assertEqual(tech_result.primary_index, "technical_docs")
        
        # CSV data
        data_result = self.classifier.classify_document(
            "data.csv",
            "id,value\n1,100\n2,200"
        )
        
        self.assertEqual(data_result.primary_index, "structured_data")
    
    def test_error_handling(self):
        """Test error handling and fallback classification"""
        # Test with problematic content
        result = self.classifier.classify_document("", "")
        
        # Should not raise exception and should return fallback
        self.assertIsInstance(result, ClassificationResult)
        # Fallback should be general text with low confidence
        self.assertLessEqual(result.confidence, 0.5)
    
    def test_classification_statistics(self):
        """Test classification statistics tracking"""
        # Perform several classifications
        test_docs = [
            ("test1.py", "def func(): pass"),
            ("test2.csv", "a,b\n1,2"),
            ("test3.txt", "Some text content")
        ]
        
        for file_path, content in test_docs:
            self.classifier.classify_document(file_path, content)
        
        stats = self.classifier.get_classification_stats()
        
        self.assertEqual(stats["total_documents"], 3)
        self.assertIn("document_type_distribution", stats)
        self.assertIn("confidence_stats", stats)
        self.assertIn("performance_metrics", stats)
        
        # Check that confidence stats are valid
        confidence_stats = stats["confidence_stats"]
        self.assertGreaterEqual(confidence_stats["average"], 0.0)
        self.assertLessEqual(confidence_stats["average"], 1.0)
        self.assertGreaterEqual(confidence_stats["min"], 0.0)
        self.assertLessEqual(confidence_stats["max"], 1.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        self.topology_manager = IndexTypologyManager()
        self.classifier = DocumentTypeClassifier(self.topology_manager)
    
    def test_empty_content(self):
        """Test classification with empty content"""
        result = self.classifier.classify_document("empty.txt", "")
        
        # Should handle gracefully
        self.assertIsInstance(result, ClassificationResult)
        self.assertEqual(result.document_type, DocumentType.GENERAL_TEXT)
        self.assertLessEqual(result.confidence, 0.5)
    
    def test_very_long_content(self):
        """Test classification with very long content"""
        long_content = "This is a test. " * 10000  # Very long repetitive content
        
        result = self.classifier.classify_document("long.txt", long_content)
        
        # Should handle without errors
        self.assertIsInstance(result, ClassificationResult)
        self.assertGreater(result.confidence, 0.0)
    
    def test_binary_like_content(self):
        """Test with binary-like or encoded content"""
        binary_content = "����\x00\x01\x02\x03������"
        
        result = self.classifier.classify_document("binary.dat", binary_content)
        
        # Should classify as general text with low confidence
        self.assertIsInstance(result, ClassificationResult)
        self.assertLessEqual(result.confidence, 0.5)
    
    def test_unknown_file_extension(self):
        """Test with unknown file extensions"""
        result = self.classifier.classify_document(
            "unknown.xyz", 
            "Some content in unknown format"
        )
        
        # Should still classify but possibly with lower confidence
        self.assertIsInstance(result, ClassificationResult)
        self.assertIsInstance(result.document_type, DocumentType)
    
    def test_multilingual_content(self):
        """Test with non-English content"""
        spanish_content = """
        Documentación de la API
        
        Esta es la documentación para nuestra API REST.
        
        ```python
        def obtener_datos():
            return {"mensaje": "Hola mundo"}
        ```
        """
        
        result = self.classifier.classify_document("docs_es.md", spanish_content)
        
        # Should still detect technical content despite language
        self.assertIn(result.document_type, [DocumentType.TECHNICAL_DOC, DocumentType.CODE_DOC])


class TestPerformance(unittest.TestCase):
    """Test performance characteristics"""
    
    def setUp(self):
        self.topology_manager = IndexTypologyManager()
        self.classifier = DocumentTypeClassifier(self.topology_manager)
    
    def test_classification_speed(self):
        """Test that classification completes in reasonable time"""
        content = "Test document content for performance testing."
        
        start_time = datetime.now()
        result = self.classifier.classify_document("perf_test.txt", content)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Should complete within 1 second for simple content
        self.assertLess(processing_time, 1.0)
        self.assertIsInstance(result, ClassificationResult)
    
    def test_batch_performance(self):
        """Test batch processing performance"""
        # Create 50 test documents
        documents = [
            (f"test_{i}.txt", f"Test document content number {i}")
            for i in range(50)
        ]
        
        start_time = datetime.now()
        results = self.classifier.classify_batch(documents)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Should process all documents in reasonable time
        self.assertEqual(len(results), 50)
        self.assertLess(processing_time, 10.0)  # Should complete in under 10 seconds
        
        # Verify all results are valid
        for result in results:
            self.assertIsInstance(result, ClassificationResult)


def run_all_tests():
    """Run all test suites"""
    print("Running Document Type Classifier Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestContentAnalyzer,
        TestRuleBasedClassifier, 
        TestDocumentTypeClassifier,
        TestEdgeCases,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'-' * 60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 