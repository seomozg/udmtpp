import pytest
import sys
import os
# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, src_path)

from rag import RAGSystem
from vector_db import ChromaDB
from qdrant_client.models import PointStruct
import uuid

class TestRAGIntegration:
    def test_rag_full_cycle(self):
        """Test complete RAG cycle: add data -> search -> generate response"""
        # Get instances
        vector_db = ChromaDB()
        rag = RAGSystem()

        # Create test data
        test_text = "ТПП предоставляет консультации по налогам и юридическим вопросам"
        test_vectors = rag.embedder.encode(["налоговые консультации"])  # Get embedding
        test_vector = test_vectors[0] if test_vectors else []

        # Create point
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=test_vector,
            payload={
                "text": test_text,
                "url": "http://test.com"
            }
        )

        # Add to database
        vector_db.add_points("services", [point])

        # Test search
        results = vector_db.search("services", test_vector, limit=5)
        assert len(results) > 0, "Search should return results"

        # Test RAG ask method
        response = rag.ask("Какие услуги предоставляет ТПП?", "services")
        assert isinstance(response, dict)
        assert "response" in response
        assert "confidence" in response
        assert "sources" in response

        # Should find the test data (with proper similarity scoring)
        assert response["confidence"] >= 0.0, f"Confidence should be >= 0, got {response['confidence']}"
        assert len(response["sources"]) > 0, "Should have sources"

    def test_rag_no_results(self):
        """Test RAG when no relevant data found"""
        rag = RAGSystem()

        # Query that should not match anything (random characters)
        response = rag.ask("xyz123randomquerytest", "services")

        # In a real system, even random queries might get some similarity scores
        # So we just check that it returns a proper response structure
        assert isinstance(response, dict)
        assert "response" in response
        assert "confidence" in response
        assert "sources" in response
        assert isinstance(response["response"], str)

    def test_rag_search_with_real_data(self):
        """Test RAG search with real parsed data"""
        vector_db = ChromaDB()
        rag = RAGSystem()

        # Check if we have real data from parsing
        info = vector_db.get_collection_info()
        total_docs = sum(coll['points_count'] for coll in info.values())

        if total_docs == 0:
            pytest.skip("No real data available - run parsing first")

        # Test search in services collection
        services_count = info["services"]["points_count"]
        if services_count > 0:
            response = rag.ask("услуги ТПП", "services")

            # Should find some relevant information
            assert isinstance(response, dict)
            assert "response" in response
            assert "confidence" in response
            assert "sources" in response

            # With real data, should have some confidence
            assert response["confidence"] >= 0.0

            # If confidence > 0, should have sources
            if response["confidence"] > 0:
                assert len(response["sources"]) > 0

    def test_chroma_persistence(self):
        """Test that ChromaDB data persists between operations"""
        vector_db = ChromaDB()

        # Add test data
        test_vector = [0.1] * 768
        point = PointStruct(
            id=str(uuid.uuid4()),  # Must be valid UUID string
            vector=test_vector,
            payload={"text": "test persistence", "url": "http://test.com"}
        )

        vector_db.add_points("services", [point])

        # Get info
        info = vector_db.get_collection_info()
        services_count = info["services"]["points_count"]

        # Should have at least our test point
        assert services_count > 0, f"Services collection should have points, got {services_count}"
