"""
Tests for RAG system integration
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, src_path)

from rag_system import RAGSystem
from vector_db import ChromaDB


class TestRAGIntegration:
    @patch('rag_system.ChromaDB')
    @patch('rag_system.EmbeddingModel')
    def test_rag_full_cycle_mock(self, mock_embed_class, mock_db_class):
        """Test complete RAG cycle with mocked components"""
        # Setup mocks
        mock_db = MagicMock()
        mock_db.collection_configs = {"test": "Test collection"}
        mock_db.search.return_value = [
            {"id": "1", "score": 0.8, "payload": {"text": "test content", "url": "test.com", "category": "test"}}
        ]
        mock_db_class.return_value = mock_db

        mock_embed = MagicMock()
        mock_embed.encode.return_value = [[0.1] * 768]
        mock_embed_class.return_value = mock_embed

        # Mock AI response
        with patch('rag_system.requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'choices': [{'message': {'content': 'Тестовый ответ'}}]
            }
            mock_post.return_value = mock_response

            rag = RAGSystem()
            response = rag.ask("тестовый вопрос", "test")

            assert isinstance(response, dict)
            assert "response" in response
            assert "confidence" in response
            assert "sources" in response

    @patch('rag_system.ChromaDB')
    @patch('rag_system.EmbeddingModel')
    def test_rag_no_results(self, mock_embed_class, mock_db_class):
        """Test RAG when no relevant data found"""
        # Setup mocks with no results
        mock_db = MagicMock()
        mock_db.collection_configs = {"test": "Test collection"}
        mock_db.search.return_value = []
        mock_db_class.return_value = mock_db

        mock_embed = MagicMock()
        mock_embed.encode.return_value = [[0.1] * 768]
        mock_embed_class.return_value = mock_embed

        rag = RAGSystem()
        response = rag.ask("xyz123randomquerytest", "test")

        # Should return low confidence response
        assert isinstance(response, dict)
        assert "response" in response
        assert response["confidence"] == 0.0
        assert len(response["sources"]) == 0

    def test_real_data_integration(self):
        """Test with real data if available"""
        vector_db = ChromaDB()
        rag = RAGSystem()

        # Check if we have real data
        info = vector_db.get_collection_info()
        total_docs = sum(coll['points_count'] for coll in info.values())

        if total_docs == 0:
            pytest.skip("No real data available - run parsing first")

        # Test with real data (mock AI to avoid external calls)
        with patch('rag_system.requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'choices': [{'message': {'content': 'Найдена информация'}}]
            }
            mock_post.return_value = mock_response

            response = rag.ask("услуги ТПП", "services")

            assert isinstance(response, dict)
            assert "response" in response
            assert "confidence" in response
            assert "sources" in response

    def test_vector_db_operations(self):
        """Test ChromaDB operations"""
        vector_db = ChromaDB()

        # Test getting collection info
        info = vector_db.get_collection_info()
        assert isinstance(info, dict)

        # All collections should be present
        expected_collections = {"719", "support", "services", "membership", "events", "cooperation", "site"}
        actual_collections = set(info.keys())
        assert expected_collections == actual_collections

        # Each collection should have proper structure
        for coll_name, coll_info in info.items():
            assert "name" in coll_info
            assert "description" in coll_info
            assert "vectors_count" in coll_info
            assert "points_count" in coll_info
            assert coll_info["name"] == coll_name
