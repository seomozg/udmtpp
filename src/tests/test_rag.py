"""
Tests for RAG (Retrieval-Augmented Generation) system
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from unittest.mock import patch, MagicMock


class TestRAGSystem:
    """Test RAG system functionality"""

    @pytest.fixture
    def mock_chroma_db(self):
        """Mock ChromaDB for testing"""
        mock_db = MagicMock()
        mock_db.collection_configs = {
            "test": "Test collection"
        }
        mock_db.search.return_value = [
            {"id": "1", "score": 0.8, "payload": {"text": "test content", "url": "test.com", "category": "test"}}
        ]
        return mock_db

    @pytest.fixture
    def mock_embedder(self):
        """Mock EmbeddingModel for testing"""
        mock_emb = MagicMock()
        mock_emb.encode.return_value = [[0.1, 0.2, 0.3]]
        return mock_emb

    @patch('rag_system.ChromaDB')
    @patch('rag_system.EmbeddingModel')
    def test_query_expansion(self, mock_embed_class, mock_db_class, mock_chroma_db, mock_embedder):
        """Test query expansion functionality"""
        mock_db_class.return_value = mock_chroma_db
        mock_embed_class.return_value = mock_embedder

        rag = RAGSystem()

        result = rag.expand_query("мероприятия")

        # Should return expanded query
        assert isinstance(result, str)
        assert "мероприятия" in result
        assert "семинары" in result  # From expansion templates

    @patch('rag_system.ChromaDB')
    @patch('rag_system.EmbeddingModel')
    def test_search_success(self, mock_embed_class, mock_db_class, mock_chroma_db, mock_embedder):
        """Test successful search"""
        mock_db_class.return_value = mock_chroma_db
        mock_embed_class.return_value = mock_embedder

        rag = RAGSystem()

        result = rag.search("test query", "test", 5)

        # Should return search results
        assert isinstance(result, dict)
        assert "query" in result
        assert "results" in result
        assert "confidence" in result
        assert result["confidence"] > 0

    @patch('rag_system.ChromaDB')
    @patch('rag_system.EmbeddingModel')
    def test_search_empty_results(self, mock_embed_class, mock_db_class, mock_chroma_db, mock_embedder):
        """Test search with no results"""
        mock_chroma_db.search.return_value = []
        mock_db_class.return_value = mock_chroma_db
        mock_embed_class.return_value = mock_embedder

        rag = RAGSystem()

        result = rag.search("test query", "test", 5)

        # Should return empty results with confidence 0
        assert isinstance(result, dict)
        assert len(result["results"]) == 0
        assert result["confidence"] == 0.0

    @patch('rag_system.ChromaDB')
    @patch('rag_system.EmbeddingModel')
    @patch('rag_system.requests.post')
    def test_ask_success(self, mock_post, mock_embed_class, mock_db_class, mock_chroma_db, mock_embedder):
        """Test successful ask with AI response"""
        mock_db_class.return_value = mock_chroma_db
        mock_embed_class.return_value = mock_embedder

        # Mock AI response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Тестовый ответ от AI"}}]
        }
        mock_post.return_value = mock_response

        rag = RAGSystem()

        result = rag.ask("тестовый вопрос")

        # Should return complete response
        assert isinstance(result, dict)
        assert "query" in result
        assert "response" in result
        assert "confidence" in result
        assert "sources" in result
        assert result["response"] == "Тестовый ответ от AI"

    @patch('rag_system.ChromaDB')
    @patch('rag_system.EmbeddingModel')
    def test_ask_low_confidence(self, mock_embed_class, mock_db_class, mock_chroma_db, mock_embedder):
        """Test ask with low confidence (no AI call)"""
        # Mock low confidence results
        mock_chroma_db.search.return_value = [
            {"id": "1", "score": 0.1, "payload": {"text": "low relevance", "url": "test.com", "category": "test"}}
        ]
        mock_db_class.return_value = mock_chroma_db
        mock_embed_class.return_value = mock_embedder

        rag = RAGSystem()

        result = rag.ask("тестовый вопрос")

        # Should return low confidence response
        assert isinstance(result, dict)
        assert "response" in result
        assert result["confidence"] < 0.5  # Low confidence
        assert "информация отсутствует" in result["response"].lower()

    @patch('rag_system.ChromaDB')
    @patch('rag_system.EmbeddingModel')
    @patch('rag_system.requests.post')
    def test_ask_ai_error(self, mock_post, mock_embed_class, mock_db_class, mock_chroma_db, mock_embedder):
        """Test ask with AI API error"""
        mock_db_class.return_value = mock_chroma_db
        mock_embed_class.return_value = mock_embedder

        # Mock AI API error
        mock_post.side_effect = Exception("API Error")

        rag = RAGSystem()

        result = rag.ask("тестовый вопрос")

        # Should handle error gracefully
        assert isinstance(result, dict)
        assert "error" in result or "Ошибка" in result.get("response", "")

    def test_calculate_confidence(self):
        """Test confidence calculation"""
        rag = RAGSystem()

        # Test with results
        results = [
            {"score": 0.8},
            {"score": 0.9},
            {"score": 0.7}
        ]
        confidence = rag._calculate_confidence(results)
        assert isinstance(confidence, float)
        assert confidence > 0

        # Test with empty results
        confidence = rag._calculate_confidence([])
        assert confidence == 0.0

    def test_prepare_context(self):
        """Test context preparation"""
        rag = RAGSystem()

        results = [
            {"text": "test text 1", "url": "test1.com"},
            {"text": "test text 2", "url": "test2.com"}
        ]

        context = rag._prepare_context(results)

        assert isinstance(context, str)
        assert "test text 1" in context
        assert "test text 2" in context
        assert "test1.com" in context
        assert "test2.com" in context

    def test_extract_sources(self):
        """Test source extraction"""
        rag = RAGSystem()

        results = [
            {"url": "test1.com", "score": 0.8, "category": "test"},
            {"url": "test2.com", "score": 0.9, "category": "test"}
        ]

        sources = rag._extract_sources(results)

        assert isinstance(sources, list)
        assert len(sources) == 2
        assert sources[0]["url"] == "test1.com"
        assert sources[0]["score"] == 0.8
        assert sources[1]["url"] == "test2.com"
        assert sources[1]["score"] == 0.9

    @patch('rag_system.ChromaDB')
    def test_get_collection_info(self, mock_db_class):
        """Test getting collection info"""
        mock_db = MagicMock()
        mock_db.get_collection_info.return_value = {"test": {"name": "test", "points_count": 10}}
        mock_db_class.return_value = mock_db

        rag = RAGSystem()

        info = rag.get_collection_info()

        assert isinstance(info, dict)
        assert "test" in info
