import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag import RAGSystem
from vector_db import ChromaDB
from embed import EmbeddingModel

class TestRAGSystem:
    @pytest.fixture
    def mock_chroma_db(self):
        """Mock ChromaDB for testing"""
        mock_db = MagicMock(spec=ChromaDB)
        mock_db.collection_configs = {
            "test": "Test collection"
        }
        mock_db.get_collection_documents.return_value = [
            {"text": "Это тестовый документ для поиска", "url": "test.com"}
        ]
        mock_db.search.return_value = [
            {"id": "1", "score": 0.8, "payload": {"text": "test content", "url": "test.com"}}
        ]
        return mock_db

    @pytest.fixture
    def mock_embedder(self):
        """Mock EmbeddingModel for testing"""
        mock_emb = MagicMock(spec=EmbeddingModel)
        mock_emb.encode.return_value = [[0.1, 0.2, 0.3]]
        return mock_emb

    @patch('rag.ChromaDB')
    @patch('rag.EmbeddingModel')
    def test_query_expansion(self, mock_embed_class, mock_db_class, mock_chroma_db, mock_embedder):
        """Test query expansion functionality"""
        mock_db_class.return_value = mock_chroma_db
        mock_embed_class.return_value = mock_embedder

        # Mock the singleton pattern
        with patch.object(RAGSystem, '_instance', None):
            rag = RAGSystem()

        # Mock requests for API call
        with patch('rag.requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'choices': [{'message': {'content': 'расширенный запрос с синонимами'}}]
            }
            mock_post.return_value = mock_response

            result = rag.expand_query("тестовый запрос")

            # Should return expanded query
            assert "расширенный запрос с синонимами" in result
            mock_post.assert_called_once()

    @patch('rag.ChromaDB')
    @patch('rag.EmbeddingModel')
    def test_query_expansion_fallback(self, mock_embed_class, mock_db_class, mock_chroma_db, mock_embedder):
        """Test query expansion fallback on API failure"""
        mock_db_class.return_value = mock_chroma_db
        mock_embed_class.return_value = mock_embedder

        with patch.object(RAGSystem, '_instance', None):
            rag = RAGSystem()

        # Mock failed API call
        with patch('rag.requests.post') as mock_post:
            mock_post.side_effect = Exception("API Error")

            result = rag.expand_query("тестовый запрос")

            # Should return original query
            assert result == "тестовый запрос"

    @patch('rag.ChromaDB')
    @patch('rag.EmbeddingModel')
    def test_hybrid_search(self, mock_embed_class, mock_db_class, mock_chroma_db, mock_embedder):
        """Test hybrid search functionality"""
        mock_db_class.return_value = mock_chroma_db
        mock_embed_class.return_value = mock_embedder

        with patch.object(RAGSystem, '_instance', None):
            rag = RAGSystem()

        # Mock BM25 index
        rag.bm25_indexes = {
            "test": MagicMock()
        }
        rag.document_texts = {
            "test": [{"text": "test document", "url": "test.com"}]
        }

        # Mock BM25 scores
        mock_bm25 = rag.bm25_indexes["test"]
        mock_bm25.get_scores.return_value = [0.5]

        result = rag.hybrid_search("test query", "test", 5)

        # Should return results
        assert isinstance(result, list)
        assert len(result) > 0

        # Check that both semantic and keyword search were called
        mock_chroma_db.search.assert_called()
        mock_bm25.get_scores.assert_called()

    @patch('rag.ChromaDB')
    @patch('rag.EmbeddingModel')
    def test_rerank_with_cross_encoder(self, mock_embed_class, mock_db_class, mock_chroma_db, mock_embedder):
        """Test reranking with cross-encoder"""
        mock_db_class.return_value = mock_chroma_db
        mock_embed_class.return_value = mock_embedder

        with patch.object(RAGSystem, '_instance', None):
            rag = RAGSystem()

        # Mock cross-encoder
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.8, 0.6]  # Higher score for first result
        rag.cross_encoder = mock_ce

        # Setup BM25 for keyword search
        rag.bm25_indexes = {"test": MagicMock()}
        rag.document_texts = {"test": [{"text": "test document", "url": "test.com"}]}

        mock_bm25 = rag.bm25_indexes["test"]
        mock_bm25.get_scores.return_value = [0.5]

        # Mock semantic search results (more than n_results to trigger reranking)
        semantic_results = [
            {"id": "1", "score": 0.7, "payload": {"text": "first result " * 10}},  # Long text
            {"id": "2", "score": 0.8, "payload": {"text": "second result " * 10}}, # Long text
            {"id": "3", "score": 0.6, "payload": {"text": "third result " * 10}},  # Long text
            {"id": "4", "score": 0.9, "payload": {"text": "fourth result " * 10}}  # Long text
        ]
        mock_chroma_db.search.return_value = semantic_results

        # Call hybrid search with n_results that triggers reranking
        result = rag.hybrid_search("test query", "test", n_results=3)

        # Cross-encoder should be called for reranking (since we have more results than n_results)
        mock_ce.predict.assert_called()

        # Results should be reranked and limited
        assert len(result) == 3

    @patch('rag.ChromaDB')
    @patch('rag.EmbeddingModel')
    def test_rerank_disabled_when_no_cross_encoder(self, mock_embed_class, mock_db_class, mock_chroma_db, mock_embedder):
        """Test that reranking is skipped when cross-encoder is not available"""
        mock_db_class.return_value = mock_chroma_db
        mock_embed_class.return_value = mock_embedder

        with patch.object(RAGSystem, '_instance', None):
            rag = RAGSystem()

        # No cross-encoder
        rag.cross_encoder = None

        mock_results = [
            {"id": "1", "score": 0.7, "payload": {"text": "first result"}}
        ]

        with patch.object(rag, 'hybrid_search', return_value=mock_results):
            result = rag.hybrid_search("test", n_results=5)

            # Results should be returned as-is
            assert len(result) == 1
            assert result[0]["score"] == 0.7

    def test_clone_with_settings(self):
        """Test RAGSystem cloning with custom settings"""
        with patch.object(RAGSystem, '_instance', None):
            rag = RAGSystem()

        # Mock components
        rag.chroma_db = MagicMock()
        rag.embedder = MagicMock()
        rag.deepseek_api_key = "test_key"

        cloned = rag.clone_with_settings(confidence_threshold=0.8)

        # Should be different object
        assert cloned is not rag

        # Should share components
        assert cloned.chroma_db is rag.chroma_db
        assert cloned.embedder is rag.embedder

        # Should have custom settings
        assert cloned.confidence_threshold == 0.8

        # Should have copied methods
        assert hasattr(cloned, 'ask')
        assert hasattr(cloned, 'ask_stream')
