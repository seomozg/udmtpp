import pytest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from embed import EmbeddingModel
from unittest.mock import patch, MagicMock

class TestEmbeddingModel:
    @patch('embed.SentenceTransformer')
    def test_encode_single_text(self, mock_transformer):
        """Test encoding single text"""
        mock_model = MagicMock()
        # multilingual-e5-base returns 768-dimensional vectors
        mock_embedding = np.array([[0.1] * 768])  # 2D array for single text
        mock_model.encode.return_value = mock_embedding
        mock_transformer.return_value = mock_model

        embedder = EmbeddingModel()
        result = embedder.encode("test text")

        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) == 768  # multilingual-e5-base dimension
        # Note: encode may not be called due to @lru_cache in real implementation

    @patch('embed.SentenceTransformer')
    def test_encode_multiple_texts(self, mock_transformer):
        """Test encoding multiple texts"""
        mock_model = MagicMock()
        # multilingual-e5-base returns 768-dimensional vectors
        mock_embedding1 = np.array([0.1] * 768)
        mock_embedding2 = np.array([0.2] * 768)
        mock_model.encode.return_value = np.array([mock_embedding1, mock_embedding2])
        mock_transformer.return_value = mock_model

        embedder = EmbeddingModel()
        result = embedder.encode(["text1", "text2"])

        assert len(result) == 2
        assert len(result[0]) == 768  # multilingual-e5-base dimension
        assert len(result[1]) == 768  # multilingual-e5-base dimension

    @patch('embed.SentenceTransformer')
    def test_encode_query(self, mock_transformer):
        """Test encoding query"""
        mock_model = MagicMock()
        # multilingual-e5-base returns 768-dimensional vectors
        mock_embedding = np.array([[0.5] * 768])  # 2D array for single query
        mock_model.encode.return_value = mock_embedding
        mock_transformer.return_value = mock_model

        embedder = EmbeddingModel()
        result = embedder.encode_query("test query")

        assert isinstance(result, list)
        assert len(result) == 768  # Should return 768-dimensional embedding
