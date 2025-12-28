import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import chunk_text, calculate_confidence

class TestUtils:
    def test_chunk_text_basic(self):
        """Test basic text chunking"""
        text = "This is a test text that should be chunked into smaller pieces."
        chunks = chunk_text(text, chunk_size=20, overlap=5)

        assert len(chunks) > 1
        assert all(len(chunk) <= 20 for chunk in chunks)

        # Check overlap
        if len(chunks) > 1:
            assert chunks[0][-5:] == chunks[1][:5]

    def test_chunk_text_short(self):
        """Test chunking with short text"""
        text = "Short text"
        chunks = chunk_text(text, chunk_size=100)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_calculate_confidence_with_scores(self):
        """Test confidence calculation with scores"""
        scores = [0.8, 0.9, 0.7]
        confidence = calculate_confidence(scores)

        assert abs(confidence - 0.8) < 0.001  # Average of scores (with floating point tolerance)

    def test_calculate_confidence_empty(self):
        """Test confidence calculation with empty list"""
        confidence = calculate_confidence([])
        assert confidence == 0.0
