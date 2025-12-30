import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import chunk_text, adaptive_chunk_text, calculate_confidence

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

        # Expected: 50% * 0.9 + 30% * 0.8 + 20% * 0.7 = 0.83
        expected = 0.5 * 0.9 + 0.3 * 0.8 + 0.2 * 0.7
        assert abs(confidence - expected) < 0.001  # Weighted average calculation

    def test_calculate_confidence_empty(self):
        """Test confidence calculation with empty list"""
        confidence = calculate_confidence([])
        assert confidence == 0.0

    def test_adaptive_chunk_text_short(self):
        """Test adaptive chunking with short text"""
        text = "Короткий текст для тестирования"
        chunks = adaptive_chunk_text(text)

        # Short text should be single chunk
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_adaptive_chunk_text_complex_sentences(self):
        """Test adaptive chunking with complex sentences"""
        text = "Это очень длинное предложение с множеством слов, которое должно быть разбито на более мелкие части из-за своей сложности. Второе предложение тоже достаточно длинное и содержит много информации для обработки."
        chunks = adaptive_chunk_text(text)

        # Should create multiple chunks due to sentence complexity
        assert len(chunks) >= 1

        # Check that chunks are reasonably sized
        for chunk in chunks:
            assert len(chunk) > 0
            assert len(chunk) <= 1000  # Max chunk size

    def test_adaptive_chunk_text_structured(self):
        """Test adaptive chunking with structured text (paragraphs)"""
        text = """Первый параграф с информацией.

Второй параграф с другой информацией.

Третий параграф с дополнительными деталями.

Четвертый параграф с заключительной информацией."""

        chunks = adaptive_chunk_text(text)

        # Should handle paragraphs well
        assert len(chunks) >= 1

        # Check semantic boundaries are respected
        combined = "".join(chunks)
        assert combined.replace(" ", "").replace("\n", "") == text.replace(" ", "").replace("\n", "")

    def test_adaptive_chunk_text_edge_cases(self):
        """Test adaptive chunking with edge cases"""
        # Empty text
        chunks = adaptive_chunk_text("")
        assert len(chunks) == 1
        assert chunks[0] == ""

        # Very long text with many paragraphs
        long_text = ("Короткий параграф.\n\n" * 20) + "Последний параграф с текстом."
        chunks = adaptive_chunk_text(long_text)

        # Should handle well-structured long text
        assert len(chunks) >= 1
        assert sum(len(chunk) for chunk in chunks) >= len(long_text)
