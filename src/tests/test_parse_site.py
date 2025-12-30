import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from site_parser import SiteParser
from unittest.mock import patch, MagicMock

class TestSiteParser:
    @patch('site_parser.requests.get')
    def test_parse_sitemap(self, mock_get):
        """Test parsing sitemap XML"""
        mock_response = MagicMock()
        mock_response.content = '''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://udmtpp.ru/page1</loc></url>
    <url><loc>https://udmtpp.ru/page2</loc></url>
</urlset>'''
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        parser = SiteParser()
        urls = parser.parse_sitemap()

        assert len(urls) == 2
        assert "https://udmtpp.ru/page1" in urls
        assert "https://udmtpp.ru/page2" in urls

    @patch('site_parser.requests.get')
    def test_extract_text_from_url(self, mock_get):
        """Test text extraction from HTML"""
        mock_response = MagicMock()
        mock_response.content = '''<html><body>
        <h1>Test Page</h1>
        <p>This is test content about services.</p>
        <script>console.log('ignore me');</script>
        </body></html>'''
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        parser = SiteParser()
        text = parser.extract_text_from_url("https://test.com")

        assert "Test Page" in text
        assert "test content" in text
        assert "services" in text
        assert "console.log" not in text  # Scripts should be removed

    @patch('site_parser.uuid.uuid4')
    @patch('site_parser.EmbeddingModel.encode')
    def test_process_url(self, mock_encode, mock_uuid):
        """Test processing single URL"""
        # Mock dependencies
        mock_uuid.return_value = "test-uuid"
        mock_encode.return_value = [[0.1] * 768]

        parser = SiteParser()

        # Mock text extraction and categorization
        with patch.object(parser, 'extract_text_from_url', return_value="Test content about services"):
            with patch('site_parser.categorize_content', return_value="services"):
                points = parser.process_url("https://test.com")

        # Should return list of points
        assert isinstance(points, list)
        assert len(points) > 0

        # Check point structure
        point = points[0]
        assert hasattr(point, 'id')
        assert hasattr(point, 'vector')
        assert hasattr(point, 'payload')
        assert point.payload["category"] == "services"

    @patch('site_parser.uuid.uuid4')
    @patch('site_parser.EmbeddingModel.encode')
    def test_process_urls(self, mock_encode, mock_uuid):
        """Test processing multiple URLs"""
        # Mock dependencies
        mock_uuid.return_value = "test-uuid"
        mock_encode.return_value = [[0.1] * 768]

        parser = SiteParser()

        # Mock text extraction and categorization
        with patch.object(parser, 'extract_text_from_url', return_value="Test content"):
            with patch('site_parser.categorize_content', return_value="services"):
                result = parser.process_urls(["https://test.com"], limit=1)

        # Should return processing statistics
        assert isinstance(result, dict)
        assert "processed_urls" in result
        assert "total_points" in result
        assert "points" in result
        assert result["processed_urls"] == 1
