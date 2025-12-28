import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parse_site import SiteParser
from unittest.mock import patch, MagicMock

class TestSiteParser:
    @patch('parse_site.requests.get')
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

    @patch('parse_site.requests.get')
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

    def test_categorize_content(self):
        """Test content categorization"""
        parser = SiteParser()

        # Test 719 category
        url_719 = "https://udmtpp.ru/719-consultation"
        text_719 = "Консультации по 719 постановлению"
        assert parser.categorize_content(url_719, text_719) == "719"

        # Test services category
        url_services = "https://udmtpp.ru/services"
        text_services = "Наши услуги включают консультации"
        assert parser.categorize_content(url_services, text_services) == "services"

        # Test default category
        url_default = "https://udmtpp.ru/about"
        text_default = "Общая информация о компании"
        assert parser.categorize_content(url_default, text_default) == "site"

    @patch('parse_site.uuid.uuid4')
    @patch('parse_site.EmbeddingModel.encode')
    def test_process_urls(self, mock_encode, mock_uuid):
        """Test processing URLs and adding to vector DB"""
        # Mock dependencies
        mock_uuid.return_value = "test-uuid"
        mock_encode.return_value = [[0.1] * 768]

        # Mock ChromaDB
        mock_vector_db = MagicMock()
        parser = SiteParser(vector_client=mock_vector_db)

        # Mock text extraction
        with patch.object(parser, 'extract_text_from_url', return_value="Test content about services"):
            with patch.object(parser, 'categorize_content', return_value="services"):
                parser.process_urls(["https://test.com"])

        # Verify ChromaDB was called
        assert mock_vector_db.add_points.called
        call_args = mock_vector_db.add_points.call_args
        assert call_args[0][0] == "services"  # collection name
        assert len(call_args[0][1]) == 1  # one point added
