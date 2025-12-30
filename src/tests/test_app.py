import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

client = TestClient(app)

class TestFastAPIApp:
    def test_home_page(self):
        """Test home page loads"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_collections_page(self):
        """Test collections page loads"""
        response = client.get("/collections")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_chat_page(self):
        """Test chat page loads"""
        response = client.get("/chat")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_upload_page(self):
        """Test upload page loads"""
        response = client.get("/upload")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_get_collections_api(self):
        """Test collections API returns data"""
        response = client.get("/api/collections")
        assert response.status_code == 200

        data = response.json()
        assert "collections" in data
        assert isinstance(data["collections"], dict)

    @patch('api_routes.rag_system.ask')
    def test_chat_api_success(self, mock_ask):
        """Test successful chat API call"""
        mock_ask.return_value = {
            "query": "test query",
            "response": "test response",
            "confidence": 0.8,
            "sources": [{"url": "http://test.com", "score": 0.9}]
        }

        response = client.post("/api/chat", data={
            "query": "test query",
            "collection": "services"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "test response"
        assert data["confidence"] == 0.8
        assert len(data["sources"]) == 1

    def test_chat_api_missing_query(self):
        """Test chat API with missing query"""
        response = client.post("/api/chat", data={})
        assert response.status_code == 422  # Validation error

    @patch('api_routes.chroma_db')
    def test_parse_site_api(self, mock_vector_db):
        """Test parse site API"""
        # Mock the parser and vector db
        with patch('api_routes.SiteParser') as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser.parse_sitemap.return_value = ["http://test1.com", "http://test2.com"]
            mock_parser.process_urls_with_ai_plan.return_value = {
                "processed_urls": 2,
                "total_points": 10,
                "points": [],
                "ai_mappings": {}
            }
            mock_parser_class.return_value = mock_parser

            mock_vector_db.get_collection_info.return_value = {
                "services": {"points_count": 10}
            }

            response = client.post("/api/parse-site")

            assert response.status_code == 200
            data = response.json()
            assert "saved_points" in data
            assert "status" in data
            assert data["status"] == "completed"

    def test_upload_file_api(self):
        """Test file upload API"""
        test_file_content = b"test file content"
        files = {"file": ("test.txt", test_file_content, "text/plain")}
        data = {"category": "services"}

        with patch('api_routes.chroma_db') as mock_vector_db:
            with patch('api_routes.EmbeddingModel') as mock_embed_class:
                with patch('api_routes.semantic_chunk_text', return_value=["chunk1", "chunk2"]):
                    mock_embed = MagicMock()
                    mock_embed.encode.return_value = [[0.1] * 768, [0.2] * 768]
                    mock_embed_class.return_value = mock_embed

                    response = client.post("/api/upload", files=files, data=data)

                    assert response.status_code == 200
                    data = response.json()
                    assert "message" in data
                    assert "chunks_count" in data
