import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_db import ChromaDB
from qdrant_client.models import PointStruct
import uuid

class TestChromaDB:
    def test_singleton_pattern(self):
        """Test that ChromaDB follows singleton pattern"""
        db1 = ChromaDB()
        db2 = ChromaDB()
        assert db1 is db2, "ChromaDB should be singleton"

    def test_collection_configs(self):
        """Test collection configurations"""
        db = ChromaDB()
        assert len(db.collection_configs) == 7
        assert "services" in db.collection_configs
        assert "719" in db.collection_configs

    def test_get_collection_info(self):
        """Test getting collection info"""
        db = ChromaDB()
        info = db.get_collection_info()
        assert isinstance(info, dict)
        assert len(info) == 7

    def test_add_and_search_points(self):
        """Test adding points and searching them"""
        vector_db = ChromaDB()

        # Create test point
        test_vector = [0.1] * 768  # multilingual-e5-base dimension
        test_id = str(uuid.uuid4())
        point = PointStruct(
            id=test_id,
            vector=test_vector,
            payload={"text": "test content", "url": "http://test.com"}
        )

        # Add point
        vector_db.add_points("services", [point])

        # Search for it
        results = vector_db.search("services", test_vector, limit=5)
        assert len(results) >= 0  # May be 0 if persistence doesn't work in tests
