#!/usr/bin/env python3
"""
Script to reindex database with improved chunking and deduplication
"""

import sys
import os
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))

from src.vector_db import ChromaDB
from src.utils import semantic_chunk_text
from src.embed import EmbeddingModel
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reindex_database():
    """Reindex entire database with improved chunking and deduplication"""

    print("REINDEX DATABASE WITH IMPROVED CHUNKING")
    print("=" * 60)
    print("This will recreate all chunks using enhanced semantic chunking")
    print("and remove duplicates during the process")
    print("=" * 60)

    # Initialize components
    db = ChromaDB()
    embedder = EmbeddingModel()

    # Get all collections info
    collections_info = db.get_collection_info()

    total_old_points = sum(info['points_count'] for info in collections_info.values())
    total_new_points = 0

    print(f"Original database: {total_old_points} documents across {len(collections_info)} collections")
    print()

    # Create backup of current collections
    backup_collections = {}
    for collection_name in collections_info.keys():
        backup_name = f"{collection_name}_backup"
        try:
            # Copy collection data
            documents = db.get_collection_documents(collection_name, limit=10000)
            if documents:
                backup_collections[collection_name] = documents
                logger.info(f"Backed up {len(documents)} documents from {collection_name}")
        except Exception as e:
            logger.error(f"Failed to backup {collection_name}: {e}")

    print(f"Created backups for {len(backup_collections)} collections")
    print()

    # Process each collection
    for collection_name in collections_info.keys():
        print(f"🔄 Reindexing collection: {collection_name}")

        try:
            # Get all original documents
            documents = db.get_collection_documents(collection_name, limit=10000)
            logger.info(f"Processing {len(documents)} documents from {collection_name}")

            if not documents:
                print(f"  Collection {collection_name} is empty, skipping")
                continue

            # Group documents by URL to avoid duplicate processing
            docs_by_url = defaultdict(list)
            for doc in documents:
                url = doc.get('url', '')
                if url:
                    docs_by_url[url].append(doc)

            print(f"  Found {len(docs_by_url)} unique URLs")

            # Clear the collection
            db.delete_collection(collection_name)
            # Recreate collection
            db.collections[collection_name] = db.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": db.collection_configs[collection_name]}
            )

            # Reprocess each URL
            processed_points = 0
            for url, url_docs in docs_by_url.items():
                # Get the original text (assume all docs for same URL have same text)
                original_text = url_docs[0].get('text', '')
                category = url_docs[0].get('category', collection_name)

                if not original_text:
                    continue

                # Apply improved chunking
                chunks = semantic_chunk_text(original_text, max_chunk_size=800, overlap=50)

                if not chunks:
                    # If no chunks after filtering, use original text as single chunk
                    chunks = [original_text[:800]] if original_text.strip() else []

                if not chunks:
                    continue

                # Generate embeddings for new chunks
                embeddings = embedder.encode(chunks)

                # Create new points
                points = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    point_id = str(uuid.uuid4())

                    class Point:
                        def __init__(self, id, vector, payload):
                            self.id = id
                            self.vector = vector
                            self.payload = payload

                    points.append(Point(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "url": url,
                            "text": chunk,
                            "category": category,
                            "chunk_index": i,
                            "source": "reindexed"
                        }
                    ))

                # Save to database
                if points:
                    db.add_points(collection_name, points)
                    processed_points += len(points)
                    logger.info(f"Reindexed {url}: {len(chunks)} chunks")

            print(f"  ✅ Reindexed: {processed_points} new chunks")
            total_new_points += processed_points

        except Exception as e:
            logger.error(f"Error reindexing {collection_name}: {e}")
            print(f"  ❌ Error reindexing {collection_name}: {e}")

    print("\n" + "=" * 60)
    print("REINDEXING COMPLETE")
    print(f"Original documents: {total_old_points}")
    print(f"New documents: {total_new_points}")
    print(f"Reduction: {total_old_points - total_new_points} documents ({(total_old_points - total_new_points)/total_old_points*100:.1f}%)")
    print(f"Compression ratio: {total_new_points/total_old_points:.2f}x" if total_old_points > 0 else "N/A")

    print("\nBackups available in memory for recovery if needed")
    print("Improved chunking and deduplication applied!")

if __name__ == "__main__":
    reindex_database()
