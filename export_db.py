#!/usr/bin/env python3
"""
Script to export all data from ChromaDB to text file
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.vector_db import ChromaDB
import json

def export_db_to_text():
    """Export all ChromaDB data to text file"""

    # Initialize ChromaDB
    db = ChromaDB()

    # Get collection info
    collections_info = db.get_collection_info()

    output_file = "chroma_db_export.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== CHROMA DB EXPORT ===\n\n")
        f.write("Export timestamp: " + str(__import__('datetime').datetime.now()) + "\n\n")

        total_collections = len(collections_info)
        total_points = sum(info['points_count'] for info in collections_info.values())

        f.write(f"Total collections: {total_collections}\n")
        f.write(f"Total points: {total_points}\n\n")

        f.write("=== COLLECTIONS INFO ===\n")
        for name, info in collections_info.items():
            f.write(f"Collection: {name}\n")
            f.write(f"  Description: {info['description']}\n")
            f.write(f"  Points count: {info['points_count']}\n\n")

        f.write("=== DOCUMENTS ===\n\n")

        for collection_name in collections_info.keys():
            f.write(f"--- COLLECTION: {collection_name} ---\n\n")

            # Get all documents (set high limit)
            documents = db.get_collection_documents(collection_name, limit=10000)

            for i, doc in enumerate(documents, 1):
                f.write(f"Document {i}:\n")
                f.write(f"  ID: {doc['id']}\n")
                f.write(f"  URL: {doc['url']}\n")
                f.write(f"  Category: {doc['category']}\n")
                f.write(f"  Chunk index: {doc['chunk_index']}\n")
                f.write("  Text:\n")
                f.write(f"    {doc['text']}\n")
                f.write("  Metadata:\n")
                f.write(f"    {json.dumps(doc['metadata'], ensure_ascii=False, indent=4)}\n")
                f.write("\n" + "="*80 + "\n\n")

        f.write("=== EXPORT COMPLETE ===")

    print(f"Export completed. Data saved to {output_file}")
    print(f"Total collections: {total_collections}")
    print(f"Total points: {total_points}")

if __name__ == "__main__":
    export_db_to_text()
