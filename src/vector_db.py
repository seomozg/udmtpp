import chromadb
from typing import List, Dict, Any, Optional
import logging
import os
import uuid

from config import COLLECTION_CONFIGS, CHROMA_DB_DIR

logger = logging.getLogger(__name__)

class ChromaDB:
    """ChromaDB wrapper for vector storage and retrieval"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            # Initialize persistent ChromaDB - use config path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)  # Go up from src/ to project root
            persist_dir = os.path.join(project_root, CHROMA_DB_DIR)
            os.makedirs(persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir)
            logger.info(f"Initialized ChromaDB with persistence at {persist_dir}")

            # Collection configurations from config
            self.collection_configs = {name: config["name"] for name, config in COLLECTION_CONFIGS.items()}

            # Initialize collections
            self.collections = {}

            # First, try to get existing collections
            existing_collections = self.client.list_collections()
            existing_names = {col.name: col for col in existing_collections}

            for collection_name in self.collection_configs.keys():
                try:
                    if collection_name in existing_names:
                        # Use existing collection
                        collection = existing_names[collection_name]
                        logger.info(f"Loaded existing collection {collection_name}")
                    else:
                        # Create new collection
                        collection = self.client.get_or_create_collection(
                            name=collection_name,
                            metadata={"description": self.collection_configs[collection_name]}
                        )
                        logger.info(f"Created new collection {collection_name}")

                    self.collections[collection_name] = collection
                except Exception as e:
                    logger.error(f"Error initializing collection {collection_name}: {e}")

            self.initialized = True

    def add_points(self, collection_name: str, points):
        """Add points to ChromaDB collection"""
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return

        collection = self.collections[collection_name]
        logger.info(f"Adding {len(points)} points to {collection_name}")

        # Convert points to ChromaDB format
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for point in points:
            ids.append(str(point.id))
            embeddings.append(point.vector)

            # Convert payload to metadata
            metadata = {}
            if hasattr(point, 'payload') and point.payload:
                for key, value in point.payload.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    else:
                        metadata[key] = str(value)

            metadatas.append(metadata)
            documents.append(point.payload.get('text', '') if hasattr(point, 'payload') else '')

        try:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.info(f"Successfully added {len(points)} points to {collection_name}")
        except Exception as e:
            logger.error(f"Error adding points to {collection_name}: {e}")

    def search(self, collection_name: str, query_vector: List[float], limit: int = 5, query_text: str = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in ChromaDB"""
        logger.info(f"Searching in {collection_name}, vector length: {len(query_vector)}, query_text: {query_text}")

        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return []

        collection = self.collections[collection_name]

        try:
            # First try semantic search
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                include=['distances', 'metadatas', 'documents']
            )

            formatted_results = []
            if results['ids'] and len(results['ids']) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    # ChromaDB returns cosine distance (0-2), convert to similarity score (0-1)
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    similarity_score = max(0.0, 1.0 - (distance / 2.0))  # Convert to 0-1 range

                    formatted_results.append({
                        "id": doc_id,
                        "score": similarity_score,
                        "payload": results['metadatas'][0][i] if results['metadatas'] else {}
                    })

            # Always try metadata search for exact text matches if we have query_text
            if query_text:
                logger.info(f"Low semantic scores, trying metadata search for: {query_text}")

                # Search for documents containing the query in metadata
                all_docs = collection.get(include=['metadatas', 'documents'], limit=1000)
                metadata_matches = []

                query_lower = query_text.lower()
                documents_list = all_docs.get('documents', [])
                metadatas_list = all_docs.get('metadatas', [])

                for i, metadata in enumerate(metadatas_list):
                    if metadata:
                        url = metadata.get('url', '').lower()
                        doc_text = documents_list[i] if i < len(documents_list) else ""
                        doc_text = doc_text or ""
                        doc_text_lower = doc_text.lower()

                        score = 0.0
                        match_type = "none"

                        # Check if query_text is in URL (highest priority)
                        if query_lower in url:
                            score = 0.95
                            match_type = "url_exact"
                        # Check if query_text is in document text
                        elif query_lower in doc_text_lower:
                            score = 0.85
                            match_type = "text_exact"

                        if score > 0:
                            metadata_matches.append({
                                "id": all_docs['ids'][i],
                                "score": score,
                                "payload": metadata,
                                "match_type": match_type
                            })

                if metadata_matches:
                    logger.info(f"Found {len(metadata_matches)} metadata matches")
                    # Add metadata matches to results
                    formatted_results.extend(metadata_matches[:limit])
                    # Sort by score and take top results
                    formatted_results.sort(key=lambda x: x.get('score', 0), reverse=True)
                    formatted_results = formatted_results[:limit]

            # Debug logging
            logger.info(f"ChromaDB raw results: ids={len(results.get('ids', [[]])[0])}, distances={len(results.get('distances', [[]])[0])}")
            for i, result in enumerate(formatted_results[:3]):
                payload = result.get('payload', {})
                doc_text = payload.get('text', '')[:100] if payload else "No payload"
                match_type = result.get('match_type', 'semantic')
                logger.info(f"Result {i}: score={result.get('score', 0):.4f}, match_type={match_type}, text='{doc_text}...'")

            logger.info(f"Search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Search error in {collection_name}: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all collections"""
        info = {}
        total_points = 0

        for collection_name, description in self.collection_configs.items():
            try:
                if collection_name in self.collections:
                    count = self.collections[collection_name].count()
                    total_points += count

                    info[collection_name] = {
                        "name": collection_name,
                        "description": description,
                        "vectors_count": count,
                        "points_count": count
                    }
                else:
                    info[collection_name] = {
                        "name": collection_name,
                        "description": description,
                        "vectors_count": 0,
                        "points_count": 0
                    }
            except Exception as e:
                logger.warning(f"Could not get info for {collection_name}: {e}")
                info[collection_name] = {
                    "name": collection_name,
                    "description": description,
                    "vectors_count": 0,
                    "points_count": 0
                }

        logger.info(f"Total points across all collections: {total_points}")
        return info

    def get_collection_documents(self, collection_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get documents from a collection"""
        logger.info(f"Getting documents from {collection_name}, limit: {limit}")

        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return []

        collection = self.collections[collection_name]

        try:
            # Get all documents with metadata
            results = collection.get(
                include=['metadatas', 'documents'],
                limit=limit
            )

            documents = []
            if results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    documents.append({
                        "id": doc_id,
                        "text": results['documents'][i] if results['documents'] else "",
                        "metadata": results['metadatas'][i] if results['metadatas'] else {},
                        "url": results['metadatas'][i].get('url', '') if results['metadatas'] else '',
                        "filename": results['metadatas'][i].get('filename', '') if results['metadatas'] else '',
                        "category": results['metadatas'][i].get('category', '') if results['metadatas'] else '',
                        "chunk_index": results['metadatas'][i].get('chunk_index', 0) if results['metadatas'] else 0
                    })

            logger.info(f"Retrieved {len(documents)} documents from {collection_name}")
            return documents

        except Exception as e:
            logger.error(f"Error getting documents from {collection_name}: {e}")
            return []

    def delete_points(self, collection_name: str, point_ids: List[str]):
        """Delete points from ChromaDB collection by IDs"""
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return False

        if not point_ids:
            logger.warning("No point IDs provided for deletion")
            return True

        collection = self.collections[collection_name]
        logger.info(f"Deleting {len(point_ids)} points from {collection_name}")

        try:
            collection.delete(ids=point_ids)
            logger.info(f"Successfully deleted {len(point_ids)} points from {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting points from {collection_name}: {e}")
            return False

    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            if collection_name in self.collections:
                del self.collections[collection_name]
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
