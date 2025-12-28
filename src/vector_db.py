import chromadb
from typing import List, Dict, Any, Optional
import logging
import os
import uuid

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
            # Initialize persistent ChromaDB
            persist_dir = os.path.join(os.getcwd(), "chroma_db")
            os.makedirs(persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir)
            logger.info(f"Initialized ChromaDB with persistence at {persist_dir}")

            # Collection configurations
            self.collection_configs = {
                "719": "Консультации по 719-ПП / Акт СТ",
                "support": "Меры поддержки бизнеса",
                "services": "Услуги ТПП",
                "membership": "Членство в ТПП",
                "events": "Мероприятия, обучение",
                "cooperation": "Поиск партнёров / коопераций",
                "site": "Общий контент сайта"
            }

            # Initialize collections
            self.collections = {}
            for collection_name in self.collection_configs.keys():
                try:
                    collection = self.client.get_or_create_collection(
                        name=collection_name,
                        metadata={"description": self.collection_configs[collection_name]}
                    )
                    self.collections[collection_name] = collection
                    logger.info(f"Collection {collection_name} ready")
                except Exception as e:
                    logger.error(f"Error creating collection {collection_name}: {e}")

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

    def search(self, collection_name: str, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors in ChromaDB"""
        logger.info(f"Searching in {collection_name}, vector length: {len(query_vector)}")

        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return []

        collection = self.collections[collection_name]

        try:
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

    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            if collection_name in self.collections:
                del self.collections[collection_name]
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
