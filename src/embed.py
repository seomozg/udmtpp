from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from utils import get_env_var

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self):
        model_name = get_env_var("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings for text(s)"""
        if isinstance(texts, str):
            texts = [texts]

        logger.info(f"Encoding {len(texts)} text(s)")
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        logger.debug(f"Encoding query: {query[:50]}...")
        embedding = self.model.encode([query], normalize_embeddings=True)
        return embedding[0].tolist()
