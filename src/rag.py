import requests
import json
import logging
from typing import List, Dict, Any, Optional
import sys
import os

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils import get_env_var, calculate_confidence
from vector_db import ChromaDB
from embed import EmbeddingModel

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.chroma_db = ChromaDB()
        self.embedder = EmbeddingModel()
        self.deepseek_api_key = get_env_var("DEEPSEEK_API_KEY")
        self.deepseek_base_url = get_env_var("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.top_k = int(get_env_var("TOP_K", "5"))
        self.confidence_threshold = float(get_env_var("CONFIDENCE_THRESHOLD", "0.7"))

        self.system_prompt = """
Ты — ассистент ТПП УР.
Отвечай только на основе предоставленного контекста. Не придумывай факты.
Если информации недостаточно — попроси уточнить вопрос.
Если найденные данные неуверенны (confidence < 0.7) — уточни вопрос.
Будь полезным и точным в ответах.
"""

    def search_context(self, query: str, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Search for relevant context"""
        logger.info(f"Searching for query: '{query}' in collection: {collection_name}")

        # Use same encoding method as documents for consistency
        query_embeddings = self.embedder.encode([query])
        query_embedding = query_embeddings[0] if query_embeddings else []
        logger.info(f"Query embedding shape: {len(query_embedding)}")

        if collection_name:
            collections = [collection_name]
        else:
            collections = list(self.chroma_db.collection_configs.keys())

        all_results = []
        for coll in collections:
            try:
                logger.info(f"Searching in collection '{coll}'")
                # Use ChromaDB search method
                results = self.chroma_db.search(coll, query_embedding, self.top_k)
                logger.info(f"Found {len(results)} results in {coll}")
                all_results.extend(results)

            except Exception as e:
                logger.error(f"Error searching in {coll}: {e}")

        logger.info(f"Total results found: {len(all_results)}")

        # Sort by score and take top results
        all_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = all_results[:self.top_k]

        logger.info(f"Top {len(top_results)} results selected")

        # Calculate confidence
        scores = [r['score'] for r in top_results]
        confidence = calculate_confidence(scores)

        logger.info(f"Calculated confidence: {confidence}")

        context_texts = [r['payload']['text'] for r in top_results]

        return {
            "context": "\n\n".join(context_texts),
            "confidence": confidence,
            "sources": [
                {
                    "url": r['payload']['url'],
                    "score": r['score'],
                    "category": r['payload'].get('category', 'unknown')
                }
                for r in top_results
            ]
        }

    def generate_response(self, query: str, context: str, confidence: float) -> str:
        """Generate response using DeepSeek"""
        if confidence < self.confidence_threshold:
            return "Извините, я не нашел достаточно надежной информации для ответа на ваш вопрос. Пожалуйста, уточните вопрос или обратитесь к специалистам ТПП УР."

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"}
        ]

        try:
            response = requests.post(
                f"{self.deepseek_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.deepseek_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 1000
                },
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content'].strip()

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Произошла ошибка при обработке запроса. Попробуйте позже."

    def ask(self, query: str, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Main method to ask questions"""
        logger.info(f"Processing query: {query}")

        # Search context
        search_result = self.search_context(query, collection_name)

        # Generate response
        response = self.generate_response(query, search_result["context"], search_result["confidence"])

        return {
            "query": query,
            "response": response,
            "confidence": search_result["confidence"],
            "sources": search_result["sources"]
        }
