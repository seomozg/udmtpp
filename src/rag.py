
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
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import re
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class RAGSystem:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.chroma_db = ChromaDB()
            self.embedder = EmbeddingModel()
            self.deepseek_api_key = get_env_var("DEEPSEEK_API_KEY")
            self.deepseek_base_url = get_env_var("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            self.top_k = int(get_env_var("TOP_K", "5"))
            self.confidence_threshold = float(get_env_var("CONFIDENCE_THRESHOLD", "0.4"))

            # Initialize BM25 indexes for hybrid search
            self.bm25_indexes = {}
            self.document_texts = {}
            self._build_bm25_indexes()

            # Initialize cross-encoder for reranking
            try:
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("Cross-encoder model loaded for reranking")
            except Exception as e:
                logger.warning(f"Could not load cross-encoder: {e}, reranking disabled")
                self.cross_encoder = None

            self.system_prompt = """
Ты — ассистент ТПП УР.
Отвечай только на основе предоставленного контекста. Не придумывай факты.
Если информации недостаточно — попроси уточнить вопрос.
Если найденные данные неуверенны (confidence < 0.7) — уточни вопрос.
Будь полезным и точным в ответах.
"""
            self.initialized = True

    def clone_with_settings(self, confidence_threshold: Optional[float] = None):
        """Create a clone with custom settings without reinitializing heavy components"""
        # Create new instance but share heavy components
        clone = RAGSystem.__new__(RAGSystem)
        clone.chroma_db = self.chroma_db  # Share ChromaDB instance
        clone.embedder = self.embedder    # Share embedder instance
        clone.deepseek_api_key = self.deepseek_api_key
        clone.deepseek_base_url = self.deepseek_base_url
        clone.top_k = self.top_k
        clone.confidence_threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        clone.system_prompt = self.system_prompt
        clone.initialized = True
        # Copy methods and data
        clone.search_context = self.search_context
        clone.generate_response = self.generate_response
        clone.generate_response_stream = self.generate_response_stream
        clone.expand_query = self.expand_query
        clone.ask = self.ask
        clone.ask_stream = self.ask_stream
        clone._build_bm25_indexes = self._build_bm25_indexes
        clone.bm25_indexes = self.bm25_indexes
        clone.document_texts = self.document_texts
        return clone

    def _build_bm25_indexes(self):
        """Build BM25 indexes for all collections"""
        try:
            # Download NLTK data if needed
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)

            for collection_name in self.chroma_db.collection_configs.keys():
                try:
                    # Get all documents from collection
                    documents = self.chroma_db.get_collection_documents(collection_name, limit=10000)
                    if documents:
                        # Extract texts and tokenize
                        texts = [doc['text'] for doc in documents]
                        tokenized_texts = []

                        for text in texts:
                            # Simple tokenization for Russian text
                            tokens = re.findall(r'\b\w+\b', text.lower())
                            tokenized_texts.append(tokens)

                        # Build BM25 index
                        if tokenized_texts:
                            self.bm25_indexes[collection_name] = BM25Okapi(tokenized_texts)
                            self.document_texts[collection_name] = documents
                            logger.info(f"Built BM25 index for {collection_name} with {len(tokenized_texts)} documents")
                        else:
                            logger.warning(f"No documents found for {collection_name}")
                    else:
                        logger.warning(f"No documents in collection {collection_name}")

                except Exception as e:
                    logger.error(f"Error building BM25 index for {collection_name}: {e}")

        except Exception as e:
            logger.error(f"Error building BM25 indexes: {e}")

    def hybrid_search(self, query: str, collection_name: Optional[str] = None, n_results: int = 5, collections_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and keyword search"""
        logger.info(f"Hybrid search for query: '{query}'")

        if collection_name:
            collections = [collection_name]
        elif collections_filter:
            collections = collections_filter
        else:
            collections = list(self.chroma_db.collection_configs.keys())

        all_results = []

        for coll in collections:
            try:
                # Semantic search
                query_embedding = self.embedder.encode([query])[0]
                semantic_results = self.chroma_db.search(coll, query_embedding, n_results * 2)  # Get more for reranking

                # Keyword search with BM25
                keyword_results = []
                if coll in self.bm25_indexes and coll in self.document_texts:
                    bm25 = self.bm25_indexes[coll]
                    documents = self.document_texts[coll]

                    # Tokenize query
                    query_tokens = re.findall(r'\b\w+\b', query.lower())

                    if query_tokens:
                        # Get BM25 scores
                        bm25_scores = bm25.get_scores(query_tokens)

                        # Create keyword results
                        for i, score in enumerate(bm25_scores):
                            if score > 0:
                                doc = documents[i]
                                keyword_results.append({
                                    'id': f"bm25_{i}",
                                    'score': score / 10.0,  # Normalize BM25 score to 0-1 range
                                    'payload': {
                                        'text': doc['text'],
                                        'url': doc.get('url', 'N/A'),
                                        'category': coll,
                                        'filename': doc.get('filename', ''),
                                        'chunk_index': doc.get('chunk_index', 0)
                                    }
                                })

                        # Sort by BM25 score and take top
                        keyword_results.sort(key=lambda x: x['score'], reverse=True)
                        keyword_results = keyword_results[:n_results]

                # Combine results with weighted scoring
                combined_results = []
                semantic_weight = 0.7
                keyword_weight = 0.3

                # Create lookup for semantic results
                semantic_lookup = {r['payload']['text'][:100]: r for r in semantic_results}

                # Process keyword results and combine with semantic
                for kw_result in keyword_results[:n_results//2]:  # Take half from keyword
                    text_key = kw_result['payload']['text'][:100]
                    if text_key in semantic_lookup:
                        # Combine scores if same document
                        sem_result = semantic_lookup[text_key]
                        combined_score = (sem_result['score'] * semantic_weight +
                                        kw_result['score'] * keyword_weight)
                        combined_results.append({
                            'id': sem_result['id'],
                            'score': combined_score,
                            'payload': sem_result['payload']
                        })
                    else:
                        # Only keyword result
                        combined_results.append({
                            'id': kw_result['id'],
                            'score': kw_result['score'] * keyword_weight,
                            'payload': kw_result['payload']
                        })

                # Add remaining semantic results
                for sem_result in semantic_results:
                    text_key = sem_result['payload']['text'][:100]
                    if text_key not in [r['payload']['text'][:100] for r in combined_results]:
                        combined_results.append({
                            'id': sem_result['id'],
                            'score': sem_result['score'] * semantic_weight,
                            'payload': sem_result['payload']
                        })

                all_results.extend(combined_results)

            except Exception as e:
                logger.error(f"Error in hybrid search for {coll}: {e}")

        # Rerank top results with cross-encoder if available
        if self.cross_encoder and len(all_results) > n_results:
            try:
                top_candidates = all_results[:min(len(all_results), n_results * 2)]  # Get more for reranking
                if len(top_candidates) > 1:
                    # Prepare pairs for cross-encoder
                    pairs = [[query, result['payload']['text']] for result in top_candidates]

                    # Get cross-encoder scores
                    ce_scores = self.cross_encoder.predict(pairs)

                    # Update scores with cross-encoder results
                    for i, result in enumerate(top_candidates):
                        # Combine original score with cross-encoder score
                        original_score = result['score']
                        ce_score = ce_scores[i]
                        # Normalize cross-encoder score (usually -10 to 10) to 0-1
                        normalized_ce = (ce_score + 10) / 20.0
                        # Weighted combination
                        result['score'] = 0.7 * original_score + 0.3 * normalized_ce

                    # Re-sort by updated scores
                    top_candidates.sort(key=lambda x: x['score'], reverse=True)
                    all_results = top_candidates

                logger.info(f"Reranked {len(top_candidates)} results with cross-encoder")
            except Exception as e:
                logger.warning(f"Cross-encoder reranking failed: {e}, using original ranking")

        # Sort all results by final score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:n_results]

    def search_context(self, query: str, collection_name: Optional[str] = None, n_results: int = 5, collections_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """Search for relevant context using hybrid search"""
        logger.info(f"Searching for query: '{query}' in collection: {collection_name}, n_results: {n_results}")

        # Use hybrid search instead of pure semantic search
        top_results = self.hybrid_search(query, collection_name, n_results, collections_filter)

        logger.info(f"Hybrid search returned {len(top_results)} results")

        # Calculate confidence
        scores = [r['score'] for r in top_results]
        confidence = calculate_confidence(scores)

        logger.info(f"Calculated confidence: {confidence}")

        context_texts = [r['payload']['text'] for r in top_results]

        # Convert all numpy types to Python native types for JSON serialization
        from utils import convert_numpy_types

        return {
            "context": "\n\n".join(context_texts),
            "confidence": confidence,
            "sources": [
                {
                    "url": r['payload'].get('url', 'N/A'),
                    "score": convert_numpy_types(r['score']),
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

    async def generate_response_stream(self, query: str, context: str, confidence: float, confidence_threshold: Optional[float] = None):
        """Generate streaming response using DeepSeek (async version)"""
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        if confidence < threshold:
            yield "Извините, я не нашел достаточно надежной информации для ответа на ваш вопрос. Пожалуйста, уточните вопрос или обратитесь к специалистам ТПП УР."
            return

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"}
        ]

        logger.info(f"Calling DeepSeek API with streaming: {len(context)} chars context")
        try:
            # Use aiohttp for async HTTP requests
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.deepseek_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.deepseek_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": messages,
                        "temperature": 0.1,
                        "max_tokens": 1000,
                        "stream": True
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    logger.info(f"DeepSeek API response status: {response.status}")

                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"DeepSeek API error: {response.status} - {error_text}")
                        yield f"Ошибка API: {response.status}"
                        return

                    # Process streaming response
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if not line:
                            continue

                        if line.startswith('data: '):
                            data = line[6:]  # Remove 'data: ' prefix
                            if data == '[DONE]':
                                break

                            try:
                                chunk = json.loads(data)
                                if chunk.get('choices') and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield f"\n\nПроизошла ошибка при обработке запроса: {str(e)}"

    def expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms"""
        try:
            prompt = f"""Расширь запрос, добавив синонимы и связанные термины.
Исходный запрос: "{query}"

Верни расширенный запрос на русском языке, который будет более эффективен для поиска.
Добавь ключевые слова, синонимы, связанные понятия.

Пример:
Исходный: "услуги ТПП"
Расширенный: "услуги ТПП поддержка бизнеса консультации предпринимателей содействие развитию"

Верни только расширенный запрос, без объяснений."""

            response = requests.post(
                f"{self.deepseek_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.deepseek_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 100
                },
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                expanded = result['choices'][0]['message']['content'].strip()
                logger.info(f"Query expanded: '{query}' -> '{expanded}'")
                return expanded
            else:
                logger.warning("Query expansion failed, using original")
                return query

        except Exception as e:
            logger.warning(f"Query expansion error: {e}, using original")
            return query

    def ask(self, query: str, collection_name: Optional[str] = None, n_results: int = 5, collections_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """Main method to ask questions"""
        logger.info(f"Processing query: {query} with n_results={n_results}")

        # Expand query for better search
        expanded_query = self.expand_query(query)
        search_query = expanded_query if len(expanded_query.split()) > len(query.split()) else query

        # Search context
        search_result = self.search_context(search_query, collection_name, n_results=n_results, collections_filter=collections_filter)

        # Generate response
        response = self.generate_response(query, search_result["context"], search_result["confidence"])

        return {
            "query": query,
            "response": response,
            "confidence": search_result["confidence"],
            "sources": search_result["sources"]
        }

    async def ask_stream(self, query: str, collection_name: Optional[str] = None, n_results: int = 5, collections_filter: Optional[List[str]] = None, confidence_threshold: Optional[float] = None):
        """Main method to ask questions with streaming response"""
        logger.info(f"Processing streaming query: {query} with n_results={n_results}")

        # Use provided confidence_threshold or default
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold

        # Expand query for better search
        expanded_query = self.expand_query(query)
        search_query = expanded_query if len(expanded_query.split()) > len(query.split()) else query

        # Search context
        search_result = self.search_context(search_query, collection_name, n_results=n_results, collections_filter=collections_filter)

        # Yield metadata first
        yield {
            "type": "metadata",
            "query": query,
            "confidence": search_result["confidence"],
            "sources": search_result["sources"]
        }

        # Stream response
        response_parts = []
        async for chunk in self.generate_response_stream(query, search_result["context"], search_result["confidence"], threshold):
            response_parts.append(chunk)
            yield {
                "type": "content",
                "content": chunk
            }

        # Yield final response
        full_response = "".join(response_parts)
        yield {
            "type": "end",
            "full_response": full_response
        }
