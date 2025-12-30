"""
RAG (Retrieval-Augmented Generation) system for UdmTPP
Handles query processing, search, and AI response generation
"""

import logging
from typing import Dict, List, Any, Optional
import requests

from config import (
    DEFAULT_N_RESULTS,
    CONFIDENCE_THRESHOLD,
    MAX_TOKENS,
    LLM_MODEL,
    DEEPSEEK_API_KEY,
    QUERY_EXPANSION_TEMPLATES,
    EMBEDDING_MODEL,
    RAG_ANSWER_PROMPT,
    LOW_CONFIDENCE_MESSAGE
)
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: BM25 not available. Install with: pip install rank-bm25")

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    print("Warning: CrossEncoder not available. Install with: pip install sentence-transformers")

from vector_db import ChromaDB
from embed import EmbeddingModel

logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG system for question answering"""

    def __init__(self):
        self.vector_db = ChromaDB()
        self.embedder = EmbeddingModel()
        self.bm25_indexes = {}
        self.document_texts = {}
        self.cross_encoder = None

        # Initialize BM25 indexes for hybrid search
        if BM25_AVAILABLE:
            self._build_bm25_indexes()
        else:
            logger.warning("BM25 not available - using semantic search only")

        # Initialize Cross-Encoder lazily (on first use)
        self.cross_encoder = None
        self.cross_encoder_initialized = False

    def _build_bm25_indexes(self):
        """Build BM25 indexes for all collections"""
        logger.info("Building BM25 indexes for hybrid search...")

        for collection_name in self.vector_db.collection_configs.keys():
            try:
                # Get all documents from collection
                documents = self.vector_db.get_collection_documents(collection_name, limit=10000)

                if documents:
                    # Extract texts for BM25
                    texts = [doc.get('text', '') for doc in documents if doc.get('text')]
                    if texts:
                        # Tokenize texts for BM25
                        tokenized_texts = [text.lower().split() for text in texts]
                        bm25 = BM25Okapi(tokenized_texts)
                        self.bm25_indexes[collection_name] = bm25
                        self.document_texts[collection_name] = texts
                        logger.info(f"Built BM25 index for {collection_name} with {len(texts)} documents")
                    else:
                        logger.warning(f"No texts found for BM25 index in {collection_name}")
                else:
                    logger.warning(f"No documents found for BM25 index in {collection_name}")

            except Exception as e:
                logger.error(f"Error building BM25 index for {collection_name}: {e}")

    def expand_query(self, query: str) -> str:
        """
        Expand query with related terms

        Args:
            query: Original query

        Returns:
            Expanded query
        """
        query_lower = query.lower()
        expanded_terms = []

        # Add original query
        expanded_terms.append(query)

        # Add related terms from templates
        for category, terms in QUERY_EXPANSION_TEMPLATES.items():
            if any(term in query_lower for term in terms):
                expanded_terms.extend(terms)

        # Remove duplicates and join
        unique_terms = list(set(expanded_terms))
        return " ".join(unique_terms)

    def hybrid_search(self, query: str, collection: Optional[str] = None, n_results: int = DEFAULT_N_RESULTS, alpha: float = 0.7) -> Dict[str, Any]:
        """
        Hybrid search combining semantic and keyword-based search

        Args:
            query: Search query
            collection: Specific collection to search in (None for all)
            n_results: Number of results to return
            alpha: Weight for semantic search (0-1), 1-alpha for keyword search

        Returns:
            Search results
        """
        try:
            # Expand query
            expanded_query = self.expand_query(query)
            logger.info(f"Query expanded: '{query}' -> '{expanded_query}'")

            # Generate embedding for the query
            query_embedding = self.embedder.encode([expanded_query])[0]

            # Get candidates from semantic search (more candidates for reranking)
            semantic_limit = n_results * 3
            semantic_results = []

            if collection:
                semantic_results = self.vector_db.search(collection, query_embedding, limit=semantic_limit, query_text=expanded_query)
                for result in semantic_results:
                    result["collection"] = collection
            else:
                for coll_name in self.vector_db.collection_configs.keys():
                    results = self.vector_db.search(coll_name, query_embedding, limit=semantic_limit, query_text=expanded_query)
                    for result in results:
                        result["collection"] = coll_name
                    semantic_results.extend(results)

            # Get BM25 scores if available
            if BM25_AVAILABLE and semantic_results:
                bm25_scores = {}

                # Group results by collection for BM25 scoring
                for result in semantic_results:
                    coll_name = result.get("collection", "")
                    if coll_name and coll_name in self.bm25_indexes:
                        doc_text = result.get("payload", {}).get("text", "")
                        if doc_text:
                            # Get BM25 score for this document
                            query_tokens = expanded_query.lower().split()
                            bm25 = self.bm25_indexes[coll_name]

                            # Find document index in BM25 corpus
                            try:
                                doc_index = self.document_texts[coll_name].index(doc_text)
                                scores = bm25.get_scores(query_tokens)
                                bm25_score = scores[doc_index] if doc_index < len(scores) else 0.0
                                bm25_scores[result.get("id", "")] = bm25_score
                            except (ValueError, IndexError):
                                bm25_scores[result.get("id", "")] = 0.0

                # Combine semantic and BM25 scores
                for result in semantic_results:
                    semantic_score = result.get("score", 0.0)
                    bm25_score = bm25_scores.get(result.get("id", ""), 0.0)

                    # Normalize BM25 score (BM25 scores can be large)
                    if bm25_score > 0:
                        bm25_score = min(bm25_score / 10.0, 1.0)  # Normalize to 0-1 range

                    # Weighted combination
                    combined_score = alpha * semantic_score + (1 - alpha) * bm25_score
                    result["combined_score"] = combined_score
                    result["semantic_score"] = semantic_score
                    result["bm25_score"] = bm25_score

                # Sort by combined score
                semantic_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
            else:
                # Fallback to semantic search only
                for result in semantic_results:
                    result["combined_score"] = result.get("score", 0.0)

            # Apply Cross-Encoder reranking if available
            if self.cross_encoder and len(semantic_results) > n_results:
                semantic_results = self._rerank_with_cross_encoder(expanded_query, semantic_results, n_results)

            # Take top results
            final_results = semantic_results[:n_results]

            # Calculate confidence
            confidence = self._calculate_confidence(final_results)

            return {
                "query": query,
                "expanded_query": expanded_query,
                "results": final_results,
                "confidence": confidence,
                "n_results": len(final_results),
                "search_type": "hybrid_ce" if (BM25_AVAILABLE and self.cross_encoder) else ("hybrid" if BM25_AVAILABLE else "semantic")
            }

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return {
                "query": query,
                "error": str(e),
                "results": [],
                "confidence": 0.0,
                "n_results": 0
            }

    def search(self, query: str, collection: Optional[str] = None, n_results: int = DEFAULT_N_RESULTS) -> Dict[str, Any]:
        """
        Search for relevant documents (uses hybrid search if available)

        Args:
            query: Search query
            collection: Specific collection to search in (None for all)
            n_results: Number of results to return

        Returns:
            Search results
        """
        # Use hybrid search if BM25 is available, otherwise fallback to semantic
        if BM25_AVAILABLE and self.bm25_indexes:
            return self.hybrid_search(query, collection, n_results)
        else:
            return self._semantic_search(query, collection, n_results)

    def _semantic_search(self, query: str, collection: Optional[str] = None, n_results: int = DEFAULT_N_RESULTS) -> Dict[str, Any]:
        """
        Pure semantic search (fallback when BM25 not available)

        Args:
            query: Search query
            collection: Specific collection to search in (None for all)
            n_results: Number of results to return

        Returns:
            Search results
        """
        try:
            # Expand query
            expanded_query = self.expand_query(query)
            logger.info(f"Query expanded: '{query}' -> '{expanded_query}'")

            # Generate embedding for the query
            query_embedding = self.embedder.encode([expanded_query])[0]

            # Search in specified collection or all collections
            all_results = []
            if collection:
                # Search in specific collection
                results = self.vector_db.search(collection, query_embedding, limit=n_results)
                all_results.extend(results)
            else:
                # Search in all collections and combine results
                for coll_name in self.vector_db.collection_configs.keys():
                    results = self.vector_db.search(coll_name, query_embedding, limit=n_results)
                    # Add collection info to results
                    for result in results:
                        result["collection"] = coll_name
                    all_results.extend(results)

                # Sort by score and take top results
                all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
                all_results = all_results[:n_results]

            # Calculate confidence
            confidence = self._calculate_confidence(all_results)

            return {
                "query": query,
                "expanded_query": expanded_query,
                "results": all_results,
                "confidence": confidence,
                "n_results": len(all_results),
                "search_type": "semantic"
            }

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return {
                "query": query,
                "error": str(e),
                "results": [],
                "confidence": 0.0,
                "n_results": 0
            }

    def _calculate_confidence(self, results: List[Dict]) -> float:
        """
        Calculate confidence score from search results

        Args:
            results: Search results

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not results:
            return 0.0

        # Use average similarity score as confidence
        scores = [result.get("score", 0.0) for result in results]
        return sum(scores) / len(scores) if scores else 0.0

    def ask(self, query: str, collection: Optional[str] = None, n_results: int = 30, stream: bool = False):
        """
        Answer question using RAG

        Args:
            query: User question
            collection: Specific collection to search in
            n_results: Number of search results to use
            stream: Whether to use streaming response

        Returns:
            Answer with sources and confidence, or streaming response object
        """
        try:
            # Search for relevant documents
            search_results = self.search(query, collection, n_results)

            if search_results["confidence"] < CONFIDENCE_THRESHOLD:
                if stream:
                    # For streaming, return a simple response object that can be iterated
                    import json
                    class StreamResponse:
                        def __iter__(self):
                            # Send metadata
                            metadata_chunk = {
                                "type": "metadata",
                                "confidence": search_results["confidence"],
                                "sources": []
                            }
                            yield f"data: {json.dumps(metadata_chunk, ensure_ascii=False)}\n\n"

                            # Send content
                            content_chunk = {
                                "type": "content",
                                "content": LOW_CONFIDENCE_MESSAGE
                            }
                            yield f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"

                            # Send end
                            end_chunk = {
                                "type": "end",
                                "full_response": LOW_CONFIDENCE_MESSAGE
                            }
                            yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
                            yield "data: [DONE]\n\n"

                    return StreamResponse()
                else:
                    return {
                        "query": query,
                        "response": LOW_CONFIDENCE_MESSAGE,
                        "confidence": search_results["confidence"],
                        "sources": [],
                        "search_results": search_results
                    }

            # Generate answer using AI
            context = self._prepare_context(search_results["results"])

            if stream:
                # For streaming, return a generator that yields SSE chunks
                return self._generate_streaming_answer(query, context, search_results)
            else:
                response_text = self._generate_answer(query, context)
                sources = self._extract_sources(search_results["results"])

                return {
                    "query": query,
                    "response": response_text,
                    "confidence": search_results["confidence"],
                    "sources": sources,
                    "search_results": search_results
                }

        except Exception as e:
            logger.error(f"Error in ask: {e}")
            if stream:
                import json
                class ErrorStreamResponse:
                    def __iter__(self):
                        error_chunk = {
                            "type": "error",
                            "error": str(e)
                        }
                        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"

                return ErrorStreamResponse()
            else:
                return {
                    "query": query,
                    "response": f"Произошла ошибка при обработке запроса: {str(e)}",
                    "confidence": 0.0,
                    "sources": [],
                    "error": str(e)
                }

    def _prepare_context(self, results: List[Dict]) -> str:
        """
        Prepare context from search results

        Args:
            results: Search results

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, result in enumerate(results, 1):
            payload = result.get("payload", {})
            text = payload.get("text", "")
            url = payload.get("url", "")
            score = result.get("score", 0.0)

            context_parts.append(f"[Источник {i}] (релевантность: {score:.3f})\nURL: {url}\nТекст: {text}\n")

        return "\n".join(context_parts)

    def _generate_streaming_answer(self, query: str, context: str, search_results: Dict[str, Any]):
        """
        Generate streaming answer using AI

        Args:
            query: User question
            context: Search context
            search_results: Search results for metadata

        Returns:
            Generator yielding SSE chunks
        """
        import json

        try:
            # Extract sources for metadata
            sources = self._extract_sources(search_results["results"])

            # Send metadata first
            metadata_chunk = {
                "type": "metadata",
                "confidence": search_results["confidence"],
                "sources": sources
            }
            yield f"data: {json.dumps(metadata_chunk, ensure_ascii=False)}\n\n"

            prompt = RAG_ANSWER_PROMPT.format(
                context=context,
                query=query
            )

            if not DEEPSEEK_API_KEY:
                error_chunk = {
                    "type": "content",
                    "content": "AI ключ не настроен. Обратитесь к администратору."
                }
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                end_chunk = {
                    "type": "end",
                    "full_response": "AI ключ не настроен. Обратитесь к администратору."
                }
                yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.1,
                    "stream": True
                },
                timeout=60,
                stream=True
            )

            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        full_response += content
                                        # Send content chunk
                                        content_chunk = {
                                            "type": "content",
                                            "content": content
                                        }
                                        yield f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"
                            except json.JSONDecodeError:
                                continue

                # Send end chunk with full response
                end_chunk = {
                    "type": "end",
                    "full_response": full_response
                }
                yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            else:
                logger.error(f"DeepSeek API error: {response.status_code}")
                error_chunk = {
                    "type": "content",
                    "content": f"Ошибка API: {response.status_code}"
                }
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                end_chunk = {
                    "type": "end",
                    "full_response": f"Ошибка API: {response.status_code}"
                }
                yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error in streaming answer generation: {e}")
            error_chunk = {
                "type": "content",
                "content": f"Ошибка при генерации ответа: {str(e)}"
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
            end_chunk = {
                "type": "end",
                "full_response": f"Ошибка при генерации ответа: {str(e)}"
            }
            yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    def _generate_answer(self, query: str, context: str, stream: bool = False) -> str:
        """
        Generate answer using AI

        Args:
            query: User question
            context: Search context
            stream: Whether to use streaming response

        Returns:
            AI-generated answer
        """
        try:
            prompt = RAG_ANSWER_PROMPT.format(
                context=context,
                query=query
            )

            if not DEEPSEEK_API_KEY:
                return "AI ключ не настроен. Обратитесь к администратору."

            request_data = {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.1
            }

            if stream:
                request_data["stream"] = True

            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=request_data,
                timeout=60,
                stream=stream
            )

            if response.status_code == 200:
                if stream:
                    # For streaming, return the response object
                    return response
                else:
                    result = response.json()
                    return result["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"DeepSeek API error: {response.status_code}")
                return f"Ошибка API: {response.status_code}"

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Ошибка при генерации ответа: {str(e)}"

    def _extract_sources(self, results: List[Dict]) -> List[Dict]:
        """
        Extract sources from search results

        Args:
            results: Search results

        Returns:
            List of source dictionaries
        """
        sources = []
        for result in results:
            payload = result.get("payload", {})
            source = {
                "url": payload.get("url", ""),
                "score": result.get("score", 0.0),
                "category": payload.get("category", "unknown")
            }
            sources.append(source)

        return sources

    def _rerank_with_cross_encoder(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        Rerank candidates using Cross-Encoder (lazy initialization)

        Args:
            query: Search query
            candidates: Candidate results to rerank
            top_k: Number of top results to return

        Returns:
            Reranked results
        """
        if not candidates:
            return candidates

        # Lazy initialization of Cross-Encoder
        if not self.cross_encoder_initialized:
            if CROSS_ENCODER_AVAILABLE:
                try:
                    logger.info("Initializing Cross-Encoder for reranking...")
                    self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                    self.cross_encoder_initialized = True
                    logger.info("Cross-Encoder initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize Cross-Encoder: {e}")
                    self.cross_encoder_initialized = True  # Don't try again
                    return candidates
            else:
                logger.warning("Cross-Encoder not available - skipping reranking")
                return candidates

        if not self.cross_encoder:
            return candidates

        try:
            logger.info(f"Reranking {len(candidates)} candidates with Cross-Encoder")

            # Prepare query-document pairs for Cross-Encoder
            query_doc_pairs = []
            for candidate in candidates:
                doc_text = candidate.get("payload", {}).get("text", "")
                if doc_text:
                    query_doc_pairs.append([query, doc_text])

            if not query_doc_pairs:
                return candidates

            # Get Cross-Encoder scores
            scores = self.cross_encoder.predict(query_doc_pairs)

            # Add rerank scores to candidates
            for i, candidate in enumerate(candidates):
                if i < len(scores):
                    candidate["rerank_score"] = float(scores[i])
                    # Update combined score with rerank score (70% original + 30% rerank)
                    original_score = candidate.get("combined_score", candidate.get("score", 0.0))
                    candidate["final_score"] = 0.7 * original_score + 0.3 * scores[i]

            # Sort by final score
            candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)

            logger.info(f"Cross-Encoder reranking completed, returning top {top_k}")
            return candidates[:top_k]

        except Exception as e:
            logger.error(f"Error in Cross-Encoder reranking: {e}")
            return candidates

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about all collections

        Returns:
            Collection information
        """
        return self.vector_db.get_collection_info()
