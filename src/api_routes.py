"""
API routes for UdmTPP RAG system
Handles all REST API endpoints
"""

import logging
import os
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
import json

from config import (
    DEFAULT_N_RESULTS,
    CONFIDENCE_THRESHOLD,
    MAX_TOKENS
)
from rag_system import RAGSystem
from site_parser import SiteParser
from vector_db import ChromaDB
from utils import semantic_chunk_text
from embed import EmbeddingModel
from categorizer import categorize_content
import uuid

logger = logging.getLogger(__name__)

# Initialize components
chroma_db = ChromaDB()
rag_system = RAGSystem()

# Create router
api_router = APIRouter()


@api_router.post("/chat")
async def chat_api(
    query: str = Form(...),
    collection: Optional[str] = Form(None),
    temperature: float = Form(0.1),
    max_tokens: int = Form(1000),
    n_results: int = Form(5),
    confidence_threshold: float = Form(0.4)
):
    """API endpoint for chat"""
    try:
        result = rag_system.ask(
            query=query,
            collection=collection,
            n_results=n_results
        )

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Chat API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@api_router.get("/chat/stream")
async def chat_stream_api(
    query: str,
    collection: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1000,
    n_results: int = 5,
    confidence_threshold: float = 0.4
):
    """Streaming API endpoint for chat"""
    try:
        stream_response = rag_system.ask(query, collection, n_results, stream=True)

        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            stream_response,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )

    except Exception as e:
        logger.error(f"Chat streaming API error: {e}")
        # Return error response
        async def error_stream():
            error_chunk = {
                "type": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )


@api_router.get("/collections")
async def get_collections_api():
    """Get collections info API"""
    try:
        collections_info = chroma_db.get_collection_info()
        return JSONResponse(content={"collections": collections_info})
    except Exception as e:
        logger.error(f"Collections API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@api_router.get("/collections/{collection_id}/documents")
async def get_collection_documents_api(collection_id: str, limit: int = 50):
    """Get documents from a specific collection"""
    try:
        documents = chroma_db.get_collection_documents(collection_id, limit)

        return JSONResponse(content={
            "collection_id": collection_id,
            "documents": documents,
            "total_count": len(documents)
        })
    except Exception as e:
        logger.error(f"Collection documents API error: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")


@api_router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    category: str = Form(...)
):
    """Upload and process document"""
    try:
        # Read file content
        content = await file.read()

        # Extract text based on file type
        filename = (file.filename or "").lower()

        if filename.endswith('.pdf'):
            # PDF processing
            try:
                from PyPDF2 import PdfReader
                from io import BytesIO
                pdf_stream = BytesIO(content)
                pdf_reader = PdfReader(pdf_stream)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            except ImportError:
                raise HTTPException(status_code=400, detail="PDF processing not available")

        elif filename.endswith(('.doc', '.docx')):
            # DOC/DOCX processing
            try:
                from docx import Document
                from io import BytesIO
                doc_stream = BytesIO(content)
                doc = Document(doc_stream)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
            except ImportError:
                raise HTTPException(status_code=400, detail="DOC processing not available")

        else:
            # Plain text
            text = content.decode('utf-8')

        # Process text
        embedder = EmbeddingModel()
        chunks = semantic_chunk_text(text, max_chunk_size=800, overlap=50)
        embeddings = embedder.encode(chunks)

        # Create points
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
                    "filename": file.filename,
                    "text": chunk,
                    "category": category,
                    "chunk_index": i,
                    "source": "upload"
                }
            ))

        # Save to database
        chroma_db.add_points(category, points)

        return JSONResponse(content={
            "message": f"Файл {file.filename} успешно загружен и обработан",
            "chunks_count": len(chunks)
        })

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при загрузке файла")


@api_router.post("/parse-site")
async def parse_site_api():
    """Trigger site parsing"""
    try:
        logger.info("Starting site parsing via API")

        parser = SiteParser()
        urls = parser.parse_sitemap()
        logger.info(f"Found {len(urls)} URLs in sitemap")

        # Process limited URLs (increased limit for better database coverage)
        urls_to_process = urls[:100]  # Increased from 30 to 100
        logger.info(f"Processing {len(urls_to_process)} URLs from sitemap")

        result = parser.process_urls_with_ai_plan(urls_to_process, limit=len(urls_to_process))

        # Save points to database
        if result["points"]:
            logger.info(f"Saving {len(result['points'])} points to ChromaDB")
            # Group points by collection
            points_by_collection = {}
            for point in result["points"]:
                category = point.payload["category"]
                if category not in points_by_collection:
                    points_by_collection[category] = []
                points_by_collection[category].append(point)

            # Save to each collection
            for collection_name, points in points_by_collection.items():
                logger.info(f"Saving {len(points)} points to collection '{collection_name}'")
                try:
                    chroma_db.add_points(collection_name, points)
                    logger.info(f"Successfully saved {len(points)} points to '{collection_name}'")
                except Exception as e:
                    logger.error(f"Error saving points to collection '{collection_name}': {e}")
                    raise  # Re-raise to fail the request
        else:
            logger.warning("No points to save - result['points'] is empty")

        # Get updated statistics
        collections_info = chroma_db.get_collection_info()

        return JSONResponse(content={
            "message": f"Обработано {result['processed_urls']} из {len(urls)} URL с сайта (с AI-планированием)",
            "saved_points": result['total_points'],
            "collections": collections_info,
            "ai_mappings_count": len(result.get('ai_mappings', {})),
            "status": "completed"
        })

    except Exception as e:
        logger.error(f"Parse site error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/parse-site-full")
async def parse_site_full_api(limit: int = 200):
    """Trigger full site parsing with configurable limit"""
    try:
        logger.info(f"Starting full site parsing via API with limit {limit}")

        parser = SiteParser()
        urls = parser.parse_sitemap()
        logger.info(f"Found {len(urls)} URLs in sitemap")

        # Process URLs with specified limit
        urls_to_process = urls[:limit]
        logger.info(f"Processing {len(urls_to_process)} URLs from sitemap")

        result = parser.process_urls_with_ai_plan(urls_to_process, limit=len(urls_to_process))

        # Save points to database
        if result["points"]:
            logger.info(f"Saving {len(result['points'])} points to ChromaDB")
            # Group points by collection
            points_by_collection = {}
            for point in result["points"]:
                category = point.payload["category"]
                if category not in points_by_collection:
                    points_by_collection[category] = []
                points_by_collection[category].append(point)

            # Save to each collection
            for collection_name, points in points_by_collection.items():
                logger.info(f"Saving {len(points)} points to collection '{collection_name}'")
                try:
                    chroma_db.add_points(collection_name, points)
                    logger.info(f"Successfully saved {len(points)} points to '{collection_name}'")
                except Exception as e:
                    logger.error(f"Error saving points to collection '{collection_name}': {e}")
                    raise  # Re-raise to fail the request
        else:
            logger.warning("No points to save - result['points'] is empty")

        # Get updated statistics
        collections_info = chroma_db.get_collection_info()

        return JSONResponse(content={
            "message": f"Обработано {result['processed_urls']} из {len(urls)} URL с сайта (полный парсинг)",
            "saved_points": result['total_points'],
            "collections": collections_info,
            "ai_mappings_count": len(result.get('ai_mappings', {})),
            "status": "completed"
        })

    except Exception as e:
        logger.error(f"Parse site full error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/rebuild-from-cache")
async def rebuild_from_cache_api():
    """Rebuild database from existing cache files"""
    try:
        logger.info("Starting rebuild from cache")

        parser = SiteParser()
        cache_dir = parser.cache_dir

        if not os.path.exists(cache_dir):
            return JSONResponse(content={
                "message": "Cache directory not found",
                "status": "error"
            })

        # Find all cache files
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.txt')]
        logger.info(f"Found {len(cache_files)} cache files")

        total_points = 0
        processed_files = 0

        for cache_file in cache_files:
            try:
                filepath = os.path.join(cache_dir, cache_file)

                # Parse cache file
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract URL from first line
                lines = content.split('\n')
                url = ""
                text_content = ""

                if lines and lines[0].startswith('URL: '):
                    url = lines[0][5:].strip()

                    # Find content start
                    content_start = False
                    for line in lines[3:]:  # Skip URL, empty line, timestamp, empty line
                        if line.startswith('=') * 50:  # Content separator
                            content_start = True
                            continue
                        if content_start:
                            text_content += line + '\n'

                if url and text_content.strip():
                    # Process the cached content
                    chunks = semantic_chunk_text(text_content, max_chunk_size=800, overlap=50)

                    # Categorize content
                    category = categorize_content(url, text_content)

                    # Generate embeddings
                    embedder = EmbeddingModel()
                    embeddings = embedder.encode(chunks)

                    # Create points
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
                                "source": "cache"
                            }
                        ))

                    # Save to database
                    chroma_db.add_points(category, points)
                    total_points += len(points)
                    processed_files += 1

                    logger.info(f"Processed cache file {cache_file}: {len(points)} points")

            except Exception as e:
                logger.error(f"Error processing cache file {cache_file}: {e}")

        # Get updated statistics
        collections_info = chroma_db.get_collection_info()

        return JSONResponse(content={
            "message": f"Перестроено из кеша: {processed_files} файлов, {total_points} точек данных",
            "processed_files": processed_files,
            "saved_points": total_points,
            "collections": collections_info,
            "status": "completed"
        })

    except Exception as e:
        logger.error(f"Rebuild from cache error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
