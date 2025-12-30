from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import logging
import sys
import time
import json
from typing import Optional

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils import setup_logging
from vector_db import ChromaDB
from rag import RAGSystem
from parse_site import SiteParser

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="ТПП УР AI-Chat Assistant", version="1.0.0")

# Create static directory
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)

# Mount static files
import os
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=templates_dir)

# Initialize components
chroma_db = ChromaDB()
rag_system = RAGSystem()  # Global singleton instance

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main page with navigation"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/collections", response_class=HTMLResponse)
async def collections_page(request: Request):
    """Collections management page"""
    collections_info = chroma_db.get_collection_info()
    return templates.TemplateResponse("collections.html", {
        "request": request,
        "collections": collections_info
    })

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """AI Chat interface"""
    collections_info = chroma_db.get_collection_info()
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "collections": collections_info
    })

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Upload documents page"""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/api/chat")
async def chat_api(
    query: str = Form(...),
    collection: Optional[str] = Form(None),
    temperature: float = Form(0.1),
    max_tokens: int = Form(1000),
    n_results: int = Form(5),
    confidence_threshold: float = Form(0.7),
    collections_filter: Optional[str] = Form(None)
):
    """API endpoint for chat"""
    try:
        # Parse collections filter if provided
        collections_list = None
        if collections_filter:
            collections_list = collections_filter.split(',') if collections_filter else None

        # Create custom RAG instance with user settings (clone for performance)
        custom_rag = rag_system.clone_with_settings(confidence_threshold=confidence_threshold)

        result = custom_rag.ask(query, collection, n_results=n_results, collections_filter=collections_list)

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/chat/stream")
async def chat_stream_api(
    query: str,
    collection: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1000,
    n_results: int = 5,
    confidence_threshold: float = 0.7,
    collections_filter: Optional[str] = None
):
    """Streaming API endpoint for chat"""
    # Parse collections filter if provided
    collections_list = None
    if collections_filter:
        collections_list = collections_filter.split(',') if collections_filter else None

    # Create custom RAG instance with user settings (clone for performance)
    custom_rag = rag_system.clone_with_settings(confidence_threshold=confidence_threshold)

    async def generate_stream():
        try:
            async for chunk in custom_rag.ask_stream(query, collection, n_results=n_results, collections_filter=collections_list):
                if chunk["type"] == "metadata":
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                elif chunk["type"] == "content":
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                elif chunk["type"] == "end":
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Chat streaming API error: {e}")
            error_chunk = {
                "type": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

@app.get("/api/collections")
async def get_collections_api():
    """Get collections info API"""
    try:
        collections_info = chroma_db.get_collection_info()
        return JSONResponse(content={"collections": collections_info})
    except Exception as e:
        logger.error(f"Collections API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/collections/{collection_id}/documents")
async def get_collection_documents_api(collection_id: str, limit: int = 50):
    """Get documents from a specific collection"""
    try:
        # Get documents from ChromaDB
        documents = chroma_db.get_collection_documents(collection_id, limit)

        return JSONResponse(content={
            "collection_id": collection_id,
            "documents": documents,
            "total_count": len(documents)
        })
    except Exception as e:
        logger.error(f"Collection documents API error: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@app.post("/api/upload")
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
            # PDF file processing
            from PyPDF2 import PdfReader
            from io import BytesIO

            pdf_stream = BytesIO(content)
            pdf_reader = PdfReader(pdf_stream)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

        elif filename.endswith(('.doc', '.docx')):
            # DOC/DOCX file processing
            from docx import Document
            from io import BytesIO

            doc_stream = BytesIO(content)
            doc = Document(doc_stream)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"

        else:
            # Plain text file
            text = content.decode('utf-8')

        # Chunk and embed with semantic chunking
        from utils import semantic_chunk_text
        from embed import EmbeddingModel
        import uuid

        embedder = EmbeddingModel()
        chunks = semantic_chunk_text(text, max_chunk_size=800, overlap=50)
        embeddings = embedder.encode(chunks)

        # Create ChromaDB compatible points
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            # Create simple point object for ChromaDB
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

        # Add to ChromaDB
        chroma_db.add_points(category, points)

        return JSONResponse(content={
            "message": f"Файл {file.filename} успешно загружен и обработан",
            "chunks_count": len(chunks)
        })

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при загрузке файла")

@app.post("/api/parse-site")
async def parse_site_api():
    """Trigger site parsing"""
    try:
        logger.info("Starting site parsing via API")
        # Use shared ChromaDB instance from web app
        # SiteParser already imported at top
        parser = SiteParser(vector_client=chroma_db)  # Pass shared instance
        urls = parser.parse_sitemap()
        logger.info(f"Found {len(urls)} URLs in sitemap")

        # Process all URLs from sitemap
        urls_to_process = urls  # Process all URLs
        logger.info(f"Processing {len(urls_to_process)} URLs from sitemap")
        parser.process_urls(urls_to_process)

        # Get real statistics from ChromaDB
        collections_info = chroma_db.get_collection_info()
        total_saved = sum(info['points_count'] for info in collections_info.values())

        return JSONResponse(content={
            "message": f"Обработано {len(urls_to_process)} из {len(urls)} URL с сайта",
            "saved_points": total_saved,
            "collections": collections_info,
            "status": "completed"
        })
    except Exception as e:
        logger.error(f"Parse site error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rebuild-from-cache")
async def rebuild_from_cache_api():
    """Rebuild database from site_cache using AI categorization"""
    try:
        logger.info("Starting rebuild from cache with AI categorization")

        # Clear and recreate all collections
        logger.info("Clearing and recreating all collections...")
        for collection_name in chroma_db.collection_configs.keys():
            try:
                # Delete existing collection if it exists
                try:
                    chroma_db.client.delete_collection(collection_name)
                    logger.info(f"Deleted collection: {collection_name}")
                except Exception:
                    pass  # Collection might not exist, that's ok

                # Create new collection
                collection = chroma_db.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": chroma_db.collection_configs[collection_name]}
                )
                chroma_db.collections[collection_name] = collection
                logger.info(f"Created collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Could not recreate {collection_name}: {e}")

        # Longer delay to ensure collections are created
        time.sleep(2)

        # Process files from site_cache
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "site_cache")
        if not os.path.exists(cache_dir):
            raise HTTPException(status_code=404, detail="site_cache directory not found")

        # Load categories cache to skip already processed files
        categories_cache = load_categories_cache()
        processed_urls = set(categories_cache.keys())

        all_files = [f for f in os.listdir(cache_dir) if f.endswith('.txt')]
        files = []

        # Filter out already processed files
        for filename in all_files:
            filepath = os.path.join(cache_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.split('\n', 2)
                url = lines[0].replace('URL: ', '') if lines[0].startswith('URL: ') else filename
                cache_key = url if url != filename else filename

                if cache_key not in processed_urls:
                    files.append(filename)
            except Exception as e:
                logger.warning(f"Could not check file {filename}: {e}")
                continue

        logger.info(f"Processing {len(files)} remaining files out of {len(all_files)} total (skipping {len(processed_urls)} already processed)")

        processed_count = 0
        total_points = 0

        for i, filename in enumerate(files, 1):
            filepath = os.path.join(cache_dir, filename)
            print(f"📄 [{i}/{len(files)}] Processing: {filename}")

            try:
                # Read file content
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract URL and text
                lines = content.split('\n', 2)
                url = lines[0].replace('URL: ', '') if lines[0].startswith('URL: ') else filename
                text = lines[2] if len(lines) > 2 else content

                # AI categorization using DeepSeek with caching
                cache_key = url if url != filename else filename
                category = categorize_with_ai(text, chroma_db.collection_configs, cache_key)
                print(f"   🏷️  AI categorized as: {category}")

                # Chunk and embed with semantic chunking
                from utils import semantic_chunk_text
                from embed import EmbeddingModel
                import uuid

                embedder = EmbeddingModel()
                chunks = semantic_chunk_text(text, max_chunk_size=800, overlap=50)
                embeddings = embedder.encode(chunks)

                # Create points
                points = []
                for j, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    point_id = str(uuid.uuid4())
                    # Create simple point object for ChromaDB
                    class Point:
                        def __init__(self, id, vector, payload):
                            self.id = id
                            self.vector = vector
                            self.payload = payload

                    points.append(Point(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "filename": filename,
                            "url": url,
                            "text": chunk,
                            "category": category,
                            "chunk_index": j,
                            "source": "cache_rebuild"
                        }
                    ))

                # Save to database
                if points:
                    chroma_db.add_points(category, points)
                    print(f"   💾 Saved {len(points)} points to {category}")

                processed_count += 1
                total_points += len(points)

                print(f"   📊 Progress: {processed_count}/{len(files)} files, {total_points} points")
                print("-"*50)

            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                print(f"   ❌ Error: {e}")
                continue

        # Final statistics
        collections_info = chroma_db.get_collection_info()

        print("="*60)
        print("🎉 ПЕРЕСТРОЙКА ЗАВЕРШЕНА!")
        print(f"📊 Обработано файлов: {processed_count}/{len(files)}")
        print(f"💾 Сохранено точек данных: {total_points}")
        print("="*60)

        return JSONResponse(content={
            "message": f"Перестроено из кеша: {processed_count} файлов",
            "saved_points": total_points,
            "collections": collections_info,
            "status": "completed"
        })

    except Exception as e:
        logger.error(f"Rebuild from cache error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def load_categories_cache():
    """Load categories cache from file"""
    try:
        with open('categories_cache.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_categories_cache(cache):
    """Save categories cache to file"""
    try:
        with open('categories_cache.json', 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save categories cache: {e}")

def categorize_with_ai(text: str, collection_configs: dict, cache_key: str) -> str:
    """Use DeepSeek AI to categorize text content with caching"""
    try:
        # Load existing cache
        cache = load_categories_cache()

        # Check if we already have this category cached
        if cache_key in cache:
            cached_category = cache[cache_key]
            if cached_category in collection_configs:
                logger.info(f"Using cached category for {cache_key}: {cached_category}")
                return cached_category

        from utils import get_env_var

        # Prepare prompt
        categories_desc = "\n".join([f"- {name}: {desc}" for name, desc in collection_configs.items()])

        prompt = f"""Проанализируй следующий текст и определи, к какой категории он относится.

Доступные категории:
{categories_desc}

Текст для анализа:
{text[:2000]}...

Верни ТОЛЬКО название категории (одно слово), без объяснений."""

        # Call DeepSeek API
        import requests
        deepseek_api_key = get_env_var("DEEPSEEK_API_KEY")
        deepseek_base_url = get_env_var("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        logger.info(f"Making AI request for categorization of {cache_key}")
        response = requests.post(
            f"{deepseek_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {deepseek_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 50
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            category = result['choices'][0]['message']['content'].strip()

            # Validate category
            if category in collection_configs:
                # Cache the result
                cache[cache_key] = category
                save_categories_cache(cache)
                logger.info(f"Cached new category for {cache_key}: {category}")
                return category

        # Fallback to default
        logger.warning(f"Invalid category from AI for {cache_key}, using default")
        cache[cache_key] = "site"
        save_categories_cache(cache)
        return "site"

    except Exception as e:
        logger.warning(f"AI categorization failed for {cache_key}: {e}, using default category")
        # Even on error, try to cache the default
        try:
            cache = load_categories_cache()
            cache[cache_key] = "site"
            save_categories_cache(cache)
        except:
            pass
        return "site"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
