"""
Web routes for UdmTPP RAG system
Handles all HTML page endpoints
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

from vector_db import ChromaDB

# Initialize components
chroma_db = ChromaDB()

# Setup templates
templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=templates_dir)

# Create router
web_router = APIRouter()


@web_router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main page with navigation"""
    return templates.TemplateResponse(request, "index.html")


@web_router.get("/collections", response_class=HTMLResponse)
async def collections_page(request: Request):
    """Collections management page"""
    collections_info = chroma_db.get_collection_info()
    return templates.TemplateResponse(request, "collections.html", {
        "collections": collections_info
    })


@web_router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """AI Chat interface"""
    collections_info = chroma_db.get_collection_info()
    return templates.TemplateResponse(request, "chat.html", {
        "collections": collections_info
    })


@web_router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Upload documents page"""
    return templates.TemplateResponse(request, "upload.html")
