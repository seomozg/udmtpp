"""
Main FastAPI application for UdmTPP RAG system
Simplified and modular architecture
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging
import sys

# Add src and root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add both src and root directories to Python path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import setup_logging
from config import API_HOST, API_PORT, RELOAD
from web_routes import web_router
from api_routes import api_router

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ТПП УР AI-Chat Assistant",
    version="1.0.0",
    description="RAG система для Торгово-промышленной палаты Удмуртской Республики"
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include routers
app.include_router(web_router)  # Web pages
app.include_router(api_router, prefix="/api")  # API endpoints


if __name__ == "__main__":
    uvicorn.run("src.app:app", host=API_HOST, port=API_PORT, reload=RELOAD)
