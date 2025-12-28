import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_env_var(key: str, default: str = None) -> str:
    """Get environment variable with fallback"""
    return os.getenv(key, default)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into chunks with overlap"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break

    return chunks

def calculate_confidence(scores: List[float]) -> float:
    """Calculate average confidence from similarity scores"""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
