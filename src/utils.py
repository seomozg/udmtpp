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

def get_env_var(key: str, default: str = "") -> str:
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

def adaptive_chunk_text(text: str) -> List[str]:
    """Adaptive chunking based on content analysis"""
    import re

    # Analyze content structure
    sentences = re.split(r'[.!?]+\s+', text)
    paragraphs = text.split('\n\n')

    # Calculate content metrics
    avg_sentence_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
    num_paragraphs = len([p for p in paragraphs if p.strip()])
    total_length = len(text)

    # Adaptive chunk sizing based on content
    if total_length < 500:
        # Short content - single chunk
        chunk_size = total_length
        overlap = 0
    elif avg_sentence_length > 100:
        # Complex sentences - smaller chunks
        chunk_size = 800
        overlap = 150
    elif num_paragraphs > 10:
        # Well-structured content - medium chunks
        chunk_size = 1200
        overlap = 200
    else:
        # Default chunking
        chunk_size = 1000
        overlap = 150

    # Semantic boundary-aware chunking
    return semantic_chunk_text(text, chunk_size, overlap)

def semantic_chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunk text respecting semantic boundaries"""
    import re

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # Try to find good semantic boundary within chunk_size window
        end = start + chunk_size

        if end >= len(text):
            # Last chunk
            chunks.append(text[start:])
            break

        # Look for paragraph boundaries first
        para_match = re.search(r'\n\s*\n', text[start:end+200])
        if para_match and para_match.end() + start < end + 100:
            end = start + para_match.end()
        else:
            # Look for sentence boundaries
            sentence_match = re.search(r'[.!?]\s+', text[start:end+50])
            if sentence_match and sentence_match.end() + start < end + 50:
                end = start + sentence_match.end()

        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

        # Calculate next start with overlap
        overlap_start = max(0, end - overlap)
        next_para = re.search(r'\n\s*\n', text[overlap_start:])
        if next_para:
            start = overlap_start + next_para.end()
        else:
            start = overlap_start

        # Safety check to prevent infinite loop
        if start >= len(text):
            break
        elif len(chunks) > 100:  # Emergency break
            chunks.append(text[start:])
            break

    return chunks

def calculate_confidence(scores: List[float]) -> float:
    """Calculate average confidence from similarity scores"""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
