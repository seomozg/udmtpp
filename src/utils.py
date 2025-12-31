import os
import logging
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config import LOGGING_CONFIG, MAX_CHUNK_SIZE, CHUNK_OVERLAP

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG["level"]),
        format=LOGGING_CONFIG["format"]
    )

def get_env_var(key: str, default: str = "") -> str:
    """Get environment variable with fallback"""
    return os.getenv(key, default)

def chunk_text(text: str, chunk_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
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

def adaptive_chunk_text(text: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> List[str]:
    """Adaptive chunking based on text length and structure"""
    text_length = len(text)

    # For very short texts, return as single chunk
    if text_length <= max_chunk_size // 4:
        return [text]

    # For medium texts, use semantic chunking
    if text_length <= max_chunk_size * 2:
        return semantic_chunk_text(text, max_chunk_size=max_chunk_size)

    # For long texts, use smaller chunks to ensure better retrieval
    small_chunk_size = max_chunk_size // 2
    return semantic_chunk_text(text, max_chunk_size=small_chunk_size)

def remove_repeated_content(text: str) -> str:
    """Remove common repeated content like headers, footers, contact info"""
    lines = text.split('\n')
    cleaned_lines = []
    seen_lines = set()

    # Common patterns to remove (case insensitive)
    remove_patterns = [
        r'.*\b(пн—чт|пт|выходной|сб|вс|телефон|email|адрес)\b.*',  # Contact info
        r'.*\b(ул\.|улица|г\.|город|индекс|почта)\b.*',  # Address info
        r'.*\b(политика конфиденциальности|обратная связь|задать вопрос)\b.*',  # Form headers
        r'.*\b(поля отмеченные.*обязательны|согласие на обработку|персональных данных)\b.*',  # Legal forms
        r'.*\b(отправить|нажмимая.*кнопку|я даю.*согласие)\b.*',  # Form buttons
        r'^[\s]*$',  # Empty lines
        r'^[-=\s]*$',  # Separator lines
    ]

    for line in lines:
        line_lower = line.lower().strip()

        # Skip if line matches removal patterns
        should_remove = False
        for pattern in remove_patterns:
            if re.search(pattern, line_lower, re.IGNORECASE):
                should_remove = True
                break

        # Skip if line was seen before (simple deduplication)
        if line_lower in seen_lines:
            should_remove = True

        if not should_remove and line.strip():
            cleaned_lines.append(line)
            seen_lines.add(line_lower)

    return '\n'.join(cleaned_lines)

def semantic_chunk_text(text: str, max_chunk_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Enhanced semantic chunking with content filtering and deduplication"""
    # First, remove repeated content
    text = remove_repeated_content(text)

    # Ensure minimum chunk size after cleaning
    if len(text) <= max_chunk_size:
        return [text] if text.strip() else []

    # Try NLTK first, fallback to regex
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        # Split by sentences
        sentences = nltk.sent_tokenize(text)
    except ImportError:
        # Fallback: split by punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text] if text.strip() else []

    chunks = []
    current_chunk = ""
    current_length = 0
    chunk_texts_seen = set()  # For deduplication within chunks

    for sentence in sentences:
        sentence_length = len(sentence)

        # Skip very short sentences that might be noise
        if sentence_length < 10:
            continue

        # If single sentence is too long, split it by words
        if sentence_length > max_chunk_size:
            # Split long sentence by words
            words = sentence.split()
            temp_chunk = ""
            temp_length = 0

            for word in words:
                word_with_space = word + " "
                if temp_length + len(word_with_space) > max_chunk_size and temp_chunk:
                    # Save current temp chunk
                    chunk_candidate = temp_chunk.strip()
                    if chunk_candidate and chunk_candidate not in chunk_texts_seen:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = chunk_candidate
                        current_length = temp_length
                        chunk_texts_seen.add(chunk_candidate)
                    temp_chunk = word_with_space
                    temp_length = len(word_with_space)
                else:
                    temp_chunk += word_with_space
                    temp_length += len(word_with_space)

            # Add remaining temp chunk
            if temp_chunk:
                chunk_candidate = temp_chunk.strip()
                if chunk_candidate and chunk_candidate not in chunk_texts_seen:
                    if current_length + temp_length > max_chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = chunk_candidate
                        current_length = temp_length
                        chunk_texts_seen.add(chunk_candidate)
                    else:
                        current_chunk += " " + chunk_candidate
                        current_length += temp_length + 1
                        chunk_texts_seen.add(chunk_candidate)

        # Normal sentence processing
        elif current_length + sentence_length > max_chunk_size and current_chunk:
            # Save current chunk if it's unique
            chunk_candidate = current_chunk.strip()
            if chunk_candidate and chunk_candidate not in chunk_texts_seen:
                chunks.append(chunk_candidate)
                chunk_texts_seen.add(chunk_candidate)

            # Start new chunk with overlap if possible
            if len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:].strip()
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
            else:
                current_chunk = sentence
                current_length = sentence_length
        else:
            # Add to current chunk
            if current_chunk:
                current_chunk += " " + sentence
                current_length += sentence_length + 1  # +1 for space
            else:
                current_chunk = sentence
                current_length = sentence_length

    # Add final chunk if it's unique
    if current_chunk:
        chunk_candidate = current_chunk.strip()
        if chunk_candidate and chunk_candidate not in chunk_texts_seen:
            chunks.append(chunk_candidate)

    # Final deduplication - remove duplicate chunks across all chunks
    unique_chunks = []
    seen_chunks = set()
    for chunk in chunks:
        if chunk not in seen_chunks and len(chunk.strip()) > 20:  # Minimum chunk length
            unique_chunks.append(chunk)
            seen_chunks.add(chunk)

    # Ensure we have at least one chunk if original text was meaningful
    return unique_chunks if unique_chunks else ([text[:max_chunk_size]] if text.strip() else [])



def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization"""
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'dtype') and obj.dtype in (np.float32, np.float64):
            return float(obj)
        elif hasattr(obj, 'dtype') and obj.dtype in (np.int32, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    except ImportError:
        # If numpy is not available, return as-is
        return obj

def calculate_confidence(scores: List[float], top_k: int = 3) -> float:
    """Calculate confidence from top similarity scores"""
    if not scores:
        return 0.0

    # Sort scores in descending order and take top_k
    sorted_scores = sorted(scores, reverse=True)
    top_scores = sorted_scores[:top_k]

    # Calculate weighted average: higher weight for top results
    if len(top_scores) == 1:
        return convert_numpy_types(top_scores[0])
    elif len(top_scores) == 2:
        return convert_numpy_types(0.7 * top_scores[0] + 0.3 * top_scores[1])
    else:
        # For 3+ scores: 50% top, 30% second, 20% third
        return convert_numpy_types(0.5 * top_scores[0] + 0.3 * top_scores[1] + 0.2 * top_scores[2])
