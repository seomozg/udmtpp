"""
Content categorization module for UdmTPP RAG system
Handles both rule-based and AI-powered categorization
"""

import logging
from typing import Optional
import requests
import sys
import os

# Add root directory to path for config import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import (
    CATEGORIZATION_PROMPT,
    COLLECTION_CONFIGS,
    DEEPSEEK_API_KEY
)

logger = logging.getLogger(__name__)


def categorize_content(url: str, text: str) -> str:
    """
    Categorize content using hybrid approach: rules + AI

    Args:
        url: Page URL
        text: Page text content

    Returns:
        Category name
    """
    url_lower = url.lower()
    text_lower = text.lower()

    # Priority 1: Explicit rules for clear cases
    for category, config in COLLECTION_CONFIGS.items():
        # Check URL keywords
        if any(keyword in url_lower for keyword in config.get("url_keywords", [])):
            return category

        # Check text keywords
        if any(keyword in text_lower for keyword in config.get("text_keywords", [])):
            return category

    # Priority 2: AI categorization for ambiguous content
    return _ai_categorize(url, text)


def _ai_categorize(url: str, text: str) -> str:
    """
    Use AI to categorize ambiguous content

    Args:
        url: Page URL
        text: Page text content

    Returns:
        Category name
    """
    try:
        # Prepare prompt
        categories_text = "\n".join([
            f"- {cat}: {config['name']}"
            for cat, config in COLLECTION_CONFIGS.items()
        ])

        prompt = CATEGORIZATION_PROMPT.format(
            input_section=f"URL: {url}\n\nТекст контента (первые 1000 символов):\n{text[:1000]}",
            categories=categories_text
        )

        # Call DeepSeek API
        if not DEEPSEEK_API_KEY:
            logger.warning("DEEPSEEK_API_KEY not found, using site as default")
            return "site"

        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": 0.1
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            category = result["choices"][0]["message"]["content"].strip().lower()

            # Validate response
            valid_categories = set(COLLECTION_CONFIGS.keys())
            if category in valid_categories:
                logger.info(f"AI categorized '{url}' as '{category}'")
                return category
            else:
                logger.warning(f"AI returned invalid category '{category}', using site")
                return "site"
        else:
            logger.warning(f"DeepSeek API error: {response.status_code}, using site")
            return "site"

    except Exception as e:
        logger.error(f"Error in AI categorization: {e}, using site")
        return "site"


def get_collection_config(collection_name: str) -> Optional[dict]:
    """
    Get collection configuration

    Args:
        collection_name: Name of the collection

    Returns:
        Collection configuration dict or None
    """
    return COLLECTION_CONFIGS.get(collection_name)


def get_all_collections() -> dict:
    """
    Get all collection configurations

    Returns:
        Dict of all collections
    """
    return COLLECTION_CONFIGS.copy()
