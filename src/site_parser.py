"""
Site parsing module for UdmTPP RAG system
Handles website crawling, content extraction and processing
"""

import os
import time
import logging
from typing import List, Set, Dict
import requests
from bs4 import BeautifulSoup
import uuid
import json

from config import (
    SITE_URL,
    SITEMAP_URL,
    CACHE_DIR,
    MAX_CHUNK_SIZE,
    CHUNK_OVERLAP,
    REQUEST_TIMEOUT,
    RATE_LIMIT_DELAY,
    CATEGORIZATION_PROMPT,
    DEEPSEEK_API_KEY,
    COLLECTION_CONFIGS
)
from utils import semantic_chunk_text
from embed import EmbeddingModel
from categorizer import categorize_content

logger = logging.getLogger(__name__)


class SiteParser:
    """Handles website parsing and content processing"""

    def __init__(self):
        self.site_url = SITE_URL
        self.sitemap_url = SITEMAP_URL
        self.embedder = EmbeddingModel()
        self.visited_urls: Set[str] = set()

        # Create cache directory
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def parse_sitemap(self) -> List[str]:
        """Parse sitemap and return list of URLs"""
        logger.info(f"Parsing sitemap: {self.sitemap_url}")
        try:
            response = requests.get(self.sitemap_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'xml')
            urls = []

            for url in soup.find_all('url'):
                loc = url.find('loc')
                if loc:
                    urls.append(loc.text.strip())

            logger.info(f"Found {len(urls)} URLs in sitemap")
            return urls
        except Exception as e:
            logger.error(f"Error parsing sitemap: {e}")
            return []

    def extract_text_from_url(self, url: str) -> str:
        """Extract text content from a URL"""
        if url in self.visited_urls:
            return ""

        try:
            logger.info(f"Extracting text from: {url}")
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()

            # Get text
            text = soup.get_text(separator=' ', strip=True)

            # Save text to local cache file
            self.save_text_to_cache(url, text)

            self.visited_urls.add(url)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {url}: {e}")
            return ""

    def save_text_to_cache(self, url: str, text: str):
        """Save extracted text to local cache file"""
        try:
            # Create filename from URL
            filename = url.replace('https://', '').replace('http://', '').replace('/', '_').replace(':', '_') + '.txt'
            filepath = os.path.join(self.cache_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"URL: {url}\n\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("Content:\n")
                f.write("="*50 + "\n")
                f.write(text)

            logger.info(f"Saved text to cache: {filepath}")

        except Exception as e:
            logger.error(f"Error saving text to cache for {url}: {e}")

    def process_url(self, url: str) -> List[dict]:
        """
        Process a single URL and return data points

        Args:
            url: URL to process

        Returns:
            List of data points for ChromaDB
        """
        text = self.extract_text_from_url(url)
        if not text:
            return []

        text_length = len(text)
        logger.info(f"Extracted {text_length} characters from {url}")

        # Chunk text
        chunks = semantic_chunk_text(text, max_chunk_size=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        logger.info(f"Split into {len(chunks)} chunks")

        # Categorize content
        category = categorize_content(url, text)
        logger.info(f"Categorized as: {category}")

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedder.encode(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")

        # Create points
        points = []
        for j, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())

            # Create Point object for ChromaDB
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
                    "chunk_index": j
                }
            ))

        logger.info(f"Created {len(points)} data points")
        return points

    def process_urls(self, urls: List[str], limit: int = 50) -> dict:
        """
        Process multiple URLs

        Args:
            urls: List of URLs to process
            limit: Maximum number of URLs to process

        Returns:
            Processing statistics
        """
        urls_to_process = urls[:limit]
        total_urls = len(urls_to_process)
        processed_count = 0
        saved_points_total = 0

        print(f"🚀 НАЧИНАЕМ ПАРСИНГ {total_urls} СТРАНИЦ САЙТА")
        print("="*60)

        all_points = []

        for i, url in enumerate(urls_to_process, 1):
            print(f"📄 [{i}/{total_urls}] Обработка: {url}")

            points = self.process_url(url)
            all_points.extend(points)

            saved_points_total += len(points)
            processed_count += 1

            print(f"   📊 Прогресс: {processed_count}/{total_urls} страниц, {saved_points_total} точек сохранено")
            print("-"*50)

            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)

        print("="*60)
        print(f"🎉 ПАРСИНГ ЗАВЕРШЕН!")
        print(f"📊 РЕЗУЛЬТАТЫ:")
        print(f"   • Обработано страниц: {processed_count}/{total_urls}")
        print(f"   • Создано точек данных: {saved_points_total}")
        print(f"   • Среднее точек на страницу: {saved_points_total/processed_count:.1f}" if processed_count > 0 else "")
        print("="*60)

        return {
            "processed_urls": processed_count,
            "total_points": saved_points_total,
            "points": all_points
        }

    def ai_plan_categorization(self, urls: List[str]) -> Dict[str, str]:
        """
        Use AI to plan categorization of all URLs from sitemap

        Args:
            urls: List of URLs to categorize

        Returns:
            Dict mapping URL to category
        """
        if not urls:
            return {}

        logger.info(f"Planning categorization for {len(urls)} URLs using AI")

        try:
            # Prepare categories description
            categories_text = "\n".join([
                f"- {cat}: {config['name']}"
                for cat, config in COLLECTION_CONFIGS.items()
            ])

            # Prepare URLs list (limit to reasonable size for API)
            urls_text = "\n".join(urls[:100])  # Limit to first 100 URLs for API constraints

            prompt = CATEGORIZATION_PROMPT.format(
                input_section=f"Список URL:\n{urls_text}",
                categories=categories_text
            )

            # Call DeepSeek API
            if not DEEPSEEK_API_KEY:
                logger.warning("DEEPSEEK_API_KEY not found, using rule-based categorization")
                return self._rule_based_categorization(urls)

            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.1
                },
                timeout=120  # Longer timeout for complex task
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()

                # Try to parse JSON response
                try:
                    parsed = json.loads(content)
                    mappings = parsed.get("mappings", {})

                    # Validate categories
                    valid_categories = set(COLLECTION_CONFIGS.keys())
                    validated_mappings = {}

                    for url, category in mappings.items():
                        if category in valid_categories:
                            validated_mappings[url] = category
                        else:
                            # Fallback to rule-based for invalid categories
                            validated_mappings[url] = self._categorize_url_by_rules(url)

                    logger.info(f"AI planned categorization for {len(validated_mappings)} URLs")
                    return validated_mappings

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse AI response as JSON: {e}")
                    return self._rule_based_categorization(urls)
            else:
                logger.warning(f"AI categorization API error: {response.status_code}")
                return self._rule_based_categorization(urls)

        except Exception as e:
            logger.error(f"Error in AI categorization planning: {e}")
            return self._rule_based_categorization(urls)

    def _rule_based_categorization(self, urls: List[str]) -> Dict[str, str]:
        """
        Fallback rule-based categorization for URLs

        Args:
            urls: List of URLs to categorize

        Returns:
            Dict mapping URL to category
        """
        mappings = {}
        for url in urls:
            mappings[url] = self._categorize_url_by_rules(url)
        return mappings

    def _categorize_url_by_rules(self, url: str) -> str:
        """
        Categorize URL using simple rules

        Args:
            url: URL to categorize

        Returns:
            Category name
        """
        url_lower = url.lower()

        # Apply rules from COLLECTION_CONFIGS
        for category, config in COLLECTION_CONFIGS.items():
            # Check URL keywords
            if any(keyword in url_lower for keyword in config.get("url_keywords", [])):
                return category

        # Default category
        return "site"

    def process_urls_with_ai_plan(self, urls: List[str], limit: int = 50) -> dict:
        """
        Process URLs using AI-planned categorization

        Args:
            urls: List of URLs to process
            limit: Maximum number of URLs to process

        Returns:
            Processing statistics
        """
        urls_to_process = urls[:limit]
        total_urls = len(urls_to_process)

        print(f"🤖 AI-ПЛАНИРОВАНИЕ КАТЕГОРИЗАЦИИ")
        print(f"   Анализируем {total_urls} URL...")
        print("-"*50)

        # Step 1: AI planning
        category_mappings = self.ai_plan_categorization(urls_to_process)

        print(f"✅ AI определил категории для {len(category_mappings)} страниц")
        print()

        # Step 2: Process URLs based on AI plan
        processed_count = 0
        saved_points_total = 0
        all_points = []

        print(f"🚀 НАЧИНАЕМ ПАРСИНГ ПО ПЛАНУ AI")
        print("="*60)

        for i, url in enumerate(urls_to_process, 1):
            planned_category = category_mappings.get(url, "site")
            print(f"📄 [{i}/{total_urls}] {url}")
            print(f"   🎯 Запланированная категория: {planned_category}")

            # Process URL
            points = self.process_url_with_category(url, planned_category)
            all_points.extend(points)

            saved_points_total += len(points)
            processed_count += 1

            print(f"   📊 Создано точек: {len(points)}")
            print("-"*50)

            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)

        print("="*60)
        print(f"🎉 ПАРСИНГ ПО AI-ПЛАНУ ЗАВЕРШЕН!")
        print(f"📊 РЕЗУЛЬТАТЫ:")
        print(f"   • Обработано страниц: {processed_count}/{total_urls}")
        print(f"   • Создано точек данных: {saved_points_total}")
        print(f"   • Среднее точек на страницу: {saved_points_total/processed_count:.1f}" if processed_count > 0 else "")
        print("="*60)

        return {
            "processed_urls": processed_count,
            "total_points": saved_points_total,
            "points": all_points,
            "ai_mappings": category_mappings
        }

    def process_url_with_category(self, url: str, planned_category: str) -> List[dict]:
        """
        Process a single URL with pre-planned category

        Args:
            url: URL to process
            planned_category: Category determined by AI

        Returns:
            List of data points for ChromaDB
        """
        text = self.extract_text_from_url(url)
        if not text:
            return []

        text_length = len(text)
        logger.info(f"Extracted {text_length} characters from {url}")

        # Chunk text
        chunks = semantic_chunk_text(text, max_chunk_size=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        logger.info(f"Split into {len(chunks)} chunks")

        # Use AI-planned category instead of rule-based categorization
        category = planned_category
        logger.info(f"Using AI-planned category: {category}")

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedder.encode(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")

        # Create points
        points = []
        for j, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())

            # Create Point object for ChromaDB
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
                    "chunk_index": j,
                    "categorization_method": "ai_planned"
                }
            ))

        logger.info(f"Created {len(points)} data points")
        return points
