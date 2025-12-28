import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import logging
import sys
import os
from typing import List, Dict, Set
import uuid
import logging

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from utils import get_env_var, chunk_text
from embed import EmbeddingModel

# Import ChromaDB for compatibility
from vector_db import ChromaDB

# Simple PointStruct for compatibility
class PointStruct:
    def __init__(self, id, vector, payload):
        self.id = id
        self.vector = vector
        self.payload = payload

logger = logging.getLogger(__name__)

class SiteParser:
    def __init__(self, vector_client=None):
        self.site_url = get_env_var("SITE_URL")
        self.sitemap_url = get_env_var("SITEMAP_URL")
        self.embedder = EmbeddingModel()
        self.visited_urls: Set[str] = set()

        # Create cache directory for storing page content
        self.cache_dir = os.path.join(os.getcwd(), "site_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Use provided vector client or create new one
        if vector_client:
            self.client = vector_client
        else:
            # Fallback for standalone usage (not recommended)
            from vector_db import ChromaDB
            self.client = ChromaDB()

        # Create collections
        self.collection_configs = {
            "719": "Консультации по 719-ПП / Акт СТ",
            "support": "Меры поддержки бизнеса",
            "services": "Услуги ТПП",
            "membership": "Членство в ТПП",
            "events": "Мероприятия, обучение",
            "cooperation": "Поиск партнёров / коопераций",
            "site": "Общий контент сайта"
        }

    def parse_sitemap(self) -> List[str]:
        """Parse sitemap and return list of URLs"""
        logger.info(f"Parsing sitemap: {self.sitemap_url}")
        try:
            response = requests.get(self.sitemap_url)
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
            response = requests.get(url, timeout=30)
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

    def categorize_content(self, url: str, text: str) -> str:
        """Categorize content based on URL and text analysis"""
        url_lower = url.lower()
        text_lower = text.lower()

        categories = {
            "719": ["719", "акт ст", "постановление"],
            "support": ["поддержка", "субсидии", "гранты", "помощь"],
            "services": ["услуги", "консультации", "сертификаты"],
            "membership": ["членство", "вступить", "участие"],
            "events": ["мероприятия", "обучение", "семинары", "конференции"],
            "cooperation": ["партнёры", "кооперация", "сотрудничество"],
        }

        for category, keywords in categories.items():
            if any(keyword in url_lower or keyword in text_lower for keyword in keywords):
                return category

        return "site"  # Default category

    def process_urls(self, urls: List[str]):
        """Process URLs and add to ChromaDB"""
        total_urls = len(urls)
        processed_count = 0
        saved_points_total = 0

        print(f"🚀 НАЧИНАЕМ ПАРСИНГ {total_urls} СТРАНИЦ САЙТА")
        print("="*60)

        for i, url in enumerate(urls, 1):
            print(f"📄 [{i}/{total_urls}] Обработка: {url}")

            text = self.extract_text_from_url(url)
            if not text:
                print(f"   ❌ Пропущено: не удалось извлечь текст")
                continue

            text_length = len(text)
            print(f"   📝 Извлечено {text_length} символов текста")

            # Chunk text
            chunks = chunk_text(text)
            print(f"   ✂️  Разбито на {len(chunks)} чанков")

            # Categorize
            category = self.categorize_content(url, text)
            print(f"   🏷️  Категория: {category}")

            # Generate embeddings
            print(f"   🧠 Генерация эмбеддингов...")
            embeddings = self.embedder.encode(chunks)
            print(f"   ✅ Эмбеддинги созданы: {len(embeddings)} векторов")

            # Create points
            points = []
            for j, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid4())
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "url": url,
                        "text": chunk,
                        "category": category,
                        "chunk_index": j
                    }
                ))

            print(f"   📦 Создано {len(points)} точек данных")

            # Add to ChromaDB collection immediately
            if points:
                print(f"   💾 Сохранение в коллекцию '{category}'...")
                self.client.add_points(category, points)
                print(f"   ✅ Успешно сохранено {len(points)} точек в {category}")

            saved_points_total += len(points)
            processed_count += 1

            print(f"   📊 Прогресс: {processed_count}/{total_urls} страниц, {saved_points_total} точек сохранено")
            print("-"*50)

            # Rate limiting
            time.sleep(1)

        print("="*60)
        print(f"🎉 ПАРСИНГ ЗАВЕРШЕН!")
        print(f"📊 РЕЗУЛЬТАТЫ:")
        print(f"   • Обработано страниц: {processed_count}/{total_urls}")
        print(f"   • Сохранено точек данных: {saved_points_total}")
        print(f"   • Среднее точек на страницу: {saved_points_total/processed_count:.1f}" if processed_count > 0 else "")
        print("="*60)



def main():
    """Main entry point"""
    from utils import setup_logging
    setup_logging()

    print("🚀 Запуск парсера сайта udmtpp.ru")
    print("📊 Находим URL из sitemap...")

    parser = SiteParser()
    urls = parser.parse_sitemap()

    print(f"📋 Найдено {len(urls)} URL в sitemap")
    print("🔄 Начинаем обработку страниц...")

    # Process first 5 URLs for demo
    urls_to_process = urls[:5]
    parser.process_urls(urls_to_process)

    print("✅ Парсинг завершен!")
    print(f"📁 Текстовый кеш сохранен в папку: {parser.cache_dir}")
    print(f"🗄️ Данные сохранены в ChromaDB: {len(urls_to_process)} страниц")

if __name__ == "__main__":
    main()
