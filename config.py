"""
Configuration file for UdmTPP RAG system
All data and settings are centralized here
"""

import os
from typing import Dict, List

# API Keys and URLs
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
SITE_URL = "https://udmtpp.ru"
SITEMAP_URL = "https://udmtpp.ru/sitemap.xml"

# Model settings
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
LLM_MODEL = "deepseek-chat"

# Chunking settings
MAX_CHUNK_SIZE = 800
CHUNK_OVERLAP = 50

# Search settings
DEFAULT_N_RESULTS = 5
CONFIDENCE_THRESHOLD = 0.25
MAX_TOKENS = 1000

# Collection configurations (обновлено на основе анализа sitemap.xml)
COLLECTION_CONFIGS = {
    "719": {
        "name": "Консультации по 719-ПП / Акт СТ",
        "url_keywords": ["719", "akt-st", "promyshlennoy-produktsii"],
        "text_keywords": ["реестр", "промышленной продукции", "акт ст", "постановление", "сертификат", "продукции"]
    },
    "support": {
        "name": "Меры поддержки бизнеса",
        "url_keywords": ["podderzhka", "grant", "subsidii", "lgoty"],
        "text_keywords": ["субсидии", "гранты", "льготы", "поддержка бизнеса", "финансовая поддержка", "государственная поддержка"]
    },
    "services": {
        "name": "Услуги ТПП",
        "url_keywords": ["uslugi", "ekspertiza", "oczenka", "sertifikaty", "yuridicheskie-uslugi", "ocenka-imushhestva", "kursyi", "investitsionnyj-konsalting"],
        "text_keywords": ["услуги тпп", "экспертиза", "оценка", "сертификация", "юридические услуги", "консультации", "услуги"]
    },
    "membership": {
        "name": "Членство в ТПП",
        "url_keywords": ["членство", "vstupit", "utpp-chleny", "kak-stat-chlenom-utpp"],
        "text_keywords": ["членство", "вступление", "член тпп", "присоединение"]
    },
    "events": {
        "name": "Мероприятия, обучение",
        "url_keywords": ["meropriyat", "seminar", "konferenc", "forum", "plan-meropriyatij", "videokonferenciyu", "eksportnyix-seminarax"],
        "text_keywords": ["мероприятие", "конференция", "форум", "семинар", "выставка", "круглый стол"]
    },
    "cooperation": {
        "name": "Поиск партнёров / коопераций",
        "url_keywords": ["kommercheskoe-predlozhenie", "partner", "kooperacziya", "sotrudnichestvo"],
        "text_keywords": ["партнеры", "сотрудничество", "деловое партнерство", "кооперация", "альянс"]
    },
    "site": {
        "name": "Общий контент сайта",
        "url_keywords": [],
        "text_keywords": []
    }
}


# Categorization prompt template (universal for single content or sitemap)
CATEGORIZATION_PROMPT = """
Определи наиболее подходящую категорию для контента сайта ТПП.

{input_section}

Доступные категории:
{categories}

Проанализируй URL и содержание текста. Определи, к какой категории этот контент относится больше всего.

Если предоставлен список URL, верни результат в формате JSON:
{{
  "mappings": {{
    "url1": "category1",
    "url2": "category2",
    ...
  }}
}}

Если одиночный URL с текстом, верни только название категории (одно слово из списка: 719, support, services, membership, events, cooperation, site).
"""

# RAG answer generation prompt template
RAG_ANSWER_PROMPT = """Контекст:
{context}

Вопрос: {query}

ВАЖНЫЕ ИНСТРУКЦИИ:
- Отвечайте ТОЛЬКО на основе предоставленного контекста
- Начинайте ответ НЕМЕДЛЕННО с полезной информации
- НЕ ИСПОЛЬЗУЙТЕ вводные фразы типа "На основе контекста", "Согласно информации", "Исходя из предоставленных данных"
- Будьте конкретны и полезны
- Если информации недостаточно, скажите кратко без лишних слов

Ответ (начните сразу с сути):"""

# Low confidence response message
LOW_CONFIDENCE_MESSAGE = "Информация отсутствует или недостаточно релевантна для данного запроса."

# Query expansion templates (extended)
QUERY_EXPANSION_TEMPLATES = {
    "greetings": ["привет", "здравствуй", "добрый день", "доброе утро", "добрый вечер", "здравствуйте", "доброго времени"],
    "events": ["мероприятия", "семинары", "конференции", "вебинары", "обучение", "курсы", "тренинги", "форумы", "выставки", "круглые столы"],
    "services": ["услуги", "экспертиза", "оценка", "сертификация", "консультации", "аудит", "проверка", "осмотр", "анализ", "диагностика"],
    "support": ["поддержка", "субсидии", "гранты", "льготы", "финансирование", "помощь", "стимулирование", "компенсации", "возмещение"],
    "membership": ["членство", "вступление", "участие", "члены", "партнеры", "ассоциация", "союз", "присоединение"],
    "cooperation": ["партнеры", "сотрудничество", "поставщики", "подрядчики", "контрагенты", "партнерство", "альянс", "взаимодействие"],
    "documents": ["документы", "сертификаты", "лицензии", "разрешения", "акты", "справки", "удостоверения"],
    "consulting": ["консультации", "советы", "рекомендации", "помощь", "сопровождение", "содействие"],
    "legal": ["юридические", "правовые", "законодательство", "нормативные", "регуляторные", "договорные"],
    "business": ["бизнес", "предпринимательство", "коммерция", "экономика", "производство", "торговля", "строительство", "строительные"]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

# File paths
CACHE_DIR = "site_cache"
CHROMA_DB_DIR = "chroma_db"

# Collection names (for backward compatibility)
COLLECTION_NAMES = list(COLLECTION_CONFIGS.keys())

# Rate limiting
REQUEST_TIMEOUT = 30
RATE_LIMIT_DELAY = 1

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8001
RELOAD = True
