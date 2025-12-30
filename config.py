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
CONFIDENCE_THRESHOLD = 0.4
MAX_TOKENS = 1000

# Collection configurations (обновлено на основе анализа sitemap.xml)
COLLECTION_CONFIGS = {
    "services": {
        "name": "Услуги ТПП",
        "description": "Услуги ТПП",
        "url_keywords": ["uslugi", "ekspertiza", "oczenka", "sertifikaty", "yuridicheskie-uslugi", "ocenka-imushhestva", "eksperty-i-otsenshhiki"],
        "text_keywords": ["услуги тпп", "экспертиза", "оценка", "сертификация", "юридические услуги", "консультации", "услуги"]
    },
    "committees": {
        "name": "Комитеты и комиссии ТПП УР",
        "url_keywords": ["komitetyi-tpp-ur"],
        "text_keywords": ["комитет", "комиссия", "экспертный совет", "рабочая группа"]
    },
    "membership": {
        "name": "Членство в ТПП",
        "url_keywords": ["членство", "vstupit", "utpp-chleny", "kak-stat-chlenom-utpp"],
        "text_keywords": ["членство", "вступление", "член тпп", "присоединение"]
    },
    "education": {
        "name": "Образовательная деятельность",
        "url_keywords": ["svedeniya-ob-obrazovatelnoj-organizaczii", "obrazovanie"],
        "text_keywords": ["образование", "обучение", "курсы", "семинары", "тренинги", "повышение квалификации"]
    },
    "events": {
        "name": "Мероприятия и форумы",
        "url_keywords": ["meropriyat", "seminar", "konferenc", "forum", "vef"],
        "text_keywords": ["мероприятие", "конференция", "форум", "выставка", "круглый стол", "семинар"]
    },
    "cooperation": {
        "name": "Деловое сотрудничество",
        "url_keywords": ["kommercheskoe-predlozhenie", "partner", "kooperacziya", "sotrudnichestvo"],
        "text_keywords": ["партнеры", "сотрудничество", "деловое партнерство", "кооперация", "альянс"]
    },
    "support": {
        "name": "Меры поддержки бизнеса",
        "url_keywords": ["podderzhka", "grant", "subsidii"],
        "text_keywords": ["субсидии", "гранты", "льготы", "поддержка бизнеса", "финансовая поддержка", "государственная поддержка"]
    },
    "legal": {
        "name": "Юридические услуги",
        "url_keywords": ["yurid-konsultatsii", "nalogovoe-pravo", "antimonopolnoe-pravo"],
        "text_keywords": ["юридические услуги", "правовое", "законодательство", "нормативные акты", "консультации"]
    },
    "news": {
        "name": "Новости и пресс-релизы",
        "url_keywords": ["news"],
        "text_keywords": ["новости", "пресс-релиз", "сообщение", "анонс"]
    },
    "about": {
        "name": "О ТПП УР",
        "url_keywords": ["o-soyuze-udmurtskaya-utpp", "ob-organizatsii", "respublika-udmurtiya"],
        "text_keywords": ["торгово-промышленная палата", "удмуртская республика", "о союзе", "история"]
    },
    "site": {
        "name": "Общий контент сайта",
        "url_keywords": [],
        "text_keywords": []
    }
}


# Categorization prompt template (обновлено для новых категорий)
CATEGORIZATION_PROMPT = """
Определи наиболее подходящую категорию для контента сайта ТПП УР.

{input_section}

Доступные категории:
{categories}

ВАЖНЫЕ УТОЧНЕНИЯ:
- Услуги ТПП (экспертиза, оценка, сертификация, консультации) → services
- Комитеты и комиссии ТПП УР → committees
- Членство в ТПП, вступление → membership
- Образовательная деятельность, курсы, семинары → education
- Мероприятия, конференции, форумы, выставки → events
- Деловое сотрудничество, партнеры, кооперация → cooperation
- Меры поддержки бизнеса, субсидии, гранты → support
- Юридические услуги, правовые консультации → legal
- Новости, пресс-релизы, анонсы → news
- Информация о ТПП УР, республике, истории → about
- Общий контент сайта → site

Проанализируй URL и содержание текста. Определи, к какой категории этот контент относится больше всего.

Если предоставлен список URL, верни результат в формате JSON:
{{
  "mappings": {{
    "url1": "category1",
    "url2": "category2",
    ...
  }}
}}

Если одиночный URL с текстом, верни только название категории (одно слово из списка: services, committees, membership, education, events, cooperation, support, legal, news, about, site).
"""

# RAG answer generation prompt template
RAG_ANSWER_PROMPT = """На основе предоставленного контекста ответьте на вопрос пользователя.

Контекст:
{context}

Вопрос: {query}

Инструкции:
- Отвечайте только на основе предоставленного контекста
- Если в контексте нет информации для ответа, скажите об этом
- Будьте конкретны и полезны
- Укажите источники информации, если применимо

Ответ:"""

# Low confidence response message
LOW_CONFIDENCE_MESSAGE = "На основе предоставленного контекста информация отсутствует или недостаточно релевантна."

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
