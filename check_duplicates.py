#!/usr/bin/env python3
"""
Script to check for duplicate documents in ChromaDB
"""

import sys
import os
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(__file__))

from src.vector_db import ChromaDB

def check_duplicates():
    """Check for duplicate documents in ChromaDB"""

    # Initialize ChromaDB
    db = ChromaDB()

    # Get collection info
    collections_info = db.get_collection_info()

    print("🔍 ПРОВЕРКА НА ДУБЛИКАТЫ В БАЗЕ ДАННЫХ")
    print("=" * 60)

    total_duplicates = 0
    total_documents = 0

    for collection_name in collections_info.keys():
        print(f"\n📂 Анализ коллекции: {collection_name}")
        print("-" * 40)

        # Get all documents
        documents = db.get_collection_documents(collection_name, limit=10000)
        total_documents += len(documents)

        if not documents:
            print("  Коллекция пуста")
            continue

        # Group by text content
        text_groups = defaultdict(list)

        for doc in documents:
            text = doc.get('text', '').strip()
            if text:  # Only check non-empty texts
                text_groups[text].append(doc)

        # Find duplicates
        duplicates_found = 0
        duplicate_texts = []

        for text, docs in text_groups.items():
            if len(docs) > 1:
                duplicates_found += len(docs)
                duplicate_texts.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'count': len(docs),
                    'docs': docs
                })

        if duplicate_texts:
            print(f"  ❌ Найдено {duplicates_found} дублированных документов")
            print(f"  📊 Уникальных текстов: {len(text_groups)}")
            print(f"  📈 Коэффициент дублирования: {duplicates_found/len(documents)*100:.1f}%")

            # Show top duplicates
            duplicate_texts.sort(key=lambda x: x['count'], reverse=True)
            for i, dup in enumerate(duplicate_texts[:3], 1):  # Show top 3
                print(f"    {i}. \"{dup['text']}\" - {dup['count']} копий")
        else:
            print(f"  ✅ Дубликатов не найдено")
            print(f"  📊 Уникальных текстов: {len(text_groups)}")

        total_duplicates += duplicates_found

    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"  • Всего документов: {total_documents}")
    print(f"  • Дублированных документов: {total_duplicates}")
    if total_documents > 0:
        print(f"  • Процент дубликатов: {total_duplicates/total_documents*100:.2f}%")

    if total_duplicates > 0:
        print("\n⚠️  РЕКОМЕНДАЦИИ:")
        print("  • Рассмотрите удаление дублированных документов")
        print("  • Проверьте логику парсинга для предотвращения дублей")
        print("  • Возможно, нужно улучшить chunking или дедупликацию")
    else:
        print("\n✅ База данных не содержит полных дублей текстов!")

if __name__ == "__main__":
    check_duplicates()
