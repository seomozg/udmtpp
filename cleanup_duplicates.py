#!/usr/bin/env python3
"""
Script to clean duplicate documents from ChromaDB
"""

import sys
import os
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))

from src.vector_db import ChromaDB

def cleanup_duplicates():
    """Remove duplicate documents from ChromaDB"""

    # Initialize ChromaDB
    db = ChromaDB()

    # Get collection info
    collections_info = db.get_collection_info()

    print("🧹 ОЧИСТКА ДУБЛИКАТОВ В БАЗЕ ДАННЫХ")
    print("=" * 60)
    print("⚠️  ВНИМАНИЕ: Эта операция необратима!")
    print("Рекомендуется сделать резервную копию перед запуском")
    print("=" * 60)

    total_removed = 0
    total_processed = 0

    for collection_name in collections_info.keys():
        print(f"\n📂 Очистка коллекции: {collection_name}")
        print("-" * 40)

        # Get all documents
        documents = db.get_collection_documents(collection_name, limit=10000)
        total_processed += len(documents)

        if not documents:
            print("  Коллекция пуста")
            continue

        # Group by text content
        text_groups = defaultdict(list)

        for doc in documents:
            text = doc.get('text', '').strip()
            if text:  # Only check non-empty texts
                text_groups[text].append(doc)

        # Find and remove duplicates
        duplicates_to_remove = []

        for text, docs in text_groups.items():
            if len(docs) > 1:
                # Keep the first document, remove the rest
                for doc in docs[1:]:
                    duplicates_to_remove.append(doc['id'])

        if duplicates_to_remove:
            print(f"  🗑️  Найдено {len(duplicates_to_remove)} дублированных документов")
            print(f"  📊 Уникальных текстов: {len(text_groups)}")

            # Remove duplicates from ChromaDB
            # Note: ChromaDB doesn't have a direct delete by ID method in this wrapper
            # We would need to implement batch delete or recreate collection
            print(f"  ❌ Удаление дублей не реализовано в текущей версии")
            print(f"  💡 Для очистки нужно пересоздать коллекцию без дублей")

        else:
            print(f"  ✅ Дубликатов не найдено")
            print(f"  📊 Уникальных текстов: {len(text_groups)}")

        total_removed += len(duplicates_to_remove)

    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ОЧИСТКИ:")
    print(f"  • Обработано документов: {total_processed}")
    print(f"  • Найдено дублей: {total_removed}")
    if total_processed > 0:
        print(f"  • Процент дублей: {total_removed/total_processed*100:.2f}%")

    print("\n💡 РЕКОМЕНДАЦИИ:")
    print("  • Для полной очистки дублей нужно:")
    print("    1. Экспортировать уникальные документы")
    print("    2. Очистить коллекции")
    print("    3. Импортировать только уникальные документы")
    print("  • Или улучшить логику парсинга и chunking")

if __name__ == "__main__":
    cleanup_duplicates()
