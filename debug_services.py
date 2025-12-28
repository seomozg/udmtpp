#!/usr/bin/env python3
"""Debug script for ChromaDB services collection"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from vector_db import ChromaDB
import numpy as np

def main():
    db = ChromaDB()

    print("=== ChromaDB Services Collection Debug ===")

    # Попробуем поиск в services для диагностики
    try:
        test_vector = np.random.rand(768).tolist()
        results = db.search('services', test_vector, limit=1)
        print(f"✅ Поиск в services работает: {len(results)} результатов")
    except Exception as e:
        print(f"❌ Ошибка поиска в services: {str(e)}")

    # Проверим состояние коллекции
    try:
        info = db.get_collection_info()
        print("\n📊 Состояние коллекций:")
        for name, data in info.items():
            points_count = data.get('points_count', 0)
            print(f"  {name}: {points_count} документов")
    except Exception as e:
        print(f"❌ Ошибка получения информации: {str(e)}")

    # Попробуем получить count напрямую
    try:
        if 'services' in db.collections:
            count = db.collections['services'].count()
            print(f"\n🔢 Прямой count для services: {count}")
        else:
            print("\n❌ Коллекция services не найдена в db.collections")
    except Exception as e:
        print(f"❌ Ошибка прямого count: {str(e)}")

if __name__ == "__main__":
    main()
