#!/usr/bin/env python3
"""Пример клиента для тестирования API"""
import requests
import json
import sys
from pathlib import Path

API_BASE_URL = "http://localhost:8000/api/v1"


def test_process_document(file_path: str, template: str = None):
    """Тест обработки одного документа"""
    print(f"\n📄 Обработка документа: {file_path}")
    
    url = f"{API_BASE_URL}/process"
    
    with open(file_path, 'rb') as f:
        files = {'file': (Path(file_path).name, f, 'application/pdf')}
        data = {}
        if template:
            data['template'] = template
        
        response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Успешно обработано")
        print(f"   Document ID: {result.get('document_id')}")
        print(f"   Качество: {result.get('quality_report', {}).get('overall_quality', 0):.2f}")
        print(f"   Применено исправлений: {len(result.get('corrections_applied', []))}")
        
        # Показать критические поля
        critical_fields = result.get('extracted_data', {}).get('critical_fields', {})
        if critical_fields:
            print("\n   Критические поля:")
            for field, data in critical_fields.items():
                status = "✅" if data.get('valid') else "❌"
                print(f"   {status} {field}: {data.get('value')}")
        
        return result
    else:
        print(f"❌ Ошибка: {response.status_code}")
        print(response.text)
        return None


def test_health():
    """Тест проверки здоровья сервиса"""
    print("\n🏥 Проверка здоровья сервиса...")
    response = requests.get(f"{API_BASE_URL}/health")
    
    if response.status_code == 200:
        print("✅ Сервис работает")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    else:
        print(f"❌ Ошибка: {response.status_code}")


def test_corrections_db():
    """Тест получения базы исправлений"""
    print("\n📚 Получение базы исправлений...")
    response = requests.get(f"{API_BASE_URL}/corrections_db")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Найдено исправлений: {result.get('total_count', 0)}")
        corrections = result.get('corrections', {})
        if corrections:
            print("\n   Примеры исправлений:")
            for original, corrected in list(corrections.items())[:5]:
                print(f"   '{original}' -> '{corrected}'")
    else:
        print(f"❌ Ошибка: {response.status_code}")


def test_confirm_correction():
    """Тест подтверждения исправления"""
    print("\n✏️  Подтверждение исправления...")
    url = f"{API_BASE_URL}/confirm_correction"
    
    data = {
        "original": "ТестОшибка",
        "corrected": "ТестИсправление",
        "add_to_db": True
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Исправление добавлено: {result.get('message')}")
    else:
        print(f"❌ Ошибка: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    print("🚀 Тестирование OCR API\n")
    
    # Проверка здоровья
    test_health()
    
    # Получение базы исправлений
    test_corrections_db()
    
    # Тест обработки документа (если указан путь)
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        template = sys.argv[2] if len(sys.argv) > 2 else None
        test_process_document(file_path, template)
    else:
        print("\n💡 Для тестирования обработки документа укажите путь:")
        print("   python test_client.py <путь_к_файлу> [шаблон]")
    
    # Тест подтверждения исправления
    # test_confirm_correction()
