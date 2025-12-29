#!/usr/bin/env python3
"""
Тестирование Фазы 1: Базовый OCR + правила
Обработка документов из датасета
"""
import sys
from pathlib import Path

# Добавление корня приложения в путь
app_root = Path(__file__).parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from core.processor import DocumentPipeline
import json
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_single_document(file_path: str, document_type: str = None):
    """Тестирование обработки одного документа"""
    print("\n" + "=" * 80)
    print(f"ОБРАБОТКА ДОКУМЕНТА: {Path(file_path).name}")
    print("=" * 80)
    
    if not Path(file_path).exists():
        print(f"❌ Файл не найден: {file_path}")
        return None
    
    try:
        # Создание пайплайна (только Фаза 1)
        pipeline = DocumentPipeline(use_ml=False, use_active_learning=False)
        
        print(f"📄 Файл: {file_path}")
        print(f"📋 Тип документа: {document_type or 'не указан'}")
        print("\n⏳ Обработка...")
        
        # Обработка документа
        result = pipeline.process(
            file_path=file_path,
            template=document_type,
            required_fields=["ogrn", "inn", "date"]
        )
        
        # Вывод результатов
        print("\n" + "-" * 80)
        print("РЕЗУЛЬТАТЫ ОБРАБОТКИ")
        print("-" * 80)
        
        print(f"\n✅ Document ID: {result['document_id']}")
        print(f"📅 Дата обработки: {result['processing_date']}")
        print(f"📊 Качество: {result['quality_report']['overall_quality']:.2%}")
        print(f"🔍 Уверенность OCR: {result['quality_report'].get('ocr_confidence', 0):.2%}")
        print(f"⚠️  Требует проверки: {'Да' if result['needs_review'] else 'Нет'}")
        
        # Критические поля
        print("\n" + "-" * 80)
        print("КРИТИЧЕСКИЕ ПОЛЯ")
        print("-" * 80)
        critical_fields = result['extracted_data']['critical_fields']
        
        if critical_fields:
            for field_name, field_data in critical_fields.items():
                status = "✅" if field_data['valid'] else "❌"
                print(f"\n{status} {field_name.upper()}:")
                print(f"   Значение: {field_data['value'] or '(не найдено)'}")
                print(f"   Уверенность: {field_data['confidence']:.2%}")
                print(f"   Валидно: {'Да' if field_data['valid'] else 'Нет'}")
                if field_data.get('suggested_correction'):
                    print(f"   💡 Предложенное исправление: {field_data['suggested_correction']}")
        else:
            print("Критические поля не найдены")
        
        # Примененные исправления
        print("\n" + "-" * 80)
        print("ПРИМЕНЕННЫЕ ИСПРАВЛЕНИЯ")
        print("-" * 80)
        corrections = result.get('corrections_applied', [])
        if corrections:
            for i, correction in enumerate(corrections, 1):
                print(f"{i}. '{correction['from']}' -> '{correction['to']}' "
                      f"(уверенность: {correction['confidence']:.2%}, метод: {correction.get('method', 'unknown')})")
        else:
            print("Исправления не применялись")
        
        # Проблемы качества
        print("\n" + "-" * 80)
        print("ПРОБЛЕМЫ КАЧЕСТВА")
        print("-" * 80)
        issues = result['quality_report'].get('issues', [])
        if issues:
            for issue in issues:
                print(f"⚠️  {issue.get('type', 'unknown')}: {issue.get('message', '')}")
        else:
            print("✅ Проблем не обнаружено")
        
        # Полный текст (первые 300 символов)
        print("\n" + "-" * 80)
        print("ПОЛНЫЙ ТЕКСТ (первые 300 символов)")
        print("-" * 80)
        full_text = result['extracted_data']['full_text']
        print(full_text[:300] + ("..." if len(full_text) > 300 else ""))
        
        return result
        
    except Exception as e:
        print(f"\n❌ Ошибка при обработке: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_documents(documents_dir: str, document_type: str = None, limit: int = 3):
    """Тестирование обработки нескольких документов"""
    print("\n" + "=" * 80)
    print(f"ПАКЕТНАЯ ОБРАБОТКА ДОКУМЕНТОВ")
    print("=" * 80)
    
    documents_path = Path(documents_dir)
    if not documents_path.exists():
        print(f"❌ Директория не найдена: {documents_dir}")
        return
    
    # Поиск PDF и изображений
    image_files = []
    for ext in ['*.pdf', '*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(documents_path.glob(ext)))
    
    if not image_files:
        print(f"❌ Не найдено изображений/PDF в {documents_dir}")
        return
    
    # Ограничение количества
    image_files = sorted(image_files)[:limit]
    
    print(f"📁 Найдено файлов: {len(image_files)}")
    print(f"📋 Тип документа: {document_type or 'не указан'}")
    
    results = []
    for i, file_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Обработка: {file_path.name}")
        result = test_single_document(str(file_path), document_type)
        if result:
            results.append(result)
    
    # Сводная статистика
    print("\n" + "=" * 80)
    print("СВОДНАЯ СТАТИСТИКА")
    print("=" * 80)
    
    if results:
        avg_quality = sum(r['quality_report']['overall_quality'] for r in results) / len(results)
        total_corrections = sum(len(r.get('corrections_applied', [])) for r in results)
        needs_review_count = sum(1 for r in results if r.get('needs_review', False))
        
        print(f"\n📊 Обработано документов: {len(results)}")
        print(f"📈 Среднее качество: {avg_quality:.2%}")
        print(f"✏️  Всего применено исправлений: {total_corrections}")
        print(f"⚠️  Требуют проверки: {needs_review_count}")
        
        # Статистика по полям
        all_fields = {}
        for result in results:
            for field_name, field_data in result['extracted_data']['critical_fields'].items():
                if field_name not in all_fields:
                    all_fields[field_name] = {'found': 0, 'valid': 0, 'total': 0}
                all_fields[field_name]['total'] += 1
                if field_data['value']:
                    all_fields[field_name]['found'] += 1
                if field_data['valid']:
                    all_fields[field_name]['valid'] += 1
        
        if all_fields:
            print("\n📋 Статистика по полям:")
            for field_name, stats in all_fields.items():
                found_pct = (stats['found'] / stats['total']) * 100 if stats['total'] > 0 else 0
                valid_pct = (stats['valid'] / stats['total']) * 100 if stats['total'] > 0 else 0
                print(f"   {field_name.upper()}: найдено {stats['found']}/{stats['total']} ({found_pct:.1f}%), "
                      f"валидно {stats['valid']}/{stats['total']} ({valid_pct:.1f}%)")
    
    return results


def main():
    """Основная функция"""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ФАЗЫ 1: БАЗОВЫЙ OCR + ПРАВИЛА")
    print("=" * 80)
    
    # Путь к датасету
    dataset_root = Path("../Датасет/Наборы однотипных документов со сканами")
    
    # Тест 1: Один документ
    print("\n🔬 ТЕСТ 1: Обработка одного документа")
    test_file = dataset_root / "Акт АОСР" / "1 АОСР.pdf"
    if test_file.exists():
        test_single_document(str(test_file), "Акт АОСР")
    else:
        print(f"❌ Тестовый файл не найден: {test_file}")
    
    # Тест 2: Пакетная обработка
    print("\n\n🔬 ТЕСТ 2: Пакетная обработка документов")
    aosr_dir = dataset_root / "Акт АОСР"
    if aosr_dir.exists():
        test_multiple_documents(str(aosr_dir), "Акт АОСР", limit=3)
    else:
        print(f"❌ Директория не найдена: {aosr_dir}")
    
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)


if __name__ == "__main__":
    main()
