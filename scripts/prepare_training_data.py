#!/usr/bin/env python3
"""
Подготовка данных для обучения ML моделей из датасета
"""
import sys
from pathlib import Path

# Добавление корня приложения в путь
app_root = Path(__file__).parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from utils.dataset_loader import DatasetLoader
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_classifier_data():
    """Подготовка данных для обучения классификатора"""
    print("\n" + "=" * 80)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ КЛАССИФИКАТОРА ДОКУМЕНТОВ")
    print("=" * 80)
    
    dataset_root = Path("../Датасет")
    loader = DatasetLoader(str(dataset_root))
    
    # Получение всех пар документов
    all_pairs = []
    doc_types = loader.get_all_document_types()
    
    print(f"\n📋 Найдено типов документов: {len(doc_types)}")
    
    for doc_type in doc_types:
        pairs = loader.find_document_pairs(doc_type)
        all_pairs.extend(pairs)
        print(f"   {doc_type}: {len(pairs)} пар")
    
    print(f"\n✅ Всего пар для обучения: {len(all_pairs)}")
    
    # Сохранение информации о данных
    training_info = {
        'total_pairs': len(all_pairs),
        'document_types': doc_types,
        'pairs_by_type': {doc_type: len(loader.find_document_pairs(doc_type)) 
                          for doc_type in doc_types}
    }
    
    output_path = Path("data/training_data/classifier_info.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Информация сохранена: {output_path}")
    
    return all_pairs, doc_types


def prepare_spell_correction_data():
    """Подготовка данных для обучения исправления опечаток"""
    print("\n" + "=" * 80)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ ИСПРАВЛЕНИЯ ОПЕЧАТОК")
    print("=" * 80)
    
    dataset_root = Path("../Датасет")
    loader = DatasetLoader(str(dataset_root))
    
    # Получение пар (изображение + эталон)
    pairs = loader.find_document_pairs()
    
    print(f"\n📁 Найдено пар документов: {len(pairs)}")
    
    # Обработка документов для создания пар (OCR текст -> эталонный текст)
    from core.ocr_engine import OCREngine
    
    ocr_engine = OCREngine()
    training_pairs = []
    
    print("\n⏳ Обработка документов...")
    
    for i, pair in enumerate(pairs[:20], 1):  # Ограничиваем для теста
        print(f"   [{i}/{min(20, len(pairs))}] {Path(pair['image_path']).name}")
        
        try:
            # OCR распознавание
            ocr_result = ocr_engine.process_file(pair['image_path'])
            ocr_text = ocr_result['text']
            
            # Загрузка эталонного текста
            reference_text = loader.load_reference_text(pair['reference_path'])
            
            if ocr_text.strip() and reference_text.strip():
                training_pairs.append({
                    'ocr_text': ocr_text[:500],  # Ограничиваем длину
                    'reference_text': reference_text[:500],
                    'document_type': pair['document_type']
                })
        except Exception as e:
            logger.warning(f"Ошибка при обработке {pair['image_path']}: {str(e)}")
    
    print(f"\n✅ Создано пар для обучения: {len(training_pairs)}")
    
    # Сохранение данных
    output_path = Path("data/training_data/spell_correction_pairs.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Данные сохранены: {output_path}")
    
    return training_pairs


def main():
    """Основная функция"""
    print("=" * 80)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ ML МОДЕЛЕЙ")
    print("=" * 80)
    
    # Подготовка данных для классификатора
    pairs, doc_types = prepare_classifier_data()
    
    # Подготовка данных для исправления опечаток
    spell_pairs = prepare_spell_correction_data()
    
    print("\n" + "=" * 80)
    print("ПОДГОТОВКА ЗАВЕРШЕНА")
    print("=" * 80)
    print("\n📊 Итоги:")
    print(f"   - Пар для классификатора: {len(pairs)}")
    print(f"   - Типов документов: {len(doc_types)}")
    print(f"   - Пар для исправления опечаток: {len(spell_pairs)}")
    print("\n💡 Следующие шаги:")
    print("   1. Запустите: python scripts/train_classifier.py")
    print("   2. Для исправления опечаток нужна дообучка T5 (более сложная задача)")


if __name__ == "__main__":
    main()
