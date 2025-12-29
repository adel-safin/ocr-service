#!/usr/bin/env python3
"""
Тестирование обученной модели классификатора
"""
import sys
from pathlib import Path

# Добавление корня приложения в путь
app_root = Path(__file__).parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from core.processor import DocumentPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_classifier():
    """Тестирование классификатора"""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ОБУЧЕННОГО КЛАССИФИКАТОРА")
    print("=" * 80)
    
    # Создание пайплайна с ML
    pipeline = DocumentPipeline(use_ml=True, use_active_learning=False)
    
    if not pipeline.document_classifier:
        print("\n❌ Классификатор не загружен!")
        print("   Запустите: python scripts/train_classifier.py")
        return
    
    print("\n✅ Классификатор загружен")
    
    if hasattr(pipeline, 'class_mapping'):
        classes = pipeline.class_mapping.get('class_to_idx', {})
        print(f"   Классов: {len(classes)}")
        print(f"   Типы документов:")
        for i, doc_type in enumerate(list(classes.keys())[:10], 1):
            print(f"      {i}. {doc_type}")
    
    # Тестирование на реальных документах
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ НА ДОКУМЕНТАХ")
    print("=" * 80)
    
    test_files = [
        "../Датасет/Наборы однотипных документов со сканами/Акт АОСР/1 АОСР.pdf",
        "../Датасет/Наборы однотипных документов со сканами/Акт внешнего осмотра/1 АКТ.pdf",
        "../Датасет/Наборы однотипных документов со сканами/Акт входного контроля/1 АВК.jpg",
    ]
    
    for test_file in test_files:
        if not Path(test_file).exists():
            continue
        
        print(f"\n📄 Файл: {Path(test_file).name}")
        
        try:
            device = 'cuda' if hasattr(__import__('torch'), 'cuda') and __import__('torch').cuda.is_available() else 'cpu'
            doc_type_idx, confidence = pipeline.document_classifier.predict(test_file, device=device)
            
            if hasattr(pipeline, 'class_mapping'):
                predicted_type = pipeline.class_mapping['idx_to_class'].get(doc_type_idx, 'unknown')
                print(f"   Предсказанный тип: {predicted_type}")
                print(f"   Уверенность: {confidence:.2%}")
                
                # Проверка правильности
                expected_type = Path(test_file).parent.name
                is_correct = predicted_type == expected_type
                status = "✅" if is_correct else "❌"
                print(f"   {status} Ожидался: {expected_type}")
        except Exception as e:
            print(f"   ❌ Ошибка: {str(e)}")
    
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)


if __name__ == "__main__":
    test_classifier()
