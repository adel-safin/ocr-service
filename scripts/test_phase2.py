#!/usr/bin/env python3
"""
Тестирование Фазы 2: Машинное обучение
Демонстрация ML компонентов
"""
import sys
from pathlib import Path

# Добавление корня приложения в путь
app_root = Path(__file__).parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from core.processor import DocumentPipeline
from utils.dataset_loader import DatasetLoader
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_ml_components():
    """Тестирование ML компонентов"""
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ФАЗЫ 2: МАШИННОЕ ОБУЧЕНИЕ")
    print("=" * 80)
    
    # Тест 1: Работа с датасетом
    print("\n🔬 ТЕСТ 1: Загрузка датасета")
    print("-" * 80)
    
    dataset_root = Path("../Датасет")
    loader = DatasetLoader(str(dataset_root))
    
    # Получение всех типов документов
    doc_types = loader.get_all_document_types()
    print(f"📋 Найдено типов документов: {len(doc_types)}")
    for i, doc_type in enumerate(doc_types[:10], 1):
        print(f"   {i}. {doc_type}")
    
    # Поиск пар документов
    print("\n🔍 Поиск пар документов (изображение + эталон)...")
    pairs = loader.find_document_pairs("Акт АОСР")
    print(f"✅ Найдено пар: {len(pairs)}")
    
    if pairs:
        print("\nПримеры пар:")
        for i, pair in enumerate(pairs[:3], 1):
            print(f"   {i}. {pair['base_name']}")
            print(f"      Изображение: {Path(pair['image_path']).name}")
            print(f"      Эталон: {Path(pair['reference_path']).name}")
    
    # Тест 2: Обработка с ML
    print("\n\n🔬 ТЕСТ 2: Обработка документа с ML компонентами")
    print("-" * 80)
    
    if pairs:
        test_file = pairs[0]['image_path']
        print(f"📄 Обработка: {Path(test_file).name}")
        
        try:
            # Создание пайплайна с ML (Фаза 2)
            pipeline = DocumentPipeline(use_ml=True, use_active_learning=False)
            
            print("\n⏳ Обработка с ML компонентами...")
            result = pipeline.process(
                file_path=test_file,
                template="Акт АОСР",
                required_fields=["ogrn", "inn", "date"]
            )
            
            # Вывод результатов
            print("\n" + "-" * 80)
            print("РЕЗУЛЬТАТЫ С ML")
            print("-" * 80)
            
            print(f"\n✅ Document ID: {result['document_id']}")
            print(f"📊 Качество: {result['quality_report']['overall_quality']:.2%}")
            print(f"🔍 Уверенность OCR: {result['quality_report'].get('ocr_confidence', 0):.2%}")
            
            # ML компоненты
            if 'ml_quality_score' in result['quality_report']:
                print(f"🤖 ML оценка качества: {result['quality_report']['ml_quality_score']:.2%}")
            
            # Исправления
            corrections = result.get('corrections_applied', [])
            ml_corrections = [c for c in corrections if c.get('method') == 'ml_transformer']
            rule_corrections = [c for c in corrections if c.get('method') != 'ml_transformer']
            
            print(f"\n✏️  Всего исправлений: {len(corrections)}")
            print(f"   - По правилам: {len(rule_corrections)}")
            print(f"   - ML исправления: {len(ml_corrections)}")
            
            if ml_corrections:
                print("\n🤖 ML исправления:")
                for i, correction in enumerate(ml_corrections[:3], 1):
                    print(f"   {i}. '{correction['from'][:30]}...' -> '{correction['to'][:30]}...'")
            
            # Критические поля
            print("\n📋 Критические поля:")
            for field_name, field_data in result['extracted_data']['critical_fields'].items():
                status = "✅" if field_data['valid'] else "❌"
                print(f"   {status} {field_name.upper()}: {field_data['value'] or '(не найдено)'} "
                      f"(валидно: {field_data['valid']})")
            
        except Exception as e:
            print(f"❌ Ошибка: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Тест 3: Сравнение Фаза 1 vs Фаза 2
    print("\n\n🔬 ТЕСТ 3: Сравнение Фаза 1 vs Фаза 2")
    print("-" * 80)
    
    if pairs:
        test_file = pairs[0]['image_path']
        
        print(f"\n📄 Тестовый файл: {Path(test_file).name}")
        
        # Фаза 1
        print("\n📌 Фаза 1 (без ML):")
        pipeline_phase1 = DocumentPipeline(use_ml=False, use_active_learning=False)
        result_phase1 = pipeline_phase1.process(test_file, "Акт АОСР")
        
        print(f"   Качество: {result_phase1['quality_report']['overall_quality']:.2%}")
        print(f"   Исправлений: {len(result_phase1.get('corrections_applied', []))}")
        
        # Фаза 2
        print("\n🤖 Фаза 2 (с ML):")
        try:
            pipeline_phase2 = DocumentPipeline(use_ml=True, use_active_learning=False)
            result_phase2 = pipeline_phase2.process(test_file, "Акт АОСР")
            
            print(f"   Качество: {result_phase2['quality_report']['overall_quality']:.2%}")
            print(f"   Исправлений: {len(result_phase2.get('corrections_applied', []))}")
            
            # Сравнение
            quality_diff = result_phase2['quality_report']['overall_quality'] - result_phase1['quality_report']['overall_quality']
            corrections_diff = len(result_phase2.get('corrections_applied', [])) - len(result_phase1.get('corrections_applied', []))
            
            print(f"\n📊 Разница:")
            print(f"   Качество: {quality_diff:+.2%}")
            print(f"   Исправлений: {corrections_diff:+d}")
            
        except Exception as e:
            print(f"   ⚠️  ML компоненты недоступны: {str(e)}")
            print("   Используется только Фаза 1")


def main():
    """Основная функция"""
    test_ml_components()
    
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)
    print("\n💡 Следующий шаг: Запустите test_phase3.py для тестирования активного обучения")


if __name__ == "__main__":
    main()
