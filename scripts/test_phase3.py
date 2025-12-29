#!/usr/bin/env python3
"""
Тестирование Фазы 3: Активное обучение
Демонстрация системы обратной связи и автоматического обучения
"""
import sys
from pathlib import Path

# Добавление корня приложения в путь
app_root = Path(__file__).parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from core.processor import DocumentPipeline
from services.active_learning import ActiveLearningSystem
from utils.dataset_loader import DatasetLoader
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_active_learning():
    """Тестирование активного обучения"""
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ФАЗЫ 3: АКТИВНОЕ ОБУЧЕНИЕ")
    print("=" * 80)
    
    # Инициализация системы активного обучения
    active_learning = ActiveLearningSystem()
    
    # Тест 1: Обработка документов с активным обучением
    print("\n🔬 ТЕСТ 1: Обработка документов с активным обучением")
    print("-" * 80)
    
    dataset_root = Path("../Датасет")
    loader = DatasetLoader(str(dataset_root))
    pairs = loader.find_document_pairs("Акт АОСР")
    
    if not pairs:
        print("❌ Не найдено пар документов для тестирования")
        return
    
    # Обработка нескольких документов
    print(f"\n📁 Обработка {min(3, len(pairs))} документов...")
    
    pipeline = DocumentPipeline(use_ml=True, use_active_learning=True)
    
    processed_docs = []
    for i, pair in enumerate(pairs[:3], 1):
        print(f"\n[{i}/{min(3, len(pairs))}] Обработка: {Path(pair['image_path']).name}")
        
        try:
            result = pipeline.process(
                file_path=pair['image_path'],
                template="Акт АОСР"
            )
            processed_docs.append(result)
            
            print(f"   ✅ Обработано (ID: {result['document_id'][:20]}...)")
            print(f"   📊 Качество: {result['quality_report']['overall_quality']:.2%}")
            print(f"   ✏️  Исправлений: {len(result.get('corrections_applied', []))}")
            
        except Exception as e:
            print(f"   ❌ Ошибка: {str(e)}")
    
    # Тест 2: Отправка feedback
    print("\n\n🔬 ТЕСТ 2: Отправка обратной связи")
    print("-" * 80)
    
    if processed_docs:
        # Симуляция feedback от пользователя
        print("\n📝 Симуляция feedback от пользователя...")
        
        for doc in processed_docs[:2]:
            # Feedback по исправлению
            corrections = doc.get('corrections_applied', [])
            if corrections:
                correction = corrections[0]
                feedback_data = {
                    "correction": {
                        "original": correction.get('from', ''),
                        "corrected": correction.get('to', ''),
                        "document_id": doc['document_id'],
                        "confidence": 1.0
                    }
                }
                
                result = active_learning.process_feedback(feedback_data)
                print(f"   ✅ Feedback по исправлению отправлен для {doc['document_id'][:20]}...")
                if result.get('corrections_added'):
                    print(f"      Добавлено исправлений: {len(result['corrections_added'])}")
            
            # Feedback по качеству
            quality_rating = doc['quality_report']['overall_quality']
            feedback_data = {
                "quality": {
                    "document_id": doc['document_id'],
                    "rating": quality_rating,
                    "issues": [] if quality_rating > 0.8 else ["Низкое качество изображения"]
                }
            }
            
            active_learning.process_feedback(feedback_data)
            print(f"   ✅ Оценка качества отправлена: {quality_rating:.2%}")
    
    # Тест 3: Статистика и анализ
    print("\n\n🔬 ТЕСТ 3: Статистика и анализ")
    print("-" * 80)
    
    stats = active_learning.get_learning_statistics()
    feedback_stats = stats['feedback_statistics']
    
    print("\n📊 Статистика обратной связи:")
    print(f"   Всего feedback записей: {feedback_stats['total_feedback']}")
    print(f"   Исправлений: {feedback_stats['corrections_count']}")
    print(f"   Применено: {feedback_stats['applied_corrections']}")
    print(f"   Ожидают применения: {feedback_stats['pending_corrections']}")
    print(f"   Размер базы исправлений: {stats['corrections_db_size']}")
    
    # Анализ паттернов
    print("\n🔍 Анализ паттернов:")
    analysis = active_learning.analyze_feedback_patterns()
    
    if analysis['common_errors']:
        print(f"   Частые ошибки: {len(analysis['common_errors'])}")
        for i, error in enumerate(analysis['common_errors'][:3], 1):
            print(f"      {i}. '{error['original']}' -> '{error['corrected']}' "
                  f"({error['count']} раз)")
    
    if analysis['recommendations']:
        print("\n💡 Рекомендации:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    # Тест 4: Автоматическое обновление
    print("\n\n🔬 ТЕСТ 4: Автоматическое обновление базы исправлений")
    print("-" * 80)
    
    candidates = active_learning.feedback_collector.get_unapplied_corrections(
        min_confidence=0.7,
        min_occurrences=1  # Для теста снижаем порог
    )
    
    print(f"\n📋 Кандидаты для автоматического добавления: {len(candidates)}")
    
    if candidates:
        print("\nПримеры кандидатов:")
        for i, candidate in enumerate(candidates[:3], 1):
            print(f"   {i}. '{candidate['original']}' -> '{candidate['corrected']}'")
            print(f"      Встречается: {candidate['occurrences']} раз")
            print(f"      Уверенность: {candidate['avg_confidence']:.2%}")
        
        print("\n🔄 Применение автоматического обновления...")
        active_learning._auto_update_corrections()
        
        # Проверка результата
        new_stats = active_learning.get_learning_statistics()
        print(f"   ✅ Размер базы исправлений: {new_stats['corrections_db_size']}")
    else:
        print("   ℹ️  Кандидаты не найдены (нужно больше feedback данных)")
    
    # Тест 5: Экспорт данных
    print("\n\n🔬 ТЕСТ 5: Экспорт данных для обучения")
    print("-" * 80)
    
    export_path = active_learning.export_training_data()
    print(f"✅ Данные экспортированы: {export_path}")
    
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ АКТИВНОГО ОБУЧЕНИЯ ЗАВЕРШЕНО")
    print("=" * 80)
    print("\n💡 Система готова к использованию!")
    print("   - Feedback автоматически собирается при обработке")
    print("   - База исправлений обновляется автоматически")
    print("   - Используйте analyze_feedback.py для анализа")


def main():
    """Основная функция"""
    test_active_learning()


if __name__ == "__main__":
    main()
