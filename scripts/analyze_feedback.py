#!/usr/bin/env python3
"""Скрипт для анализа обратной связи и применения улучшений"""
import sys
from pathlib import Path

# Добавление корня приложения в путь
app_root = Path(__file__).parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from services.active_learning import ActiveLearningSystem
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Основная функция анализа"""
    active_learning = ActiveLearningSystem()
    
    print("=" * 60)
    print("АНАЛИЗ ОБРАТНОЙ СВЯЗИ И АКТИВНОЕ ОБУЧЕНИЕ")
    print("=" * 60)
    
    # Статистика
    print("\n📊 СТАТИСТИКА")
    print("-" * 60)
    stats = active_learning.get_learning_statistics()
    feedback_stats = stats['feedback_statistics']
    
    print(f"Всего feedback записей: {feedback_stats['total_feedback']}")
    print(f"Исправлений: {feedback_stats['corrections_count']}")
    print(f"  - Применено: {feedback_stats['applied_corrections']}")
    print(f"  - Ожидают применения: {feedback_stats['pending_corrections']}")
    print(f"Оценок качества: {feedback_stats['quality_ratings_count']}")
    print(f"Feedback по типам документов: {feedback_stats['document_types_feedback_count']}")
    print(f"Размер базы исправлений: {stats['corrections_db_size']}")
    
    # Анализ паттернов
    print("\n🔍 АНАЛИЗ ПАТТЕРНОВ")
    print("-" * 60)
    analysis = active_learning.analyze_feedback_patterns()
    
    # Частые ошибки
    if analysis['common_errors']:
        print("\nЧастые ошибки:")
        for i, error in enumerate(analysis['common_errors'][:10], 1):
            print(f"  {i}. '{error['original']}' -> '{error['corrected']}' "
                  f"(встречается {error['count']} раз)")
    else:
        print("Частые ошибки не найдены")
    
    # Точность классификации типов документов
    if analysis['document_type_accuracy']:
        print("\nТочность классификации типов документов:")
        for doc_type, stats in analysis['document_type_accuracy'].items():
            accuracy_pct = stats['accuracy'] * 100
            print(f"  {doc_type}: {accuracy_pct:.1f}% "
                  f"({stats['correct']}/{stats['total']})")
    
    # Рекомендации
    if analysis['recommendations']:
        print("\n💡 РЕКОМЕНДАЦИИ")
        print("-" * 60)
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Автоматическое обновление
    print("\n🔄 АВТОМАТИЧЕСКОЕ ОБНОВЛЕНИЕ")
    print("-" * 60)
    
    candidates = active_learning.feedback_collector.get_unapplied_corrections(
        min_confidence=0.7,
        min_occurrences=2
    )
    
    if candidates:
        print(f"Найдено {len(candidates)} кандидатов для автоматического добавления:")
        for i, candidate in enumerate(candidates[:5], 1):
            print(f"  {i}. '{candidate['original']}' -> '{candidate['corrected']}' "
                  f"(встречается {candidate['occurrences']} раз, "
                  f"уверенность: {candidate['avg_confidence']:.2f})")
        
        # Применение автоматического обновления
        print("\nПрименение автоматического обновления...")
        active_learning._auto_update_corrections()
        print("✅ Автоматическое обновление выполнено")
    else:
        print("Кандидаты для автоматического обновления не найдены")
    
    # Экспорт данных для обучения
    print("\n💾 ЭКСПОРТ ДАННЫХ")
    print("-" * 60)
    export_path = active_learning.export_training_data()
    print(f"Данные для обучения экспортированы: {export_path}")
    
    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 60)


if __name__ == "__main__":
    main()
