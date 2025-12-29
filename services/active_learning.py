"""Система активного обучения"""
import logging
from typing import Dict, List, Optional, Any
from collections import Counter
from datetime import datetime
import json
import os
import sys
from pathlib import Path

# Добавление корня приложения в путь
app_root = Path(__file__).parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from services.feedback_collector import FeedbackCollector
from core.correctors import AutoCorrectionSystem
from config.settings import settings

logger = logging.getLogger(__name__)


class ActiveLearningSystem:
    """Система активного обучения на основе обратной связи"""
    
    def __init__(self):
        """Инициализация системы активного обучения"""
        self.feedback_collector = FeedbackCollector()
        self.corrector = AutoCorrectionSystem()
        self.auto_update_enabled = True
        self.min_occurrences_for_auto_update = 2
        self.min_confidence_for_auto_update = 0.7
    
    def process_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка обратной связи и автоматическое обновление
        
        Args:
            feedback_data: Данные обратной связи
            
        Returns:
            Результат обработки
        """
        results = {
            'corrections_added': [],
            'corrections_updated': [],
            'document_types_learned': [],
            'quality_improvements': []
        }
        
        # Обработка исправлений
        if 'correction' in feedback_data:
            correction_feedback = feedback_data['correction']
            feedback_id = self.feedback_collector.add_correction_feedback(
                original=correction_feedback.get('original'),
                corrected=correction_feedback.get('corrected'),
                document_id=correction_feedback.get('document_id'),
                context=correction_feedback.get('context'),
                user_id=correction_feedback.get('user_id'),
                confidence=correction_feedback.get('confidence', 1.0)
            )
            
            # Автоматическое обновление базы исправлений
            if self.auto_update_enabled:
                self._auto_update_corrections()
                results['corrections_added'].append(feedback_id)
        
        # Обработка оценки качества
        if 'quality' in feedback_data:
            quality_feedback = feedback_data['quality']
            self.feedback_collector.add_quality_rating(
                document_id=quality_feedback.get('document_id'),
                rating=quality_feedback.get('rating'),
                issues=quality_feedback.get('issues'),
                user_id=quality_feedback.get('user_id')
            )
        
        # Обработка типа документа
        if 'document_type' in feedback_data:
            type_feedback = feedback_data['document_type']
            self.feedback_collector.add_document_type_feedback(
                document_id=type_feedback.get('document_id'),
                predicted_type=type_feedback.get('predicted_type'),
                actual_type=type_feedback.get('actual_type'),
                user_id=type_feedback.get('user_id')
            )
            results['document_types_learned'].append(type_feedback.get('actual_type'))
        
        return results
    
    def _auto_update_corrections(self):
        """Автоматическое обновление базы исправлений"""
        # Получение кандидатов для автоматического добавления
        candidates = self.feedback_collector.get_unapplied_corrections(
            min_confidence=self.min_confidence_for_auto_update,
            min_occurrences=self.min_occurrences_for_auto_update
        )
        
        if not candidates:
            return
        
        logger.info(f"Найдено {len(candidates)} кандидатов для автоматического обновления")
        
        # Добавление в базу исправлений
        added_count = 0
        feedback_ids_to_mark = []
        
        for candidate in candidates:
            # Проверка, не существует ли уже такое исправление
            if candidate['original'] not in self.corrector.corrections_db:
                self.corrector.add_correction(
                    candidate['original'],
                    candidate['corrected'],
                    confirm=False
                )
                added_count += 1
                feedback_ids_to_mark.extend(candidate['feedback_ids'])
                logger.info(f"Автоматически добавлено исправление: '{candidate['original']}' -> '{candidate['corrected']}' "
                           f"(встречается {candidate['occurrences']} раз, уверенность: {candidate['avg_confidence']:.2f})")
        
        # Отметка как примененных
        if feedback_ids_to_mark:
            self.feedback_collector.mark_corrections_applied(feedback_ids_to_mark)
            logger.info(f"Автоматически обновлено {added_count} исправлений в базе")
    
    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """
        Анализ паттернов в обратной связи
        
        Returns:
            Словарь с анализом паттернов
        """
        analysis = {
            'common_errors': [],
            'document_type_accuracy': {},
            'quality_trends': [],
            'recommendations': []
        }
        
        # Анализ частых ошибок
        corrections = self.feedback_collector.feedback_data.get('corrections', [])
        if corrections:
            error_counter = Counter()
            for correction in corrections:
                if not correction.get('applied', False):
                    error_counter[(correction['original'], correction['corrected'])] += 1
            
            analysis['common_errors'] = [
                {
                    'original': orig,
                    'corrected': corr,
                    'count': count
                }
                for (orig, corr), count in error_counter.most_common(10)
            ]
        
        # Анализ точности классификации типов документов
        doc_type_feedback = self.feedback_collector.feedback_data.get('document_types', [])
        if doc_type_feedback:
            type_accuracy = {}
            for feedback in doc_type_feedback:
                predicted = feedback.get('predicted_type', 'unknown')
                if predicted not in type_accuracy:
                    type_accuracy[predicted] = {'correct': 0, 'total': 0}
                
                type_accuracy[predicted]['total'] += 1
                if feedback.get('correct', False):
                    type_accuracy[predicted]['correct'] += 1
            
            for doc_type, stats in type_accuracy.items():
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
                analysis['document_type_accuracy'][doc_type] = {
                    'accuracy': accuracy,
                    'correct': stats['correct'],
                    'total': stats['total']
                }
        
        # Рекомендации
        if analysis['common_errors']:
            analysis['recommendations'].append(
                f"Найдено {len(analysis['common_errors'])} частых ошибок. "
                "Рекомендуется добавить их в базу автозамен."
            )
        
        low_accuracy_types = [
            doc_type for doc_type, stats in analysis['document_type_accuracy'].items()
            if stats['accuracy'] < 0.7 and stats['total'] >= 5
        ]
        if low_accuracy_types:
            analysis['recommendations'].append(
                f"Низкая точность классификации для типов: {', '.join(low_accuracy_types)}. "
                "Рекомендуется дообучение модели."
            )
        
        return analysis
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Получение статистики обучения"""
        stats = self.feedback_collector.get_statistics()
        analysis = self.analyze_feedback_patterns()
        
        return {
            'feedback_statistics': stats,
            'pattern_analysis': analysis,
            'auto_update_enabled': self.auto_update_enabled,
            'corrections_db_size': len(self.corrector.corrections_db)
        }
    
    def export_training_data(self, output_path: Optional[str] = None) -> str:
        """
        Экспорт данных для обучения моделей
        
        Args:
            output_path: Путь для сохранения
            
        Returns:
            Путь к экспортированному файлу
        """
        if output_path is None:
            output_path = os.path.join(settings.DATA_DIR, "training_data_export.json")
        
        training_data = {
            'corrections': [
                {
                    'original': c['original'],
                    'corrected': c['corrected'],
                    'context': c.get('context'),
                    'confidence': c.get('confidence', 1.0)
                }
                for c in self.feedback_collector.feedback_data.get('corrections', [])
                if not c.get('applied', False)
            ],
            'document_types': self.feedback_collector.feedback_data.get('document_types', []),
            'quality_ratings': self.feedback_collector.feedback_data.get('quality_ratings', []),
            'export_timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Данные для обучения экспортированы: {output_path}")
        return output_path
