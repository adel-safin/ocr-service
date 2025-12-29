"""Генератор обратной связи для активного обучения"""
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeedbackGenerator:
    """Генератор обратной связи для операторов"""
    
    def __init__(self):
        """Инициализация генератора обратной связи"""
        pass
    
    def generate_feedback(self, processing_result: Dict) -> Dict[str, any]:
        """
        Генерация обратной связи на основе результатов обработки
        
        Args:
            processing_result: Результаты обработки документа
            
        Returns:
            Словарь с рекомендациями и предупреждениями
        """
        feedback = {
            'warnings': [],
            'suggestions': [],
            'errors': [],
            'confidence_score': processing_result.get('quality_report', {}).get('overall_quality', 0.0)
        }
        
        # Проверка качества
        quality_report = processing_result.get('quality_report', {})
        if quality_report.get('needs_review'):
            feedback['warnings'].append({
                'type': 'quality',
                'message': 'Документ требует ручной проверки из-за низкого качества',
                'severity': 'medium'
            })
        
        # Проверка критических полей
        critical_fields = processing_result.get('extracted_data', {}).get('critical_fields', {})
        invalid_fields = [name for name, data in critical_fields.items() if not data.get('valid', False)]
        
        if invalid_fields:
            feedback['errors'].append({
                'type': 'validation',
                'message': f'Невалидные поля: {", ".join(invalid_fields)}',
                'fields': invalid_fields,
                'severity': 'high'
            })
        
        # Предложения по исправлениям
        corrections = processing_result.get('corrections_applied', [])
        if corrections:
            feedback['suggestions'].append({
                'type': 'corrections',
                'message': f'Применено {len(corrections)} автоматических исправлений',
                'count': len(corrections)
            })
        
        # Новые предложения исправлений
        new_suggestions = processing_result.get('new_corrections_suggested', [])
        if new_suggestions:
            feedback['suggestions'].append({
                'type': 'new_corrections',
                'message': f'Найдено {len(new_suggestions)} новых паттернов для автозамены',
                'suggestions': new_suggestions
            })
        
        return feedback
